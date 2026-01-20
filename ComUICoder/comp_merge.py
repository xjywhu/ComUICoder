
from collections import defaultdict
from skimage.metrics import structural_similarity as ssim
import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
import lpips
import cv2
import json

def parse_key(key):
    # key 格式：folderIdx_fileIdx_name_idx_idx
    key = os.path.splitext(key)[0]
    parts = key.split("_")
    file = parts[0]
    tail = parts[1:]

    # There may exist one or more {_idx}
    idx_count = 0
    for p in reversed(tail):
        if p.isdigit():
            idx_count += 1
        else:
            break
    if idx_count > 0:
        name_parts = tail[:-idx_count]
    else:
        name_parts = tail
    name = "_".join(name_parts) if name_parts else ""
    return file, name



##########################################################
# 组与组之间 LPIPS + size 的相似度
##########################################################

class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        rx, ry = self.find(x), self.find(y)
        if rx != ry:
            self.parent[ry] = rx

def group_similarity(lpips_model, groupA, groupB, use_ssim=True,
                     weights={"lpips": 0.5, "size": 0.25, "ssim": 0.25}):

    ########################################################
    # 1. LPIPS 平均
    ########################################################
    lpips_scores = []
    for a in groupA:
        for b in groupB:
            with torch.no_grad():
                dist = lpips_model(a["img"], b["img"]).item()
            lpips_scores.append(1 - dist)

    lpips_mean = float(np.mean(lpips_scores)) if lpips_scores else 0
    ########################################################
    # 2. bbox area similarity 平均
    ########################################################
    def size_similarity(box1, box2, aspect_tol=0.3):
        w1 = box1[2] - box1[0]
        h1 = box1[3] - box1[1]
        w2 = box2[2] - box2[0]
        h2 = box2[3] - box2[1]

        if w1 <= 0 or h1 <= 0 or w2 <= 0 or h2 <= 0:
            return 0.0

        is_vert_1 = h1 > w1
        is_vert_2 = h2 > w2
        if is_vert_1 != is_vert_2:
            return 0.0

        area1 = w1 * h1
        area2 = w2 * h2
        area_sim = min(area1, area2) / max(area1, area2)

        ar1 = w1 / h1
        ar2 = w2 / h2
        ar_sim = min(ar1, ar2) / max(ar1, ar2)
        return area_sim * ar_sim

    size_scores = []
    for a in groupA:
        for b in groupB:
            s = size_similarity(a["bbox"], b["bbox"])
            size_scores.append(s)
    size_mean = float(np.mean(size_scores)) if size_scores else 0

    ########################################################
    # 3. SSIM 平均 (可选)
    ########################################################
    ssim_mean = 0.0
    if use_ssim and weights.get("ssim", 0) > 0:
        ssim_scores = []
        for a in groupA:
            for b in groupB:
                s = ssim(a["img_gray"], b["img_gray"], data_range=255)
                ssim_scores.append(s)
        ssim_mean = float(np.mean(ssim_scores)) if ssim_scores else 0
    ########################################################
    # 4. 综合相似度加权
    ########################################################
    sim = (
        weights.get("lpips", 0) * lpips_mean +
        weights.get("size", 0)  * size_mean  +
        weights.get("ssim", 0)  * ssim_mean
    )
    return sim

def split_group_by_consistency(items, lpips_model, sim_threshold=0.5):
    """
    items: 原始 group（同 file + 同 name）
    return: List[List[item]]  拆分后的子组
    """
    n = len(items)
    if n <= 1:
        return [items]

    uf = UnionFind(n)

    for i in range(n):
        for j in range(i + 1, n):
            sim = group_similarity(lpips_model, [items[i]], [items[j]])
            if sim >= sim_threshold:
                uf.union(i, j)

    clusters = {}
    for i in range(n):
        r = uf.find(i)
        clusters.setdefault(r, []).append(items[i])

    return list(clusters.values())

##########################################################
# 主函数：组 → 组 相似度 + 聚合
##########################################################

def compute_group_merge(item_dict, device, merge_threshold=0.8, graph_threshold=0.6):

    lpips_model = lpips.LPIPS(net='alex').to(device)
    ######################################################
    # Step 1：group by {folder} {file} {name}
    ######################################################

    groups_by_file = {}

    for key, (crop_path, bbox, json_path) in item_dict.items():
        file, name = parse_key(key)

        groups_by_file.setdefault(file, {})
        groups_by_file[file].setdefault(name, [])
        groups_by_file[file][name].append({
            "key": key,
            "path": crop_path,
            "bbox": bbox,
            "json_path": json_path
        })

    for _, name_groups in groups_by_file.items():
        for name, items in name_groups.items():
            for it in items:
                img = Image.open(it["path"]).convert("RGB").resize((256, 256))
                t = torch.tensor(np.array(img)).float() / 255.0
                t = t.permute(2, 0, 1).unsqueeze(0).to(device)
                it["img"] = t
                img_gray = img.convert("L")
                it["img_gray"] = np.array(img_gray)  # uint8, H×W

    # split intra-class
    new_groups_by_file = {}
    for file, name_groups in groups_by_file.items():
        new_groups_by_file[file] = {}
        for name, items in name_groups.items():
            if len(items) == 1:
                new_groups_by_file[file][name] = items
                continue
            sub_groups = split_group_by_consistency(
                items,
                lpips_model,
                sim_threshold=0.5
            )
            if len(sub_groups) == 1:
                new_groups_by_file[file][name] = sub_groups[0]
            else:
                for idx, sub in enumerate(sub_groups):
                    new_name = f"{name}_{idx}"
                    new_groups_by_file[file][new_name] = sub
    groups_by_file = new_groups_by_file

    ######################################################
    # Step 3：组与组之间计算平均相似度（跨 file）
    ######################################################

    # 全局组列表（顺序固定，便于 df）
    group_keys = []  # [(file, name)]
    group_items = []  # [items list]

    for file, name_groups in groups_by_file.items():
        for name, items in name_groups.items():
            group_keys.append((file, name))
            group_items.append(items)
    G = len(group_keys)
    sim_matrix = np.zeros((G, G))
    merge_list=[]
    for i in range(G):
        file_i, name_i = group_keys[i]
        items_i = group_items[i]

        for j in range(i, G):
            file_j, name_j = group_keys[j]
            items_j = group_items[j]

            # 同一个 file 的组不比较
            if file_i == file_j:
                sim = 0.0
            else:
                sim = group_similarity(
                    lpips_model,
                    items_i,
                    items_j,
                    device,
                    weights={"lpips": 0.5, "size": 0.25, "ssim": 0.25}
                )

                if sim >= merge_threshold:
                    merge_list.append(((file_i, name_i), (file_j, name_j)))
                elif graph_threshold < sim < merge_threshold:
                    len_i = len(items_i)
                    len_j = len(items_j)
                    can_merge = False
                    if len_i > 1 and len_j > 1:
                        continue
                    if len_i == 1 and len_j > 1:
                        for k in range(len_j):
                            if is_subgraph_directed(items_i[0], items_j[k]):
                                can_merge = True
                                break
                    elif len_j == 1 and len_i > 1:
                        for k in range(len_i):
                            if is_subgraph_directed(items_i[k], items_j[0]):
                                can_merge = True
                                break
                    elif len_i == 1 and len_j == 1:
                        if is_subgraph_directed(items_i[0], items_j[0]):
                            can_merge = True
                    if can_merge:
                        print((file_i, name_i), (file_j, name_j))
                        merge_list.append(((file_i, name_i), (file_j, name_j)))


            sim_matrix[i, j] = sim
            sim_matrix[j, i] = sim

    group_labels = [f"{file}_{name}" for file, name in group_keys]
    df = pd.DataFrame(
        np.round(sim_matrix, 3),
        index=group_labels,
        columns=group_labels
    )

    df.to_csv("similarity_matrix.csv", encoding="utf-8-sig")

    ######################################################
    # Step 5：合并组
    ######################################################

    gid = {g: i for i, g in enumerate(group_keys)}
    uf = UnionFind(G)

    for g1, g2 in merge_list:
        uf.union(gid[g1], gid[g2])

    clusters = {}
    for i, g in enumerate(group_keys):
        root = uf.find(i)
        clusters.setdefault(root, []).append(g)

    ######################################################
    # Step 6：输出 clusters_paths
    ######################################################

    clusters_paths = []
    for group_list in clusters.values():
        merged_items = []
        for file, name in group_list:
            merged_items.extend(groups_by_file[file][name])
        clusters_paths.append([it["path"] for it in merged_items])

    return df, clusters_paths


######################################################
# tag
######################################################


import networkx as nx
from networkx.algorithms import isomorphism


def build_digraph_from_json(data):
    G = nx.DiGraph()
    connections = data["connections"]

    for u, neighs in connections.items():
        for d, v in neighs.items():
            if v is None:
                continue
            u = str(u)
            v = str(v)
            G.add_edge(u, v, direction=d)

    return G


def extract_node_info_from_json(data):
    node_info = {}

    def parse_bbox(pos):
        return (
            int(pos["column_min"]),
            int(pos["row_min"]),
            int(pos["column_max"]),
            int(pos["row_max"])
        )

    # compos
    for comp in data.get("compos", []):
        nid = str(comp["id"])
        bbox = parse_bbox(comp["position"])
        node_info[nid] = {
            "class": comp["class"],   # Block
            "bbox": bbox,
            "text": None
        }

    # texts
    for text in data.get("texts", []):
        nid = str(text["id"])
        bbox = parse_bbox(text["position"])
        node_info[nid] = {
            "class": text["class"],   # Text
            "bbox": bbox,
            "text": text.get("text_content", "")
        }

    return node_info

def is_subgraph_directed(data1, data2):
    def choose_big_small(data1, data2):
        with open(data1["json_path"], "r") as f:
            json1 = json.load(f)
        with open(data2["json_path"], "r") as f:
            json2 = json.load(f)
        img1 = data1["path"]
        img2 = data2["path"]
        G1 = build_digraph_from_json(json1)
        G2 = build_digraph_from_json(json2)
        if G1.number_of_edges() > G2.number_of_edges():
            return G1, json1, img1, G2, json2, img2
        elif G1.number_of_edges() < G2.number_of_edges():
            return G2, json2, img2, G1, json1, img1
        else:
            if G1.number_of_nodes() >= G2.number_of_nodes():
                return G1, json1, img1, G2, json2, img2
            else:
                return G2, json2, img2, G1, json1, img1

    def edge_match(e1, e2):
        return e1["direction"] == e2["direction"]

    def mapping_geometry( small_to_big, small_node_info, big_node_info,
                          small_img_path, big_img_path, ratio_thresh=0.5, size_thresh=0.8, align_thresh=30):
        small_img = cv2.imread(small_img_path)
        big_img = cv2.imread(big_img_path)

        H1, W1 = small_img.shape[:2]
        H2, W2 = big_img.shape[:2]

        img_area1 = H1 * W1
        img_area2 = H2 * W2
        for s_node, b_node in small_to_big.items():
            b1 = small_node_info[s_node]["bbox"]
            b2 = big_node_info[b_node]["bbox"]
            x1_min, y1_min, x1_max, y1_max = b1
            x2_min, y2_min, x2_max, y2_max = b2
            area1 = (x1_max - x1_min) * (y1_max - y1_min)
            area2 = (x2_max - x2_min) * (y2_max - y2_min)
            ratio1 = area1 / img_area1
            ratio2 = area2 / img_area2
            ratio_sim = min(ratio1, ratio2) / max(ratio1, ratio2)

            if ratio_sim < ratio_thresh:
                return False

            # if not (abs(x1_min - x2_min) <= align_thresh or abs(y1_min - y2_min) <= align_thresh or
            #     abs(x1_max - x2_max) <= align_thresh or abs(y1_max - y2_max) <= align_thresh):
            #     return False

        return True

    G_big, big_data, big_image, G_small, small_data, small_image = choose_big_small(data1, data2)

    GM = isomorphism.DiGraphMatcher(
        G_big,
        G_small,
        edge_match=edge_match
    )
    for mapping in GM.subgraph_isomorphisms_iter():
        small_to_big = {v: k for k, v in mapping.items()}
        big_node_info = extract_node_info_from_json(big_data)
        small_node_info = extract_node_info_from_json(small_data)
        if mapping_geometry(small_to_big,small_node_info,big_node_info, small_image, big_image):
            print("✅ 找到结构 + 几何一致的匹配")
            print(small_to_big)
            return True

    return False




from itertools import combinations

JSON_DIR = r"D:\py_code\fyp\UIED\data\output\merge"

def load_all_jsons(json_dir):
    json_files = [
        os.path.join(json_dir, f)
        for f in os.listdir(json_dir)
        if f.endswith(".json")
    ]

    data_list = []
    for path in json_files:
        with open(path, "r") as f:
            data = json.load(f)
        data_list.append((os.path.basename(path), data))

    return data_list


if __name__ == "__main__":
    all_data = load_all_jsons(JSON_DIR)

    print(f"📦 共加载 {len(all_data)} 个 JSON")

    for (name1, data1), (name2, data2) in combinations(all_data, 2):
        print("=" * 60)
        print(f"🔍 Comparing {name1}  vs  {name2}")

        matched = is_subgraph_directed(data1, data2)
        print("✅ Match:" if matched else "❌ No Match")


