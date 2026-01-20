from PIL import Image
import lpips
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import clip
from skimage.metrics import structural_similarity as ssim
import os
import json
from bs4 import BeautifulSoup
from collections import Counter
from scipy.stats import mannwhitneyu, ttest_ind, wilcoxon
from collections import defaultdict


import re
import numpy as np
import pandas as pd
from PIL import Image
import torch
import lpips

##########################################################
# 工具：解析 key
##########################################################

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

def compute_group_merge(item_dict, device, merge_threshold=0.8, graph_threshold = 0.6):

    lpips_model = lpips.LPIPS(net='alex').to(device)
    ######################################################
    # Step 1：group by {folder} {file} {name}
    ######################################################

    groups_by_file = {}

    for key, (crop_path, bbox) in item_dict.items():
        file, name = parse_key(key)

        groups_by_file.setdefault(file, {})
        groups_by_file[file].setdefault(name, [])
        groups_by_file[file][name].append({
            "key": key,
            "path": crop_path,
            "bbox": bbox
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
    print(group_keys)
    G = len(group_keys)
    sim_matrix = np.zeros((G, G))
    print(G)
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
                    lpips_model, items_i, items_j, device,
                    weights={"lpips": 0.5, "size": 0.25, "ssim": 0.25}
                )

                if sim >= merge_threshold:
                    merge_list.append(((file_i, name_i), (file_j, name_j)))
                elif graph_threshold < sim < merge_threshold:

                    merge_list.append()

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

def merge_list_to_clusters(n, merge_list):
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        pa, pb = find(a), find(b)
        if pa != pb:
            parent[pb] = pa

    for i, j in merge_list:
        union(i, j)
    clusters = defaultdict(list)
    for i in range(n):
        clusters[find(i)].append(i)

    return list(clusters.values())


def resize_to_same(img1, img2):

    target_w = max(img1.width, img2.width)
    target_h = max(img1.height, img2.height)

    img1_resized = img1.resize((target_w, target_h), Image.Resampling.LANCZOS)
    img2_resized = img2.resize((target_w, target_h), Image.Resampling.LANCZOS)

    return np.array(img1_resized, dtype=np.float32), np.array(img2_resized, dtype=np.float32)


def compute_merge_table(item_dict,  device,
                        merge_threshold=0.8, use_clip=False, use_lpips=True, use_size=True, use_ssim=True):
    keys = list(item_dict.keys())
    crop_paths = [item_dict[k][0] for k in keys]
    bboxes = [item_dict[k][1] for k in keys]
    n = len(crop_paths)
    sim_matrix = np.zeros((n, n))

    weights = {
        "lpips": 0.4 if use_lpips else 0.0,
        "size": 0.3 if use_size else 0.0,
        "ssim": 0.3 if use_ssim else 0.0,
        "clip": 0.0 if use_ssim else 0.0,
    }

    if use_lpips and weights["lpips"] > 0:
        lpips_model = lpips.LPIPS(net='alex').to(device)
        images_rgb = [Image.open(p).convert("RGB") for p in crop_paths]
        target_size = 256

        def resize_lpips(img):
            return img.resize((target_size, target_size), Image.Resampling.LANCZOS)

        images_tensor = []
        for img in images_rgb:
            img = resize_lpips(img)
            img = torch.tensor(np.array(img)).float() / 255.0
            img = img.permute(2, 0, 1).unsqueeze(0).to(device)
            images_tensor.append(img)

        for i in range(n):
            for j in range(i, n):
                with torch.no_grad():
                    dist = lpips_model(images_tensor[i], images_tensor[j]).item()
                s = 1 - dist
                sim_matrix[i, j] += s * weights["lpips"]
                if i != j:
                    sim_matrix[j, i] += s * weights["lpips"]

    # 1. CLIP embedding
    # if use_clip:
    #     images = [preprocess(Image.open(p)).unsqueeze(0).to(device) for p in crop_paths]
    #     with torch.no_grad():
    #         image_features = torch.cat([model.encode_image(img) for img in images])
    #     image_features /= image_features.norm(dim=-1, keepdim=True)
    #     clip_sim = (image_features @ image_features.T).cpu().numpy()
    #     sim_matrix += clip_sim

    # 2. IoU
    if use_size and weights["size"] > 0:
        def size_similarity(box1, box2):
            area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
            area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
            return min(area1, area2) / max(area1, area2)

        for i in range(n):
            for j in range(i, n):
                s = size_similarity(bboxes[i], bboxes[j])
                sim_matrix[i, j] += s * weights["size"]
                if i != j:
                    sim_matrix[j, i] += s * weights["size"]

        # SSIM
    if use_ssim and weights["ssim"] > 0:
        images_pil = [Image.open(p).convert("L") for p in crop_paths]
        for i in range(n):
            for j in range(i, n):
                img1 = images_pil[i]
                img2 = images_pil[j]
                img1_np, img2_np = resize_to_same(img1, img2)
                s = ssim(img1_np, img2_np, data_range=255)
                sim_matrix[i, j] += s * weights["ssim"]
                if i != j:
                    sim_matrix[j, i] += s * weights["ssim"]


    # total_weights = use_clip + use_lpips + use_size + use_ssim
    # if total_weights > 0:
    #     sim_matrix /= total_weights

    # 生成 merge 列表
    merge_list = []
    for i in range(n):
        for j in range(i+1, n):
            if sim_matrix[i, j] >= merge_threshold:
                merge_list.append((i, j))

    sim_matrix = np.round(sim_matrix, 2)
    df = pd.DataFrame(sim_matrix, columns=keys, index=keys)
    df.to_csv("similarity_matrix.csv", encoding="utf-8-sig")

    clusters = merge_list_to_clusters(n, merge_list)
    clusters_keys = [[keys[i] for i in cls] for cls in clusters]
    clusters_paths = [[crop_paths[i] for i in cls] for cls in clusters]
    return sim_matrix, clusters_paths


def mae(img_gt_path, img_gen_path):
    img_gt = Image.open(img_gt_path).convert("RGB")
    img_gen = Image.open(img_gen_path).convert("RGB")

    img_gt_np, img_gen_np = resize_to_same(img_gt, img_gen)

    return float(np.mean(np.abs(img_gt_np - img_gen_np)))


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

def json_has_directed_subgraph(json1, json2):
    G1 = build_digraph_from_json(json1)
    G2 = build_digraph_from_json(json2)

    def choose_big_small(G1, G2):
        # compare edges first
        e1, e2 = G1.number_of_edges(), G2.number_of_edges()
        if e1 > e2:
            return G1, G2
        if e1 < e2:
            return G2, G1
        # if the numbers of edges are same, then compare nodes
        n1, n2 = G1.number_of_nodes(), G2.number_of_nodes()
        if n1 > n2:
            return G1, G2
        else:
            return G2, G1

    def is_subgraph_directed(G_big, G_small):
        def edge_match(e1, e2):
            return e1["direction"] == e2["direction"]

        GM = isomorphism.DiGraphMatcher(
            G_big,
            G_small,
            edge_match=edge_match
        )
        return GM.subgraph_is_isomorphic()

    G_big, G_small = choose_big_small(G1, G2)
    return is_subgraph_directed(G_big, G_small)











def classify_height(height):
    if height > 5000:
        return "long"
    elif height >= 2000:
        return "medium"
    else:
        return "short"

def html_tree_bleu(html_ref: str, html_pred: str):

    def get_1height_subtrees(soup):
        subtrees = []

        def traverse(node):
            if node.name is not None:
                children_tags = [
                    child.name for child in node.children
                    if getattr(child, "name", None) is not None
                ]
                subtrees.append((node.name, tuple(children_tags)))
                for child in node.children:
                    if getattr(child, "name", None) is not None:
                        traverse(child)

        traverse(soup)
        return subtrees

    soup_ref = BeautifulSoup(html_ref, "html.parser")
    soup_pred = BeautifulSoup(html_pred, "html.parser")

    ref_subtrees = get_1height_subtrees(soup_ref)
    pred_subtrees = get_1height_subtrees(soup_pred)

    if not ref_subtrees:
        return 0.0

    ref_counter = Counter(ref_subtrees)
    pred_counter = Counter(pred_subtrees)

    # 计算匹配数：对每个子树取两边出现次数的最小值
    matched = sum(min(ref_counter[subtree], pred_counter.get(subtree, 0)) for subtree in ref_counter)

    tree_bleu_score = matched / len(ref_subtrees)

    return tree_bleu_score



# ====== 读取 JSON 文件 ======
def load_scores(file_path, method_name):
    with open(file_path, 'r') as f:
        data = json.load(f)

    sub = data[method_name]

    keys = sorted(
        [k for k in sub.keys() if re.match(r"^\d+\.html$", k)],
        key=lambda x: int(x.split('.')[0])
    )

    # 取出对应的 118×5 矩阵（跳过每行第一个 average）
    matrix = [sub[k][1:] for k in keys]

    return matrix

def print_multi_score(multi_score):
    _, final_size_score, final_matched_text_score, final_position_score, final_text_color_score, final_clip_score = multi_score
    print()
    print("Block-Match: ", final_size_score)
    print("Text: ", final_matched_text_score)
    print("Position: ", final_position_score)
    print("Color: ", final_text_color_score)
    print("CLIP: ", final_clip_score)
    print("--------------------------------\n")





# if __name__ == "__main__":
#     gt_dir = r"D:/py_code/fyp/VueGen/data"
#     test_dirs = {
#         "gemini_vanilla_prompting": r"D:\py_code\fyp\VueGen\testing\output_baseline",
#         "gemini_component_prompting": r"D:\py_code\fyp\VueGen\testing\output_compare",
#         "gemini_vuegen_prompting_layout": r"D:\py_code\fyp\VueGen\testing\output_vuegen_gemini",
#         "gemini_vuegen_prompting": r"D:\py_code\fyp\VueGen\testing\output_vuegen_gemini_1",
#     }
#
#     # ---- 分类：只统计一次 ----
#     size_dict = {"short": [], "medium": [], "long": []}
#     file_numbers = [f.split(".")[0] for f in os.listdir(gt_dir) if f.endswith(".png")]
#     file_numbers = sorted(file_numbers, key=lambda x: int(x))
#
#     for num in file_numbers:
#         gt_img_path = os.path.join(gt_dir, f"{num}.png")
#         img_gt = Image.open(gt_img_path)
#         h = img_gt.size[1]
#         size_class = classify_height(h)
#         size_dict[size_class].append(num)
#
#     for size, nums in size_dict.items():
#         print(f"{size}: count={len(nums)}, ids={nums}")
#
#     with open(r"D:\py_code\fyp\Design2Code\Design2Code\metrics\res_dict_testset_final_wo_layout.json", "r") as f:
#         res_dict = json.load(f)
#
#     # 按 size 输出
#     for key in res_dict:
#         print(f"\nMethod: {key}")
#         for size, nums in size_dict.items():
#             if not nums:
#                 continue
#             # 将编号转成 json key
#             json_keys = [f"{num}.html" for num in nums if f"{num}.html" in res_dict[key]]
#             if not json_keys:
#                 continue
#             # 提取对应的值
#             values = [res_dict[key][jk] for jk in json_keys]
#             current_res = np.mean(np.array(values), axis=0)
#             print(f"--- {size} ---")
#             print_multi_score(current_res)

    # # ---- 收集每个方法、每个类别指标 ----
    # results = {}
    # for method, pred_dir in test_dirs.items():
    #     results[method] = {size: {"MAE": [], "TreeBLEU": []} for size in size_dict}
    #
    #     for size, nums in size_dict.items():
    #         for num in nums:
    #             pred_img_path = os.path.join(pred_dir, f"{num}.png")
    #             pred_html_path = os.path.join(pred_dir, f"{num}.html")
    #             gt_img_path = os.path.join(gt_dir, f"{num}.png")
    #             gt_html_path = os.path.join(gt_dir, f"{num}.html")
    #
    #             if not (os.path.exists(pred_img_path) and os.path.exists(pred_html_path)):
    #                 continue
    #
    #             mae_score = mae(gt_img_path, pred_img_path)
    #
    #             with open(gt_html_path, "r", encoding="utf-8") as f:
    #                 html_gt = f.read()
    #             with open(pred_html_path, "r", encoding="utf-8") as f:
    #                 html_pred = f.read()
    #             tree_bleu_score = html_tree_bleu(html_gt, html_pred)
    #
    #             results[method][size]["MAE"].append(mae_score)
    #             results[method][size]["TreeBLEU"].append(tree_bleu_score)
    #
    # # ---- 统计平均和显著性 ----
    # baseline_method = "gemini_vanilla_prompting"
    #
    # for method in results:
    #     if method == baseline_method:
    #         continue
    #     print(f"\nMethod: {method} vs baseline ({baseline_method})")
    #
    #     # ---- 分类别显著性 ----
    #     for size in results[method]:
    #         method_mae = results[method][size]["MAE"]
    #         baseline_mae = results[baseline_method][size]["MAE"]
    #         method_bleu = results[method][size]["TreeBLEU"]
    #         baseline_bleu = results[baseline_method][size]["TreeBLEU"]
    #
    #         if len(method_mae) == 0 or len(baseline_mae) == 0:
    #             print(f"  {size}: no data")
    #             continue
    #
    #         mae_avg_method = np.mean(method_mae)
    #         mae_avg_baseline = np.mean(baseline_mae)
    #         bleu_avg_method = np.mean(method_bleu)
    #         bleu_avg_baseline = np.mean(baseline_bleu)
    #
    #         stat_mae, p_mae = mannwhitneyu(method_mae, baseline_mae, alternative='less')      # MAE 越小越好
    #         stat_bleu, p_bleu = mannwhitneyu(method_bleu, baseline_bleu, alternative='greater') # TreeBLEU 越大越好
    #
    #         print(f"  {size}: MAE avg={mae_avg_method:.4f} vs {mae_avg_baseline:.4f}, p={p_mae:.6f}; "
    #               f"TreeBLEU avg={bleu_avg_method:.4f} vs {bleu_avg_baseline:.4f}, p={p_bleu:.6f}")
    #
    #     # ---- 合并所有类别求整体显著性 ----
    #     method_all_mae, baseline_all_mae = [], []
    #     method_all_bleu, baseline_all_bleu = [], []
    #     for size in results[method]:
    #         method_all_mae.extend(results[method][size]["MAE"])
    #         baseline_all_mae.extend(results[baseline_method][size]["MAE"])
    #         method_all_bleu.extend(results[method][size]["TreeBLEU"])
    #         baseline_all_bleu.extend(results[baseline_method][size]["TreeBLEU"])
    #
    #     if len(method_all_mae) > 0 and len(baseline_all_mae) > 0:
    #         mae_avg_method_all = np.mean(method_all_mae)
    #         mae_avg_baseline_all = np.mean(baseline_all_mae)
    #         stat_all_mae, p_all_mae = mannwhitneyu(method_all_mae, baseline_all_mae, alternative='less')
    #
    #         bleu_avg_method_all = np.mean(method_all_bleu)
    #         bleu_avg_baseline_all = np.mean(baseline_all_bleu)
    #         stat_all_bleu, p_all_bleu = mannwhitneyu(method_all_bleu, baseline_all_bleu, alternative='greater')
    #
    #         print(f"  ALL SIZES: MAE avg={mae_avg_method_all:.4f} vs {mae_avg_baseline_all:.4f}, p={p_all_mae:.6f}; "
    #               f"TreeBLEU avg={bleu_avg_method_all:.4f} vs {bleu_avg_baseline_all:.4f}, p={p_all_bleu:.6f}")



# significant test
if __name__ == "__main__":
    gt_dir = r"D:/py_code/fyp/VueGen/data"
    baseline_file = r"D:\py_code\fyp\Design2Code\Design2Code\metrics\baseline.json"
    method_file = r"D:\py_code\fyp\Design2Code\Design2Code\metrics\merged.json"

    # ---- 分类：只统计一次 ----
    size_dict = {"short": [], "medium": [], "long": []}
    file_numbers = [f.split(".")[0] for f in os.listdir(gt_dir) if f.endswith(".png")]
    file_numbers = sorted(file_numbers, key=lambda x: int(x))

    for num in file_numbers:
        gt_img_path = os.path.join(gt_dir, f"{num}.png")
        img_gt = Image.open(gt_img_path)
        h = img_gt.size[1]
        size_class = classify_height(h)
        size_dict[size_class].append(num)

    metric_names = ["BLOCK", "TEXT", "POSITION", "COLOR", "CLIP"]

    # 先加载整个分数字典
    with open(baseline_file, "r") as f:
        baseline_all_dict = json.load(f)["gemini_vanilla_prompting"]
    with open(method_file, "r") as f:
        method_all_dict = json.load(f)["gemini_vuegen_prompting"]

    # 按 size 输出
    for size, nums in size_dict.items():
        if not nums:
            continue
        print(f"\n=== Size: {size} ===")

        baseline_size = []
        method_size = []

        for num in nums:
            key = f"{num}.html"
            if key in baseline_all_dict and key in method_all_dict:
                baseline_size.append(baseline_all_dict[key][1:6])
                method_size.append(method_all_dict[key][1:6])

        if not baseline_size or not method_size:
            print("No matching files found in JSON for this size.")
            continue

        # 转置成每个 metric 一列
        baseline_size = list(map(list, zip(*baseline_size)))
        method_size = list(map(list, zip(*method_size)))
        print(method_size)
        bins = np.linspace(0, 1, 11)
        # Mann-Whitney U 检验

        for i in range(5):
            if size == "long":
                method = method_size[i]
                baseline = baseline_size[i]

                x = range(len(method))  # index

                plt.figure(figsize=(8, 5))

                # 折线
                plt.plot(baseline, color='red', label='Baseline')
                plt.plot(method, color='green', label='VueGen')

                # 每个点
                plt.scatter(range(len(baseline)), baseline, color='red', s=10)
                plt.scatter(range(len(method)), method, color='green', s=10)

                plt.title("Pair Comparison")
                plt.xlabel("Index")
                plt.ylabel(f"{metric_names[i]}")
                plt.legend()
                plt.grid(alpha=0.3)
                plt.show()
            stat, p = wilcoxon(method_size[i], baseline_size[i], alternative='greater')
            print(f"{metric_names[i]}: p = {p:.6f}")


# if __name__ == "__main__":
#     gt_dir = r"D:/py_code/fyp/VueGen/data"
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     for i in range(1,118):
#         json_dir = f"D:/py_code/fyp/VueGen/output_seg/{i}"
#         image_path = os.path.join(image_dir, f"{i}.png")
#         json_path = os.path.join(json_dir, f"{i}_test_modify.json")
#         output_dir = os.path.join(r"D:/py_code/fyp/VueGen/output_seg", "crop", f"{i}_cropped")



    #     #crop_paths, bboxes, masked_path = crop_and_mask_image(image_path, json_path, output_dir, mask_color=(200, 200, 200))
    #     crop_paths = [
    #         os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.lower().endswith(".png")
    #     ]
    #     print(crop_paths)
    #
    #     with open(json_path, "r") as f:
    #         boxes = json.load(f)
    #
    #     crop_paths = []
    #     bboxes = []
    #     for i, item in enumerate(boxes):
    #         bbox = item["bbox_2d"]
    #         label = item.get("label", f"group_{i + 1}")
    #         bboxes.append(bbox)
    #
    #     if crop_paths:
    #         sim_matrix, merge_list = compute_merge_table(crop_paths, bboxes, device=device, merge_threshold=0.7)
    #         print(f"Image {i} sim_matrix:\n{sim_matrix}")
    #         print(f"Image {i}: merge_list = {merge_list}")