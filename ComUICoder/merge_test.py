# import cv2
# img = cv2.imread(r"D:\py_code\fyp\VueGen\data\4.png")  # 读取图片（BGR格式）
# h, w, _ = img.shape
# print(f"宽度: {w}, 高度: {h}")
from collections import defaultdict
import shutil
import cv2
import os
import json
from glob import glob
import numpy as np
import re

def draw_boxes_from_json(json_path, img_path, output_path):
    # Load image
    if not os.path.exists(img_path):
        print(f"Error: Image not found: {img_path}")
        return
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error: Failed to read image: {img_path}")
        return

    # Load JSON
    if not os.path.exists(json_path):
        print(f"Error: JSON file not found: {json_path}")
        return
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Draw bounding boxes
    for item in data:
        bbox = item.get("bbox_2d", None)
        label = item.get("label", "")
        if bbox and len(bbox) == 4:
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.putText(img, label, (x1, max(y1 - 10, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, img)
    print(f"Saved image with boxes: {output_path}")


def compute_iou(box1, box2):
    """
    Compute intersection-over-union ratio for two boxes.
    box: [x1, y1, x2, y2]
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    inter_area = inter_w * inter_h

    box1_area = max(0, box1[2]-box1[0]) * max(0, box1[3]-box1[1])
    box2_area = max(0, box2[2]-box2[0]) * max(0, box2[3]-box2[1])

    union_area = box1_area + box2_area - inter_area
    return inter_area, box1_area, box2_area, union_area


def filter_small_boxes(json_data, min_width=20, min_height=20):
    """
    Filter out small boxes likely to be text/noise.
    """
    filtered = []
    for item in json_data:
        x1, y1, x2, y2 = item
        width = x2 - x1
        height = y2 - y1
        if width >= min_width and height >= min_height:
            filtered.append(item)
    return filtered


def merge_split(heights_data,main_data):
    main_data_cleaned = []

    # Step 1. Build the list of upper and lower boxes for each split line
    line_pairs = []  # [(h, upper_idx, lower_idx, vertical_gap), ...]

    for h in heights_data["split_lines"]:
        upper_boxes = []
        lower_boxes = []

        for i, box in enumerate(main_data):
            x1, y1, x2, y2 = box["bbox_2d"]

            # If the bottom edge is close to the split line → upper box
            if abs(y2 - h) <= y_tolerance:
                upper_boxes.append((i, box))
            # If the top edge is close to the split line → lower box
            elif abs(y1 - h) <= y_tolerance:
                lower_boxes.append((i, box))

        # Step 2. Match upper and lower box pairs for each line
        for i_u, upper in upper_boxes:
            x1_u, y1_u, x2_u, y2_u = upper["bbox_2d"]

            candidates = []
            for i_l, lower in lower_boxes:
                if i_u == i_l:
                    continue
                x1_l, y1_l, x2_l, y2_l = lower["bbox_2d"]

                # Rough horizontal alignment
                if abs(x1_l - x1_u) <= x_tolerance or abs(x2_l - x2_u) <= x_tolerance:
                    gap = abs(y1_l - y2_u)
                    candidates.append((gap, i_u, i_l))

            if candidates:
                # Take the pair with the smallest gap
                candidates.sort(key=lambda x: x[0])
                gap, i_u_best, i_l_best = candidates[0]
                line_pairs.append((h, i_u_best, i_l_best, gap))

    # Step 3. Determine merge priority globally
    # If the same box appears in multiple pairs, keep the one with the smallest gap
    best_pairs = {}
    for h, i_u, i_l, gap in line_pairs:
        for idx in [i_u, i_l]:
            if idx not in best_pairs or gap < best_pairs[idx][3]:
                best_pairs[idx] = (h, i_u, i_l, gap)

    merged_indices = set()
    for h, i_u, i_l, gap in best_pairs.values():
        if i_u in merged_indices or i_l in merged_indices:
            continue

        box_u = main_data[i_u]
        box_l = main_data[i_l]

        x1_u, y1_u, x2_u, y2_u = box_u["bbox_2d"]
        x1_l, y1_l, x2_l, y2_l = box_l["bbox_2d"]

        new_box = {
            "bbox_2d": [
                min(x1_u, x1_l),
                min(y1_u, y1_l),
                max(x2_u, x2_l),
                max(y2_u, y2_l)
            ],
            "label": f"merged_{box_u.get('label', '')}_{box_l.get('label', '')}"
        }
        main_data_cleaned.append(new_box)
        merged_indices.add(i_u)
        merged_indices.add(i_l)

    # Step 4. Add back boxes that were not merged
    for idx, box in enumerate(main_data):
        if idx not in merged_indices:
            main_data_cleaned.append(box)

    return main_data_cleaned


def component_add(img, ref_boxes, main_data_merged):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(img_gray, 100, 200)  # 计算全图边缘

    added_count = 0  # 计数器

    for ref_coords in ref_boxes:
        x1_r, y1_r, x2_r, y2_r = ref_coords
        area = (x2_r - x1_r) * (y2_r - y1_r)
        if area < min_area:
            continue

        # check if this ref box is mostly inside any existing main box
        overlap_with_any = False
        for main_box in main_data_merged:
            main_coords = main_box["bbox_2d"]
            inter_area, _, _, _ = compute_iou(main_coords, ref_coords)
            if inter_area / area > 0:
                overlap_with_any = True
                break
        if overlap_with_any:
            continue

        # 在原图上检测该区域是否有明显的边界线
        patch = edges[y1_r:y2_r, x1_r:x2_r]
        if patch.size == 0:
            continue

        h, w = patch.shape
        top_edge = np.mean(patch[:max(2, int(0.02 * h)), :]) / 255.0
        bottom_edge = np.mean(patch[-max(2, int(0.02 * h)):, :]) / 255.0
        left_edge = np.mean(patch[:, :max(2, int(0.02 * w))]) / 255.0
        right_edge = np.mean(patch[:, -max(2, int(0.02 * w)):]) / 255.0

        edge_threshold = 0.1
        strong_edges = sum(e > edge_threshold for e in [top_edge, bottom_edge, left_edge, right_edge])

        if strong_edges >= 2:
            if added_count == 0:
                new_label = "added_from_ref"
            else:
                new_label = f"added_from_ref_{added_count}"
            added_count += 1

            main_data_merged.append({
                "bbox_2d": [x1_r, y1_r, x2_r, y2_r],
                "label": new_label
            })

    return main_data_merged



def data_cleaning(img, main_data):
    h, w, _ = img.shape
    filtered_main_data = []
    for main_box in main_data:
        x1_m, y1_m, x2_m, y2_m = main_box["bbox_2d"]
        if x2_m > w:
            x2_m = w
        if y2_m > h:
            y2_m = h
        width = x2_m - x1_m
        height = y2_m - y1_m
        if width <= 100 or height >= 6 * width:
            continue
        filtered_main_data.append(main_box)
    return filtered_main_data

def check_contained_boxes(main_data):
    adjusted_data = []
    n = len(main_data)
    for i in range(n):
        main_data[i]["bbox_2d"] = list(main_data[i]["bbox_2d"])
        x1_i, y1_i, x2_i, y2_i = main_data[i]["bbox_2d"]

        a_coords = x1_i, y1_i, x2_i, y2_i

        is_contained = False
        for j in range(n):
            if i == j:
                continue
            main_data[j]["bbox_2d"] = list(main_data[j]["bbox_2d"])
            x1_j, y1_j, x2_j, y2_j = main_data[j]["bbox_2d"]
            b_coords = x1_j, y1_j, x2_j, y2_j

            # vertical adjust
            inter_area, main_area, ref_area, _ = compute_iou(a_coords, b_coords)
            if inter_area / main_area == 1:
                is_contained = True
                break

        if not is_contained:
            adjusted_data.append(main_data[i])

    return adjusted_data



def data_overlap_adjust(main_data):
    # check overlap
    n = len(main_data)
    for i in range(n):
        main_data[i]["bbox_2d"] = list(main_data[i]["bbox_2d"])
        x1_i, y1_i, x2_i, y2_i = main_data[i]["bbox_2d"]
        a_coords = x1_i, y1_i, x2_i, y2_i
        for j in range(n):
            if i == j:
                continue
            main_data[j]["bbox_2d"] = list(main_data[j]["bbox_2d"])
            x1_j, y1_j, x2_j, y2_j = main_data[j]["bbox_2d"]
            b_coords = x1_j, y1_j, x2_j, y2_j
            # vertical adjust
            inter_area, main_area, ref_area, _ = compute_iou(a_coords, b_coords)
            if inter_area > 0:
                print("wow")

                vert_overlap = min(abs(y2_i - y1_j), abs(y1_i - y2_j))
                hori_overlap = min(abs(x2_i - x1_j), abs(x1_i - x2_j))

                if vert_overlap > 0 and hori_overlap > 0:

                    if vert_overlap >= hori_overlap:
                        if x1_i < x1_j:
                            main_data[i]["bbox_2d"][2] = min(x2_i, x1_j)
                        else:
                            main_data[j]["bbox_2d"][2] = min(x2_j, x1_i)

                    else:
                        if y1_i < y1_j:
                            main_data[i]["bbox_2d"][3] = min(y2_i, y1_j)
                        else:
                            main_data[j]["bbox_2d"][3] = min(y2_j, y1_i)


                    print(f"Adjusted overlap between box {i} and {j}")

    # check illegal boxes
    adjusted_data=[]
    for main_box in main_data:
        main_box["bbox_2d"] = list(main_box["bbox_2d"])
        x1_i, y1_i, x2_i, y2_i = main_box["bbox_2d"]
        if x2_i - x1_i <= 30 or y2_i - y1_i <= 30:
            continue
        adjusted_data.append(main_box)

    return adjusted_data


def merge_align(main_data):
    """
    Merge boxes that are tightly aligned and have same width (for vertical merge)
    or same height (for horizontal merge). Works iteratively to handle chains of boxes.
    """
    changed = True

    while changed:
        changed = False
        n = len(main_data)
        merged_indices = set()

        for i in range(n):
            if i in merged_indices:
                continue
            x1_i, y1_i, x2_i, y2_i = main_data[i]["bbox_2d"]
            for j in range(i + 1, n):
                if j in merged_indices:
                    continue
                x1_j, y1_j, x2_j, y2_j = main_data[j]["bbox_2d"]

                # merge horizontally
                if abs(y1_i - y1_j) < 10 and abs(y2_i - y2_j) < 10:
                    if abs(x2_i - x1_j) < 10 or abs(x2_j - x1_i) < 10:
                        new_box = {
                            "bbox_2d": [
                                min(x1_i, x1_j),
                                min(y1_i, y1_j),
                                max(x2_i, x2_j),
                                max(y2_i, y2_j)
                            ],
                            "label": f"merged_{main_data[i].get('label', '')}_{main_data[j].get('label', '')}"
                        }
                        main_data[i] = new_box
                        merged_indices.add(j)
                        changed = True

                elif abs(x1_i - x1_j) < 10 and abs(x2_i - x2_j) < 10:
                    if abs(y2_i - y1_j) < 10 or abs(y2_j - y1_i) < 10:
                        new_box = {
                            "bbox_2d": [
                                min(x1_i, x1_j),
                                min(y1_i, y1_j),
                                max(x2_i, x2_j),
                                max(y2_i, y2_j)
                            ],
                            "label": f"merged_{main_data[i].get('label', '')}_{main_data[j].get('label', '')}"
                        }
                        main_data[i] = new_box
                        merged_indices.add(j)
                        changed = True

        main_data = [box for idx, box in enumerate(main_data) if idx not in merged_indices]
    return main_data


def adjust_main_boxes(main_json_path, ref_json_path, split_h_path, org_img_path, output_path):
    with open(split_h_path, "r", encoding="utf-8") as f:
        heights_data = json.load(f)
    with open(main_json_path, "r", encoding="utf-8") as f:
        main_data = json.load(f)
    with open(ref_json_path, "r", encoding="utf-8") as f:
        ref_data = json.load(f)
    img = cv2.imread(org_img_path)  # 读取图片（BGR格式）
    h, w, _ = img.shape


    # reference (LayoutCoder) box format should be like:
    # { ..., "compos": {"column_min": 116, "row_min": 1565, "column_max": 576, "row_max": 1586}, ...}
    ref_boxes = []
    for item in ref_data["compos"]:
        x1 = item["position"]["column_min"]
        y1 = item["position"]["row_min"]
        x2 = item["position"]["column_max"]
        y2 = item["position"]["row_max"]
        ref_boxes.append([x1, y1, x2, y2])
    #ref_boxes = filter_small_boxes(ref_boxes, min_width=min_width, min_height=min_height)
    print(len(main_data))
    main_data = data_cleaning(img, main_data)
    print(len(main_data))
    main_data = check_contained_boxes(main_data)

    for main_box in main_data:
        x1_m, y1_m, x2_m, y2_m = main_box["bbox_2d"]
        main_coords = x1_m, y1_m, x2_m, y2_m
        # Collect all ref boxes that are mostly inside this main box
        inside_refs = []
        for ref_coords in ref_boxes:
            inter_area, main_area, ref_area, _ = compute_iou(main_coords, ref_coords)
            if ref_area == 0:
                continue

            overlap_ratio_ref = inter_area / ref_area

            # If at least 50% of the reference box is inside main box
            if overlap_ratio_ref >= 0.5:
                inside_refs.append(ref_coords)
        # If we found at least one such reference box, merge them
        if inside_refs:
            x1_new = min(x1 for x1, _, _, _ in inside_refs)
            y1_new = min(y1 for _, y1, _, _ in inside_refs)
            x2_new = max(x2 for _, _, x2, _ in inside_refs)
            y2_new = max(y2 for _, _, _, y2 in inside_refs)
            candidate_coords = [x1_new, y1_new, x2_new, y2_new]
            inter_area, main_area, ref_area, _ = compute_iou(main_coords, candidate_coords)
            if inter_area / min(main_area, ref_area) >= 0.5:
                main_coords = candidate_coords
        # Update main box after merging
        main_box["bbox_2d"] = main_coords


    main_data = merge_split(heights_data,main_data)
    print(len(main_data))
    main_data = component_add(img,ref_boxes,main_data)
    print(len(main_data))
    main_data = check_contained_boxes(main_data)
    print(len(main_data))
    main_data = data_overlap_adjust(main_data)
    print(len(main_data))
    main_data = merge_align(main_data)
    print(len(main_data))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(main_data, f, ensure_ascii=False, indent=2)

    #print(f"Adjusted boxes saved to: {output_path}")


def copy_img(src_base, dst_dir):
    # Destination directory
    os.makedirs(dst_dir, exist_ok=True)

    for subfolder in os.listdir(src_base):
        subfolder_path = os.path.join(src_base, subfolder)
        if not os.path.isdir(subfolder_path) or subfolder == "testing":
            continue

        # 查找 *_test_modify.png 文件
        png_files = glob(os.path.join(subfolder_path, "*_test_modify.png"))
        for file_path in png_files:
            # 构造目标路径
            filename = os.path.basename(file_path)
            dst_path = os.path.join(dst_dir, filename)
            # 复制文件
            shutil.copy(file_path, dst_path)
            #print(f"Copied {file_path} -> {dst_path}")

    print("All files have been copied to the testing folder.")


def numeric_sort_key(name):
    """Extract number from filename/foldername for sorting"""
    match = re.search(r'\d+', name)
    return int(match.group()) if match else float('inf')


def merge_split_jsons_with_offset(base_dir,base_name):

    split_dir = os.path.join(base_dir, f"{base_name}_split")

    split_lines_path = os.path.join(base_dir, "split_lines.json")
    with open(split_lines_path, "r", encoding="utf-8") as f:
        split_lines_data = json.load(f)
    split_offsets = split_lines_data.get("split_lines", [])

    json_files = [f for f in os.listdir(split_dir) if f.endswith("_positions_merge.json")]
    json_files = sorted(json_files, key=numeric_sort_key)
    print(f"Found {len(json_files)} json files in {split_dir}")

    merged_data = []
    label_counter = defaultdict(int)

    for idx, jf in enumerate(json_files):
        jpath = os.path.join(split_dir, jf)
        with open(jpath, "r", encoding="utf-8") as f:
            data = json.load(f)

        offset = 0
        if idx > 0 and idx - 1 < len(split_offsets):
            offset = split_offsets[idx - 1]

        for obj in data:
            x1, y1, x2, y2 = obj["bbox_2d"]
            obj["bbox_2d"] = [x1, y1 + offset, x2, y2 + offset]

            original_label = obj["label"]
            original_label = re.sub(r'(_\d+)+$', '', original_label)
            count = label_counter[original_label]
            if count == 0:
                new_label = original_label
            else:
                new_label = f"{original_label}_{count}"

            label_counter[original_label] += 1
            obj["label"] = new_label

            merged_data.append(obj)

    merged_path = os.path.join(base_dir, f"{base_name}.json")
    with open(merged_path, "w", encoding="utf-8") as f:
        json.dump(merged_data, f, indent=2, ensure_ascii=False)

    print(f"✅ Merged {len(json_files)} files with offsets → {merged_path}")
    return merged_path


x_tolerance=30
y_tolerance=30
min_area=100000
min_width=20
min_height=20

if __name__ == "__main__":
    for i in range(1,119):
        # if i not in [32,39,43]:  # ,99 in [1,2,3,4,5,7] or
        #     continue
        split_h_path = f"D:/py_code/fyp/VueGen/output_seg/{i}/split_lines.json"
        json_path = f"D:/py_code/fyp/VueGen/output_seg/{i}/{i}.json"
        img_path = f"D:/py_code/fyp/VueGen/data/{i}.png"
        #output_path = f"D:/py_code/fyp/VueGen/output_seg/{i}/{i}_with_boxes.png"
        output_json_path = f"D:/py_code/fyp/VueGen/output_seg/{i}/{i}_test_modify.json"
        output_image_path = f"D:/py_code/fyp/VueGen/output_seg/{i}/{i}_test_modify.png"
        ref_json_path = f"D:/py_code/fyp/LayoutCoder/data/output/merge/{i}.json"

        adjust_main_boxes(json_path, ref_json_path, split_h_path, img_path, output_json_path)


        draw_boxes_from_json(output_json_path, img_path, output_image_path)

    src_base = r"D:\py_code\fyp\VueGen\output_seg"
    dst_dir = os.path.join(src_base, "testing_add")
    os.makedirs(dst_dir,exist_ok=True)
    copy_img(src_base,dst_dir)



