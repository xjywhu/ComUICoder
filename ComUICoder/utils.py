# import os
# import shutil
# from PIL import Image
# import numpy as np
#
# # 原始根目录，包含子文件夹
# root = r"C:\Users\dianj\Downloads\image\image"
#
# # 目标目录
# target_dir = r"C:\Users\dianj\Downloads\ttt"
# os.makedirs(target_dir, exist_ok=True)
#
# # 子文件夹名
# subfolders = [str(i) for i in range(1, 8)]
#
# all_heights = []
#
# for sub in subfolders:
#     sub_path = os.path.join(root, sub)
#
#     if not os.path.isdir(sub_path):
#         print(f"Skipping {sub_path}, not a folder.")
#         continue
#
#     for fname in os.listdir(sub_path):
#         fpath = os.path.join(sub_path, fname)
#
#         # 判断是否是图片
#         if not fname.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
#             continue
#
#         # 读取图片高度
#         try:
#             img = Image.open(fpath)
#             w, h = img.size
#             all_heights.append(h)
#             img.close()  # ✅ 关闭文件句柄
#         except Exception as e:
#             print(f"Error reading image: {fpath}, {e}")
#             continue
#
#         # 移动到目标文件夹
#         dst = os.path.join(target_dir, fname)
#         try:
#             shutil.move(fpath, dst)
#         except PermissionError:
#             try:
#                 # 如果 move 失败，先复制再删除
#                 shutil.copy2(fpath, dst)
#                 os.remove(fpath)
#             except Exception as e2:
#                 print(f"Failed to move {fpath} to {dst}: {e2}")
#
#     print(f"Finished moving images from {sub_path}")
#
# # 统计图片高度
# if all_heights:
#     heights = np.array(all_heights)
#     print("====== HEIGHT STATISTICS ======")
#     print(f"count: {len(heights)}")
#     print(f"mean: {heights.mean():.2f}")
#     print(f"std:  {heights.std():.2f}")
#     print(f"min:  {heights.min()}")
#     print(f"max:  {heights.max()}")
# else:
#     print("No images found.")
import json

# 读取两个 JSON 文件
with open(r"D:\py_code\fyp\Design2Code\Design2Code\metrics\res_dict_testset_final_wlayout.json", "r", encoding="utf-8") as f:
    data1 = json.load(f)

with open(r"D:\py_code\fyp\Design2Code\Design2Code\metrics\res_dict_testset_final_wo_layout.json", "r", encoding="utf-8") as f:
    data2 = json.load(f)

import json
import os

# 假设新的 JSON 文件路径
output_path = "merged.json"

# 如果文件不存在，创建一个带框架的空 JSON
if not os.path.exists(output_path):
    new_data = {"gemini_vuegen_prompting": {}}
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(new_data, f, indent=4)
else:
    # 如果文件已存在，就读取
    with open(output_path, "r", encoding="utf-8") as f:
        new_data = json.load(f)

# 读取两个原始 JSON 文件
with open(r"D:\py_code\fyp\Design2Code\Design2Code\metrics\res_dict_testset_final_wlayout.json", "r", encoding="utf-8") as f:
    data1 = json.load(f)

with open(r"D:\py_code\fyp\Design2Code\Design2Code\metrics\res_dict_testset_final_wo_layout.json", "r", encoding="utf-8") as f:
    data2 = json.load(f)

# 对比每个 idx.html 的第一个数，把更大的放到新的 JSON
dict1 = data1.get("gemini_vuegen_prompting", {})
dict2 = data2.get("gemini_vuegen_prompting", {})

all_keys = set(dict1.keys()).union(dict2.keys())

for key in all_keys:
    val1 = dict1.get(key)
    val2 = dict2.get(key)

    if val1 is not None and val2 is not None:
        if val1[0] >= val2[0]:
            new_data["gemini_vuegen_prompting"][key] = val1
        else:
            new_data["gemini_vuegen_prompting"][key] = val2
    elif val1 is not None:
        new_data["gemini_vuegen_prompting"][key] = val1
    elif val2 is not None:
        new_data["gemini_vuegen_prompting"][key] = val2

# 保存最终结果
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(new_data, f, indent=4)

print(f"✅ Merged JSON saved to {output_path}")
