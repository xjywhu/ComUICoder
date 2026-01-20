import os
from PIL import Image, ImageDraw

root_dir = r"D:\py_code\fyp\VueGen\output_seg"
output_dir = os.path.join(root_dir, "merged")
os.makedirs(output_dir, exist_ok=True)

for folder_name in os.listdir(root_dir):
    folder_path = os.path.join(root_dir, folder_name)
    if not os.path.isdir(folder_path):
        continue  # 只处理子文件夹

    # 找出所有 split_xxx_with_boxes.png
    images = [f for f in os.listdir(folder_path) if f.endswith("_with_boxes.png")]

    # 按照 i 排序
    def extract_index(filename):
        return int(filename.split("_")[1])
    images.sort(key=extract_index)

    # 打开所有图片
    pil_images = [Image.open(os.path.join(folder_path, f)) for f in images]

    # 获取拼接后的尺寸（竖直拼接）
    widths = [im.width for im in pil_images]
    heights = [im.height for im in pil_images]
    total_height = sum(heights)
    max_width = max(widths)

    merged_img = Image.new("RGB", (max_width, total_height), "white")
    draw = ImageDraw.Draw(merged_img)

    # 按顺序贴到大图里，并画红色分隔线
    y_offset = 0
    for idx, im in enumerate(pil_images):
        merged_img.paste(im, (0, y_offset))

        # 除了最后一张，每次贴完后画一条横线
        if idx < len(pil_images) - 1:
            y_offset += im.height
            draw.line([(0, y_offset), (max_width, y_offset)], fill="red", width=8)

        else:
            y_offset += im.height

    # 保存
    out_path = os.path.join(output_dir, f"{folder_name}_merged.png")
    merged_img.save(out_path)
    print(f"✅ Merged {folder_name} -> {out_path}")
