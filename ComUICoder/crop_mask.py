import json
from PIL import Image, ImageDraw, ImageFont
import os
import torch
def crop_and_mask_image(image_path, json_path, output_dir, mask_color=(255, 255, 255), target_ratio=0.3):

    os.makedirs(output_dir, exist_ok=True)

    img = Image.open(image_path).convert("RGB")
    img_masked = img.copy()
    draw = ImageDraw.Draw(img_masked)

    with open(json_path, "r") as f:
        boxes = json.load(f)


    items=[]
    for i, item in enumerate(boxes):
        bbox = item["bbox_2d"]
        label = item.get("label")
        #bboxes.append(bbox)
        # crop
        cropped = img.crop((bbox[0], bbox[1], bbox[2], bbox[3]))
        cropped_path = os.path.join(output_dir, f"{label}.png")
        cropped.save(cropped_path)
        #crop_paths.append(cropped_path)
        items.append([cropped_path,bbox])
        # mask
        draw.rectangle(bbox, fill=mask_color)

        # bbox 尺寸
        box_width = bbox[2] - bbox[0]
        box_height = bbox[3] - bbox[1]

        # 初始字体大小
        font_size = 10
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except:
            font = ImageFont.load_default()

        while True:
            bbox_text = draw.textbbox((0, 0), label, font=font)
            text_w = bbox_text[2] - bbox_text[0]
            text_h = bbox_text[3] - bbox_text[1]
            text_area = text_w * text_h
            box_area = box_width * box_height

            if text_w > box_width/2 or text_h > box_height/2:
                font_size-=20
                break
            font_size += 10
            try:
                font = ImageFont.truetype("arial.ttf", font_size)
            except:
                font = ImageFont.load_default()
                break  # 用默认字体就不再增大

        # 文本居中
        text_x = bbox[0] + (box_width - text_w) / 2
        text_y = bbox[1] + (box_height - text_h) / 2
        draw.text((text_x, text_y), label, fill=(0, 0, 0), font=font)

    masked_path = os.path.join(output_dir, "masked_image.png")
    img_masked.save(masked_path)

    print(f"✅ 共保存 {len(boxes)} 个裁剪图像和 1 张遮罩图到 {output_dir}/")
    return items, masked_path

if __name__ == "__main__":
    image_dir = r"D:/py_code/fyp/VueGen/data"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for i in range(3,4):
        json_dir = f"D:/py_code/fyp/VueGen/output_seg/{i}"
        image_path = os.path.join(image_dir, f"{i}.png")
        json_path = os.path.join(json_dir, f"{i}_test_modify.json")
        output_dir = os.path.join(r"D:/py_code/fyp/VueGen/output_seg", "crop", f"{i}_cropped")
        #items, masked_path = crop_and_mask_image(image_path, json_path, output_dir, mask_color=(200, 200, 200))
