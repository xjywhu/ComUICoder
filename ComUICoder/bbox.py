from typing import List, Optional
from PIL import Image, ImageDraw, ImageFont


class BoundingBox:
    def __init__(self, bbox_2d: List[int], label: str = None):
        self.bbox_2d = bbox_2d  # [x1, y1, x2, y2]
        self.label = label

# Function to draw bounding boxes
def plot_bounding_boxes(image_path: str, bounding_boxes: List[BoundingBox], output_path: Optional[str] = None) -> Image.Image:
    with Image.open(image_path) as im:
        im_with_boxes = im.copy().convert('RGBA')
        draw = ImageDraw.Draw(im_with_boxes)

        bright_colors = [
            (255, 0, 0),       # 红
            (0, 0, 255),       # 蓝
            (0, 200, 0),       # 绿
            (255, 165, 0),     # 橙
            (128, 0, 128),     # 紫
            (0, 255, 255),     # 青
            (255, 20, 147),    # 粉红
            (0, 0, 0)          # 黑色 (用于对比强烈场景)
        ]

        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except IOError:
            font = ImageFont.load_default()

        for i, bbox in enumerate(bounding_boxes):
            x1, y1, x2, y2 = bbox.bbox_2d

            # 从显眼调色盘里取颜色
            color = bright_colors[i % len(bright_colors)]

            # 画矩形框（加粗）
            for offset in range(4):
                draw.rectangle(
                    [(x1-offset, y1-offset), (x2+offset, y2+offset)],
                    outline=color
                )

            # 画 label
            if bbox.label:
                try:
                    text_width, text_height = font.getsize(bbox.label)
                except AttributeError:
                    try:
                        left, top, right, bottom = font.getbbox(bbox.label)
                        text_width, text_height = right - left, bottom - top
                    except AttributeError:
                        text_width, text_height = len(bbox.label) * 8, 20

                # 背景白底
                draw.rectangle(
                    [(x1, y1-text_height-10), (x1+text_width+10, y1)],
                    fill=(255, 255, 255, 200)
                )
                draw.text((x1+5, y1-text_height-5), bbox.label, fill=color, font=font)

        if output_path:
            im_with_boxes.save(output_path)

        return im_with_boxes