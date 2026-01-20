import google.generativeai as genai
import io
from IPython.display import Markdown
import textwrap
import base64
import requests
import os
import re
from bs4 import BeautifulSoup
import numpy as np
from PIL import Image
import json
import random
import time
from pathlib import Path
import tiktoken
import subprocess
import shutil
from selenium import webdriver
from selenium.webdriver.firefox.service import Service as FirefoxService
from selenium.webdriver.firefox.options import Options as FirefoxOptions
from bbox import plot_bounding_boxes, BoundingBox




def process_single_json(json_file_path: str, image_path: str, output_dir: str, image_id: str) -> str:
    """
    Process a single JSON file: convert normalized bbox to pixel coordinates,
    group by label, and compute minimum bounding box for each group.
    (reprocess model's output)

    Args:
        json_file_path (str): Path to the input JSON file
        image_path (str): image path
        output_dir (str): Directory to save the processed JSON

    Returns:
        list: List of group bounding boxes [{'bbox_2d': [...], 'label': ...}, ...]
    """

    output_path = os.path.join(output_dir, f"{image_id}_positions_merge.json")
    json_file = os.path.basename(json_file_path)
    print(f"Processing image {image_id}...")

    with Image.open(image_path) as img:
        width, height = img.size
    print(width,height)

    # Load JSON
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            boxes_data = json.load(f)
        print(f"  - Loaded {len(boxes_data)} bounding boxes")
    except Exception as e:
        print(f"  - Error loading JSON file: {e}")
        return False

    if not boxes_data:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(boxes_data, f, indent=2)
        print(f"  - No bounding boxes in {json_file}, skipping")
        return False

    # Group by label but only merge consecutive same labels
    groups = []  # each item: {"label": xxx, "boxes": []}
    last_label = None

    for box in boxes_data:
        if "bbox_2d" in box and "label" in box:
            label = box["label"]

            # convert normalized to absolute pixel coords
            y1, x1, y2, x2 = box["bbox_2d"]
            abs_x1 = int(x1 / 1000 * width)
            abs_y1 = int(y1 / 1000 * height)
            abs_x2 = int(x2 / 1000 * width)
            abs_y2 = int(y2 / 1000 * height)
            abs_box = [abs_x1, abs_y1, abs_x2, abs_y2]

            # If same label and consecutive → merge into last group
            if groups and label == last_label:
                groups[-1]["boxes"].append(abs_box)
            else:
                # new group segment
                groups.append({"label": label, "boxes": [abs_box]})

            last_label = label

    # Now compute min bounding box for each group segment
    group_boxes = []
    for g in groups:
        boxes_array = np.array(g["boxes"])
        min_x = int(np.min(boxes_array[:, 0]))
        min_y = int(np.min(boxes_array[:, 1]))
        max_x = int(np.max(boxes_array[:, 2]))
        max_y = int(np.max(boxes_array[:, 3]))

        group_boxes.append({
            "bbox_2d": [min_x, min_y, max_x, max_y],
            "label": g["label"]
        })

    if not group_boxes:
        print(f"  - No valid group bounding boxes for image {image_id}")
        return False

    for i, box in enumerate(group_boxes):
        print(f"  - Group bounding box {i+1}: coordinates {box['bbox_2d']}, label: {box['label']}")

    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(group_boxes, f, indent=2)
        print(f"  - Saved group bounding boxes to: {output_path}")
    except Exception as e:
        print(f"  - Error saving JSON file: {e}")

    return output_path
# def process_single_json(json_file_path: str, image_path: str, output_dir: str, image_id: str) -> str:
#     """
#     Process a JSON file:
#     - Convert normalized bbox (0-1000) to pixel coordinates
#     - Keep all bboxes (NO merging)
#     - For same-label boxes: rename as label_1, label_2...
#     """
#
#     output_path = os.path.join(output_dir, f"{image_id}_positions_merge.json")
#     json_file = os.path.basename(json_file_path)
#     print(f"Processing image {image_id}...")
#
#     # Load image size
#     with Image.open(image_path) as img:
#         width, height = img.size
#     print(width, height)
#
#     # Load JSON
#     try:
#         with open(json_file_path, 'r', encoding='utf-8') as f:
#             boxes_data = json.load(f)
#         print(f"  - Loaded {len(boxes_data)} bounding boxes")
#     except Exception as e:
#         print(f"  - Error loading JSON file: {e}")
#         return False
#
#     # If empty — just save empty file
#     if not boxes_data:
#         with open(output_path, 'w', encoding='utf-8') as f:
#             json.dump([], f, indent=2)
#         print("  - Empty JSON, saved empty list.")
#         return False
#
#     # Used to count how many times each label appeared
#     label_counter = {}
#
#     # Output list (NO merging)
#     output_boxes = []
#
#     # Process each individual bbox
#     for box in boxes_data:
#         if "bbox_2d" not in box or "label" not in box:
#             continue
#
#         label = box["label"]
#
#         # Increment counter
#         if label not in label_counter:
#             label_counter[label] = 1
#         else:
#             label_counter[label] += 1
#
#         # New label with index
#         new_label = f"{label}_{label_counter[label]}"
#
#         # Convert normalized to pixel coords
#         y1, x1, y2, x2 = box["bbox_2d"]
#         abs_x1 = int(x1 / 1000 * width)
#         abs_y1 = int(y1 / 1000 * height)
#         abs_x2 = int(x2 / 1000 * width)
#         abs_y2 = int(y2 / 1000 * height)
#
#         # Save this box as-is (no merge)
#         output_boxes.append({
#             "bbox_2d": [abs_x1, abs_y1, abs_x2, abs_y2],
#             "label": new_label
#         })
#
#     # Save
#     try:
#         with open(output_path, 'w', encoding='utf-8') as f:
#             json.dump(output_boxes, f, indent=2)
#         print(f"  - Saved {len(output_boxes)} boxes to: {output_path}")
#     except Exception as e:
#         print(f"  - Error saving JSON file: {e}")
#
#     return output_path



def find_outer_template(text):
    start_tag = "<template"
    end_tag = "</template>"
    start_idx = text.find(start_tag)
    if start_idx == -1:
        return None

    start_tag_end = text.find(">", start_idx)

    count = 1
    idx = start_tag_end + 1
    while count > 0:
        next_start = text.find(start_tag, idx)
        next_end = text.find(end_tag, idx)

        if next_start != -1 and next_start < next_end:
            count += 1
            idx = next_start + 9
        else:
            count -= 1
            idx = next_end + 11

    return text[start_idx:idx]


def post_process(vue_dir):

    app_path = os.path.join(vue_dir, "App.vue")
    with open(app_path, "r", encoding="utf-8") as f:
        content = f.read()

    content = re.sub(r'(?i)<style\s+scoped\s*>', '<style>', content)
    content = re.sub(r"\.group_\d+_placeholder::before\s*\{[\s\S]*?\}", "", content)
    pattern = r"import\s*\{\s*ref\s*\}\s*from\s*['\"]vue['\"]\s*;"
    matches = list(re.finditer(pattern, content))
    print(f"Found {len(matches)} import {{ ref }} statements in {app_path}. Removing duplicates...")
    if matches:
        first_end = matches[0].end()
        new_content = content[:first_end]
        last_index = first_end
        for match in matches[1:]:
            new_content += content[last_index:match.start()]
            last_index = match.end()
        new_content += content[last_index:]
        content = new_content

    # delete duplicate import
    import_pattern = r"^import\s+\w+\s+from\s+['\"][^'\"]+\.vue['\"]\s*;\s*$"
    seen = set()
    lines = content.splitlines()
    new_lines = []
    for line in lines:
        if re.match(import_pattern, line):
            if line not in seen:
                seen.add(line)
                new_lines.append(line)
            else:
                pass
        else:
            new_lines.append(line)
    content = "\n".join(new_lines)

    with open(app_path, "w", encoding="utf-8") as f:
        f.write(content)



def rename_duplicate_style(app_content, snippet_style, snippet_template):
    existing_style_match = re.search(r"<style[^>]*>([\s\S]*?)</style>", app_content)
    existing_style = existing_style_match.group(1) if existing_style_match else ""
    existing_classes = set(re.findall(r"\.([\w-]+)", existing_style))
    snippet_classes = re.findall(r"\.([\w-]+)", snippet_style)
    rename_map = {}
    for cls in snippet_classes:
        new_cls = cls
        counter = 1
        while new_cls in existing_classes:
            new_cls = f"{cls}-{counter}"
            counter += 1
        if new_cls != cls:
            rename_map[cls] = new_cls
            existing_classes.add(new_cls)
            snippet_style = re.sub(rf"\b{cls}\b", new_cls, snippet_style)

    for old_cls, new_cls in rename_map.items():
        snippet_template = re.sub(rf'\b{old_cls}\b', new_cls, snippet_template)
        print(f"Renamed duplicate style '{old_cls}' → '{new_cls}'")

    return snippet_style, snippet_template



# def rename_duplicate_consts(app_content, snippet_script, snippet_template):
#
#     existing_consts = re.findall(r"\bconst\s+(\w+)\s*=", app_content)
#     print(existing_consts)
#     existing_consts = set(existing_consts)
#     snippet_consts = re.findall(r"\bconst\s+(\w+)\s*=", snippet_script)
#
#     rename_map = {}
#     for name in snippet_consts:
#         new_name = name
#         counter = 1
#         while new_name in existing_consts:
#             new_name = f"{name}_{counter}"
#             counter += 1
#
#         if new_name != name:
#             rename_map[name] = new_name
#             existing_consts.add(new_name)
#             snippet_script = re.sub(rf"\b{name}\b", new_name, snippet_script)
#     for old, new in rename_map.items():
#         snippet_template = re.sub(
#             rf'(v-for="[\w_$]+\s+in\s+){old}(\b)',
#             rf'\1{new}\2',
#             snippet_template
#         )
#         print(f"Renamed duplicate const '{old}' → '{new}'")
#
#     return snippet_script, snippet_template

from bs4 import BeautifulSoup

def rename_template_vars(template: str, rename_map: dict) -> str:
    import re

    def replace_expr(expr: str) -> str:
        for old, new in rename_map.items():
            expr = re.sub(
                rf'\b{old}\b(?=\s*[\.\[\(]|$)',  # cardData 或 cardData.xxx
                new,
                expr
            )
        return expr

    template = re.sub(
        r'(:\w+\s*=\s*)"([^"]+)"',
        lambda m: m.group(1) + '"' + replace_expr(m.group(2)) + '"',
        template
    )

    template = re.sub(
        r'(v-(?:if|for|show|bind)\s*=\s*)"([^"]+)"',
        lambda m: m.group(1) + '"' + replace_expr(m.group(2)) + '"',
        template
    )

    template = re.sub(
        r'{{([^}]+)}}',
        lambda m: '{{ ' + replace_expr(m.group(1)) + ' }}',
        template
    )

    return template


def rename_duplicate_consts(app_content, snippet_script, snippet_template):
    """
    对 snippet_script 中的 const 与 app_content 中重复的变量加序号重命名，
    并同步更新 snippet_template 中的所有引用。
    """
    # 找到已有的 const
    existing_consts = set(re.findall(r"\bconst\s+(\w+)\s*=", app_content))
    snippet_consts = re.findall(r"\bconst\s+(\w+)\s*=", snippet_script)

    rename_map = {}

    for name in snippet_consts:
        new_name = name
        counter = 1
        while new_name in existing_consts:
            new_name = f"{name}_{counter}"
            counter += 1

        if new_name != name:
            rename_map[name] = new_name
            existing_consts.add(new_name)
            snippet_script = re.sub(rf"\b{name}\b", new_name, snippet_script)

    if rename_map:
        snippet_template = rename_template_vars(snippet_template, rename_map)

    return snippet_script, snippet_template




def remove_export_default_block(code):
    lines = code.splitlines()
    start, end = None, None
    for i, line in enumerate(lines):
        if re.match(r'^\s*export\s+default\s*\{', line):
            start = i
        elif start is not None and re.match(r'^\}', line):
            end = i
            break
    if start is not None and end is not None:
        del lines[start:end + 1]

    return "\n".join(lines)


# def bg_color_eliminate(snippet_template, app_content):
#     match = re.search(r'<div\s+class=["\']([^"\']+)["\']', snippet_template)
#     if not match:
#         return app_content
#     first_class = match.group(1).split()[0]  # 只取第一个 class
#     pattern = rf'(\.{re.escape(first_class)}\s*\{{[^}}]*\}})'
#     def remove_bg(m):
#         block = m.group(1)
#         block = re.sub(r'^\s*background-color\s*:[^;]+;\s*', '', block, flags=re.MULTILINE)
#         return block
#     app_content, count = re.subn(pattern, remove_bg, app_content, count=1, flags=re.DOTALL)
#     print(f"✅ Modified {count} CSS block(s)")
#     return app_content


def parse_vue_from_txt(txt_path, vue_dir, category_pattern, multipage_flag=False):
    with open(txt_path, "r", encoding="utf-8") as f:
        content = f.read()
    content = re.sub(r"\?(\s*):", r"\1:", content)
    base_name = os.path.basename(txt_path)

    # data cleaning: replace those content with more than one ```vue to one ```component and one ```snippet
    if "crop" in base_name:
        count=0
        def replacer(match):
            nonlocal count
            count += 1
            if count == 1:
                return "```component"
            elif count == 2:
                return "```snippet"
            else:
                print("More than two ```vue are detected, is there something wrong?")
                return match.group(0)

        content = re.sub(r"```vue", replacer, content)
    if "masked" in base_name:
        content = re.sub(r"```app", "```vue", content)
    pattern_full = re.compile(rf"```{category_pattern}\s*(.*?)\s*```", re.DOTALL)

    # There may be more than one code snippets that being matched
    matches = pattern_full.findall(content)
    id=0
    for block in matches:
        print()
        # block = re.sub(r'^\s*height:\s*\d+\s*vh;\s*$', '', block, flags=re.MULTILINE)

        # match component name
        #name_match = re.search(r'defineOptions\(\s*{[^}]*name:\s*["\']([^"\']+)["\']', block)   # for original
        name_match = re.search(r'export\s+default\s*{[^}]*?\bname\s*:\s*[\'"]([^\'"]+)[\'"]', block, re.DOTALL)
        if name_match:
            comp_name = name_match.group(1).strip()
        else:
            name_match = re.search(r'defineOptions\(\s*{[^}]*name:\s*["\']([^"\']+)["\']', block)   # for original
            if name_match:
                comp_name = name_match.group(1).strip()
            else:
                comp_name = "UnknownComponent"
        print(comp_name)

        # directly save
        if category_pattern == "component" or category_pattern == "vue":
            file_name = f"{comp_name}.vue"
            file_path = os.path.join(vue_dir, file_name)
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(block.strip())
            print(f"✅ Saved {file_name}")

        # handle snippet part
        if category_pattern == "snippet":
            if multipage_flag:
                page_number = os.path.basename(os.path.dirname(vue_dir))
                base_name = re.search(rf'<!--\s*page_{page_number}_([a-zA-Z0-9_.]+)\s*-->', block)
                if not base_name:
                    continue
                base_name=base_name.group(1)
                base_name = os.path.splitext(base_name)[0]
                print(base_name)
                block = re.sub(r'<!--\s*(page_[a-zA-Z0-9_]+)\s*-->', '', block)
            else:
                base_name = re.search(r"(.+?)_crop", os.path.basename(txt_path)).group(1)
            app_path = os.path.join(vue_dir, "App.vue")

            with open(app_path, "r", encoding="utf-8") as f:
                app_content = f.read()

            template_match = find_outer_template(block)
            script_match = re.search(r"<script[^>]*>([\s\S]*?)</script>", block)
            style_match = re.search(r"<style[^>]*>([\s\S]*?)</style>", block)
            if style_match:
                snippet_style = style_match.group(1).strip()
                snippet_style, template_match = rename_duplicate_style(
                    app_content, snippet_style, template_match
                )
                # app_content = re.sub(
                #     r"(<style[^>]*>)([\s\S]*?)(</style>)",
                #     lambda m: f"{m.group(1)}{m.group(2).strip()}\n\n{snippet_style}\n{m.group(3)}",
                #     app_content,
                # )
            if script_match:
                snippet_script = script_match.group(1).strip()
                # check import error
                pattern = r"^//.*assume\s+'([\w-]+)'\s+is\s+imported.*$"
                replacement = r'import \1 from "./\1.vue"'
                snippet_script = re.sub(pattern, replacement, snippet_script, flags=re.MULTILINE)
                # check export error
                pattern = r'export\s+default\s*\{(?:[^{}]|\{[^{}]*\})*\}'
                print("[DELETE CHECK] export exists.")
                snippet_script = remove_export_default_block(snippet_script)
                # check duplicate style
                snippet_script, template_match = rename_duplicate_consts(
                    app_content, snippet_script, template_match
                )
                app_content = re.sub(
                    r"(<script[^>]*>)([\s\S]*?)(</script>)",
                    lambda m: f"{m.group(1)}{m.group(2).strip()}\n\n{snippet_script}\n{m.group(3)}",
                    app_content,
                )

            if template_match:
                start_tag_end = template_match.find(">") + 1
                end_tag_start = template_match.rfind("</template>")
                snippet_template = template_match[start_tag_end:end_tag_start].strip()
                #  map placeholder to group_{i}_placeholder
                base_name = base_name.replace(" ", "_")
                placeholder_pattern = rf'(<(div|aside)\s+[^>]*\b{base_name}_placeholder\b[^>]*>)'

                new_div_content = r'\1\n' + snippet_template + '\n'

                app_content, count = re.subn(
                    placeholder_pattern,
                    new_div_content,
                    app_content,
                    flags=re.DOTALL
                )
                print(count)
                if count:
                    print(f"✅ Inserted snippet inside {base_name}_placeholder")
                else:
                    print("[Warn!] No placeholder found for snippet; skipped inserting template.")

                match_class = re.search(r'<div[^>]*\bclass=["\']([^"\']+)["\']', snippet_template)
                if match_class and style_match:
                    class_name = match_class.group(1).split()[0]  # 如果有多个 class，只取第一个?
                    print("Class name:", class_name)
                    insert_css = "  width: 100%;\n  height: 100%;"
                    pattern = rf'(^\.{re.escape(class_name)}\s*\{{)(.*?)(\}})'

                    def insert_properties(m):
                        before = m.group(1)
                        body = m.group(2)
                        after = m.group(3)
                        new_body = body.rstrip() + "\n" + insert_css + "\n"
                        return before + new_body + after
                    snippet_style, count = re.subn(pattern, insert_properties, snippet_style, flags=re.DOTALL)
                    app_content = re.sub(
                        r"(<style[^>]*>)([\s\S]*?)(</style>)",
                        lambda m: f"{m.group(1)}{m.group(2).strip()}\n\n{snippet_style}\n{m.group(3)}",
                        app_content,
                    )

                else:
                    print("No class found in snippet_template")
            with open(app_path, "w", encoding="utf-8") as f:
                f.write(app_content)
            print(f"✅ Updated App.vue with snippet ({comp_name})")

    return len(matches)


def parse_vue_from_txt_component(txt_path, vue_dir, pattern):
    with open(txt_path, "r", encoding="utf-8") as f:
        content = f.read()
    content = re.sub(r"\?(\s*):", r"\1:", content)
    base_name = os.path.basename(txt_path)
    # data cleaning: replace those content with more than one ```vue to one ```component and one ```snippet
    if "crop" in base_name:
        count=0
        def replacer(match):
            nonlocal count
            count += 1
            if count == 1:
                return "```component"
            elif count == 2:
                return "```snippet"
            else:
                print("More than two ```vue are detected, is there something wrong?")
                return match.group(0)

        content = re.sub(r"```vue", replacer, content)

    if "masked" in base_name:
        content = re.sub(r"```app", "```vue", content)

    pattern_full = re.compile(rf"```{pattern}\s*(.*?)\s*```", re.DOTALL)
    matches = pattern_full.findall(content)

    for block in matches:
        #block = re.sub(r'^\s*height:\s*\d+\s*vh;\s*$', '', block, flags=re.MULTILINE)
        # match component name
        #name_match = re.search(r'defineOptions\(\s*{[^}]*name:\s*["\']([^"\']+)["\']', block)   # for original
        name_match = re.search(r'export\s+default\s*{[^}]*?\bname\s*:\s*[\'"]([^\'"]+)[\'"]', block, re.DOTALL)
        if name_match:
            comp_name = name_match.group(1).strip()
        else:
            name_match = re.search(r'defineOptions\(\s*{[^}]*name:\s*["\']([^"\']+)["\']', block)   # for original
            if name_match:
                comp_name = name_match.group(1).strip()
            else:
                comp_name = "UnknownComponent"

        output_dir = os.path.join(vue_dir, "components")
        os.makedirs(output_dir, exist_ok=True)
        # directly save
        if pattern == "component" or pattern == "vue":
            file_name = f"{comp_name}.vue"

            file_path = os.path.join(output_dir, file_name)
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(block.strip())
            print(f"✅ Saved {file_name}")

        # handle snippet part
        if pattern == "snippet":
            base_name_match = re.search(r"(.+?)_crop", os.path.basename(txt_path))
            if not base_name_match:
                print("Cannot extract base name")
                return
            base_name = base_name_match.group(1)

            snippet_match = re.search(r"```snippet([\s\S]+?)```", content)
            if not snippet_match:
                print("No snippet found")
                return
            snippet_content = snippet_match.group(1).strip()

            # 分离 <template>, <script>, <style> 三部分
            # template
            template_match = find_outer_template(block)
            start_tag_end = template_match.find(">") + 1
            end_tag_start = template_match.rfind("</template>")
            template_content = template_match[start_tag_end:end_tag_start].strip()

            # script
            script_match = re.search(r"<script[^>]*>([\s\S]*?)</script>", block)
            snippet_script = script_match.group(1).strip()
            pattern = r"^//.*assume\s+'([\w-]+)'\s+is\s+imported.*$"
            replacement = r'import \1 from "./\1.vue"'
            snippet_script = re.sub(pattern, replacement, snippet_script, flags=re.MULTILINE)
            pattern = r'export\s+default\s*\{(?:[^{}]|\{[^{}]*\})*\}'
            print("[DELETE CHECK] export exists.")
            script_content = remove_export_default_block(snippet_script)

            # style
            style_match = re.search(r"<style[^>]*>([\s\S]+?)</style>", snippet_content)
            style_content = style_match.group(1).strip() if style_match else ""

            # 拼成 Vue 文件内容
            vue_content = f"""<template>
{template_content}
</template>

<script setup>
{script_content}
</script>

<style scoped>
{style_content}
</style>
            """
            output_dir=os.path.join(vue_dir,"components")
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, "App.vue")
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(vue_content)

            print(f"Written {output_file}")

    return len(matches)
