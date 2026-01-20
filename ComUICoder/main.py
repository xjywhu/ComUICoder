import textwrap
import base64
import requests
import torch
import os
import re
import sys
from bs4 import BeautifulSoup
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import json
import random
import time
from pathlib import Path
import tiktoken
import subprocess
import shutil
import importlib.util
from selenium import webdriver
from selenium.webdriver.firefox.service import Service as FirefoxService
from selenium.webdriver.firefox.options import Options as FirefoxOptions

# 添加项目根目录到 Python 路径
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAIN_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from UIED.run_single import uied

# 显式导入 main 文件夹下的 prompts.py
_main_prompts_path = os.path.join(MAIN_DIR, "prompts.py")
_spec_prompts = importlib.util.spec_from_file_location("main_prompts", _main_prompts_path)
_main_prompts = importlib.util.module_from_spec(_spec_prompts)
_spec_prompts.loader.exec_module(_main_prompts)
# 从 main/prompts.py 导入所有需要的变量
SYSTEM_PROMPT_FOR_BBOX = _main_prompts.SYSTEM_PROMPT_FOR_BBOX
COMPONENT_GEN_SYSTEM_PROMPT = _main_prompts.COMPONENT_GEN_SYSTEM_PROMPT
MASKED_GEN_SYSTEM_PROMPT = _main_prompts.MASKED_GEN_SYSTEM_PROMPT
MULTI_COMPONENT_GEN_SYSTEM_PROMPT = _main_prompts.MULTI_COMPONENT_GEN_SYSTEM_PROMPT
GENERATION_VUE_TAILWIND_SYSTEM_PROMPT = _main_prompts.GENERATION_VUE_TAILWIND_SYSTEM_PROMPT

from api_call import code_generation, bbox_json_generation

# code_generation_multipage 只在 main/api_call.py 中，需要显式导入
import importlib.util
_main_api_call_path = os.path.join(MAIN_DIR, "api_call.py")
_spec = importlib.util.spec_from_file_location("main_api_call", _main_api_call_path)
_main_api_call = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_main_api_call)
code_generation_multipage = _main_api_call.code_generation_multipage

# post_process 在 main/ProcessTools.py 中版本不需要 json_file_path 参数
_main_process_tools_path = os.path.join(MAIN_DIR, "ProcessTools.py")
_spec2 = importlib.util.spec_from_file_location("main_process_tools", _main_process_tools_path)
_main_process_tools = importlib.util.module_from_spec(_spec2)
_spec2.loader.exec_module(_main_process_tools)
post_process = _main_process_tools.post_process
process_single_json = _main_process_tools.process_single_json
parse_vue_from_txt = _main_process_tools.parse_vue_from_txt
parse_vue_from_txt_component = _main_process_tools.parse_vue_from_txt_component

# 显式导入 main 文件夹下的 segmentation.py
_main_segmentation_path = os.path.join(MAIN_DIR, "segmentation.py")
_spec3 = importlib.util.spec_from_file_location("main_segmentation", _main_segmentation_path)
_main_segmentation = importlib.util.module_from_spec(_spec3)
_spec3.loader.exec_module(_main_segmentation)
find_split_lines_color = _main_segmentation.find_split_lines_color

from bbox import plot_bounding_boxes, BoundingBox
from merge_test import merge_split_jsons_with_offset, adjust_main_boxes,draw_boxes_from_json
from crop_mask import crop_and_mask_image
from comp_merge import compute_group_merge
from eval_metrics import compute_merge_table

import matplotlib.pyplot as plt

def save_cluster_images(clusters_paths, base_tmp):
    """
    clusters_paths: List[List[str]]  # 每个 cluster 里的 image paths
    base_tmp: str
    """

    os.makedirs(base_tmp, exist_ok=True)

    for idx, cluster in enumerate(clusters_paths):
        cluster_dir = os.path.join(base_tmp, f"cluster_{idx+1}")
        os.makedirs(cluster_dir, exist_ok=True)

        for img_path in cluster:
            if not os.path.isfile(img_path):
                print(f"Warning: file not found: {img_path}")
                continue

            filename = os.path.basename(img_path)
            name, ext = os.path.splitext(filename)

            dst_path = os.path.join(cluster_dir, filename)
            counter = 1
            while os.path.exists(dst_path):
                dst_path = os.path.join(cluster_dir, f"{name}_{counter}{ext}")
                counter += 1

            shutil.copy2(img_path, dst_path)

        print(f"Cluster {idx+1} → {cluster_dir} 完成复制")


def generate_comparison_image(gt_path, before_path, after_path, output_path):
    """
    生成布局修复前后的对比图，包含 GT、修复前、修复后三张图并排显示
    
    Args:
        gt_path: Ground Truth 图片路径
        before_path: 布局修复前的截图路径
        after_path: 布局修复后的截图路径
        output_path: 对比图输出路径
    """
    # 加载图片
    gt_img = Image.open(gt_path).convert("RGB") if os.path.exists(gt_path) else None
    before_img = Image.open(before_path).convert("RGB")
    after_img = Image.open(after_path).convert("RGB")
    
    # 确定目标高度（使用最大高度）
    heights = [before_img.height, after_img.height]
    if gt_img:
        heights.append(gt_img.height)
    target_height = max(heights)
    
    # 等比例缩放图片到相同高度
    def resize_to_height(img, target_h):
        if img is None:
            return None
        ratio = target_h / img.height
        new_width = int(img.width * ratio)
        return img.resize((new_width, target_h), Image.Resampling.LANCZOS)
    
    gt_resized = resize_to_height(gt_img, target_height) if gt_img else None
    before_resized = resize_to_height(before_img, target_height)
    after_resized = resize_to_height(after_img, target_height)
    
    # 计算总宽度和标题高度
    title_height = 40
    padding = 20
    
    images = []
    titles = []
    if gt_resized:
        images.append(gt_resized)
        titles.append("Ground Truth")
    images.append(before_resized)
    titles.append("Before Layout Fix")
    images.append(after_resized)
    titles.append("After Layout Fix")
    
    total_width = sum(img.width for img in images) + padding * (len(images) + 1)
    total_height = target_height + title_height + padding * 2
    
    # 创建画布
    comparison = Image.new("RGB", (total_width, total_height), color=(255, 255, 255))
    
    # 绘制图片和标题
    draw = ImageDraw.Draw(comparison)
    
    # 尝试加载字体
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()
    
    x_offset = padding
    for img, title in zip(images, titles):
        # 绘制标题
        text_bbox = draw.textbbox((0, 0), title, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_x = x_offset + (img.width - text_width) // 2
        draw.text((text_x, padding // 2), title, fill=(0, 0, 0), font=font)
        
        # 粘贴图片
        comparison.paste(img, (x_offset, title_height + padding))
        x_offset += img.width + padding
    
    # 保存对比图
    comparison.save(output_path)
    print(f"✅ Comparison image saved to {output_path}")


def plot_distribution(block_num_list, title="Block Number Distribution"):
    plt.figure(figsize=(10, 5))

    bins = range(0, 16)
    counts, edges, patches = plt.hist(
        block_num_list, bins=bins, alpha=0.7, color="skyblue", edgecolor="black", align="left", rwidth=0.8
    )

    plt.title(title)
    plt.xlabel("Block Number")
    plt.ylabel("Frequency")

    plt.xticks(range(0, 16))

    for x, count in zip(range(0, 15), counts):
        if count > 0:
            plt.text(x, count, str(int(count)), ha="center", va="bottom", fontsize=9)

    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()


def count_tokens_from_file(file_path, model="gpt-4o-mini"):
    """
    读取文件，用正则提取 vue/jsx/html 内容再统计 token
    """
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    # 匹配 vue/jsx/html 块内容
    pattern = re.compile(r"```vue\s*(.*?)```", re.DOTALL | re.IGNORECASE)
    matches = pattern.findall(text)

    # 合并所有匹配的内容
    combined_text = "\n".join(matches)

    enc = tiktoken.encoding_for_model(model)
    tokens = enc.encode(combined_text)

    return len(tokens)


def extract_and_save_components(txt_path, block_framework="vue"):
    """
    Read a model output .txt file, extract component code blocks,
    and save them into components/{idx}/ directory.

    Args:
        txt_path (str): Path to the model output txt file (e.g., "1.txt")
        default_framework (str): Default framework to use if not detected or mismatched (default "vue")
    """
    # Use the txt file name (without extension) as idx
    idx = os.path.splitext(os.path.basename(txt_path))[0]

    with open(txt_path, "r", encoding="utf-8") as f:
        model_output = f.read()

    os.makedirs("components", exist_ok=True)

    # Regex pattern for extracting groups
    pattern = re.compile(
        r"'''(?:vue|jsx|html)\s*"
        r"Group ID:\s*(?P<id>.+?)\s*"
        r"Description:\s*(?P<desc>.+?)\s*"
        r"Framework:\s*(?P<framework>.+?)\s*"
        r"Code:\s*(?P<code>.*?)'''",
        re.DOTALL
    )

    for match in pattern.finditer(model_output):
        group_id = match.group("id").strip()
        code = match.group("code").strip()

        save_dir = os.path.join("components", idx)
        os.makedirs(save_dir, exist_ok=True)


        # Decide file extension
        if block_framework == "vue":
            ext = ".vue"

        filename = os.path.join(save_dir, f"{group_id}{ext}")
        with open(filename, "w", encoding="utf-8") as f:
            f.write(f"<!-- {match.group('desc').strip()} -->\n")
            f.write(code)

        print(f"✅ Saved {filename}")




def parse_vue_from_txt_for_designbench(base_name, txt_path, proj_dir, pattern):
    with open(txt_path, "r", encoding="utf-8") as f:
        content = f.read()

    pattern_full = re.compile(rf"```{pattern}\s*(.*?)\s*```", re.DOTALL)
    matches = pattern_full.findall(content)

    for block in matches:
        file_name = f"App.vue"
        file_path = os.path.join(proj_dir, "src", file_name)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(block.strip())

        print(f"✅ Saved {file_name}")

    return True





def generate_app_file(components_dir, vue_files, framework="vue"):

    if framework.lower() == "vue":
        app_file = os.path.join(components_dir,"App.vue")

        tmp_app_file = os.path.join(components_dir, "App.vue.tmp")
        template_lines = ["<template>", "  <div id=\"app\">"]
        script_lines = ["<script setup>"]

        for f_name in vue_files:
            comp_name = os.path.splitext(f_name)[0]
            template_lines.append(f"    <{comp_name} />")
            script_lines.append(f"import {comp_name} from './{f_name}'")

        template_lines.append("  </div>")
        template_lines.append("</template>")
        script_lines.append("</script>")

        with open(tmp_app_file, "w", encoding="utf-8") as f:
            f.write("\n".join(template_lines + script_lines))

        print(f"✅ Generated temporary App.vue at {tmp_app_file}")
        return tmp_app_file


def screenshot_from_html(html_file_path, screenshot_file, geckodriver_path):
    """
    从已有的 HTML 文件生成截图
    
    Args:
        html_file_path: HTML 文件路径
        screenshot_file: 截图保存路径
        geckodriver_path: geckodriver 路径
    """
    if not os.path.exists(html_file_path):
        raise FileNotFoundError(f"HTML file not found: {html_file_path}")
    
    options = FirefoxOptions()
    options.add_argument("--headless")
    options.binary_location = r"C:\Program Files\Mozilla Firefox\firefox.exe"
    driver = webdriver.Firefox(service=FirefoxService(geckodriver_path), options=options)
    
    try:
        # 使用 file:// 协议打开本地 HTML 文件
        html_file_uri = Path(html_file_path).as_uri()
        driver.get(html_file_uri)
        time.sleep(3)  # 等待页面加载完成
        
        driver.get_full_page_screenshot_as_file(screenshot_file)
        print(f"Screenshot saved to {screenshot_file}")
    finally:
        driver.quit()


def render_vue_with_extracted_dom(project_dir, vue_dir, html_file_path, screenshot_file, geckodriver_path):
    """
    渲染 Vue 文件并生成截图和 HTML（使用提取渲染后 DOM 的方法）
    
    关键步骤：
    1. 先从 Vite server 提取渲染后的 DOM 和样式，保存为 HTML
    2. 再用保存的 HTML 文件截图（确保截图和 HTML 内容一致）
    """
    src_dir = os.path.join(project_dir, "src")
    if not os.path.exists(project_dir):
        raise FileNotFoundError(f"Template project not found at {project_dir}, try npm init + npm install")

    copied_items = []
    for item in os.listdir(vue_dir):
        if item.lower().endswith(".vue"):
            s = os.path.join(vue_dir, item)
            d = os.path.join(src_dir, item)
            shutil.copy2(s, d)
            copied_items.append(item)

    npm_cmd = "npm"
    if os.name == "nt" and not shutil.which("npm"):
        npm_cmd = r"D:\nodejs\node_global\npm.cmd"

    proc = subprocess.Popen([npm_cmd, "run", "dev"], cwd=project_dir, shell=True)
    url = "http://localhost:5173"

    server_started = False
    for _ in range(15):
        try:
            requests.get(url)
            server_started = True
            break
        except:
            time.sleep(1)

    if not server_started:
        print("❌ Dev server did not start in time.")
        proc.terminate()
        proc.wait()
        return

    options = FirefoxOptions()
    options.add_argument("--headless")
    options.binary_location = r"C:\Program Files\Mozilla Firefox\firefox.exe"
    driver = webdriver.Firefox(service=FirefoxService(geckodriver_path), options=options)

    try:
        driver.get(url)
        time.sleep(5)
        
        # 等待页面完全加载
        ready_state = driver.execute_script("return document.readyState")
        time.sleep(2)
        
        # 检查是否有 Vite 错误
        error_html = driver.execute_script("""
            const overlay = document.querySelector('vite-error-overlay');
            if (!overlay) return null;
            const shadow = overlay.shadowRoot;
            return shadow ? shadow.innerHTML : null;
        """)
        
        # 1. 获取渲染后的完整 HTML（包含实际渲染的 DOM 和内联样式）
        try:
            rendered_html = driver.execute_script("""
                // 获取所有样式规则
                let styles = '';
                for (let sheet of document.styleSheets) {
                    try {
                        for (let rule of sheet.cssRules) {
                            styles += rule.cssText + '\\n';
                        }
                    } catch(e) {
                        // 跨域样式表无法访问，跳过
                    }
                }
                
                // 获取 #app 内部渲染后的实际 DOM 内容
                const appElement = document.getElementById('app');
                const renderedContent = appElement ? appElement.innerHTML : document.body.innerHTML;
                
                // 构建自包含的静态 HTML
                let html = '<!DOCTYPE html>\\n';
                html += '<html lang="en">\\n<head>\\n';
                html += '<meta charset="UTF-8">\\n';
                html += '<meta name="viewport" content="width=device-width, initial-scale=1.0">\\n';
                html += '<title>Generated Page</title>\\n';
                html += '<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">\\n';
                html += '<style>\\n' + styles + '\\n</style>\\n';
                html += '</head>\\n<body>\\n';
                html += '<div id="app">' + renderedContent + '</div>\\n';
                html += '</body>\\n</html>';
                return html;
            """)
        except Exception as e:
            print(f"Warning: Failed to extract rendered HTML: {e}")
            rendered_html = driver.page_source
        
        if error_html:
            rendered_html += f"\n<!-- Vite Error Overlay -->\n<div id='__vite_error_overlay__'>{error_html}</div>\n"

        # 2. 保存 HTML 文件
        with open(html_file_path, "w", encoding="utf-8") as f:
            f.write(rendered_html)
        print(f"HTML saved to {html_file_path}")
        
        # 3. 用保存的 HTML 文件截图（确保截图和 HTML 内容一致）
        html_file_uri = Path(html_file_path).as_uri()
        driver.get(html_file_uri)
        time.sleep(3)  # 等待静态 HTML 加载完成
        driver.get_full_page_screenshot_as_file(screenshot_file)
        print(f"Screenshot saved to {screenshot_file}")
        
    finally:
        for item in copied_items:
            path = os.path.join(src_dir, item)
            if os.path.exists(path):
                os.remove(path)
        driver.quit()
        proc.terminate()
        proc.wait()


def render_single_vue_file(project_dir, vue_dir, html_file_path, screenshot_file, geckodriver_path):
    src_dir = os.path.join(project_dir, "src")
    if not os.path.exists(project_dir):
        raise FileNotFoundError(f"Template project not found at {project_dir}, try npm init + npm install")

    copied_items = []
    for item in os.listdir(vue_dir):
        if item.lower().endswith(".vue"):
            s = os.path.join(vue_dir, item)
            d = os.path.join(src_dir, item)
            shutil.copy2(s, d)
            copied_items.append(item)

    npm_cmd = "npm"
    if os.name == "nt" and not shutil.which("npm"):
        npm_cmd = r"D:\nodejs\node_global\npm.cmd"

    proc = subprocess.Popen([npm_cmd, "run", "dev"], cwd=project_dir, shell=True)
    url = "http://localhost:5173"

    server_started = False
    for _ in range(10):
        try:
            requests.get(url)
            server_started = True
            break
        except:
            time.sleep(1)

    if not server_started:
        print("❌ Dev server did not start in time.")
        proc.terminate()
        proc.wait()
        return

    options = FirefoxOptions()
    options.add_argument("--headless")
    options.binary_location = r"C:\Program Files\Mozilla Firefox\firefox.exe"
    driver = webdriver.Firefox(service=FirefoxService(geckodriver_path), options=options)

    try:
        driver.get(url)
        time.sleep(5)
        
        # 等待页面完全加载
        ready_state = driver.execute_script("return document.readyState")
        time.sleep(2)
        
        # 检查是否有 Vite 错误
        error_html = driver.execute_script("""
            const overlay = document.querySelector('vite-error-overlay');
            if (!overlay) return null;
            const shadow = overlay.shadowRoot;
            return shadow ? shadow.innerHTML : null;
        """)
        
        # 1. 先在 Vite dev server 上截图（这是真实的渲染效果）
        driver.get_full_page_screenshot_as_file(screenshot_file)
        print(f"Screenshot saved to {screenshot_file}")
        
        # 2. 获取渲染后的完整 HTML（包含实际渲染的 DOM 和内联样式）
        try:
            rendered_html = driver.execute_script("""
                // 获取所有样式规则
                let styles = '';
                for (let sheet of document.styleSheets) {
                    try {
                        for (let rule of sheet.cssRules) {
                            styles += rule.cssText + '\\n';
                        }
                    } catch(e) {
                        // 跨域样式表无法访问，跳过
                    }
                }
                
                // 获取 #app 内部渲染后的实际 DOM 内容
                const appElement = document.getElementById('app');
                const renderedContent = appElement ? appElement.innerHTML : document.body.innerHTML;
                
                // 构建自包含的静态 HTML
                let html = '<!DOCTYPE html>\\n';
                html += '<html lang="en">\\n<head>\\n';
                html += '<meta charset="UTF-8">\\n';
                html += '<meta name="viewport" content="width=device-width, initial-scale=1.0">\\n';
                html += '<title>Generated Page</title>\\n';
                html += '<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">\\n';
                html += '<style>\\n' + styles + '\\n</style>\\n';
                html += '</head>\\n<body>\\n';
                html += '<div id="app">' + renderedContent + '</div>\\n';
                html += '</body>\\n</html>';
                return html;
            """)
        except Exception as e:
            print(f"Warning: Failed to extract rendered HTML: {e}")
            rendered_html = driver.page_source
        
        if error_html:
            rendered_html += f"\n<!-- Vite Error Overlay -->\n<div id='__vite_error_overlay__'>{error_html}</div>\n"

        with open(html_file_path, "w", encoding="utf-8") as f:
            f.write(rendered_html)
        print(f"HTML saved to {html_file_path}")
    finally:
        for item in copied_items:
            path = os.path.join(src_dir, item)
            if os.path.exists(path):
                os.remove(path)
        driver.quit()
        proc.terminate()
        proc.wait()


def extract_component_name(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    name_match = re.search(r'export\s+default\s*{[^}]*?\bname\s*:\s*[\'"]([^\'"]+)[\'"]', content, re.DOTALL)
    if name_match:
        comp_name = name_match.group(1).strip()
    else:
        name_match = re.search(r'defineOptions\(\s*{[^}]*name:\s*["\']([^"\']+)["\']', content)  # for original
        if name_match:
            comp_name = name_match.group(1).strip()
        else:
            comp_name = "UnknownComponent"
    return comp_name


# def render_single_vue_file(project_dir, html_file_path, screenshot_file, geckodriver_path, start_port=5173, max_tries=10):
#     src_dir = os.path.join(project_dir, "src")
#     app_vue = os.path.join(src_dir, "App.vue")
#     if not os.path.exists(app_vue):
#         raise FileNotFoundError(f"{app_vue} 不存在，Vite 需要入口文件")
#
#     npm_cmd = "npm"
#     if os.name == "nt" and not shutil.which("npm"):
#         npm_cmd = r"D:\nodejs\node_global\npm.cmd"
#
#     for port in range(start_port, start_port + max_tries):
#         url = f"http://localhost:{port}"
#         try:
#             requests.get(url, timeout=1)
#         except:
#             selected_port = port
#             break
#     else:
#         selected_port = start_port
#
#
#     # 启动 Vite dev server
#     env = os.environ.copy()
#     env["PORT"] = str(selected_port)
#     print(f"启动 Vite dev server，端口 {selected_port} ...")
#     proc = subprocess.Popen([npm_cmd, "run", "dev"], cwd=project_dir, shell=True, env=env)
#
#     server_up = False
#     for _ in range(10):
#         try:
#             r = requests.get(url, timeout=1)
#             if r.status_code:
#                 server_up = True
#                 break
#         except:
#             time.sleep(1)
#
#     if not server_up:
#         print("❌ Dev server 启动失败")
#         proc.terminate()
#         proc.wait()
#         return
#
#     # 保存 HTML
#     print(f"使用 single-file 保存 HTML 到 {html_file_path} ...")
#     os.system(f"npx single-file {url} {html_file_path}")
#
#     proc.terminate()
#     proc.wait()
#
#     print("✅ HTML 和截图保存完成")


def segmentation_bbox_call(root_dir, base_name):
    #seg_dir = os.path.join("segmentation", base_name,f"splits_{base_name}")
    seg_dir = os.path.join(root_dir, f"{base_name}_split")

    filename_list = [f for f in os.listdir(seg_dir) if f.lower().endswith((".png", ".jpg", ".jpeg")) and "with_boxes" not in f.lower()]
    print(filename_list)
    for idx, file in enumerate(filename_list):
        image_path = os.path.join(seg_dir, file)
        file_base_name = os.path.splitext(file)[0]
        json_file_path = os.path.join(seg_dir, f"{file_base_name}.json")
        image_path_with_box_merge = os.path.join(seg_dir, f"{file_base_name}_with_boxes.png")
        json_path_with_box_merge = os.path.join(seg_dir, f"{file_base_name}_positions_merge.json")
        
        # 调用 LLM 生成 bbox
        success_flag = bbox_json_generation(SYSTEM_PROMPT_FOR_BBOX, idx, image_path, json_file_path, model="gemini-2.5-pro")
        
        group_boxes = None
        if success_flag:
            # Reprocess the model's output (Resize & Group)
            merge_json_path = process_single_json(json_file_path, image_path, seg_dir, file_base_name)
            if merge_json_path:
                with open(merge_json_path , "r", encoding="utf-8") as f:
                    group_boxes = json.load(f)
            if not group_boxes:
                print("No group boxes found, skipping plotting.")
            else:
                bounding_boxes = [BoundingBox(b["bbox_2d"], b.get("label", None)) for b in group_boxes]
                plot_bounding_boxes(image_path, bounding_boxes, image_path_with_box_merge)
                print(f"Plotted bounding boxes saved to {image_path_with_box_merge}")


def cropped_vue_gen_call(crop_dir, id, base_name, proj_dir, webdriver_path):
    filename_list = [f for f in os.listdir(crop_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    # generate frame first
    filename_list.sort(key=lambda fname: 0 if os.path.splitext(fname)[0].lower() == "masked_image" else 1)
    print(filename_list)

    vue_dir = os.path.join(crop_dir, "vue")
    os.makedirs(vue_dir, exist_ok=True)
    result_dir = os.path.join(crop_dir, "result")
    os.makedirs(result_dir, exist_ok=True)
    screenshot_file_path = os.path.join(result_dir, f"{base_name}.png")
    html_file_path = os.path.join(result_dir, f"{base_name}.html")


    fail_cases=[]
    name_list = []   # in case model generate duplicate names
    success_flag = True
    for idx, file in enumerate(filename_list):
        image_path = os.path.join(crop_dir, file)
        base_name = os.path.splitext(file)[0]
        print(base_name)
        # if base_name == "masked_image":
        #     continue
        output_file_for_code = os.path.join(crop_dir, f"{base_name}.txt")
        # skip if the file has already existed
        if os.path.exists(output_file_for_code):
            if base_name != "masked_image":
                name = extract_component_name(output_file_for_code)
                name_list.append(name)
                print(name_list)
                parse_vue_from_txt(output_file_for_code, vue_dir, "component")
                parse_vue_from_txt(output_file_for_code, vue_dir, "snippet")
            else:
                parse_vue_from_txt(output_file_for_code, vue_dir, "vue")
            continue

        else:
            if base_name != "masked_image":
                success_flag = code_generation(COMPONENT_GEN_SYSTEM_PROMPT, idx, image_path, output_file_for_code,
                                            name_list, model="gemini-2.5-pro")
                if not success_flag:
                    break
                name = extract_component_name(output_file_for_code)
                name_list.append(name)
                print(name_list)
                parse_vue_from_txt(output_file_for_code, vue_dir, "snippet")
                parse_vue_from_txt(output_file_for_code, vue_dir, "component")
            else:
                success_flag = code_generation(MASKED_GEN_SYSTEM_PROMPT, idx, image_path, output_file_for_code,
                                               model="gemini-2.5-pro")
                if not success_flag:
                    break
                parse_vue_from_txt(output_file_for_code, vue_dir, "vue")

    if not success_flag:
        fail_cases.append(id)
    print(fail_cases)

    post_process(vue_dir)
    render_single_vue_file(proj_dir, vue_dir, html_file_path, screenshot_file_path, webdriver_path)



def cropped_vue_gen_call_multi(clusters_keys, output_dir):
    fail_cases=[]
    name_list = []   # in case model generate duplicate names
    success_flag = True
    name_list=[]
    for idx, cluster in enumerate(clusters_keys):
        id=idx
        # if base_name == "masked_image":
        #     continue
        output_file_for_code = os.path.join(output_dir, f"cluster_{idx+1}.txt")
        group_name_list=[]
        page_dirs = set()
        for path in cluster:
            norm_path = os.path.normpath(path)
            page_dir = os.path.dirname(os.path.dirname(norm_path))
            page_dirs.add(page_dir)
            file_group = os.path.basename(page_dir)
            basename = os.path.splitext(os.path.basename(norm_path))[0]
            name = f"page_{file_group}_{basename}"
            group_name_list.append(name)
        print(page_dirs)
        print(group_name_list)
        # skip if the file has already existed
        # if os.path.exists(output_file_for_code):
        #     parse_vue_from_txt(output_file_for_code, vue_dir, "vue")
        #     continue
        if not os.path.exists(output_file_for_code):
            success_flag = code_generation_multipage(MULTI_COMPONENT_GEN_SYSTEM_PROMPT, idx, cluster, output_file_for_code,
                                                 group_name_list, name_list, model="gemini-2.5-pro")
        name = extract_component_name(output_file_for_code)
        name_list.append(name)
        if not success_flag:
            break
        for page_dir in page_dirs:
            vue_dir = os.path.join(page_dir, "vue")
            os.makedirs(vue_dir, exist_ok=True)
            parse_vue_from_txt(output_file_for_code, vue_dir, "component")
            len=parse_vue_from_txt(output_file_for_code, vue_dir, "snippet", multipage_flag=True)
            print(len)

    if not success_flag:
        fail_cases.append(id)
    print(fail_cases)


def component_render(base_name, proj_dir, webdriver_path):
    crop_dir = r"D:\py_code\fyp\VueGen\output_seg\crop"
    crop_dir = os.path.join(crop_dir, f"{base_name}_cropped")

    filename_list = [f for f in os.listdir(crop_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    # generate frame first
    filename_list.sort(key=lambda fname: 0 if os.path.splitext(fname)[0].lower() == "masked_image" else 1)
    print(filename_list)


    result_dir = os.path.join(crop_dir, "components")
    os.makedirs(result_dir, exist_ok=True)
    vue_dir = os.path.join(crop_dir, "vue")
    os.makedirs(vue_dir, exist_ok=True)

    for idx, file in enumerate(filename_list):

        image_path = os.path.join(crop_dir, file)
        base_name = os.path.splitext(file)[0]
        screenshot_file_path = os.path.join(result_dir, f"{base_name}.png")
        html_file_path = os.path.join(result_dir, f"{base_name}.html")
        print(base_name)
        output_file_for_code = os.path.join(crop_dir, f"{base_name}.txt")

        # the following code is designed to render component part
        if base_name == "masked_image":
            continue
        try:
            parse_vue_from_txt_component(output_file_for_code, vue_dir, "component")
            parse_vue_from_txt_component(output_file_for_code, vue_dir, "snippet")
            vue_component_dir=os.path.join(vue_dir,"components")
            render_single_vue_file(proj_dir, vue_component_dir, html_file_path, screenshot_file_path, webdriver_path)
        except:
            continue

    #shutil.rmtree(vue_component_dir, ignore_errors=True)






# seemingly useless??
def seg_code_gen_call(root_dir, proj_dir, webdriver_path, image_files):

    num_list = []
    total_images = len(image_files)
    for idx, filename in enumerate(image_files, start=1):
        if idx != 1:  # ,99
            continue

        print(f"Processing image {idx}/{total_images}...")
        base_name, _ = os.path.splitext(filename)
        print(base_name)

        seg_dir = os.path.join("segmentation", base_name, f"splits_{base_name}")
        filename_list = [f for f in os.listdir(seg_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]

        output_dir = os.path.join(root_dir, base_name)
        os.makedirs(output_dir, exist_ok=True)
        vue_dir = os.path.join(output_dir, "vue")
        os.makedirs(vue_dir, exist_ok=True)
        json_file_path = os.path.join(output_dir, f"{base_name}_positions.json")
        image_path_with_box_org = os.path.join(output_dir, f"{base_name}_with_boxes_org.png")
        image_path_with_box_merge = os.path.join(output_dir, f"{base_name}_with_boxes_merge.png")
        output_file_for_code = os.path.join(output_dir, f"{base_name}_all_components.txt")
        screenshot_file_path = os.path.join(output_dir, f"{base_name}_all_components_screenshot_vue_based.png")
        html_file_path = os.path.join(output_dir, f"{base_name}.html")

        for idx, file in enumerate(filename_list):
            image_path = os.path.join(seg_dir, file)
            base_name = os.path.splitext(file)[0]
            json_file_path = os.path.join(output_dir, f"{base_name}.json")
            image_path_with_box_merge = os.path.join(output_dir, f"{base_name}_with_boxes.png")

            # draw bounding boxes
            merge_json_path = process_single_json(json_file_path, image_path, output_dir, base_name)
            if merge_json_path:
                with open(merge_json_path, "r", encoding="utf-8") as f:
                    group_boxes = json.load(f)

            if not group_boxes:
                print("No group boxes found, skipping plotting.")
                image_path_with_box = image_path
            else:

                bounding_boxes = [BoundingBox(b["bbox_2d"], b.get("label", None)) for b in group_boxes]

                plotted_image = plot_bounding_boxes(image_path, bounding_boxes, image_path_with_box_merge)
                print(f"Plotted bounding boxes saved to {image_path_with_box_merge}")

        # code gen
        code_generation(GENERATION_VUE_TAILWIND_SYSTEM_PROMPT, idx, image_path, output_file_for_code,  model="gemini-2.5-pro")
        parse_vue_from_txt(output_file_for_code, proj_dir)
        #token_count = count_tokens_from_file(output_file_for_code, model="gpt-4o-mini")
        #print(token_count)
        #num_list.append(token_count)
        parse_vue_from_txt_for_designbench(base_name, output_file_for_code, proj_dir)

        render_single_vue_file(proj_dir,vue_dir, html_file_path, screenshot_file_path, webdriver_path)  # 渲染
        continue



def main():
    data_dir = "./data"
    multipage_dir = "./multipage_data"
    seg_dir = "./output_seg"
    proj_dir= "D:/py/vue_template"                                        #"D:/py/vue_template"
    proj_dir_for_db="D:/py_code/fyp/DesignBench/web/my-vue-app"
    webdriver_path="D:/py_code/fyp/geckodriver.exe"

    # detect all images (jpg/png) in folder
    image_files = sorted(
        [f for f in os.listdir(multipage_dir) if f.lower().endswith(('.jpg', '.png'))],
        key=lambda x: int(re.findall(r'\d+', x)[0])
    )

    total_images = len(image_files)

    num = 0
    num_list=[]
    fail_cases=[]

    for idx, filename in enumerate(image_files, start=1):
        if idx > 118 or idx<89:    #,idx > 3 or
            continue
        print(f"Processing image {idx}/{total_images}...")
        base_name, _ = os.path.splitext(filename)


        # generate bbox
        #segmentation_bbox_call(seg_dir, base_name)

        # crop the image

        # generate vue code by cropped images
        crop_dir = r"D:\py_code\fyp\VueGen\output_seg\crop"
        crop_dir = os.path.join(crop_dir, f"{base_name}_cropped")
        #cropped_vue_gen_call(crop_dir, idx, base_name, proj_dir, webdriver_path)

        component_render(base_name, proj_dir, webdriver_path)


        continue


        # TODO: Then compare the screenshot with GT

        num += 1

        if num % 10 == 0:
            print(f"Processed {num} images")
    print(round(sum(num_list)/118,4))
    print("_" * 80)
    print(f"Processing complete, processed {num} images")


def main_multi():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # 基础路径配置
    base_dir = r"C:\Users\Shuoqi\Documents\GitHub\VueGen\VueGen"
    multipage_dir = os.path.join(base_dir, "2")  # 测试数据目录
    proj_dir = os.path.join(base_dir, "my-vue-app")  # Vue 模板项目
    seg_dir = os.path.join(base_dir, "output_test")  # 输出目录
    proj_dir_for_db = os.path.join(base_dir, "my-vue-app")
    
    # geckodriver 路径
    webdriver_path = r"C:\Program Files\geckodriver.exe"

    # 直接检测 2 文件夹下的图片文件（1.png, 2.png, 3.png, 4.png）
    gt_folder_path = multipage_dir
    image_files = sorted(
        [f for f in os.listdir(gt_folder_path) if f.lower().endswith(('.jpg', '.png')) and not f.endswith('_marked.png')],
        key=lambda x: int(re.findall(r'\d+', x)[0]) if re.findall(r'\d+', x) else 0
    )
    total_images = len(image_files)
    print(f"Found {total_images} images in {gt_folder_path}: {image_files}")
    
    # 使用 "main" 作为输出文件夹名
    folder = "main"
    folder_idx = 1
    all_items=[]
    item_dict = {}
    gen_folder_path = os.path.join(seg_dir, folder)  # 使用 main 作为输出文件夹名
    os.makedirs(gen_folder_path, exist_ok=True)
    
    for file_idx, filename in enumerate(image_files, start=1):
        if file_idx > 4:  # 只处理前 4 张图片进行测试
            continue
        print(f"Processing image {file_idx}/{total_images}: {filename}")
        base_name, _ = os.path.splitext(filename)
        split_save_dir = os.path.join(seg_dir, folder, str(file_idx))
        
        # Step 1: 分割图片
        img_path = os.path.join(gt_folder_path, filename)
        split_subdir = os.path.join(split_save_dir, f"{file_idx}_split")  # 分割图片保存在子目录
        os.makedirs(split_subdir, exist_ok=True)
        find_split_lines_color(img_path, basename=str(file_idx), save_dir=split_subdir)
        
        # 把 split_lines.json 也复制到父目录（merge_split_jsons_with_offset 需要）
        split_lines_src = os.path.join(split_subdir, "split_lines.json")
        split_lines_dst = os.path.join(split_save_dir, "split_lines.json")
        if os.path.exists(split_lines_src):
            shutil.copy(split_lines_src, split_lines_dst)
        
        # Step 1.5: 为分割后的图片生成 bbox
        segmentation_bbox_call(split_save_dir, str(file_idx))
        
        # Step 2: 合并分割后的 bbox JSON
        split_h_path = os.path.join(split_save_dir, "split_lines.json")
        output_json_path = os.path.join(split_save_dir, f"{file_idx}_test_modify.json")
        output_image_path = os.path.join(split_save_dir, f"{file_idx}_test_modify.png")
        ref_json_path = os.path.join(base_dir, "data", "output", str(folder_idx), "merge", f"{file_idx}.json")
        json_path = merge_split_jsons_with_offset(split_save_dir, file_idx)
        
        # 如果参考 JSON 存在，则调整 bbox；否则直接使用合并后的结果
        if os.path.exists(ref_json_path):
            adjust_main_boxes(json_path, ref_json_path, split_h_path, img_path, output_json_path)
        else:
            print(f"  Warning: Reference JSON not found at {ref_json_path}, using merged JSON directly")
            # 直接复制 json_path 到 output_json_path
            shutil.copy(json_path, output_json_path)
        
        draw_boxes_from_json(output_json_path, img_path, output_image_path)
        
        # Step 3: 裁剪图片
        crop_output_dir = os.path.join(split_save_dir, f"{file_idx}_cropped")
        crop_paths, bboxes, masked_path = crop_and_mask_image(img_path, output_json_path, crop_output_dir, mask_color=(200, 200, 200))
        items = list(zip(crop_paths, bboxes))  # 组合成 (crop_path, bbox) 的列表
        
        output_file_for_code = os.path.join(split_save_dir, f"masked_image.txt")
        vue_dir = os.path.join(split_save_dir, "vue")
        os.makedirs(vue_dir, exist_ok=True)
        
        # Step 4: 为 masked image 生成代码
        if not os.path.exists(output_file_for_code):
            code_generation(MASKED_GEN_SYSTEM_PROMPT, file_idx, masked_path, output_file_for_code, model="gemini-2.5-pro")
        parse_vue_from_txt(output_file_for_code, vue_dir, "vue")
        
        # Step 5: UIED 检测并合并组件
        uied_output_root = os.path.join(split_save_dir, "uied")
        uied(crop_output_dir, uied_output_root, is_ip=True, is_ocr=True, is_merge=True)
        for crop_path, bbox in items:
            base = os.path.basename(crop_path)
            key = f"{file_idx}_{base}"
            json_base = base.replace(".png", ".json")
            # UIED 生成的文件名就是原始图片名（不带目录前缀）
            json_path = os.path.join(split_save_dir, "uied", "merge", json_base)
            print(json_path)
            item_dict[key] = [crop_path, bbox, json_path]
    print(item_dict)

    # Step 6: 计算组件相似度并聚类
    sim_matrix, clusters_keys = compute_group_merge(item_dict, device, merge_threshold=0.8)
    base_tmp = os.path.join(seg_dir, "tmp")
    save_cluster_images(clusters_keys, base_tmp)
    print(clusters_keys)

    output_dir = os.path.join(gen_folder_path, "tmp")
    os.makedirs(output_dir, exist_ok=True)
    cropped_vue_gen_call_multi(clusters_keys, output_dir)

    # ============================================================
    # Step 6.5: 把 cluster txt 拆分到各页面的 _cropped 目录
    # 这样 feedback_loop_for_page 可以找到对应的 txt 和 GT 图片
    # ============================================================
    print("\n" + "=" * 80)
    print("Splitting cluster txt files to page directories")
    print("=" * 80)
    
    import glob
    cluster_txt_files = glob.glob(os.path.join(output_dir, "cluster_*.txt"))
    
    for cluster_txt in cluster_txt_files:
        with open(cluster_txt, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 提取 component 部分
        component_match = re.search(r'```component\s*(.*?)\s*```', content, re.DOTALL)
        component_code = component_match.group(1).strip() if component_match else ""
        
        # 提取所有 snippet 部分
        snippet_pattern = re.compile(r'```snippet\s*<!--\s*(page_(\d+)_(.+?))\s*-->\s*(.*?)\s*```', re.DOTALL)
        
        for match in snippet_pattern.finditer(content):
            full_name = match.group(1).strip()  # page_1_banner_crop
            page_idx = match.group(2)  # 1
            component_name = match.group(3).strip()  # banner_crop
            snippet_code = match.group(4).strip()
            
            # 构建目标路径
            page_crop_dir = os.path.join(seg_dir, folder, page_idx, f"{page_idx}_cropped")
            
            if not os.path.exists(page_crop_dir):
                print(f"  Warning: {page_crop_dir} does not exist, skipping")
                continue
            
            # 检查对应的 GT 图片是否存在
            gt_image = os.path.join(page_crop_dir, f"{component_name}.png")
            if not os.path.exists(gt_image):
                print(f"  Warning: GT image not found: {gt_image}")
                continue
            
            # 保存拆分后的 txt 文件
            output_txt = os.path.join(page_crop_dir, f"{component_name}.txt")
            with open(output_txt, 'w', encoding='utf-8') as f:
                f.write(f"```component\n{component_code}\n```\n\n")
                f.write(f"```snippet\n<!-- {full_name} -->\n{snippet_code}\n```\n")
            
            print(f"  Saved: {output_txt}")
    
    print("Cluster splitting complete.")

    # ============================================================
    # Step 1: Component Fix - 对每个页面的组件进行反馈修复
    # ============================================================
    print("\n" + "=" * 80)
    print("Starting Component Fix Phase")
    print("=" * 80)
    
    for file_idx, filename in enumerate(image_files, start=1):
        if file_idx > 4:
            continue
        print(f"\n--- Component Fix for Page {file_idx} ---")
        split_save_dir = os.path.join(seg_dir, folder, str(file_idx))
        crop_dir = os.path.join(split_save_dir, f"{file_idx}_cropped")
        vue_dir_for_fix = os.path.join(split_save_dir, "vue")
        
        # 调用 feedback_loop_for_page 对组件进行修复
        from feedback import feedback_loop_for_page
        try:
            fix_result = feedback_loop_for_page(
                page_dir=crop_dir,
                vue_dir=vue_dir_for_fix,
                proj_dir=proj_dir,
                webdriver_path=webdriver_path,
                max_iterations=3,
                model="gemini-2.5-pro",
                top_k=3  # 修复得分最低的 3 个组件
            )
            print(f"Component fix for page {file_idx} complete.")
            if fix_result and 'fixed_results' in fix_result:
                for name, res in fix_result['fixed_results'].items():
                    if res:
                        print(f"  - {name}: Score = {res.get('match_score', 0):.4f}")
        except Exception as e:
            print(f"Component fix failed for page {file_idx}: {e}")

    # ============================================================
    # Step 2: 渲染整个网页 (布局修复前)
    # ============================================================
    print("\n" + "=" * 80)
    print("Rendering Full Pages Before Layout Fix")
    print("=" * 80)
    
    for file_idx, filename in enumerate(image_files, start=1):
        if file_idx > 4:
            continue
        split_save_dir = os.path.join(seg_dir, folder, str(file_idx))
        vue_dir = os.path.join(split_save_dir, "vue")
        post_process(vue_dir)
        screenshot_file_path = os.path.join(split_save_dir, f"{file_idx}_before_layout_fix.png")
        html_file_path = os.path.join(split_save_dir, f"{file_idx}_before_layout_fix.html")
        render_vue_with_extracted_dom(proj_dir, vue_dir, html_file_path, screenshot_file_path, webdriver_path)
        print(f"Rendered page {file_idx} (before layout fix): {screenshot_file_path}")

    # ============================================================
    # Step 3: Layout Fix - 对整体布局进行修复
    # ============================================================
    print("\n" + "=" * 80)
    print("Starting Layout Fix Phase")
    print("=" * 80)
    
    from layout_fix import build_messages, request_fix, extract_code_blocks, save_app_snippet
    
    for file_idx, filename in enumerate(image_files, start=1):
        if file_idx > 4:
            continue
        print(f"\n--- Layout Fix for Page {file_idx} ---")
        
        split_save_dir = os.path.join(seg_dir, folder, str(file_idx))
        vue_dir = os.path.join(split_save_dir, "vue")
        app_vue_path = os.path.join(vue_dir, "App.vue")
        
        # Ground truth image
        gt_image_path = os.path.join(gt_folder_path, f"{file_idx}.png")
        # Current rendered screenshot (before layout fix)
        pred_image_path = os.path.join(split_save_dir, f"{file_idx}_before_layout_fix.png")
        # Output for fixed response
        fixed_output_path = os.path.join(split_save_dir, f"{file_idx}_layout_fixed.txt")
        
        if not os.path.exists(app_vue_path):
            print(f"  Skipping: App.vue not found at {app_vue_path}")
            continue
        if not os.path.exists(gt_image_path):
            print(f"  Skipping: GT image not found at {gt_image_path}")
            continue
        if not os.path.exists(pred_image_path):
            print(f"  Skipping: Rendered image not found at {pred_image_path}")
            continue
        
        try:
            # 读取当前 App.vue
            with open(app_vue_path, 'r', encoding='utf-8') as f:
                app_code = f.read()
            
            # 构建消息并调用 LLM 修复布局
            messages = build_messages(app_code, Path(gt_image_path), Path(pred_image_path))
            fixed_response = request_fix(
                messages,
                model="gemini-2.5-pro",
                max_tokens=32768,
                temperature=0
            )
            
            # 保存原始响应
            with open(fixed_output_path, 'w', encoding='utf-8') as f:
                f.write(fixed_response)
            print(f"  Saved layout fix response to {fixed_output_path}")
            
            # 提取代码块并更新 App.vue
            blocks = extract_code_blocks(fixed_response)
            app_block = None
            if blocks.get("app"):
                app_block = blocks["app"][0]
            elif blocks.get("snippet"):
                app_block = blocks["snippet"][0]
            elif blocks.get("vue"):
                app_block = blocks["vue"][0]
            
            if app_block:
                save_app_snippet(app_block, Path(app_vue_path))
                print(f"  Updated App.vue with layout fix")
            else:
                print(f"  Warning: No app/snippet/vue block found in response")
                
        except Exception as e:
            print(f"  Layout fix failed for page {file_idx}: {e}")

    # ============================================================
    # Step 4: 最终渲染 (布局修复后)
    # ============================================================
    print("\n" + "=" * 80)
    print("Final Rendering After Layout Fix")
    print("=" * 80)
    
    for file_idx, filename in enumerate(image_files, start=1):
        if file_idx > 4:
            continue
        split_save_dir = os.path.join(seg_dir, folder, str(file_idx))
        vue_dir = os.path.join(split_save_dir, "vue")
        screenshot_file_path = os.path.join(split_save_dir, f"{file_idx}_after_layout_fix.png")
        html_file_path = os.path.join(split_save_dir, f"{file_idx}_after_layout_fix.html")
        render_vue_with_extracted_dom(proj_dir, vue_dir, html_file_path, screenshot_file_path, webdriver_path)
        print(f"Rendered page {file_idx} (after layout fix): {screenshot_file_path}")
    
    # ============================================================
    # Step 5: 生成布局修复前后对比图
    # ============================================================
    print("\n" + "=" * 80)
    print("Generating Layout Fix Comparison Images")
    print("=" * 80)
    
    for file_idx, filename in enumerate(image_files, start=1):
        if file_idx > 4:
            continue
        split_save_dir = os.path.join(seg_dir, folder, str(file_idx))
        gt_image_path = os.path.join(gt_folder_path, f"{file_idx}.png")
        before_path = os.path.join(split_save_dir, f"{file_idx}_before_layout_fix.png")
        after_path = os.path.join(split_save_dir, f"{file_idx}_after_layout_fix.png")
        comparison_path = os.path.join(split_save_dir, f"{file_idx}_layout_comparison.png")
        
        if os.path.exists(before_path) and os.path.exists(after_path):
            try:
                generate_comparison_image(
                    gt_path=gt_image_path,
                    before_path=before_path,
                    after_path=after_path,
                    output_path=comparison_path
                )
                print(f"Generated comparison image for page {file_idx}: {comparison_path}")
            except Exception as e:
                print(f"Failed to generate comparison for page {file_idx}: {e}")
        else:
            print(f"Skipping comparison for page {file_idx}: missing before/after images")
    
    print("\n" + "=" * 80)
    print("All processing complete!")
    print("=" * 80)
    # component_render(base_name, proj_dir, webdriver_path)


if __name__ == "__main__":
    main_multi()





    # def batch_render_vue_files(src_vue_dir, project_dir, geckodriver_path, output_dir):
    #     os.makedirs(output_dir, exist_ok=True)
    #
    #     # 匹配 vue_1_gpt-5_vue.vue 里的编号
    #     vue_files = sorted([
    #         f for f in os.listdir(src_vue_dir)
    #         if re.match(r"vue_(\d+)_claude-sonnet-4-20250514_vue\.vue$", f)
    #     ], key=lambda x: int(re.search(r"vue_(\d+)_claude-sonnet-4-20250514_vue\.vue$", x).group(1)))
    #
    #     for vue_file in vue_files:
    #         match = re.search(r"vue_(\d+)_claude-sonnet-4-20250514_vue\.vue$", vue_file)
    #         if not match:
    #             continue
    #         idx = int(match.group(1))  # 提取序号 1, 2, ..., 118
    #         if idx not in [49]:
    #             continue
    #
    #         app_vue_path = os.path.join(project_dir, "src", "App.vue")
    #         vue_src_path = os.path.join(src_vue_dir, vue_file)
    #
    #         # 覆盖到 App.vue
    #         shutil.copy(vue_src_path, app_vue_path)
    #
    #         html_file_path = os.path.join(output_dir, f"{idx}.html")
    #         screenshot_file = os.path.join(output_dir, f"{idx}.png")
    #
    #         print(f"🚀 Rendering {vue_file} → {idx}.html / {idx}.png")
    #         try:
    #             render_single_vue_file(
    #                 project_dir=project_dir,
    #                 html_file_path=html_file_path,
    #                 screenshot_file=screenshot_file,
    #                 geckodriver_path=geckodriver_path
    #             )
    #         except:
    #             continue
    #
    #
    # src_vue_dir = r"D:\py_code\fyp\VueGen\GPT-Claude-Gemini-Generation\vue-vue\claude-sonnet-4-20250514"
    # project_dir = r"D:\py_code\fyp\DesignBench\web\my-vue-app"
    # geckodriver_path = r"D:\py_code\fyp\geckodriver.exe"
    # output_dir = r"D:\py_code\fyp\VueGen\testing\output"
    #
    # batch_render_vue_files(src_vue_dir, project_dir, geckodriver_path, output_dir)