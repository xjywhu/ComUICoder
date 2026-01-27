"""
Run LayoutCoder Baseline on Multipage Data - First 3 folders
"""

import os
import sys
import json
import shutil
import time
import traceback
import cv2
import numpy as np
import gc
from os.path import join as pjoin
from tqdm.auto import tqdm

# Configuration
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
LAYOUTCODER_PATH = CURRENT_DIR
WORKSPACE_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))
DATA_DIR = os.path.join(WORKSPACE_ROOT, "data_multipage_filter")
OUTPUT_DIR = os.path.join(WORKSPACE_ROOT, "LayoutCoder_multipage_output")

# Add LayoutCoder to path
sys.path.insert(0, LAYOUTCODER_PATH)
os.chdir(LAYOUTCODER_PATH)

# ============================================================
# LayoutCoder Functions (from run_single.py)
# ============================================================

def resize_height_by_longest_edge(img_path, resize_length=800):
    # Use PIL to get size without loading text into memory (lighter weight)
    from PIL import Image
    with Image.open(img_path) as img:
        width, height = img.size
    return height

def uied(input_path_img=None, output_root=None):
    key_params = {'min-grad': 10, 'ffl-block':5, 'min-ele-area':50,
                  'merge-contained-ele':True, 'merge-line-to-paragraph':True, 'remove-bar':False}

    input_path_img = input_path_img or './data/input/real_image/360.cn_6.png'
    output_root = output_root or './data/output'

    resized_height = resize_height_by_longest_edge(input_path_img, resize_length=800)

    is_ip = True
    is_clf = False
    is_ocr = True
    is_merge = True

    if is_ocr:
        # print("====enter====[ocr]")
        import UIED.detect_text.text_detection as text
        os.makedirs(pjoin(output_root, 'ocr'), exist_ok=True)
        text.text_detection(input_path_img, output_root, show=False)
        # print("====exit====[ocr]")

    if is_ip:
        # print("====enter====[ip]")
        import UIED.detect_compo.ip_region_proposal as ip
        os.makedirs(pjoin(output_root, 'ip'), exist_ok=True)
        classifier = None
        if is_clf:
            classifier = {}
            from UIED.cnn.CNN import CNN
            classifier['Elements'] = CNN('Elements')
        ip.compo_detection(input_path_img, output_root, key_params,
                           classifier=classifier, resize_by_height=resized_height, show=False)
        # print("====exit====[ip]")

    if is_merge:
        # print("====enter====[merge]")
        import UIED.detect_merge.merge as merge
        os.makedirs(pjoin(output_root, 'merge'), exist_ok=True)
        name = os.path.splitext(os.path.basename(input_path_img))[0]
        compo_path = pjoin(output_root, 'ip', str(name) + '.json')
        ocr_path = pjoin(output_root, 'ocr', str(name) + '.json')
        merge.merge(input_path_img, compo_path, ocr_path, pjoin(output_root, 'merge'),
                    is_remove_bar=key_params['remove-bar'], is_paragraph=key_params['merge-line-to-paragraph'], show=False, max_line_gap=30)
        # print("====exit====[merge]")

def ui2code_pipeline(input_path_img, output_root, **key_params):
    from utils import layout, detect_lines, page_layout_divider, code_gen
    
    metrics = {"segmentation": 0.0, "assembly": 0.0, "optimization": 0.0}

    is_uied = key_params.get('is_uied', True)
    is_layout = key_params.get('is_layout', True)
    is_lines = key_params.get('is_lines', True)
    is_divide = key_params.get('is_divide', True)
    is_global_gen = key_params.get('is_global_gen', True)
    
    seg_start = time.time()
    # 识别text、non-text
    if is_uied: uied(input_path_img, output_root)
    # 识别分割线
    if is_lines: detect_lines.detect_sep_lines_with_lsd(input_path_img, output_root)
    # 识别layout（依赖分割线去除错误的layout)
    if is_layout: layout.process_layout(input_path_img, output_root, use_uied_img=True, is_detail_print=False, use_sep_line=True)
    # 绘制分割线
    if is_lines: detect_lines.draw_lines(input_path_img, output_root)
    # 布局分割
    if is_divide: page_layout_divider.divide_layout(input_path_img, output_root)
    metrics["segmentation"] = time.time() - seg_start
    
    # 布局分割线=>mask=>json=>layout html
    if is_global_gen: 
        # Explicitly pass refine=False here
        lc_metrics = code_gen.make_layout_code(input_path_img, output_root, force=False, refine=False)
        # Use actual assembly time from the function if available
        if lc_metrics and isinstance(lc_metrics, dict):
            metrics["assembly"] = lc_metrics.get("assembly", 0)
        else:
            # Fallback if uninstrumented
            metrics["assembly"] = 0.0
        
    return metrics

def count_atomic_nodes(node):
    """Recursively count atomic nodes in the layout structure"""
    count = 0
    if isinstance(node, dict):
        if node.get('type') == 'atomic':
            return 1
        elif node.get('type') in ['row', 'column']:
            if 'value' in node and isinstance(node['value'], list):
                for child in node['value']:
                    count += count_atomic_nodes(child)
    elif isinstance(node, list):
        for child in node:
            count += count_atomic_nodes(child)
    return count

# ============================================================
# Screenshot function
# ============================================================

def take_screenshot_html(html_path, screenshot_path):
    """Take screenshot of HTML file using Playwright"""
    try:
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1920, "height": 1080})
            # Normalize path for file:// URL
            file_url = "file:///" + html_path.replace("\\", "/")
            page.goto(file_url, wait_until="networkidle", timeout=60000)
            time.sleep(1)
            page.screenshot(path=screenshot_path, full_page=True)
            browser.close()
        return True
    except Exception as e:
        print(f"  Screenshot error: {e}")
        return False

# ============================================================
# Main runner
# ============================================================

def run_layoutcoder_multipage():
    print(f"\n{'='*60}")
    print(f"LayoutCoder Baseline - Multipage")
    print(f"{'='*60}")
    print(f"Data Dir: {DATA_DIR}")
    print(f"Output Dir: {OUTPUT_DIR}")
    print(f"{'='*60}\n")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Get list of folders, sort numerically
    all_items = os.listdir(DATA_DIR)
    folders = []
    for item in all_items:
        if os.path.isdir(os.path.join(DATA_DIR, item)) and item.isdigit():
            folders.append(int(item))
    
    folders.sort()
    # Run all folders
    target_folders = folders
    
    print(f"Target folders ({len(target_folders)}): {target_folders}")
    
    # Load existing results if available to prevent overwriting history
    results_path = os.path.join(OUTPUT_DIR, "layoutcoder_multipage_results.json")
    if os.path.exists(results_path):
        try:
            with open(results_path, "r", encoding="utf-8") as f:
                results = json.load(f)
            print(f"Loaded existing results for {len(results)} folders.")
        except Exception as e:
            print(f"Correction: Could not load existing results: {e}. Starting fresh.")
            results = {}
    else:
        results = {}
    
    for folder_id in tqdm(target_folders, desc="Folders", unit="folder"):
        folder_path = os.path.join(DATA_DIR, str(folder_id))
        output_folder_path = os.path.join(OUTPUT_DIR, str(folder_id))
        os.makedirs(output_folder_path, exist_ok=True)
        
        # Find all png files in the folder (treat 'x.png' as wildcard *.png)
        files = os.listdir(folder_path)
        png_files = [f for f in files if f.lower().endswith('.png')]
        
        def get_num(fname):
            try:
                return int(os.path.splitext(fname)[0])
            except:
                return 999999
        png_files.sort(key=get_num)
        
        folder_results = {}
        
        for png_file in tqdm(png_files, desc=f"Folder {folder_id}", leave=False):
            img_path = os.path.join(folder_path, png_file)
            file_name_no_ext = os.path.splitext(png_file)[0]
            
            # LayoutCoder output structure
            # It generates intermediate files in subfolders of output_folder_path
            # Final HTML is usually in output_folder_path/struct/{file_name_no_ext}_sep.html
            
            try:
                start_time = time.time()
                
                # Check if already done (optional, but good for resuming)
                expected_html = os.path.join(output_folder_path, "struct", f"{file_name_no_ext}_sep.html")
                
                # Run pipeline
                # We pass output_folder_path as root. 
                # It will create ocr, ip, merge, lines, layout, struct inside it.
                
                metrics = ui2code_pipeline(
                    input_path_img=img_path,
                    output_root=output_folder_path,
                    is_uied=True,
                    is_layout=True,
                    is_lines=True,
                    is_divide=True,
                    is_global_gen=True
                )
                
                elapsed = time.time() - start_time
                
                # Copy final HTML to clean name
                final_html_path = os.path.join(output_folder_path, f"{file_name_no_ext}.html")
                screenshot_path = os.path.join(output_folder_path, f"{file_name_no_ext}.png")
                json_path = os.path.join(output_folder_path, f"{file_name_no_ext}.json")
                
                if os.path.exists(expected_html):
                    shutil.copy(expected_html, final_html_path)
                    
                    # Take screenshot
                    take_screenshot_html(final_html_path, screenshot_path)
                    
                    # Process results
                    segmentation_blocks = []
                    struct_json_path = os.path.join(output_folder_path, "struct", f"{file_name_no_ext}_sep.json")
                    if os.path.exists(struct_json_path):
                        try:
                            with open(struct_json_path, 'r', encoding='utf-8') as f:
                                data = json.load(f)
                                
                                # Recursive extraction of blocks
                                def extract_blocks(node):
                                    if isinstance(node, dict):
                                        if node.get('type') == 'atomic':
                                            # Found a block
                                            position = node.get("position", {})
                                            bbox = [
                                                position.get("column_min", 0),
                                                position.get("row_min", 0),
                                                position.get("column_max", 0),
                                                position.get("row_max", 0)
                                            ]
                                            segmentation_blocks.append({
                                                "bbox": bbox,
                                                "time": node.get("time", 0),
                                                "input_token": node.get("input_token", 0),
                                                "output_token": node.get("output_token", 0),
                                                "success": True, # LayoutCoder handles success implicitly or writes " "
                                                "reason": "" if node.get("code", "").strip() else "Empty code/White page"
                                            })
                                        
                                        # Recursion
                                        if 'value' in node and isinstance(node['value'], list):
                                            for child in node['value']:
                                                extract_blocks(child)
                                    elif isinstance(node, list):
                                        for child in node:
                                            extract_blocks(child)
                                
                                if "structure" in data:
                                    extract_blocks(data["structure"])
                                    
                        except Exception as e:
                            print(f"Error processing struct json: {e}")
                    
                    # Calculate totals
                    total_block_input_tokens = sum(b["input_token"] for b in segmentation_blocks)
                    total_block_output_tokens = sum(b["output_token"] for b in segmentation_blocks)
                    total_block_request_time = sum(b["time"] for b in segmentation_blocks)
                    
                    # LayoutCoder 'assembly' metric is now physically measured in make_layout_code
                    segmentation_time = metrics.get("segmentation", 0)
                    pure_assembly_time = metrics.get("assembly", 0)
                    
                    # total_request_time = Sum of all block generation times + Assembly time
                    # total_generation_time = Actual End-to-End Wall Clock Time
                    
                    # Note: elapsed time should roughly equal: segmentation + max(blocks) + assembly
                    final_request_time = total_block_request_time + pure_assembly_time

                    result_data = {
                        "success": True,
                        "reason": "",
                        "time": {
                            "total_request_time": final_request_time,
                            "total_generation_time": elapsed,
                            "segmentation": metrics.get("segmentation", 0),
                            "assembly": pure_assembly_time,
                            "optimization": metrics.get("optimization", 0)
                        },
                        "token": {
                            "total_output": total_block_output_tokens,
                            "total_input": total_block_input_tokens,
                            "segmentation": 0,
                            "block_total": total_block_output_tokens,
                            "assembly": 0,
                            "optimization": 0
                        },
                        "segmentation_blocks_count": len(segmentation_blocks),
                        "segmentation_blocks": segmentation_blocks
                    }
                    
                    # Save individual JSON
                    with open(json_path, "w", encoding="utf-8") as f:
                        json.dump(result_data, f, indent=2, ensure_ascii=False, default=str)
                        
                    folder_results[png_file] = result_data
                    
                else:
                    err_res = {
                        "success": False, 
                        "reason": "HTML not generated",
                        "time": {"total_generation_time": elapsed}
                    }
                    with open(json_path, "w", encoding="utf-8") as f:
                        json.dump(err_res, f, indent=2, ensure_ascii=False, default=str)
                    folder_results[png_file] = err_res
                
            except Exception as e:
                tqdm.write(f"  ❌ Error processing {folder_id}/{png_file}: {e}")
                # traceback.print_exc()
                err_res = {"success": False, "reason": str(e), "time": {"total_generation_time": time.time() - start_time}}
                # Save individual JSON for error
                json_path = os.path.join(output_folder_path, f"{file_name_no_ext}.json")
                with open(json_path, "w", encoding="utf-8") as f:
                     json.dump(err_res, f, indent=2, ensure_ascii=False, default=str)
                folder_results[png_file] = err_res
            
            # Force garbage collection to free memory from large images
            gc.collect()
        
        results[folder_id] = folder_results
        
        # Save intermediate results
        results_path = os.path.join(OUTPUT_DIR, "layoutcoder_multipage_results.json")
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)

    print(f"\nCompleted. Results saved to {results_path}")

if __name__ == "__main__":
    run_layoutcoder_multipage()
