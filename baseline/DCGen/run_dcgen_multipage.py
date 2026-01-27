"""
Run DCGen Baseline on Multipage Data - First 28 folders

This script runs DCGen on the first 28 folders of the data_multipage directory.
"""

import os
import sys
import json
import shutil
import time
import re
from threading import Thread, Lock
from tqdm.auto import tqdm
import tiktoken

# Configuration
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DCGEN_PATH = CURRENT_DIR
WORKSPACE_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))
# Adjust DATA_DIR to point to data_multipage_filter
DATA_DIR = os.path.join(WORKSPACE_ROOT, "data_multipage_filter")
# Adjust OUTPUT_DIR to point to DCGen_multipage in the root
OUTPUT_DIR = os.path.join(WORKSPACE_ROOT, "DCGen_multipage_output")

API_KEY = "Your API Key"
API_BASE = "Your API Base"
MODEL = "gemini-2.5-pro"

# Add DCGen to path and change directory (required for placeholder.png)
sys.path.insert(0, DCGEN_PATH)
os.chdir(DCGEN_PATH)

# Import DCGen modules (same as experiments.py)
from utils import (
    simplify_html, get_driver, take_screenshot, encode_image, 
    GPT4, DCGenTrace, ImgSegmentation, Gemini, QwenVL, DCGenGrid, Claude
)
from openai import OpenAI

class DetailedResponse(str):
    def __new__(cls, value, info=None):
        obj = str.__new__(cls, value)
        obj.info = info or {}
        return obj

# ============================================================
# Prompts from experiments.py (EXACT COPY)
# ============================================================

prompt_direct = """Here is a prototype image of a webpage. Return a single piece of HTML and tail-wind CSS code to reproduce exactly the website. Use "placeholder.png" to replace the images. Pay attention to things like size, text, position, and color of all the elements, as well as the overall layout. Respond with the content of the HTML+tail-wind CSS code."""

prompt_dcgen = {
    "prompt_leaf": """Here is a prototype image of a container. Please fill a single piece of HTML and tail-wind CSS code to reproduce exactly the given container. Use 'placeholder.png' to replace the images. Pay attention to things like size, text, and color of all the elements, as well as the background color and layout. Here is the code for you to fill in:
    <div>
    You code here
    </div>
    Respond with only the code inside the <div> tags.""",

    "prompt_root": """Here is a prototype image of a webpage. I have an draft HTML file that contains most of the elements and their correct positions, but it has *inaccurate background*, and some missing or wrong elements. Please compare the draft and the prototype image, then revise the draft implementation. Return a single piece of accurate HTML+tail-wind CSS code to reproduce the website. Use "placeholder.png" to replace the images. Respond with the content of the HTML+tail-wind CSS code. The current implementation I have is: \n\n [CODE]"""
}

# Segmentation parameters from experiments.py (EXACT COPY)
seg_params = {
    "max_depth": 2,
    "var_thresh": 50,
    "diff_thresh": 45,
    "diff_portion": 0.9,
    "window_size": 50
}


# ============================================================
# Custom GPT4 class with our API endpoint (VueGen style)
# ============================================================

class CustomGPT4(GPT4):
    """GPT4 class using custom API endpoint - VueGen compatible"""
    def __init__(self, api_key, api_base, model="gemini-2.5-pro", patience=5):
        # Don't call super().__init__ to avoid reading key from file
        self.key = api_key
        self.api_base = api_base
        self.patience = patience
        self.model = model
        self.max_tokens = 32768
        self.name = "gemini-2.5-pro"
        # Create client like VueGen does
        self.client = OpenAI(api_key=api_key, base_url=api_base)
        self.lock = Lock()
        
        # Stats
        self.total_request_time = 0.0
        self.total_tokens = 0
    
    def ask(self, question, image_encoding=None, verbose=False):
        """Call API exactly like VueGen's api_call.py"""
        # Recreate client each time like VueGen does
        client = OpenAI(api_key=self.key, base_url=self.api_base)
        
        if image_encoding:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": question},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{image_encoding}"}
                        }
                    ]
                }
            ]
        else:
            messages = [{"role": "user", "content": question}]
        
        try:
            start_t = time.time()
            if True: # Using stream=True
                stream = client.chat.completions.create(
                    model=self.model,
                    stream=True,
                    messages=messages,
                    max_tokens=self.max_tokens,
                    timeout=600
                )
                response = ""
                for chunk in stream:
                    if chunk.choices and chunk.choices[0].delta.content:
                        response += chunk.choices[0].delta.content
                
                # Manual precise token counting since stream_options is not supported
                try:
                    encoding = tiktoken.get_encoding("cl100k_base")
                    # Count input tokens
                    input_tokens = 0
                    for msg in messages:
                        if isinstance(msg["content"], str):
                            input_tokens += len(encoding.encode(msg["content"]))
                        elif isinstance(msg["content"], list):
                            for part in msg["content"]:
                                if part.get("type") == "text":
                                    input_tokens += len(encoding.encode(part["text"]))
                                elif part.get("type") == "image_url":
                                    # Fixed approximation for image if not calculatable
                                    input_tokens += 1000 
                    
                    # Count output tokens
                    output_tokens = len(encoding.encode(response))
                    
                    with self.lock:
                        self.total_tokens += (input_tokens + output_tokens)
                except Exception as ex:
                    # Fallback if tiktoken fails
                    with self.lock:
                        self.total_tokens += len(response) // 4

            else:
                 pass 

            duration = time.time() - start_t
            with self.lock:
                self.total_request_time += duration
            
            # Check if response looks like an HTML error page
            if not response or response.strip().startswith("<!DOCTYPE html>") or "<title>502 Bad Gateway</title>" in response:
                raise ValueError("API returned HTML error page instead of content")
                
            if verbose:
                print("####################################")
                print("question:\n", question)
                print("####################################")
                print("response:\n", response)

            return DetailedResponse(response, {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
                "time": duration,
                "success": True
            })
            
        except Exception as e:
            # Check if the exception message contains HTML (e.g. Cloudflare error)
            error_str = str(e)
            if "<!DOCTYPE html>" in error_str or "<html" in error_str.lower():
                # Extract clean error if possible or just raise generic message
                if "524" in error_str and "timeout" in error_str.lower():
                     raise ValueError("API Error: Cloudflare 524 Timeout (Server side timeout)")
                else:
                     raise ValueError("API Error: Received HTML content instead of JSON (likely 5xx error)")
            
            # Re-raise to let try_ask handle it
            raise e


# ============================================================
# DCGen function from experiments.py (EXACT COPY with minor modifications)
# ============================================================

def dcgen(bot, img_path, save_path=None, max_depth=2, multi_thread=True, seg_params=None, show_progress=False, refine=False):
    """
    This code use CSS grid to assemble code
    (EXACT implementation from experiments.py)
    """
    if not show_progress:
        print(f"Running DCGen for {img_path}")
    
    t_start_seg = time.time()
    if not seg_params:
        img_seg = ImgSegmentation(img_path, max_depth=max_depth)
    else:
        img_seg = ImgSegmentation(img_path, **seg_params)
    t_seg = time.time() - t_start_seg

    dcgen_grid = DCGenGrid(img_seg, prompt_seg=prompt_dcgen["prompt_leaf"], prompt_refine=prompt_dcgen["prompt_root"])
    dcgen_grid.metrics["segmentation"] = t_seg
    
    dcgen_grid.generate_code(bot, multi_thread=multi_thread, refine=refine)
    
    if save_path:
        with open(save_path, 'w', encoding="utf-8", errors="ignore") as f:
            f.write(dcgen_grid.code)
            
    # Calculate number of nodes
    nodes_data = img_seg.to_json()
    return dcgen_grid.code, nodes_data, dcgen_grid.metrics


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
            try:
                page.goto(file_url, wait_until="networkidle", timeout=30000)
            except Exception as e:
                tqdm.write(f"  Warning: Page load timeout/error (taking screenshot anyway): {e}")
            
            time.sleep(1)
            page.screenshot(path=screenshot_path, full_page=True)
            browser.close()
        return True
    except Exception as e:
        tqdm.write(f"  ❌ Screenshot error: {e}")
        return False


# ============================================================
# Main runner
# ============================================================

def run_dcgen_multipage():
    """Run DCGen on first 3 folders of data_multipage"""
    
    print(f"\n{'='*60}")
    print(f"DCGen Baseline - Multipage")
    print(f"{'='*60}")
    print(f"Model: {MODEL}")
    print(f"Data Dir: {DATA_DIR}")
    print(f"Output Dir: {OUTPUT_DIR}")
    print(f"{'='*60}\n")
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Get list of folders, sort numerically
    all_items = os.listdir(DATA_DIR)
    folders = []
    for item in all_items:
        if os.path.isdir(os.path.join(DATA_DIR, item)) and item.isdigit():
            folders.append(int(item))
    
    folders.sort()
    
    # Process all folders
    target_folders = folders
    
    print(f"Target folders ({len(target_folders)}): {target_folders}")
    
    # Placeholder source
    placeholder_src = os.path.join(DCGEN_PATH, "data", "original", "placeholder.png")
    
    # Initialize bot
    bot = CustomGPT4(API_KEY, API_BASE, model=MODEL)
    
    results = {}
    
    for folder_id in tqdm(target_folders, desc="Folders", unit="folder"):
        folder_path = os.path.join(DATA_DIR, str(folder_id))
        output_folder_path = os.path.join(OUTPUT_DIR, str(folder_id))
        os.makedirs(output_folder_path, exist_ok=True)
        
        # Copy placeholder to output folder
        if os.path.exists(placeholder_src):
            shutil.copy(placeholder_src, os.path.join(output_folder_path, "placeholder.png"))
            
        # Find all png files in the folder
        files = os.listdir(folder_path)
        png_files = [f for f in files if f.lower().endswith('.png')]
        
        # Sort png files numerically if possible (1.png, 2.png...)
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
            
            html_path = os.path.join(output_folder_path, f"{file_name_no_ext}.html")
            screenshot_path = os.path.join(output_folder_path, f"{file_name_no_ext}.png")
            json_path = os.path.join(output_folder_path, f"{file_name_no_ext}.json")
            
            # Reset stats for this page
            bot.total_request_time = 0.0
            bot.total_tokens = 0
            
            try:
                # Run DCGen
                start_time = time.time()
                # Using multi_thread=True, refine=False
                code, nodes_data, metrics = dcgen(bot, img_path, html_path, max_depth=2, multi_thread=True, seg_params=seg_params, show_progress=False, refine=False)
                # Wait, I need to pass refine to dcgen() too, but dcgen() calls DCGenGrid.generate_code()
                # The dcgen function in this file needs modification to accept verify
                
                elapsed = time.time() - start_time
                
                # Take screenshot
                take_screenshot_html(html_path, screenshot_path)
                
                # Collect extended metrics
                segmentation_blocks = []
                total_block_input_tokens = 0
                total_block_output_tokens = 0
                
                def extract_blocks(node):
                    # Handle both list and dict formats
                    if isinstance(node, list):
                        for item in node:
                            extract_blocks(item)
                        return
                    if not isinstance(node, dict):
                        return
                    if not node.get("children"): # Leaf node
                        segmentation_blocks.append({
                            "bbox": node.get("bbox", []),
                            "time": node.get("time", 0),
                            "input_token": node.get("input_token", 0),
                            "output_token": node.get("output_token", 0),
                            "success": node.get("success", False),
                            "reason": node.get("reason", ""),
                            "level": node.get("level", 0) # Assumes 'level' is preserved in node if ImgSegmentation adds it
                        })
                    else:
                        for child in node.get("children", []):
                            extract_blocks(child)
                extract_blocks(nodes_data) # nodes_data is the full tree
                
                # Calculate totals from blocks
                for b in segmentation_blocks:
                    total_block_input_tokens += b["input_token"]
                    total_block_output_tokens += b["output_token"]
                
                result_data = {
                    "success": True,
                    "reason": "",
                    "time": {
                        "total_request_time": sum([b["time"] for b in segmentation_blocks]),
                        "total_generation_time": elapsed,
                        "segmentation": metrics.get("segmentation", 0),
                        "assembly": metrics.get("assembly", 0),
                        "optimization": metrics.get("optimization", 0)
                    },
                    "token": {
                        "total_output": bot.total_tokens - total_block_input_tokens, # Rough estimate or better sum
                         # Better:
                        "total_output": total_block_output_tokens,
                        "total_input": total_block_input_tokens,
                        "segmentation": 0, # Usually 0 for CV-based seg
                        "block_total": total_block_input_tokens + total_block_output_tokens,
                        "assembly": 0, # Since refine=False
                        "optimization": 0
                    },
                    "segmentation_blocks_count": len(segmentation_blocks),
                    "segmentation_blocks": segmentation_blocks
                }
                
                # Save individual JSON
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(result_data, f, indent=2, ensure_ascii=False, default=str)
                
            except Exception as e:
                tqdm.write(f"  ❌ Error processing {folder_id}/{png_file}: {e}")
                # Save individual JSON for error
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump({
                        "success": False, 
                        "reason": str(e),
                        "time": {"total_generation_time": time.time() - start_time}
                    }, f, indent=2, ensure_ascii=False, default=str)
        
    print(f"\nCompleted.")

if __name__ == "__main__":
    run_dcgen_multipage()
