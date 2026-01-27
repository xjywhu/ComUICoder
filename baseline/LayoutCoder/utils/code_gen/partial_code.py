"""
局部代码生成
"""
from PIL import Image

import base64
import re
import requests

import os, time
from os.path import join as pjoin
import tiktoken
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from openai import OpenAI as RealOpenAI


HTML_TAILWINDCSS_SIMPLE_PROMPT = """
You are an expert Tailwind developer
You take screenshots of a reference web page from the user, and then build single page apps
using TailwindCSS, HTML and JS.
Return only the full code in <html></html> tags.
Do not include markdown "```" or "```html" at the start or end.
"""

# direct prompt，全局代码生成
HTML_TAILWINDCSS_GLOBAL_PROMPT = """
You are an expert Tailwind developer
You take screenshots of a reference web page from the user, and then build single page apps
using Tailwind, HTML and JS.
You might also be given a screenshot(The second image) of a web page that you have already built, and asked to
update it to look more like the reference image(The first image).

- Make sure the app looks exactly like the screenshot.
- Pay close attention to background color, text color, font size, font family,
padding, margin, border, etc. Match the colors and sizes exactly.
- Use the exact text from the screenshot.
- Do not add comments in the code such as "<!-- Add other navigation links as needed -->" and "<!-- ... other news items ... -->" in place of writing the full code. WRITE THE FULL CODE.
- Repeat elements as needed to match the screenshot. For example, if there are 15 items, the code should have 15 items. DO NOT LEAVE comments like "<!-- Repeat for each news item -->" or bad things will happen.
- For images, use placeholder images from https://placehold.co and include a detailed description of the image in the alt text so that an image generation AI can generate the image later.

In terms of libraries,

- Use this script to include Tailwind: <script src="https://cdn.tailwindcss.com"></script>
- You can use Google Fonts
- Font Awesome for icons: <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.3/css/all.min.css"></link>

Return only the full code in <html></html> tags.
Do not include markdown "```" or "```html" at the start or end.
"""

HTML_CSS_PROMPT = """
Generate the corresponding HTML code based on the input webpage image.
If you cannot generate specific HTML code, please generate code based on your best understanding of the webpage description.
Return only the full code in <html></html> tags.
Do not include markdown "```" or "```html" at the start or end.
"""

# If you cannot generate specific HTML code, please generate code based on your best understanding of the webpage description.


# ours, 局部代码生成
HTML_TAILWINDCSS_LOCAL_PROMPT = """
You are an expert Tailwind developer
You take screenshots of a reference web page from the user, and then build single page apps 
using Tailwind, HTML and JS.

- Make sure the app looks exactly like the screenshot.
- Pay close attention to background color, text color, font size, font family, 
padding, margin, border, etc. Match the colors and sizes exactly.
- Use the exact text from the screenshot.
- Do not add comments in the code such as "<!-- Add other navigation links as needed -->" and "<!-- ... other news items ... -->" in place of writing the full code. WRITE THE FULL CODE.
- Repeat elements as needed to match the screenshot. For example, if there are 15 items, the code should have 15 items. DO NOT LEAVE comments like "<!-- Repeat for each news item -->" or bad things will happen.
- For images, use placeholder images from https://placehold.co and include a detailed description of the image in the alt text so that an image generation AI can generate the image later.

In terms of libraries,

- Use this script to include Tailwind: <script src="https://cdn.tailwindcss.com"></script>
- You can use Google Fonts
- Font Awesome for icons: <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.3/css/all.min.css"></link>

Get the full code in <html></html> tags.

Extract the body of the full html not including <body> tag
- Make sure the aspect ratio of the div and the image are identical
- Ensure the code can be nested within other tags, extend to fill the entire container and adapt to varying container.
- Use flex layout and relative units from Tailwind CSS.
- Apply w-full and h-full classes to the outermost div.
- Don't use max-width or max-height, and set margin and padding to 0

Return only the code in <div></div> tags.
Do not include markdown "```" or "```html" at the start or end.
"""

# 简单局部代码生成，消融点3：是否使用定制局部代码生成提示词
HTML_TAILWINDCSS_LOCAL_PROMPT_SIMPLE = """
You are an expert Tailwind developer
You take screenshots of a reference web page from the user, and then build single page apps 
using Tailwind, HTML and JS.
Return only the code in <div></div> tags.
Do not include markdown "```" or "```html" at the start or end.
"""

# os.environ["OPENAI_API_KEY"] = "xxxx"
# os.environ["OPENAI_API_KEY"] = "xxxx"
# os.environ["OPENAI_BASE_URL"] = "http://xx.xx.xx.xx:xxxx/"

# Custom API configuration for gemini-2.5-pro
CUSTOM_API_KEY = "Your API Key"
CUSTOM_API_BASE = "Your API Base"
CUSTOM_MODEL = "gemini-2.5-pro"
CUSTOM_MAX_TOKENS = 32768


class OpenAI:
    def __init__(self, is_debug=False, refine=False):
        self.is_debug = is_debug
        self.refine = refine
        self.client = RealOpenAI(
            api_key=CUSTOM_API_KEY,
            base_url=CUSTOM_API_BASE
        )
        self.model = CUSTOM_MODEL
        self.max_tokens = CUSTOM_MAX_TOKENS
        self.usage = {
            "token": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            }
        }

    def partial_code(self, image_path):
        """局部代码生成, div结尾，ours"""
        from ablation_config import is_custom_prompt
        # 消融点3: 是否定制局部代码生成提示词
        prompt = HTML_TAILWINDCSS_LOCAL_PROMPT if is_custom_prompt else HTML_TAILWINDCSS_LOCAL_PROMPT_SIMPLE
        return self.ui2code(image_path, prompt)

    def global_code(self, image_path):
        """全局代码生成，html结尾，cot prompt"""
        # return self.ui2code(image_path, HTML_CSS_PROMPT)
        return self.ui2code(image_path, HTML_TAILWINDCSS_GLOBAL_PROMPT)

    def ui2code(self, image_path, prompt):
        """
        Modified to return (result_code, usage_dict)
        Does NOT update self.usage internally to avoid race conditions in threads.
        """
        if self.is_debug:
            name = os.path.splitext(os.path.basename(image_path))[0]
            return name, {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        print(f"正在调用{self.model}, {image_path}")
        base64_image = OpenAI.encode_image(image_path)
        
        local_usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
        
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ]

        try:
            # Using stream=True
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=32768,
                stream=True,
                timeout=600 
            )
            
            response_content = ""
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    response_content += chunk.choices[0].delta.content
            
            result = response_content

            # Manual precise token counting
            try:
                encoding = tiktoken.get_encoding("cl100k_base")
                # Count input tokens
                input_tokens = 0
                for msg in messages:
                    if isinstance(msg["content"], list):
                        for part in msg["content"]:
                            if part.get("type") == "text":
                                input_tokens += len(encoding.encode(part["text"]))
                            elif part.get("type") == "image_url":
                                # Fixed approximation for image
                                input_tokens += 1000 
                
                # Count output tokens
                output_tokens = len(encoding.encode(result))
                
                local_usage["prompt_tokens"] = input_tokens
                local_usage["completion_tokens"] = output_tokens
                local_usage["total_tokens"] = (input_tokens + output_tokens)
                
            except Exception as ex:
                print(f"Token counting error: {ex}")
                # Fallback
                local_usage["total_tokens"] = len(result) // 4

        except Exception as e:
            print(f"API Error: {e}")
            raise e

        # 检查 API 返回是否为空
        if not result:
            print(f"警告: API 返回为空，图片: {image_path}")
            return "    ", local_usage

        # matched = re.findall(r"```html([^`]+)```", result)
        html_pattern = re.compile(r'<div[^>]*>.*</div>', re.DOTALL)
        matched = html_pattern.findall(result)
        if matched:
            result = matched[0]
            
        return result, local_usage

    @staticmethod
    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    @staticmethod
    def is_white_page(image_path, threadhold=0.08):
        '''
        大律法+直方图： 判断是否白屏
        感谢vivo Blog提供的白屏检测方案
        https://quickapp.vivo.com.cn/how-to-use-picture-contrast-algorithm-to-deal-with-white-screen-detection/
        '''
        import cv2
        import numpy as np

        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        white_img = np.ones_like(img, dtype=img.dtype) * 255
        dst = cv2.cvtColor(white_img, cv2.COLOR_BGR2GRAY)
        dst1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, th = cv2.threshold(dst1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        hist_base = cv2.calcHist([dst], [0], None, [256], (0, 256), accumulate=False)
        hist_test1 = cv2.calcHist([th], [0], None, [256], (0, 256), accumulate=False)

        cv2.normalize(hist_base, hist_base, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        cv2.normalize(hist_test1, hist_test1, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

        base_test1 = cv2.compareHist(hist_base, hist_test1, 3)
        print(f"空白页评估值：{base_test1}")
        return base_test1 <= threadhold


class PartialCodeMaker:
    def __init__(self, full_img_path, output_root, is_debug=False, refine=False):
        self.openai = OpenAI(is_debug, refine=refine)
        self.full_img_path = full_img_path
        self.name = os.path.splitext(os.path.basename(full_img_path))[0]
        self.full_img = Image.open(full_img_path)
        self.partial_img_root = pjoin(output_root, "struct", "partial")
        os.makedirs(self.partial_img_root, exist_ok=True)
        self.usage_lock = threading.Lock()

    def crop(self, bbox, path):
        """根据bbox切分图片，并保存"""
        crop_area = (max(0, bbox[0]), max(0, bbox[1]), min(self.full_img.width, bbox[2]), min(self.full_img.height, bbox[3]))
        cropped_img = self.full_img.crop(crop_area)
        cropped_img.save(path)

    def load_code_cache(self):
        """加载已缓存的代码"""
        cache_path = pjoin(self.partial_img_root, f'{self.name}_code_cache.json')
        if os.path.exists(cache_path):
            import json
            with open(cache_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}

    def save_code_cache(self, cache):
        """保存代码缓存"""
        cache_path = pjoin(self.partial_img_root, f'{self.name}_code_cache.json')
        import json
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)

    def code(self, structure):
        """生成structure中所有的局部代码片段 - 并行化"""
        self.code_cache = self.load_code_cache()
        
        # 1. 扫描结构，生成所有 tasks
        tasks = []
        self._scan_and_prepare(structure, tasks)
        
        # 2. 并行执行 tasks
        if tasks:
            print(f"Parallel generating {len(tasks)} blocks...")
            with ThreadPoolExecutor(max_workers=8) as executor:
                # 提交所有任务
                futures = {executor.submit(self._generate_block_safe, task): task for task in tasks}
                for future in as_completed(futures):
                    try:
                        future.result()
                    except Exception as e:
                        print(f"Task exception: {e}")
        
        self.save_code_cache(self.code_cache)
        return structure

    def _generate_block_safe(self, task):
        """单个block生成的线程安全执行函数"""
        node = task['node']
        image_path = task['image_path']
        cache_key = task['cache_key']
        
        start_t = time.time()
        # openai.partial_code now returns (code, usage)
        code, usage = self.openai.partial_code(image_path)
        end_t = time.time()
        
        # Update node stats
        node['code'] = code
        node['time'] = end_t - start_t
        node['input_token'] = usage['prompt_tokens']
        node['output_token'] = usage['completion_tokens']
        node['success'] = True
        
        # Update cache (dict assignment for single key is atomic enough for cache consistency)
        self.code_cache[cache_key] = code
        
        # Update global usage safely
        with self.usage_lock:
             self.openai.usage["token"]["prompt_tokens"] += usage['prompt_tokens']
             self.openai.usage["token"]["completion_tokens"] += usage['completion_tokens']
             self.openai.usage["token"]["total_tokens"] += usage['total_tokens']

    def _scan_and_prepare(self, structure, tasks, current_id=1):
        """
        递归遍历结构:
        1. 赋予 ID
        2. 切图
        3. 检查缓存/白屏
        4. 如果需要生成，加入 tasks 列表
        """
        # 检查是否是 atomic 组件
        if structure['type'] == 'atomic':
            cropped_path = pjoin(self.partial_img_root, f'{self.name}_part_{current_id}.png')
            cache_key = f'{self.name}_part_{current_id}'
            position = structure["position"]
            bbox = (position["column_min"], position["row_min"], position["column_max"], position["row_max"])
            self.crop(bbox, cropped_path)  # 根据bbox切图
            
            structure['id'] = current_id
            structure["size"] = (bbox[2] - bbox[0], bbox[3] - bbox[1])
            
            # 检查缓存
            if cache_key in self.code_cache:
                print(f"使用缓存: {cache_key}")
                structure['code'] = self.code_cache[cache_key]
                structure['time'] = 0
                structure['input_token'] = 0
                structure['output_token'] = 0
            elif OpenAI.is_white_page(cropped_path):
                structure['code'] = "    "  # 空白区域不消耗token
                self.code_cache[cache_key] = structure['code']
                structure['time'] = 0
                structure['input_token'] = 0
                structure['output_token'] = 0
            else:
                # 需要生成，加入任务列表
                tasks.append({
                    "node": structure,
                    "image_path": cropped_path,
                    "cache_key": cache_key,
                    "id": current_id
                })
                
            return current_id + 1  # id 递增

        # 如果不是 atomic，递归处理其嵌套的 value
        if 'value' in structure:
            for item in structure['value']:
                current_id = self._scan_and_prepare(item, tasks, current_id)

        return current_id


if __name__ == '__main__':
    openai = OpenAI(is_debug=False)
    # result = openai.partial_code("/data/username/UIED/InteractiveLayoutEditor/output/struct/partial/bilibili_part_8.png")
    # print(result)
    # Be aware: global_code returns (code, usage) now, unpack it
    result, usage = openai.global_code("/data/username/UIED/data/ours_dataset/4shared.com.png")
    with open("test.html", "w") as f:
        f.write(result)
    print(openai.usage)
