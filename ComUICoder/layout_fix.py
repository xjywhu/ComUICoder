from __future__ import annotations

import base64
import mimetypes
import os
import re
import sys
import importlib.util
from pathlib import Path
from typing import Any, Dict, List

from openai import OpenAI

# 显式导入 main 文件夹下的 prompts.py
MAIN_DIR = os.path.dirname(os.path.abspath(__file__))
_main_prompts_path = os.path.join(MAIN_DIR, "prompts.py")
_spec = importlib.util.spec_from_file_location("main_prompts", _main_prompts_path)
_main_prompts = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_main_prompts)
LAYOUT_FIX_SYSTEM_PROMPT = _main_prompts.LAYOUT_FIX_SYSTEM_PROMPT


# 代码块正则表达式
CODE_BLOCK_RE = re.compile(r"```(component|snippet|vue|app)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def encode_image(image_path: Path) -> str:
    mime, _ = mimetypes.guess_type(image_path)
    mime = mime or "image/png"
    data = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{data}"


def build_user_prompt(app_code: str) -> str:
    """构建用户提示词"""
    return f"""Here is the current App.vue:

```vue
{app_code}
```

Please compare the attached Image A (ground truth, the first image) and Image B (current broken state, the second image), then return the corrected App.vue.
"""


def build_messages(app_code: str, gt_path: Path, pred_path: Path) -> List[Dict[str, Any]]:
    user_text = build_user_prompt(app_code)
    content: List[Dict[str, Any]] = [{"type": "text", "text": user_text}]

    for label, image_path in (("Image A (Ground Truth)", gt_path), ("Image B (Broken Layout)", pred_path)):
        content.append({"type": "text", "text": f"{label}: {image_path.name}"})
        content.append({"type": "image_url", "image_url": {"url": encode_image(image_path), "detail": "high"}})

    return [
        {"role": "system", "content": LAYOUT_FIX_SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]


def request_fix(messages: List[Dict[str, Any]], model: str = "gemini-2.5-pro", 
                max_tokens: int = 32768, temperature: float = 0) -> str:
    client = OpenAI(
        api_key="sk-IUUpnlTop1iyfmdl21533e8513Cc45B38b4e2544C20d8aC7", 
        base_url="https://openkey.cloud/v1"
    )
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        stream=False,
        timeout=600
    )
    return completion.choices[0].message.content.strip()


def extract_code_blocks(raw: str) -> Dict[str, List[str]]:
    blocks: Dict[str, List[str]] = {"component": [], "snippet": [], "vue": [], "app": []}
    for match in CODE_BLOCK_RE.finditer(raw):
        block_type = (match.group(1) or "vue").lower()
        code = match.group(2).strip()
        if block_type in blocks:
            blocks[block_type].append(code)
        else:
            blocks.setdefault(block_type, []).append(code)
    return blocks


def infer_component_name(code: str) -> str:
    match = re.search(r'defineOptions\(\s*{[^}]*name\s*:\s*"([^"]+)"', code)
    if match:
        return match.group(1).strip()
    match = re.search(r"defineOptions\(\s*{[^}]*name\s*:\s*'([^']+)'", code)
    if match:
        return match.group(1).strip()
    match = re.search(r'name\s*:\s*"([^"]+)"', code)
    if match:
        return match.group(1).strip()
    match = re.search(r"name\s*:\s*'([^']+)'", code)
    if match:
        return match.group(1).strip()
    return "Component"


def save_components(blocks: List[str], vue_dir: Path) -> None:
    components_dir = vue_dir
    components_dir.mkdir(parents=True, exist_ok=True)
    for idx, block in enumerate(blocks):
        name = infer_component_name(block)
        filename = components_dir / f"{name}.vue"
        if filename.exists() and filename.is_dir():
            raise ValueError(f"Cannot overwrite directory with component file: {filename}")
        with open(filename, "w", encoding="utf-8") as fh:
            fh.write(block.strip() + "\n")
        print(f"   - Saved component: {filename}")


def save_app_snippet(app_code: str, app_path: Path) -> None:
    app_path.parent.mkdir(parents=True, exist_ok=True)
    with open(app_path, "w", encoding="utf-8") as fh:
        fh.write(app_code.strip() + "\n")


def fix_layout(app_vue_path: str, gt_image_path: str, pred_image_path: str, 
               output_path: str = None, model: str = "gemini-2.5-pro") -> bool:
    """
    修复页面布局
    
    Args:
        app_vue_path: App.vue 文件路径
        gt_image_path: Ground Truth 图片路径
        pred_image_path: 当前渲染结果图片路径
        output_path: 保存 LLM 原始响应的路径（可选）
        model: 使用的模型
    
    Returns:
        bool: 是否成功
    """
    app_path = Path(app_vue_path)
    gt_path = Path(gt_image_path)
    pred_path = Path(pred_image_path)
    
    if not app_path.exists():
        print(f"App.vue not found: {app_path}")
        return False
    if not gt_path.exists():
        print(f"GT image not found: {gt_path}")
        return False
    if not pred_path.exists():
        print(f"Predicted image not found: {pred_path}")
        return False
    
    app_code = app_path.read_text(encoding="utf-8")
    vue_dir = app_path.parent
    
    messages = build_messages(app_code, gt_path, pred_path)
    
    print(f"Requesting layout fix from {model}...")
    try:
        fixed_response = request_fix(messages, model=model)
    except Exception as err:
        print(f"Model request failed: {err}")
        return False
    
    if output_path:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(fixed_response, encoding="utf-8")
        print(f"Saved raw LLM output to {output_file}")
    
    blocks = extract_code_blocks(fixed_response)
    app_block = None
    if blocks.get("app"):
        app_block = blocks["app"][0]
    elif blocks.get("snippet"):
        app_block = blocks["snippet"][0]
    elif blocks.get("vue"):
        app_block = blocks["vue"][0]
    
    if app_block:
        save_app_snippet(app_block, app_path)
        print(f"Updated App.vue at {app_path}")
    else:
        print("Warning: No app/snippet/vue block detected for App.vue")
        return False

    component_blocks = blocks.get("component", [])
    if component_blocks:
        print("Saving component blocks...")
        save_components(component_blocks, vue_dir)
    
    print("Layout fix complete.")
    return True
