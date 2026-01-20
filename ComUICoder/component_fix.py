"""
组件修复模块 (Component Fix)
使用 LLM 修复单个组件的代码问题
"""
import os
import re
import base64
import time
from typing import Dict, List, Optional
from openai import OpenAI

from prompts import COMPONENT_FIX_SYSTEM_PROMPT


def enhanced_fix_generate(
    error_feedback: str,
    original_code_path: str,
    output_file: str,
    gt_image_path: str = None,
    generated_image_path: str = None,
    error_code_snippets: List[Dict] = None,
    system_prompt: str = None,
    model: str = "gemini-2.5-pro",
    MAX_RETRIES: int = 3
) -> bool:
    """
    Enhanced fix generation with error code snippets.
    
    Args:
        error_feedback: Natural language error feedback text or path to feedback file
        original_code_path: Path to the original Vue code file
        output_file: Path to save the fixed code
        gt_image_path: Path to ground truth image
        generated_image_path: Path to generated screenshot
        error_code_snippets: List of dictionaries containing error code snippets
            Each dict has: {'issue_num': int, 'type': str, 'vue_snippet': str, 'html_snippet': str, 'context': str}
        system_prompt: Custom system prompt for code fixing
        model: LLM model name
        MAX_RETRIES: Maximum number of API call retries
    
    Returns:
        bool: Whether the fixing was successful
    """
    print("Enhanced Fix Generation is Processing...")
    
    with open(original_code_path, "r", encoding="utf-8") as f:
        original_code = f.read()
    
    if os.path.exists(error_feedback):
        with open(error_feedback, "r", encoding="utf-8") as f:
            feedback_text = f.read()
    else:
        feedback_text = error_feedback
    
    # Build error code snippets section
    error_snippets_text = ""
    if error_code_snippets:
        error_snippets_text = "\n\n--- ERROR CODE SNIPPETS ---\n"
        error_snippets_text += "The following code snippets are related to the errors detected:\n\n"
        
        for snippet in error_code_snippets:
            issue_num = snippet.get('issue_num', '?')
            issue_type = snippet.get('type', 'UNKNOWN')
            vue_code = snippet.get('vue_snippet', '')
            html_code = snippet.get('html_snippet', '')
            context = snippet.get('context', '')
            
            error_snippets_text += f"Issue #{issue_num} ({issue_type}):\n"
            
            if context:
                error_snippets_text += f"  Context: {context}\n"
            
            if vue_code:
                error_snippets_text += f"  Vue Code:\n```vue\n{vue_code}\n```\n"
            
            if html_code:
                error_snippets_text += f"  HTML Output:\n```html\n{html_code}\n```\n"
            
            error_snippets_text += "\n"
        
        error_snippets_text += "--- END ERROR CODE SNIPPETS ---\n"
    
    if system_prompt is None:
        system_prompt = COMPONENT_FIX_SYSTEM_PROMPT

    user_prompt = f"""Here is the original Vue code that needs to be fixed:

{original_code}

---

Here is the error feedback describing what needs to be changed:

{feedback_text}
{error_snippets_text}
---

**Remember**: 
- The images provided show red boxes marking the most critical issues
- Focus on fixing these marked areas first - they represent the biggest problems
- Use the error code snippets to understand exactly which parts of your code are wrong

Please fix the code based on the feedback above.

**IMPORTANT**: Output ONLY the ```component and ```snippet code blocks. Do NOT write any explanations, descriptions, or additional text. Just the code blocks."""

    def encode_image(image_path):
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    
    messages = [{"role": "system", "content": system_prompt}]
    
    user_content = []
    
    if generated_image_path and os.path.exists(generated_image_path):
        user_content.append({"type": "text", "text": "Generated + Red Boxes (current output with errors marked #1, #2...):"})
        gen_img_base64 = encode_image(generated_image_path)
        user_content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{gen_img_base64}"}
        })
    
    if gt_image_path and os.path.exists(gt_image_path):
        user_content.append({"type": "text", "text": "GT + Red Boxes (target appearance with error locations #1, #2...):"})
        gt_img_base64 = encode_image(gt_image_path)
        user_content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{gt_img_base64}"}
        })
    
    user_content.append({"type": "text", "text": user_prompt})
    
    messages.append({"role": "user", "content": user_content})
    
    content = None
    i = 0
    
    while content is None and i < MAX_RETRIES:
        i += 1
        try:
            client = OpenAI(
                api_key="sk-IUUpnlTop1iyfmdl21533e8513Cc45B38b4e2544C20d8aC7",
                base_url="https://openkey.cloud/v1"
            )
            completion = client.chat.completions.create(
                model=model,
                stream=False,
                messages=messages,
                max_tokens=16384,
                timeout=600
            )
            
            print(f"Enhanced fix generation response:")
            print(completion.choices[0].message)
            content = completion.choices[0].message.content
        except Exception as e:
            print(f"Error in enhanced_fix_generate (attempt {i}/{MAX_RETRIES}): {e}")
            if i >= MAX_RETRIES:
                return False
            time.sleep(2)
    
    if not content:
        return False
    
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Saved enhanced fixed code to: {output_file}")
    return True


def extract_error_snippets_from_inconsistencies(inconsistencies: List[Dict]) -> List[Dict]:
    """
    Extract error code snippets from inconsistency list for use in fix generation.
    """
    snippets = []
    
    for i, incon in enumerate(inconsistencies, 1):
        snippet = {
            'issue_num': i,
            'type': incon.get('type', 'UNKNOWN'),
            'vue_snippet': incon.get('vue_code_snippet', ''),
            'html_snippet': incon.get('html_snippet', ''),
            'context': incon.get('error_code_context', '')
        }
        
        if incon.get('gt_text') and incon.get('gen_text'):
            snippet['context'] += f" | Expected text: '{incon['gt_text']}', got: '{incon['gen_text']}'"
        
        if incon.get('inconsistency_types'):
            snippet['context'] += f" | Issues: {', '.join(incon['inconsistency_types'])}"
        
        snippets.append(snippet)
    
    return snippets
