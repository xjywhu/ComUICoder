import time
from PIL import Image
import base64
import requests
import os
import re
import json
from openai import OpenAI

def code_generation(system_prompt, image_idx, image_path, output_file, name_list = None, model="gemini-2.5-pro", MAX_RETRIES=3):
    """
    Call Gemini to process a single image and save the results.
    Args:
        system_prompt (str): System prompt string
        image_idx (int): Image index
        image_path (str): Path to the image file
        output_file (str): Path to the output file (JSON or TXT)
        model: Gemini model object
        max_retries (int): Maximum number of retries if request fails
    Returns:
        bool: Whether the processing was successful
    """

    print("Code Generation is Processing...")
    if name_list != None:
        user_prompt = f"Here is the webpage screenshot. When generating component names, do **not** use any of the names in the following list: {str(name_list)}:"
    else:
        user_prompt = f"Here is the webpage screenshot."

    def encode_image(image_path):
      with open(image_path, "rb") as f:
          return base64.b64encode(f.read()).decode("utf-8")
    img_base64 = encode_image(image_path)

    content = None
    i = 0

    while content is None and i < MAX_RETRIES:
        i += 1
        client = OpenAI(api_key="sk-IUUpnlTop1iyfmdl21533e8513Cc45B38b4e2544C20d8aC7", base_url="https://openkey.cloud/v1")
        #client = OpenAI(api_key="aSvCHqIrj64VUoNKX1u3YgteHq1WQsA6", base_url="https://api.deepinfra.com/v1/openai")
        completion = client.chat.completions.create(
            model= model,
            stream=False,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": [{"type": "text", "text": user_prompt},
                                             {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_base64}"}}]}
            ],
            max_tokens=16384,
            temperature=0,
            timeout=600
        )

        print(f"Response for image {image_idx}:")
        print(completion.choices[0].message)
        content = completion.choices[0].message.content

    # content = response.candidates[0].content.parts[0].text
    if not content:
        return False

    # Directly save full content as txt
    if content:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"Saved Vue content for image {image_idx}")
        return True


def code_generation_multipage(system_prompt, image_idx, image_paths, output_file, group_name_list,
                              name_list = None, model="gemini-2.5-pro", MAX_RETRIES=3):
    """
    Call Gemini to process a single image and save the results.
    Args:
        system_prompt (str): System prompt string
        image_idx (int): Image index
        image_path (str): Path to the image file
        output_file (str): Path to the output file (JSON or TXT)
        model: Gemini model object
        max_retries (int): Maximum number of retries if request fails
    Returns:
        bool: Whether the processing was successful
    """
    print("Code Generation is Processing...")

    def encode_image(image_path):
      with open(image_path, "rb") as f:
          return base64.b64encode(f.read()).decode("utf-8")

    img_list = []
    for image_path in image_paths:
        img_list.append(encode_image(image_path))

    if name_list != None:
        user_prompt = (
            "When generating component names, do NOT use any of the names in the following list: "
            f"{name_list}. "
            "If a name would conflict with this list, reuse the base name but append a numeric index, "
            "for example: content_1, content_2, etc."
            "Here are the webpage screenshot list: "
            f"{img_list}. "
            "And the corresponding name list: "
            f"{group_name_list}. "
        )
    else:
        user_prompt = f"Here is the webpage screenshot."


    content = None
    i = 0
    content_blocks = [{"type": "text", "text": user_prompt}]

    for img_base64 in img_list:
        content_blocks.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{img_base64}"
            }
        })
    while content is None and i < MAX_RETRIES:
        i += 1
        client = OpenAI(api_key="sk-IUUpnlTop1iyfmdl21533e8513Cc45B38b4e2544C20d8aC7", base_url="https://openkey.cloud/v1")
        #client = OpenAI(api_key="aSvCHqIrj64VUoNKX1u3YgteHq1WQsA6", base_url="https://api.deepinfra.com/v1/openai")
        completion = client.chat.completions.create(
            model= model,
            stream=False,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content_blocks}
            ],
            max_tokens=16384,
            temperature=0,
            timeout=600
        )

        print(f"Response for image {image_idx}:")
        print(completion.choices[0].message)
        content = completion.choices[0].message.content

    # content = response.candidates[0].content.parts[0].text
    if not content:
        return False

    # Directly save full content as txt
    if content:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"Saved Vue content for image {image_idx}")
        return True


def bbox_json_generation(system_prompt, image_idx, image_path, output_file, model="gemini-2.5-pro", MAX_RETRIES=3):
    """
    Call Gemini to process a single image and save the results.
    Args:
        system_prompt (str): System prompt string
        image_idx (int): Image index
        image_path (str): Path to the image file
        output_file (str): Path to the output file (JSON or TXT)
        model: Gemini model object
        max_retries (int): Maximum number of retries if request fails
    Returns:
        bool: Whether the processing was successful
    """
    print("BBox Json Generation is Processing...")

    def encode_image(image_path):
      with open(image_path, "rb") as f:
          return base64.b64encode(f.read()).decode("utf-8")
    img_base64 = encode_image(image_path)

    content = None
    i = 0
    while content is None and i < MAX_RETRIES:
        i += 1
        client = OpenAI(api_key="sk-IUUpnlTop1iyfmdl21533e8513Cc45B38b4e2544C20d8aC7", base_url="https://openkey.cloud/v1")
        #client = OpenAI(api_key="aSvCHqIrj64VUoNKX1u3YgteHq1WQsA6", base_url="https://api.deepinfra.com/v1/openai")
        completion = client.chat.completions.create(
            model=model,
            stream=False,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": [{"type": "text", "text": f"Here is the webpage screenshot."},
                                             {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_base64}"}}]}
            ],
            max_tokens=16384,
            timeout=600,
            temperature=0
        )

        print(f"Response for image {image_idx}:")
        print(completion.choices[0].message)
        content = completion.choices[0].message.content

    # content = response.candidates[0].content.parts[0].text
    if not content:
        return False

    # Extract JSON code block
    else:
        match = re.search(r'```json(.*?)(```|$)', content, re.DOTALL)
        content_json = match.group(1).strip() if match else None
        if content_json:
            try:
                pred_boxes = json.loads(content_json)
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(pred_boxes, f, ensure_ascii=False, indent=2)
                print(f"Saved JSON prediction results for image {image_idx}")
                success = True
            except json.JSONDecodeError as e:
                print(f"JSON parsing error {image_idx}: {e}")
                success=False
        else:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump([], f)
            print(f"No bounding boxes detected for image {image_idx}")
            success = True
    return success

