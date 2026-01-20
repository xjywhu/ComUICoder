import os
import re
import shutil
import glob
import cv2
from typing import Dict, List, Tuple, Optional
from difflib import SequenceMatcher

from ProcessTools import parse_vue_from_txt_component


def calculate_component_similarity(gt_widgets: Dict, gen_widgets: Dict, 
                                    gt_img_shape: Tuple[int, int], 
                                    gen_img_shape: Tuple[int, int]) -> float:
    """
    Calculate similarity score between GT and generated widgets using UIED detection.
    
    Args:
        gt_widgets: Dict of widgets detected in GT image
        gen_widgets: Dict of widgets detected in generated image  
        gt_img_shape: (height, width) of GT image
        gen_img_shape: (height, width) of generated image
    
    Returns:
        Similarity score between 0 and 1
    """
    if not gt_widgets or not gen_widgets:
        return 0.0
    
    matched = 0
    total_gt = len(gt_widgets)
    
    for gt_id, gt_widget in gt_widgets.items():
        gt_texts = gt_widget.texts if hasattr(gt_widget, 'texts') else []
        gt_text = ' '.join(gt_texts).lower().strip()
        
        best_match_score = 0.0
        for gen_id, gen_widget in gen_widgets.items():
            gen_texts = gen_widget.texts if hasattr(gen_widget, 'texts') else []
            gen_text = ' '.join(gen_texts).lower().strip()
            
            # Text similarity
            if gt_text and gen_text:
                text_sim = SequenceMatcher(None, gt_text, gen_text).ratio()
            elif not gt_text and not gen_text:
                text_sim = 0.5
            else:
                text_sim = 0.0
            
            # Position similarity (normalized)
            gt_bbox = gt_widget.bbox
            gen_bbox = gen_widget.bbox
            
            gt_cx = (gt_bbox.x1 + gt_bbox.x2) / 2 / gt_img_shape[1]
            gt_cy = (gt_bbox.y1 + gt_bbox.y2) / 2 / gt_img_shape[0]
            gen_cx = (gen_bbox.x1 + gen_bbox.x2) / 2 / gen_img_shape[1]
            gen_cy = (gen_bbox.y1 + gen_bbox.y2) / 2 / gen_img_shape[0]
            
            pos_dist = ((gt_cx - gen_cx) ** 2 + (gt_cy - gen_cy) ** 2) ** 0.5
            pos_sim = max(0, 1 - pos_dist * 2)
            
            score = 0.7 * text_sim + 0.3 * pos_sim
            best_match_score = max(best_match_score, score)
        
        if best_match_score > 0.5:
            matched += best_match_score
    
    return matched / total_gt if total_gt > 0 else 0.0


# ============================================================
# 组件评估
# ============================================================

def evaluate_component_with_uied(gt_image_path: str, 
                                  generated_image_path: str,
                                  vue_code_path: str = None,
                                  output_dir: str = None) -> Dict:
    """
    Evaluate a single component using UIED detection.
    
    Args:
        gt_image_path: Path to ground truth image
        generated_image_path: Path to generated screenshot
        vue_code_path: Path to Vue code file
        output_dir: Output directory for UIED results
    
    Returns:
        Evaluation result dictionary with match_score and widget info
    """
    from uied_detection import detect_widgets_uied
    
    if output_dir is None:
        output_dir = os.path.dirname(gt_image_path)
    
    gt_img = cv2.imread(gt_image_path)
    gen_img = cv2.imread(generated_image_path) if os.path.exists(generated_image_path) else None
    
    if gt_img is None:
        print(f"Failed to load GT image: {gt_image_path}")
        return None
    
    if gen_img is None:
        print(f"Failed to load generated image: {generated_image_path}")
        return {'match_score': 0.0, 'gt_widgets': {}, 'gen_widgets': {}, 'inconsistencies': []}
    
    print(f"  Running UIED on GT image...")
    gt_widgets = detect_widgets_uied(gt_image_path, output_dir=output_dir, granularity='medium')
    
    print(f"  Running UIED on Generated image...")
    gen_widgets = detect_widgets_uied(generated_image_path, output_dir=output_dir, granularity='medium')
    
    print(f"  GT widgets: {len(gt_widgets)}, Gen widgets: {len(gen_widgets)}")
    
    gt_shape = gt_img.shape[:2]
    gen_shape = gen_img.shape[:2]
    match_score = calculate_component_similarity(gt_widgets, gen_widgets, gt_shape, gen_shape)
    
    inconsistencies = find_uied_inconsistencies(gt_widgets, gen_widgets, gt_shape, gen_shape, vue_code_path)
    
    return {
        'match_score': match_score,
        'gt_widgets': gt_widgets,
        'gen_widgets': gen_widgets,
        'inconsistencies': inconsistencies,
        'gt_image_path': gt_image_path,
        'gen_image_path': generated_image_path
    }



def find_uied_inconsistencies(gt_widgets: Dict, gen_widgets: Dict,
                               gt_shape: Tuple[int, int], gen_shape: Tuple[int, int],
                               vue_code_path: str = None) -> List[Dict]:
    """
    Find inconsistencies between GT and generated widgets.
    """
    inconsistencies = []
    vue_code = ""
    if vue_code_path and os.path.exists(vue_code_path):
        with open(vue_code_path, 'r', encoding='utf-8') as f:
            vue_code = f.read()
    
    for gt_id, gt_widget in gt_widgets.items():
        gt_texts = gt_widget.texts if hasattr(gt_widget, 'texts') else []
        gt_text = ' '.join(gt_texts).strip()
        gt_bbox = gt_widget.bbox
        
        best_match = None
        best_score = 0.0
        
        for gen_id, gen_widget in gen_widgets.items():
            gen_texts = gen_widget.texts if hasattr(gen_widget, 'texts') else []
            gen_text = ' '.join(gen_texts).strip()
            
            if gt_text and gen_text:
                text_sim = SequenceMatcher(None, gt_text.lower(), gen_text.lower()).ratio()
            elif not gt_text and not gen_text:
                text_sim = 0.5
            else:
                text_sim = 0.0
            
            if text_sim > best_score:
                best_score = text_sim
                best_match = (gen_id, gen_widget, gen_text, text_sim)
        
        if best_match is None or best_score < 0.3:
            inconsistencies.append({
                'type': 'MISSING',
                'gt_text': gt_text,
                'gen_text': '',
                'gt_bbox': [gt_bbox.x1, gt_bbox.y1, gt_bbox.x2, gt_bbox.y2],
                'error_code_context': f"Widget with text '{gt_text}' is missing in generated output",
                'vue_code_snippet': extract_code_snippet(vue_code, gt_text)
            })
        elif best_score < 0.9:
            gen_id, gen_widget, gen_text, score = best_match
            inconsistencies.append({
                'type': 'TEXT_MISMATCH',
                'gt_text': gt_text,
                'gen_text': gen_text,
                'gt_bbox': [gt_bbox.x1, gt_bbox.y1, gt_bbox.x2, gt_bbox.y2],
                'gen_bbox': [gen_widget.bbox.x1, gen_widget.bbox.y1, gen_widget.bbox.x2, gen_widget.bbox.y2],
                'error_code_context': f"Text mismatch: expected '{gt_text}', got '{gen_text}'",
                'vue_code_snippet': extract_code_snippet(vue_code, gen_text) or extract_code_snippet(vue_code, gt_text)
            })
    
    # Check for extra widgets
    for gen_id, gen_widget in gen_widgets.items():
        gen_texts = gen_widget.texts if hasattr(gen_widget, 'texts') else []
        gen_text = ' '.join(gen_texts).strip()
        
        if not gen_text:
            continue
            
        has_match = False
        for gt_id, gt_widget in gt_widgets.items():
            gt_texts = gt_widget.texts if hasattr(gt_widget, 'texts') else []
            gt_text = ' '.join(gt_texts).strip()
            
            if gt_text:
                text_sim = SequenceMatcher(None, gt_text.lower(), gen_text.lower()).ratio()
                if text_sim > 0.5:
                    has_match = True
                    break
        
        if not has_match:
            inconsistencies.append({
                'type': 'EXTRA',
                'gt_text': '',
                'gen_text': gen_text,
                'gen_bbox': [gen_widget.bbox.x1, gen_widget.bbox.y1, gen_widget.bbox.x2, gen_widget.bbox.y2],
                'error_code_context': f"Extra widget with text '{gen_text}' not in GT",
                'vue_code_snippet': extract_code_snippet(vue_code, gen_text)
            })
    
    return inconsistencies


def extract_code_snippet(vue_code: str, search_text: str, context_lines: int = 3) -> str:
    """Extract code snippet around the search text."""
    if not vue_code or not search_text:
        return ""
    
    lines = vue_code.split('\n')
    for i, line in enumerate(lines):
        if search_text.lower() in line.lower():
            start = max(0, i - context_lines)
            end = min(len(lines), i + context_lines + 1)
            return '\n'.join(lines[start:end])
    
    return ""


def annotate_image_with_errors(image_path: str, inconsistencies: List[Dict], 
                                output_path: str, is_gt: bool = True) -> str:
    """
    Annotate image with red boxes and error numbers.
    """
    img = cv2.imread(image_path)
    if img is None:
        return image_path
    
    for i, incon in enumerate(inconsistencies[:5], 1):
        bbox_key = 'gt_bbox' if is_gt else 'gen_bbox'
        bbox = incon.get(bbox_key)
        
        if bbox is None:
            continue
            
        x1, y1, x2, y2 = [int(v) for v in bbox]
        
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        
        label = f"#{i}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)
        
        cv2.rectangle(img, (x1, y1 - th - 8), (x1 + tw + 8, y1), (0, 0, 255), -1)
        cv2.putText(img, label, (x1 + 4, y1 - 4), font, font_scale, (255, 255, 255), thickness)
    
    cv2.imwrite(output_path, img)
    return output_path


def generate_error_feedback_text(inconsistencies: List[Dict], 
                                  output_path: str,
                                  vue_code_path: str = None) -> str:
    """Generate natural language error feedback text."""
    lines = []
    lines.append("=== ERROR FEEDBACK ===\n")
    lines.append(f"Total issues found: {len(inconsistencies)}\n")
    lines.append("Issues are marked with red boxes and numbers #1, #2, etc.\n\n")
    
    for i, incon in enumerate(inconsistencies[:5], 1):
        issue_type = incon.get('type', 'UNKNOWN')
        gt_text = incon.get('gt_text', '')
        gen_text = incon.get('gen_text', '')
        context = incon.get('error_code_context', '')
        
        lines.append(f"--- Issue #{i}: {issue_type} ---\n")
        
        if issue_type == 'MISSING':
            lines.append(f"Problem: Widget with text '{gt_text}' is missing from the generated output.\n")
            lines.append(f"Expected: '{gt_text}'\n")
            lines.append("Action: Add this widget to your component.\n")
        elif issue_type == 'TEXT_MISMATCH':
            lines.append(f"Problem: Text content does not match.\n")
            lines.append(f"Expected: '{gt_text}'\n")
            lines.append(f"Got: '{gen_text}'\n")
            lines.append("Action: Fix the text content to match exactly.\n")
        elif issue_type == 'EXTRA':
            lines.append(f"Problem: Extra widget '{gen_text}' not in ground truth.\n")
            lines.append("Action: Remove this unnecessary widget.\n")
        else:
            lines.append(f"Context: {context}\n")
        
        code_snippet = incon.get('vue_code_snippet', '')
        if code_snippet:
            lines.append(f"\nRelated code:\n```vue\n{code_snippet}\n```\n")
        
        lines.append("\n")
    
    feedback_text = '\n'.join(lines)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(feedback_text)
    
    return output_path



def feedback_loop_for_component(txt_path, vue_dir, gt_image_path, proj_dir, webdriver_path, 
                                 max_iterations=2, model="gemini-2.5-pro"):
    """
    Feedback loop for a single component using UIED detection.
    """
    from component_fix import enhanced_fix_generate
    from main import render_single_vue_file
    
    print("\n" + "=" * 80)
    print("Starting Feedback Loop for Component (UIED)")
    print("=" * 80 + "\n")
    
    base_name = os.path.splitext(os.path.basename(txt_path))[0]
    crop_dir = os.path.dirname(txt_path)
    
    print("Step 1: Parsing Vue components from txt...")
    parse_vue_from_txt_component(txt_path, vue_dir, "component")
    parse_vue_from_txt_component(txt_path, vue_dir, "snippet")
    
    with open(txt_path, 'r', encoding='utf-8') as f:
        content = f.read()
    name_match = re.search(r'export\s+default\s*{[^}]*?\bname\s*:\s*[\'"]([^\'"]+)[\'"]', content, re.DOTALL)
    component_name = name_match.group(1).strip() if name_match else "UnknownComponent"
    print(f"Component name: {component_name}")
    
    components_dir = os.path.join(vue_dir, "components")
    os.makedirs(components_dir, exist_ok=True)
    app_vue_path = os.path.join(components_dir, "App.vue")
    
    snippet_match = re.search(r'```snippet\s*(.*?)\s*```', content, re.DOTALL)
    if snippet_match:
        snippet_content = snippet_match.group(1).strip()
        snippet_content = re.sub(
            r"from\s+['\"]\.\/components\/([^'\"]+\.vue)['\"]",
            r"from './\1'",
            snippet_content
        )
        with open(app_vue_path, 'w', encoding='utf-8') as f:
            f.write(snippet_content)
    else:
        app_vue_content = f'''<template>
  <div id="app">
    <{component_name} />
  </div>
</template>

<script>
import {component_name} from './{component_name}.vue'

export default {{
  name: 'App',
  components: {{
    {component_name}
  }}
}}
</script>
'''
        with open(app_vue_path, 'w', encoding='utf-8') as f:
            f.write(app_vue_content)
    
    screenshot_path = os.path.join(crop_dir, f"{base_name}_generated.png")
    html_path = os.path.join(crop_dir, f"{base_name}_generated.html")
    
    print("Step 2: Rendering component screenshot...")
    render_single_vue_file(proj_dir, components_dir, html_path, screenshot_path, webdriver_path)
    
    best_score = 0.0
    best_result = None
    best_code_path = txt_path
    
    for iteration in range(1, max_iterations + 1):
        print(f"\n--- Iteration {iteration}/{max_iterations} ---")
        
        if iteration == 1:
            current_code_path = txt_path
            current_screenshot = screenshot_path
            current_html = html_path
        else:
            current_code_path = os.path.join(crop_dir, f"{base_name}_fixed_iter{iteration-1}.txt")
            current_screenshot = os.path.join(crop_dir, f"{base_name}_iter{iteration}_generated.png")
            current_html = os.path.join(crop_dir, f"{base_name}_iter{iteration}_generated.html")
            
            print(f"Rendering fixed code from iteration {iteration-1}...")
            parse_vue_from_txt_component(current_code_path, vue_dir, "component")
            parse_vue_from_txt_component(current_code_path, vue_dir, "snippet")
            
            with open(current_code_path, 'r', encoding='utf-8') as f:
                fixed_content = f.read()
            snippet_match = re.search(r'```snippet\s*(.*?)\s*```', fixed_content, re.DOTALL)
            if snippet_match:
                snippet_content = snippet_match.group(1).strip()
                snippet_content = re.sub(
                    r"from\s+['\"]\.\/components\/([^'\"]+\.vue)['\"]",
                    r"from './\1'",
                    snippet_content
                )
                with open(app_vue_path, 'w', encoding='utf-8') as f:
                    f.write(snippet_content)
            
            render_single_vue_file(proj_dir, components_dir, current_html, current_screenshot, webdriver_path)
        
        print(f"Step 3: Evaluating with UIED detection...")
        eval_result = evaluate_component_with_uied(
            gt_image_path=gt_image_path,
            generated_image_path=current_screenshot,
            vue_code_path=current_code_path,
            output_dir=crop_dir
        )
        
        if eval_result is None:
            print(f"Evaluation failed at iteration {iteration}")
            break
        
        current_score = eval_result['match_score']
        inconsistencies = eval_result['inconsistencies']
        
        print(f"Match Score: {current_score:.4f}")
        print(f"Inconsistencies: {len(inconsistencies)}")
        
        if current_score > best_score:
            best_score = current_score
            best_result = eval_result
            best_code_path = current_code_path
            shutil.copy(current_screenshot, os.path.join(crop_dir, f"{base_name}_best_generated.png"))
        
        if len(inconsistencies) == 0 or current_score >= 0.95:
            print(f"Satisfactory result at iteration {iteration}!")
            break
        
        if iteration < max_iterations:
            error_feedback_path = os.path.join(crop_dir, f"{base_name}_errors_iter{iteration}.txt")
            generate_error_feedback_text(inconsistencies, error_feedback_path, current_code_path)
            
            annotated_gt_path = os.path.join(crop_dir, f"{base_name}_annotated_gt_iter{iteration}.png")
            annotate_image_with_errors(gt_image_path, inconsistencies, annotated_gt_path, is_gt=True)
            
            annotated_gen_path = os.path.join(crop_dir, f"{base_name}_annotated_gen_iter{iteration}.png")
            annotate_image_with_errors(current_screenshot, inconsistencies, annotated_gen_path, is_gt=False)
            
            error_snippets = []
            for i, incon in enumerate(inconsistencies[:5], 1):
                error_snippets.append({
                    'issue_num': i,
                    'type': incon.get('type', 'UNKNOWN'),
                    'vue_snippet': incon.get('vue_code_snippet', ''),
                    'html_snippet': '',
                    'context': incon.get('error_code_context', '')
                })
            
            fixed_code_path = os.path.join(crop_dir, f"{base_name}_fixed_iter{iteration}.txt")
            
            success = enhanced_fix_generate(
                error_feedback=error_feedback_path,
                original_code_path=current_code_path,
                output_file=fixed_code_path,
                gt_image_path=annotated_gt_path,
                generated_image_path=annotated_gen_path,
                error_code_snippets=error_snippets,
                model=model
            )
            
            if not success:
                print(f"Failed to fix code at iteration {iteration}")
                break
        else:
            print(f"Reached maximum iterations ({max_iterations})")
    
    print("\n" + "=" * 80)
    print(f"Feedback Loop Complete - Best Score: {best_score:.4f}")
    print("=" * 80 + "\n")
    
    best_txt_path = os.path.join(crop_dir, f"{base_name}_best.txt")
    if best_code_path != txt_path:
        shutil.copy(best_code_path, best_txt_path)
    else:
        shutil.copy(txt_path, best_txt_path)
    print(f"Best code saved to: {best_txt_path}")
    
    # Cleanup
    print("\nCleaning up intermediate files...")
    cleanup_patterns = [
        f"{base_name}_fixed_iter*.txt",
        f"{base_name}_errors_iter*.txt",
        f"{base_name}_iter*_generated.png",
        f"{base_name}_annotated_*_iter*.png"
    ]
    
    for pattern in cleanup_patterns:
        for file in glob.glob(os.path.join(crop_dir, pattern)):
            try:
                os.remove(file)
                print(f"  Deleted: {os.path.basename(file)}")
            except Exception as e:
                print(f"  Warning: Could not delete {os.path.basename(file)}: {e}")
    
    return best_result



def feedback_loop_for_page(page_dir, vue_dir, proj_dir, webdriver_path,
                            max_iterations=2, model="gemini-2.5-pro", top_k=3):
    """
    Feedback loop for a full page: evaluate all components using UIED,
    then fix the top-k components with lowest scores.
    """
    from main import render_single_vue_file
    
    print("\n" + "=" * 80)
    print(f"Starting Page Feedback Loop (UIED, fixing top {top_k} lowest scores)")
    print("=" * 80 + "\n")
    
    txt_files = glob.glob(os.path.join(page_dir, "*.txt"))
    txt_files = [f for f in txt_files if not f.endswith('_best.txt') and 
                 not f.endswith('_errors.txt') and 
                 'masked_image' not in os.path.basename(f)]
    
    print(f"Found {len(txt_files)} component txt files")
    
    component_scores = []
    
    for txt_path in txt_files:
        base_name = os.path.splitext(os.path.basename(txt_path))[0]
        gt_image_path = os.path.join(page_dir, f"{base_name}.png")
        
        if not os.path.exists(gt_image_path):
            gt_image_path = os.path.join(page_dir, f"{base_name}.jpg")
        
        if not os.path.exists(gt_image_path):
            print(f"Skipping {base_name}: no GT image found")
            continue
        
        print(f"\nEvaluating component: {base_name}")
        
        parse_vue_from_txt_component(txt_path, vue_dir, "component")
        parse_vue_from_txt_component(txt_path, vue_dir, "snippet")
        
        components_dir = os.path.join(vue_dir, "components")
        os.makedirs(components_dir, exist_ok=True)
        app_vue_path = os.path.join(components_dir, "App.vue")
        
        with open(txt_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        snippet_match = re.search(r'```snippet\s*(.*?)\s*```', content, re.DOTALL)
        if snippet_match:
            snippet_content = snippet_match.group(1).strip()
            snippet_content = re.sub(
                r"from\s+['\"]\.\/components\/([^'\"]+\.vue)['\"]",
                r"from './\1'",
                snippet_content
            )
            with open(app_vue_path, 'w', encoding='utf-8') as f:
                f.write(snippet_content)
        
        screenshot_path = os.path.join(page_dir, f"{base_name}_generated.png")
        html_path = os.path.join(page_dir, f"{base_name}_generated.html")
        
        try:
            render_single_vue_file(proj_dir, components_dir, html_path, screenshot_path, webdriver_path)
            
            eval_result = evaluate_component_with_uied(
                gt_image_path=gt_image_path,
                generated_image_path=screenshot_path,
                vue_code_path=txt_path,
                output_dir=page_dir
            )
            
            score = eval_result['match_score'] if eval_result else 0.0
            
            component_scores.append({
                'base_name': base_name,
                'txt_path': txt_path,
                'gt_image_path': gt_image_path,
                'score': score,
                'eval_result': eval_result
            })
            
            print(f"  Score: {score:.4f}")
            
        except Exception as e:
            print(f"  Error evaluating {base_name}: {e}")
            component_scores.append({
                'base_name': base_name,
                'txt_path': txt_path,
                'gt_image_path': gt_image_path,
                'score': 0.0,
                'eval_result': None
            })
    
    component_scores.sort(key=lambda x: x['score'])
    
    print("\n" + "=" * 80)
    print("Component Scores (sorted by similarity, lowest first):")
    for item in component_scores:
        print(f"  {item['base_name']}: {item['score']:.4f}")
    print("=" * 80)
    
    print(f"\nRunning feedback loop on top {top_k} lowest scoring components...")
    
    results = {}
    for item in component_scores[:top_k]:
        print(f"\n{'='*60}")
        print(f"Fixing component: {item['base_name']} (initial score: {item['score']:.4f})")
        print(f"{'='*60}")
        
        result = feedback_loop_for_component(
            txt_path=item['txt_path'],
            vue_dir=vue_dir,
            gt_image_path=item['gt_image_path'],
            proj_dir=proj_dir,
            webdriver_path=webdriver_path,
            max_iterations=max_iterations,
            model=model
        )
        
        results[item['base_name']] = result
    
    print("\n" + "=" * 80)
    print("Page Feedback Loop Complete")
    print("=" * 80)
    
    return {
        'all_scores': component_scores,
        'fixed_results': results
    }
