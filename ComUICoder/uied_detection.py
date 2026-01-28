

import os
import sys
import json
import subprocess
import cv2
import numpy as np
import math
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from enum import Enum


class UIEDWidgetType(Enum):
    """Widget types that UIED can detect"""
    TEXT = "Text"
    TEXT_BUTTON = "Compo"  # UIED uses "Compo" for components/buttons
    IMAGE = "Image"
    INPUT = "Input"
    ICON = "Icon"
    BLOCK = "Block"
    UNKNOWN = "Unknown"


@dataclass
class Bbox:
    """Bounding box coordinates"""
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def center(self) -> Tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)


@dataclass
class UIEDWidget:
    """Widget detected by UIED"""
    widget_type: UIEDWidgetType
    bbox: Bbox
    texts: List[str] = field(default_factory=list)
    confidence: float = 1.0
    component_id: int = -1
    children: List[int] = field(default_factory=list)
    parent_id: Optional[int] = None
    
    @property
    def type(self) -> UIEDWidgetType:
        return self.widget_type

    @property
    def width(self) -> int:
        return self.bbox.width

    @property
    def height(self) -> int:
        return self.bbox.height

    @property
    def area(self) -> int:
        return self.bbox.area



UIED_PATH = r'C:\Users\Shuoqi\Documents\GitHub\VueGen\UIED'
UIED_PARENT_DIR = os.path.dirname(UIED_PATH)


# Granularity presets for detection
GRANULARITY_PRESETS = {
    'fine': {
        'min-grad': 4,
        'ffl-block': 5,
        'min-ele-area': 50,
        'max-word-inline-gap': 6,
        'max-line-gap': 1,
        'merge-contained-ele': False,
        'merge-line-to-paragraph': False,
        'remove-bar': False
    },
    'medium': {
        'min-grad': 10,
        'ffl-block': 5,
        'min-ele-area': 200,
        'max-word-inline-gap': 10,
        'max-line-gap': 4,
        'merge-contained-ele': True,
        'merge-line-to-paragraph': True,
        'remove-bar': False
    },
    'coarse': {
        'min-grad': 15,
        'ffl-block': 8,
        'min-ele-area': 500,
        'max-word-inline-gap': 15,
        'max-line-gap': 8,
        'merge-contained-ele': True,
        'merge-line-to-paragraph': True,
        'remove-bar': False
    }
}


def adapt_params(img_path: str, granularity: str = 'medium') -> dict:
    """
    Adapt UIED parameters based on image size and granularity preference.
    
    Args:
        img_path: Path to the input image
        granularity: 'fine', 'medium', or 'coarse' - controls detection granularity
    
    Returns:
        Dictionary of UIED parameters
    """
    org = cv2.imread(img_path)
    if org is None:
        raise ValueError(f"Cannot read image: {img_path}")
    
    H, W = org.shape[:2]
    baseline = 800
    
    # Get base parameters from preset
    if granularity not in GRANULARITY_PRESETS:
        granularity = 'medium'
    base_params = GRANULARITY_PRESETS[granularity].copy()
    
    # Scale parameters based on image height
    if H > baseline:
        scale = H / baseline
        key_params = {
            'min-grad': base_params['min-grad'],
            'ffl-block': base_params['ffl-block'],
            'min-ele-area': int(math.sqrt(base_params['min-ele-area']) * scale) ** 2,
            'max-word-inline-gap': int(base_params['max-word-inline-gap'] * scale),
            'max-line-gap': int(base_params['max-line-gap'] * scale),
            'merge-contained-ele': base_params['merge-contained-ele'],
            'merge-line-to-paragraph': base_params['merge-line-to-paragraph'],
            'remove-bar': base_params['remove-bar']
        }
    else:
        key_params = base_params.copy()
        key_params['max-line-gap'] = int(base_params['max-line-gap'] * (H / baseline))
    
    print(f"[INFO] Image H={H}, W={W}, granularity={granularity}, key_params={key_params}")
    return key_params


def resize_height_by_longest_edge(img_path: str, resize_length: int = 800) -> int:
    """Get the height for resizing based on original image."""
    org = cv2.imread(img_path)
    if org is None:
        return resize_length
    height, width = org.shape[:2]
    return height


def merge_nearby_components(components: List[Dict], threshold: int = 30) -> List[Dict]:
    """
    Merge nearby components to reduce over-segmentation.
    Similar to merge_bboxs logic in ocr.py.
    
    Args:
        components: List of component dictionaries with 'bbox' key [x1, y1, x2, y2]
        threshold: Maximum distance between components to merge
    
    Returns:
        List of merged components
    """
    if not components:
        return components
    
    # Make a copy to avoid modifying original
    results = [comp.copy() for comp in components]
    
    while True:
        merged = False
        for idx, item in enumerate(results):
            if item is None:
                continue
            
            bbox = item['bbox']
            left, top, right, bottom = bbox[0], bbox[1], bbox[2], bbox[3]
            
            for idx2, item2 in enumerate(results):
                if idx2 <= idx or item2 is None:
                    continue
                
                bbox2 = item2['bbox']
                left2, top2, right2, bottom2 = bbox2[0], bbox2[1], bbox2[2], bbox2[3]
                
                # Check if boxes are close enough to merge
                # They should not be separated by more than threshold in any direction
                if not (left2 > right + threshold or 
                        right2 < left - threshold or 
                        bottom2 < top - threshold or 
                        top2 > bottom + threshold):
                    
                    # Merge bounding boxes
                    new_left = min(left, left2)
                    new_top = min(top, top2)
                    new_right = max(right, right2)
                    new_bottom = max(bottom, bottom2)
                    
                    # Merge text content
                    text1 = item.get('text', '')
                    text2 = item2.get('text', '')
                    if text1 and text2:
                        new_text = f"{text1}\n{text2}"
                    else:
                        new_text = text1 or text2
                    
                    # Merge children
                    children1 = item.get('children', [])
                    children2 = item2.get('children', [])
                    new_children = children1 + children2
                    
                    # Update first item with merged data
                    results[idx] = {
                        'id': item.get('id', idx),
                        'bbox': [new_left, new_top, new_right, new_bottom],
                        'type': item.get('type', 'Compo'),  # Keep first item's type
                        'text': new_text,
                        'confidence': min(item.get('confidence', 1.0), item2.get('confidence', 1.0)),
                        'children': new_children,
                        'parent_id': item.get('parent_id')
                    }
                    
                    # Mark second item as merged (to be removed)
                    results[idx2] = None
                    merged = True
                    break
            
            if merged:
                break
        
        # Remove None entries
        results = [r for r in results if r is not None]
        
        if not merged:
            break
    
    # Re-assign IDs
    for i, comp in enumerate(results):
        comp['id'] = i
    
    return results


def merge_contained_components(components: List[Dict], containment_threshold: float = 0.8) -> List[Dict]:
    """
    Remove components that are mostly contained within other components.
    
    Args:
        components: List of component dictionaries
        containment_threshold: Ratio threshold for containment (0.8 = 80% contained)
    
    Returns:
        List of components with contained ones removed
    """
    if not components:
        return components
    
    def calc_overlap_ratio(inner_bbox, outer_bbox):
        """Calculate how much of inner is contained in outer."""
        x1 = max(inner_bbox[0], outer_bbox[0])
        y1 = max(inner_bbox[1], outer_bbox[1])
        x2 = min(inner_bbox[2], outer_bbox[2])
        y2 = min(inner_bbox[3], outer_bbox[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        inter_area = (x2 - x1) * (y2 - y1)
        inner_area = (inner_bbox[2] - inner_bbox[0]) * (inner_bbox[3] - inner_bbox[1])
        
        if inner_area == 0:
            return 0.0
        
        return inter_area / inner_area
    
    # Sort by area (largest first)
    sorted_comps = sorted(components, 
                          key=lambda c: (c['bbox'][2] - c['bbox'][0]) * (c['bbox'][3] - c['bbox'][1]), 
                          reverse=True)
    
    keep = []
    for comp in sorted_comps:
        is_contained = False
        for kept in keep:
            ratio = calc_overlap_ratio(comp['bbox'], kept['bbox'])
            if ratio >= containment_threshold:
                # This component is mostly contained in a larger one
                # Merge text if available
                if comp.get('text') and not kept.get('text'):
                    kept['text'] = comp.get('text')
                elif comp.get('text') and kept.get('text'):
                    kept['text'] = f"{kept['text']}\n{comp['text']}"
                is_contained = True
                break
        
        if not is_contained:
            keep.append(comp)
    
    # Re-assign IDs
    for i, comp in enumerate(keep):
        comp['id'] = i
    
    return keep

def check_uied_available() -> bool:
    """Check if UIED is available in the system"""
    if os.path.exists(UIED_PATH):
        detect_script = os.path.join(UIED_PATH, 'detect_compo', 'ip_region_proposal.py')
        return os.path.exists(detect_script)
    return False


def run_uied_detection(image_path: str, output_dir: str = None, granularity: str = 'coarse') -> str:
    """
    Run UIED detection on an image.
    
    Args:
        image_path: Path to the input image
        output_dir: Optional output directory for UIED results
        granularity: 'fine', 'medium', or 'coarse' - controls detection granularity
                    'coarse' produces fewer, larger components (less fine-grained)
                    'fine' produces more, smaller components (more fine-grained)
    
    Returns:
        Path to the JSON output file containing detection results
    """
    if not check_uied_available():
        raise RuntimeError(f"UIED not found at {UIED_PATH}. Please install UIED or set UIED_PATH environment variable.")
    
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(image_path), 'uied_output')
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Add UIED parent directory to sys.path to allow imports
    if UIED_PARENT_DIR not in sys.path:
        sys.path.insert(0, UIED_PARENT_DIR)
        
    try:
        # Use local adapt_params with granularity control
        key_params = adapt_params(image_path, granularity=granularity)
        resized_height = resize_height_by_longest_edge(image_path, resize_length=800)
        
        # Import UIED modules dynamically
        import UIED.detect_text.text_detection as text
        import UIED.detect_compo.ip_region_proposal as ip
        import UIED.detect_merge.merge as merge
        from os.path import join as pjoin
        
        # 1. Text Detection
        os.makedirs(pjoin(output_dir, 'ocr'), exist_ok=True)
        text.text_detection(image_path, output_dir, method='google')
        
        # 2. Component Detection
        os.makedirs(pjoin(output_dir, 'ip'), exist_ok=True)
        ip.compo_detection(
            image_path, 
            output_dir, 
            key_params,
            classifier=None, 
            resize_by_height=resized_height, 
            show=False
        )
        
        # 3. Merge Results
        os.makedirs(pjoin(output_dir, 'merge'), exist_ok=True)
        
        # Construct paths for merge - handle both forward and backward slashes
        name = os.path.splitext(os.path.basename(image_path))[0]
        
        compo_path = pjoin(output_dir, 'ip', str(name) + '.json')
        ocr_path = pjoin(output_dir, 'ocr', str(name) + '.json')
        
        # Check if files exist before merging
        if not os.path.exists(compo_path):
            print(f"Warning: Component detection file not found: {compo_path}")
            return None
        if not os.path.exists(ocr_path):
            print(f"Warning: OCR file not found: {ocr_path}")
            return None
        
        # Use UIED merge function (updated signature without is_remove_bar/is_paragraph)
        merge.merge(
            image_path, 
            compo_path, 
            ocr_path, 
            pjoin(output_dir, 'merge'),
            show=False
        )
        
        # The merged result is saved in output_dir/merge/name.json
        json_path = pjoin(output_dir, 'merge', str(name) + '.json')
        
        if os.path.exists(json_path):
            return json_path
        else:
            print(f"Warning: Merged JSON not found at: {json_path}")
            
    except Exception as e:
        print(f"Error running UIED: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    return None


def parse_uied_results(json_path: str, merge_threshold: int = 30, 
                       merge_contained: bool = True) -> List[Dict]:
    """
    Parse UIED detection results from JSON file.
    
    Args:
        json_path: Path to UIED output JSON file
        merge_threshold: Distance threshold for merging nearby components
        merge_contained: Whether to merge components contained in others
    
    Returns:
        List of detected components with their properties
    """
    if not os.path.exists(json_path):
        return []
    
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except UnicodeDecodeError:
        # Fallback for Windows encoding issues
        with open(json_path, 'r', encoding='gb18030', errors='replace') as f:
            data = json.load(f)
    
    components = []
    
    # UIED merged JSON format: {"compos": [...], "texts": [...], "connections": [...]}
    if isinstance(data, dict):
        # Combine compos and texts from merged output
        all_elements = []
        if 'compos' in data:
            all_elements.extend(data['compos'])
        if 'texts' in data:
            all_elements.extend(data['texts'])
        # Fallback to 'elements' key if present
        if not all_elements and 'elements' in data:
            all_elements = data['elements']
        compos = all_elements
    elif isinstance(data, list):
        compos = data
    else:
        return []
    
    for comp in compos:
        # Extract bounding box - handle UIED's various formats
        if 'position' in comp:
            pos = comp['position']
            x1 = pos.get('column_min', pos.get('x1', 0))
            y1 = pos.get('row_min', pos.get('y1', 0))
            x2 = pos.get('column_max', pos.get('x2', 0))
            y2 = pos.get('row_max', pos.get('y2', 0))
        elif 'column_min' in comp:
            x1 = comp.get('column_min', 0)
            y1 = comp.get('row_min', 0)
            x2 = comp.get('column_max', 0)
            y2 = comp.get('row_max', 0)
        elif 'bbox' in comp:
            bbox = comp['bbox']
            if isinstance(bbox, list) and len(bbox) == 4:
                x1, y1, x2, y2 = bbox
            else:
                x1 = bbox.get('x1', 0)
                y1 = bbox.get('y1', 0)
                x2 = bbox.get('x2', 0)
                y2 = bbox.get('y2', 0)
        else:
            continue
        
        # Get component type
        comp_type = comp.get('class', comp.get('type', 'Unknown'))
        
        # Get text content if available
        text = comp.get('text_content', comp.get('text', ''))
        
        # Get children and parent info
        children = comp.get('children', [])
        parent_id = comp.get('parent', None)
        
        components.append({
            'id': comp.get('id', len(components)),
            'bbox': [int(x1), int(y1), int(x2), int(y2)],
            'type': comp_type,
            'text': text,
            'confidence': comp.get('score', comp.get('confidence', 1.0)),
            'children': children,
            'parent_id': parent_id
        })
    
    # Apply post-processing to reduce over-segmentation
    if merge_contained:
        components = merge_contained_components(components, containment_threshold=0.8)
    
    if merge_threshold > 0:
        components = merge_nearby_components(components, threshold=merge_threshold)
    
    print(f"[INFO] Parsed {len(components)} components after merging")
    
    return components


def map_uied_type_to_widget_type(uied_type: str) -> UIEDWidgetType:
    """Map UIED component type string to UIEDWidgetType enum"""
    type_lower = uied_type.lower()
    
    if 'text' in type_lower:
        return UIEDWidgetType.TEXT
    elif 'button' in type_lower or 'compo' in type_lower or 'component' in type_lower:
        return UIEDWidgetType.TEXT_BUTTON
    elif 'image' in type_lower or 'img' in type_lower:
        return UIEDWidgetType.IMAGE
    elif 'input' in type_lower or 'textbox' in type_lower or 'edit' in type_lower:
        return UIEDWidgetType.INPUT
    elif 'icon' in type_lower:
        return UIEDWidgetType.ICON
    elif 'block' in type_lower or 'container' in type_lower:
        return UIEDWidgetType.BLOCK
    else:
        return UIEDWidgetType.UNKNOWN


def detect_widgets_uied(image_path: str, output_dir: str = None, granularity: str = 'coarse',
                        merge_threshold: int = 30) -> Dict[int, UIEDWidget]:
    """
    Detect widgets in an image using UIED.
    
    Args:
        image_path: Path to the image file
        output_dir: Optional output directory for UIED results
        granularity: 'fine', 'medium', or 'coarse' - controls detection granularity
                    'coarse' produces fewer, larger components (recommended for most cases)
        merge_threshold: Distance threshold for merging nearby components (default 30px)
    
    Returns:
        Dictionary mapping widget IDs to UIEDWidget objects
    """
    # Run UIED detection
    try:
        json_path = run_uied_detection(image_path, output_dir, granularity=granularity)
    except Exception as e:
        print(f"UIED detection error: {e}")
        json_path = None
    
    if json_path is None:
        print(f"UIED detection failed for {image_path}, using fallback method")
        return detect_widgets_uied_fallback(image_path, use_ocr=True)
    
    # Parse results with merging
    components = parse_uied_results(json_path, merge_threshold=merge_threshold, merge_contained=True)
    
    widgets = {}
    for i, comp in enumerate(components):
        bbox_coords = comp['bbox']
        bbox = Bbox(
            x1=bbox_coords[0],
            y1=bbox_coords[1],
            x2=bbox_coords[2],
            y2=bbox_coords[3]
        )
        
        widget_type = map_uied_type_to_widget_type(comp['type'])
        
        texts = []
        if comp.get('text'):
            texts = [comp['text']]
        
        widget = UIEDWidget(
            widget_type=widget_type,
            bbox=bbox,
            texts=texts,
            confidence=comp.get('confidence', 1.0),
            component_id=comp.get('id', i),
            children=comp.get('children', []),
            parent_id=comp.get('parent_id')
        )
        
        widgets[i] = widget
    
    # Sort by position (top-to-bottom, left-to-right)
    sorted_widgets = {}
    sorted_items = sorted(widgets.items(), key=lambda x: (x[1].bbox.y1, x[1].bbox.x1))
    for new_id, (_, widget) in enumerate(sorted_items):
        sorted_widgets[new_id] = widget
    
    return sorted_widgets


def detect_widgets_uied_fallback(image_path: str, use_ocr: bool = True) -> Dict[int, UIEDWidget]:
    """
    Fallback method to detect widgets when UIED is not available.
    Uses OpenCV-based edge detection and contour analysis.
    
    Args:
        image_path: Path to the image file
        use_ocr: Whether to use OCR for text extraction
    
    Returns:
        Dictionary mapping widget IDs to UIEDWidget objects
    """
    import pytesseract
    
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image: {image_path}")
        return {}
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Use adaptive thresholding for better edge detection
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    
    # Find contours
    contours, _ = cv2.findContours(
        cv2.bitwise_not(binary), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    
    widgets = {}
    widget_id = 0
    
    min_area = 100  # Minimum area threshold
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        
        x, y, w, h = cv2.boundingRect(contour)
        
        # Skip very thin elements
        if w < 10 or h < 10:
            continue
        
        bbox = Bbox(x1=x, y1=y, x2=x+w, y2=y+h)
        
        # Try to extract text if OCR is enabled
        texts = []
        if use_ocr:
            try:
                roi = image[y:y+h, x:x+w]
                text = pytesseract.image_to_string(roi, lang='eng+chi_sim', config='--psm 6')
                text = text.strip()
                if text:
                    texts = [text]
            except Exception:
                pass
        
        # Classify based on aspect ratio and content
        aspect_ratio = w / h if h > 0 else 1
        
        if texts:
            widget_type = UIEDWidgetType.TEXT
        elif 0.8 < aspect_ratio < 1.2 and area < 2000:
            widget_type = UIEDWidgetType.ICON
        elif w > 100 and h < 50:
            widget_type = UIEDWidgetType.INPUT
        else:
            widget_type = UIEDWidgetType.BLOCK
        
        widget = UIEDWidget(
            widget_type=widget_type,
            bbox=bbox,
            texts=texts,
            component_id=widget_id
        )
        
        widgets[widget_id] = widget
        widget_id += 1
    
    # Sort by position
    sorted_widgets = {}
    sorted_items = sorted(widgets.items(), key=lambda x: (x[1].bbox.y1, x[1].bbox.x1))
    for new_id, (_, widget) in enumerate(sorted_items):
        sorted_widgets[new_id] = widget
    
    return sorted_widgets


# Compatibility layer for evaluation.py
def convert_to_evaluation_format(uied_widgets: Dict[int, UIEDWidget]) -> Dict:
    """
    Convert UIEDWidget dictionary to format compatible with evaluation.py Widget class.
    """
    from evaluation import Widget as EvalWidget, Bbox as EvalBbox, WidgetType as EvalWidgetType
    
    type_mapping = {
        UIEDWidgetType.TEXT: EvalWidgetType.TEXT_VIEW,
        UIEDWidgetType.TEXT_BUTTON: EvalWidgetType.TEXT_BUTTON,
        UIEDWidgetType.IMAGE: EvalWidgetType.IMAGE_VIEW,
        UIEDWidgetType.INPUT: EvalWidgetType.INPUT_BOX,
        UIEDWidgetType.ICON: EvalWidgetType.ICON_BUTTON,
        UIEDWidgetType.BLOCK: EvalWidgetType.TEXT,
        UIEDWidgetType.UNKNOWN: EvalWidgetType.TEXT,
    }
    
    eval_widgets = {}
    for widget_id, uied_widget in uied_widgets.items():
        eval_type = type_mapping.get(uied_widget.widget_type, EvalWidgetType.TEXT)
        eval_bbox = EvalBbox(
            uied_widget.bbox.x1,
            uied_widget.bbox.y1,
            uied_widget.bbox.x2,
            uied_widget.bbox.y2
        )
        
        eval_widgets[widget_id] = EvalWidget(
            widget_type=eval_type,
            bbox=eval_bbox,
            texts=uied_widget.texts
        )
    
    return eval_widgets


# High-level API
def detect_gt_widgets(image_path: str, use_uied: bool = True, output_dir: str = None, 
                      granularity: str = 'coarse', merge_threshold: int = 30) -> Dict[int, UIEDWidget]:
    """
    Detect widgets in a ground truth image.
    
    This is the main API function for detecting widgets in GT images.
    Uses UIED if available, falls back to OpenCV-based detection otherwise.
    
    Args:
        image_path: Path to the ground truth image
        use_uied: Whether to try using UIED first
        output_dir: Optional output directory for UIED results
        granularity: 'fine', 'medium', or 'coarse' - controls detection granularity
                    'coarse' (default) produces fewer, larger components
                    'medium' is balanced
                    'fine' produces more, smaller components
        merge_threshold: Distance threshold for merging nearby components (default 30px)
                        Set to 0 to disable merging
    
    Returns:
        Dictionary mapping widget IDs to UIEDWidget objects
    """
    if use_uied and check_uied_available():
        print(f"Using UIED for detection (granularity={granularity}, merge_threshold={merge_threshold}): {image_path}")
        widgets = detect_widgets_uied(image_path, output_dir, granularity=granularity, 
                                      merge_threshold=merge_threshold)
        if widgets:
            return widgets
        print("UIED detection returned no results, falling back to OpenCV method")
    
    print(f"Using OpenCV fallback for detection: {image_path}")
    return detect_widgets_uied_fallback(image_path)
