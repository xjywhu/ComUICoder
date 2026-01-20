import os
import json
from typing import List, Dict, Any


# =====================================================
# Geometry utils
# =====================================================

def area(b):
    return max(0, b["x2"] - b["x1"]) * max(0, b["y2"] - b["y1"])


def intersection(a, b):
    x1 = max(a["x1"], b["x1"])
    y1 = max(a["y1"], b["y1"])
    x2 = min(a["x2"], b["x2"])
    y2 = min(a["y2"], b["y2"])
    if x2 <= x1 or y2 <= y1:
        return 0
    return (x2 - x1) * (y2 - y1)


def sort_key(b):
    return (b["y1"], b["x1"])


# =====================================================
# JSON loaders (STRICTLY your formats)
# =====================================================

def load_reference_json(path: str) -> List[Dict]:
    """
    coordinates.json format:
    {
      "1": [x1, y1, x2, y2],
      ...
    }
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    boxes = []
    for k, (x1, y1, x2, y2) in data.items():
        boxes.append({
            "id": f"{k}",
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2
        })
    return boxes


def load_predicted_json(path: str) -> List[Dict]:
    """
    predicted format:
    [
      {"bbox_2d": [x1, y1, x2, y2], "label": "..."},
      ...
    ]
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    boxes = []
    for i, item in enumerate(data):
        x1, y1, x2, y2 = item["bbox_2d"]
        boxes.append({
            "id": item.get("label"),
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2
        })
    return boxes


# =====================================================
# Matching core (ref-driven, coverage-based)
# =====================================================

def match_bboxes(ref_boxes: List[Dict], pred_boxes: List[Dict],
                 min_coverage: float = 0.3, strong_coverage: float = 0.8) -> List[Dict]:

    ref_boxes = sorted(ref_boxes, key=sort_key)
    pred_boxes = sorted(pred_boxes, key=sort_key)

    matches = []

    for ref in ref_boxes:
        ref_area = area(ref)
        candidates = []

        for pred in pred_boxes:
            inter = intersection(ref, pred)
            if inter <= 0:
                continue

            pred_area = area(pred)
            cov_ref = inter / ref_area if ref_area > 0 else 0
            cov_pred = inter / pred_area if pred_area > 0 else 0
            # if ref["id"] == "12":
            #     print(cov_pred)
            if cov_ref >= min_coverage or cov_pred >= min_coverage:
                candidates.append((pred, cov_ref, cov_pred))

        if not candidates:
            matches.append({
                "ref_id": ref["id"],
                "match_type": "unmatched",
                "pred_ids": []
            })
            continue

        strong_preds = [p for p, cr, cp in candidates if cp >= strong_coverage]
        if strong_preds:
            matches.append({
                "ref_id": ref["id"],
                "match_type": "ref_contains_preds",
                "pred_ids": [p["id"] for p in strong_preds]
            })
            continue

        strong_ref = [p for p, cr, cp in candidates if cr >= strong_coverage]
        if strong_ref:
            best = max(strong_ref, key=lambda p: intersection(ref, p))
            matches.append({
                "ref_id": ref["id"],
                "match_type": "pred_contains_ref",
                "pred_ids": [best["id"]]
            })
            continue

        best = max(candidates, key=lambda c: intersection(ref, c[0]))
        matches.append({
            "ref_id": ref["id"],
            "match_type": "partial_match",
            "pred_ids": [best[0]["id"]]
        })

    return matches


# =====================================================
# Evaluation (component counts + matching rate)
# =====================================================

def evaluate_page(ref_json: str, pred_json: str) -> Dict[str, Any]:
    ref_boxes = load_reference_json(ref_json)
    pred_boxes = load_predicted_json(pred_json)

    matches = match_bboxes(ref_boxes, pred_boxes)

    num_ref = len(ref_boxes)
    num_pred = len(pred_boxes)

    matched_ref = sum(
        1 for m in matches if m["match_type"] != "unmatched"
    )

    matching_rate = matched_ref / num_ref if num_ref > 0 else 0.0

    return {
        "num_reference_components": num_ref,
        "num_predicted_components": num_pred,
        "num_matched_reference_components": matched_ref,
        "matching_rate": matching_rate,
        "matches": matches
    }


# =====================================================
# Main: your exact folder structure
# =====================================================

def evaluate_case( out_path, case_id: str):
    ref_root = rf"D:\py_code\fyp\VueGen\multipage_data\{case_id}"
    pred_root = rf"D:\py_code\fyp\VueGen\output_multi\{case_id}"

    results = []
    total_ref = 0
    total_matched = 0

    for page_idx in range(1, 7):
        ref_json = os.path.join(
            ref_root, f"page{page_idx}", "coordinates.json"
        )
        pred_json = os.path.join(
            pred_root, str(page_idx), f"{page_idx}_test_modify.json"
        )

        if not os.path.exists(ref_json) or not os.path.exists(pred_json):
            results.append({
                "page": page_idx,
                "status": "missing_json"
            })
            continue

        page_result = evaluate_page(ref_json, pred_json)
        page_result["page"] = page_idx
        results.append(page_result)

        total_ref += page_result["num_reference_components"]
        total_matched += page_result["num_matched_reference_components"]

    overall_rate = total_matched / total_ref if total_ref > 0 else 0.0

    output = {
        "case_id": case_id,
        "overall_matching_rate": overall_rate,
        "pages": results
    }


    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"Evaluation finished. Saved to {out_path}")


# =====================================================
# Run
# =====================================================

if __name__ == "__main__":
    out_path = f"layout_matching_eval_case_1.json"
    evaluate_case( out_path, case_id="1")
