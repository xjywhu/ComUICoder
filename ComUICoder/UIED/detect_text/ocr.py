import cv2
import easyocr


def merge_bboxs(results, threshold=50):
    while True:
        flag = True
        for idx, item in enumerate(results):
            bbox, text, _ = item
            left, top, right, bottom = bbox[0][0], bbox[0][1], bbox[2][0], bbox[2][1]
            for idx2, item2 in enumerate(results):
                if idx2 == idx:
                    continue
                bbox2, text2, _ = item2
                left2, top2, right2, bottom2 = bbox2[0][0], bbox2[0][1], bbox2[2][0], bbox2[2][1]
                if not (left2 > right + threshold or right2 < left - threshold or bottom2 < top - threshold or top2 > bottom + threshold):
                    left3, top3, right3, bottom3 = min(left, left2), min(top, top2), max(right, right2), max(bottom, bottom2)
                    results[idx] = ([[left3, top3], [right3, top3], [right3, bottom3], [left3, bottom3]], f'{text}\n{text2}', None)
                    results = results[:idx2]+results[idx2+1:]
                    flag = False
                    break
            if not flag:
                break
        if flag:
            return results
def ocr_detection_google(imgpath):
    reader = easyocr.Reader(['en', 'ch_sim']) # this needs to run only once to load the model into memory
    img = cv2.imread(imgpath)
    result = reader.readtext(img)
    result = merge_bboxs(result, threshold=10)

    output = []
    for (bbox, text, _) in result:
        vertices = [{"x": int(x), "y": int(y)} for (x, y) in bbox]
        output.append({
            "description": text,
            "boundingPoly": {"vertices": vertices}
        })
    return output