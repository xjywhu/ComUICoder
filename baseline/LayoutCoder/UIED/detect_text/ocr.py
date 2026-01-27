import cv2
import easyocr

def ocr_detection_google(imgpath):
    reader = easyocr.Reader(['en']) # this needs to run only once to load the model into memory
    img = cv2.imread(imgpath)
    result = reader.readtext(img)

    for (bbox, text, prob) in result:
        print(f"检测到文字: {text}, 置信度: {prob:.2f}, 位置: {bbox}")

    output = []
    for (bbox, text, prob) in result:
        vertices = [{"x": int(x), "y": int(y)} for (x, y) in bbox]
        output.append({
            "description": text,
            "probability": float(prob),
            "boundingPoly": {"vertices": vertices}
        })
    return output