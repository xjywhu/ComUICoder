from os.path import join as pjoin
import cv2
import os
import numpy as np
import math


def resize_height_by_longest_edge(img_path, resize_length=800):
    org = cv2.imread(img_path)
    height, width = org.shape[:2]
    return height


def color_tips():
    color_map = {'Text': (0, 0, 255), 'Compo': (0, 255, 0), 'Block': (0, 255, 255), 'Text Content': (255, 0, 255)}
    board = np.zeros((200, 200, 3), dtype=np.uint8)

    board[:50, :, :] = (0, 0, 255)
    board[50:100, :, :] = (0, 255, 0)
    board[100:150, :, :] = (255, 0, 255)
    board[150:200, :, :] = (0, 255, 255)
    cv2.putText(board, 'Text', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    cv2.putText(board, 'Non-text Compo', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    cv2.putText(board, "Compo's Text Content", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    cv2.putText(board, "Block", (10, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    cv2.imshow('colors', board)


def adapt_params(img_path):
    # 读取原图大小
    org = cv2.imread(img_path)
    H, W = org.shape[:2]

    # 基准高度
    baseline = 800
    scale = H / baseline

    # baseline 参数
    base_params = {
        'min-grad': 10,
        'ffl-block': 5,
        'min-ele-area': 200,
        'max-word-inline-gap': 10,
        'max-line-gap': 4,
    }
    if H>800:
        key_params = {
            'min-grad': base_params['min-grad'],
            'ffl-block': base_params['ffl-block'],
            'min-ele-area': int(math.sqrt(base_params['min-ele-area']) * (H / 800)),
            'max-word-inline-gap': int(base_params['max-word-inline-gap'] * (H / 800)),
            'max-line-gap': int(4 * (H / 800)),
            'merge-contained-ele': True,
            'merge-line-to-paragraph': True,
            'remove-bar': False
        }
    else:
        key_params = {
            'min-grad': base_params['min-grad'],
            'ffl-block': base_params['ffl-block'],
            'min-ele-area': base_params['min-ele-area'],
            'max-word-inline-gap': base_params['max-word-inline-gap'],
            'max-line-gap': 4 * (H / 800),
            'merge-contained-ele': True,
            'merge-line-to-paragraph': True,
            'remove-bar': False
        }


    print(f"[INFO] Image H={H}, scale={scale:.2f}, key_params={key_params}")
    return key_params


def uied(folder, output_root = f'./data/output', is_ip = True, is_ocr = True, is_merge = True, is_clf = False):

    '''
        ele:min-grad: gradient threshold to produce binary map         
        ele:ffl-block: fill-flood threshold
        ele:min-ele-area: minimum area for selected elements 
        ele:merge-contained-ele: if True, merge elements contained in others
        text:max-word-inline-gap: words with smaller distance than the gap are counted as a line
        text:max-line-gap: lines with smaller distance than the gap are counted as a paragraph

        Tips:
        1. Larger *min-grad* produces fine-grained binary-map while prone to over-segment element to small pieces
        2. Smaller *min-ele-area* leaves tiny elements while prone to produce noises
        3. If not *merge-contained-ele*, the elements inside others will be recognized, while prone to produce noises
        4. The *max-word-inline-gap* and *max-line-gap* should be dependent on the input image size and resolution

        mobile: {'min-grad':4, 'ffl-block':5, 'min-ele-area':50, 'max-word-inline-gap':6, 'max-line-gap':1}
        web   : {'min-grad':3, 'ffl-block':5, 'min-ele-area':25, 'max-word-inline-gap':4, 'max-line-gap':4}
    '''

    # set input image path/{i}.png

    image_files = sorted(
                [f for f in os.listdir(folder) if f.lower().endswith(".png") and "masked_image" not in f.lower()]
                )
    for image_file in image_files:
        input_path_img=os.path.join(folder,image_file)

        key_params = adapt_params(input_path_img)

        org = cv2.imread(input_path_img)
        height, width = org.shape[:2]
        resized_height = resize_height_by_longest_edge(input_path_img, resize_length=800)
        # color_tips()



        if is_ocr:
            import UIED.detect_text.text_detection as text

            os.makedirs(pjoin(output_root, 'ocr'), exist_ok=True)
            text.text_detection(input_path_img, output_root, method='google')

        if is_ip:
            import UIED.detect_compo.ip_region_proposal as ip

            os.makedirs(pjoin(output_root, 'ip'), exist_ok=True)
            # switch of the classification func
            classifier = None
            if is_clf:
                classifier = {}
                from cnn.CNN import CNN

                # classifier['Image'] = CNN('Image')
                classifier['Elements'] = CNN('Elements')
                # classifier['Noise'] = CNN('Noise')
            ip.compo_detection(input_path_img, output_root, key_params,
                               classifier=classifier, resize_by_height=resized_height, show=False)

        if is_merge:
            import UIED.detect_merge.merge as merge

            os.makedirs(pjoin(output_root, 'merge'), exist_ok=True)
            # 使用 os.path.basename 获取文件名，去掉扩展名
            name = os.path.splitext(os.path.basename(input_path_img))[0]
            print(name)
            compo_path = pjoin(output_root, 'ip', str(name) + '.json')
            ocr_path = pjoin(output_root, 'ocr', str(name) + '.json')
            merge.merge(input_path_img, compo_path, ocr_path, pjoin(output_root, 'merge'), show=False)

if __name__ == '__main__':
    folder = r'D:/py_code/fyp/VueGen/output_multi/1/2/2_cropped'
    output_root = f'./data/output/1/1'
    is_ip = True
    is_clf = False
    is_ocr = True
    is_merge = True
    uied(folder, output_root, is_ip = True, is_ocr = True, is_merge = True, is_clf = False)
    # multipage gt
    # for i in range(1,2):
    #
    #     folder = f"D:/py_code/fyp/VueGen/multipage_data/{i}"
    #     # --- 找出该目录下所有 PNG 文件 ---
    #     image_files = sorted(
    #         [f for f in os.listdir(folder) if f.lower().endswith(".png")],
    #         key=lambda x: int(os.path.splitext(x)[0])
    #     )
    #
    #     print(f"\n=== Processing folder {i}, total {len(image_files)} images ===")
    #
    #     for file_idx, img_name in enumerate(image_files,start=1):
    #         if file_idx!=6:
    #             continue
    #         input_path_img = pjoin(folder, img_name)  # 完整路径
    #         print(f"Processing → {input_path_img}")
    #
    #         output_root = f"data/output/{i}"
    #         os.makedirs(output_root, exist_ok=True)
    #         key_params = adapt_params(input_path_img)
    #         org = cv2.imread(input_path_img)
    #         height, width = org.shape[:2]
    #         resized_height = resize_height_by_longest_edge(input_path_img, resize_length=800)
    #
    #         is_ip = True
    #         is_clf = False
    #         is_ocr = True
    #         is_merge = True
    #
    #         if is_ocr:
    #             import detect_text.text_detection as text
    #
    #             os.makedirs(pjoin(output_root, 'ocr'), exist_ok=True)
    #             text.text_detection(input_path_img, output_root, method='google')
    #
    #         if is_ip:
    #             import detect_compo.ip_region_proposal as ip
    #
    #             os.makedirs(pjoin(output_root, 'ip'), exist_ok=True)
    #
    #             classifier = None
    #             if is_clf:
    #                 classifier = {}
    #                 from cnn.CNN import CNN
    #
    #                 classifier['Elements'] = CNN('Elements')
    #
    #             ip.compo_detection(
    #                 input_path_img, output_root, key_params,
    #                 classifier=classifier, resize_by_height=resized_height, show=False
    #             )
    #
    #         if is_merge:
    #             import detect_merge.merge as merge
    #
    #             os.makedirs(pjoin(output_root, 'merge'), exist_ok=True)
    #
    #             name = os.path.splitext(img_name)[0]  # 例如 "1"
    #
    #             compo_path = pjoin(output_root, "ip", f"{i}_{file_idx}.json")
    #             ocr_path = pjoin(output_root, "ocr", f"{i}_{file_idx}.json")
    #
    #             merge.merge(
    #                 input_path_img, compo_path, ocr_path,
    #                 pjoin(output_root, 'merge'),
    #                 is_remove_bar=key_params['remove-bar'],
    #                 is_paragraph=key_params['merge-line-to-paragraph'],
    #                 show=False
    #             )


    '''
    for i in range(98,119):
        # set input image path/{i}.png
        input_path_img = f'D:/py_code/fyp/VueGen/multipage_data/{i}'
        output_root = f'data/output'
        key_params = adapt_params(input_path_img)

        org = cv2.imread(input_path_img)
        height, width = org.shape[:2]
        resized_height = resize_height_by_longest_edge(input_path_img, resize_length=800)
        #color_tips()

        is_ip = True
        is_clf = False
        is_ocr = True
        is_merge = True

        if is_ocr:
            import detect_text.text_detection as text
            os.makedirs(pjoin(output_root, 'ocr'), exist_ok=True)
            text.text_detection(input_path_img, output_root, method='google')

        if is_ip:
            import detect_compo.ip_region_proposal as ip
            os.makedirs(pjoin(output_root, 'ip'), exist_ok=True)
            # switch of the classification func
            classifier = None
            if is_clf:
                classifier = {}
                from cnn.CNN import CNN
                # classifier['Image'] = CNN('Image')
                classifier['Elements'] = CNN('Elements')
                # classifier['Noise'] = CNN('Noise')
            ip.compo_detection(input_path_img, output_root, key_params,
                               classifier=classifier, resize_by_height=resized_height, show=False)

        if is_merge:
            import detect_merge.merge as merge
            os.makedirs(pjoin(output_root, 'merge'), exist_ok=True)
            name = input_path_img.split('/')[-1][:-4]
            compo_path = pjoin(output_root, 'ip', str(name) + '.json')
            ocr_path = pjoin(output_root, 'ocr', str(name) + '.json')
            merge.merge(input_path_img, compo_path, ocr_path, pjoin(output_root, 'merge'),
                        is_remove_bar=key_params['remove-bar'], is_paragraph=key_params['merge-line-to-paragraph'], show=False)
    '''