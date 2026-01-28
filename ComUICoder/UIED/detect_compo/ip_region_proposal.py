import cv2
from os.path import join as pjoin
import time
import json
import numpy as np

import UIED.detect_compo.lib_ip.ip_preprocessing as pre
import UIED.detect_compo.lib_ip.ip_draw as draw
import UIED.detect_compo.lib_ip.ip_detection as det
import UIED.detect_compo.lib_ip.file_utils as file
import UIED.detect_compo.lib_ip.Component as Compo
from UIED.config.CONFIG_UIED import Config
C = Config()


def nesting_inspection(org, grey, compos, ffl_block):
    '''
    Inspect all big compos through block division by flood-fill
    :param ffl_block: gradient threshold for flood-fill
    :return: nesting compos
    '''
    nesting_compos = []
    for i, compo in enumerate(compos):
        if compo.height > 50:
            replace = False
            clip_grey = compo.compo_clipping(grey)
            n_compos = det.nested_components_detection(clip_grey, org, grad_thresh=ffl_block, show=False)
            Compo.cvt_compos_relative_pos(n_compos, compo.bbox.col_min, compo.bbox.row_min)

            for n_compo in n_compos:
                if n_compo.redundant:
                    compos[i] = n_compo
                    replace = True
                    break
            if not replace:
                nesting_compos += n_compos
    return nesting_compos


def compo_detection(input_img_path, output_root, uied_params,
                    resize_by_height=800, classifier=None, show=False, wai_key=0):

    start = time.perf_counter()
    PAD = 30
    # Fix: Use os.path for proper cross-platform path handling
    import os
    name = os.path.splitext(os.path.basename(input_img_path))[0]
    ip_root = file.build_directory(pjoin(output_root, "ip"))

    # *** Step 1 *** pre-processing: read img -> get binary map
    org, grey = pre.read_img(input_img_path, resize_by_height, padding=PAD)
    binary = pre.binarization(org, grad_min=int(uied_params['min-grad']))

    # *** Step 2 *** element detection
    det.rm_line(binary, show=show, wait_key=wai_key)
    uicompos = det.component_detection(binary, min_obj_area=int(uied_params['min-ele-area']))

    # *** Step 3 *** results refinement
    uicompos = det.compo_filter(uicompos, min_area=int(uied_params['min-ele-area']), img_shape=binary.shape)
    uicompos = det.merge_intersected_compos(uicompos)
    det.compo_block_recognition(binary, uicompos)
    if uied_params['merge-contained-ele']:
        uicompos = det.rm_contained_compos_not_in_block(uicompos)
    Compo.compos_update(uicompos, org.shape)
    Compo.compos_containment(uicompos)

    # *** Step 4 ** nesting inspection: check if big compos have nesting element
    uicompos += nesting_inspection(org, grey, uicompos, ffl_block=uied_params['ffl-block'])
    Compo.compos_update(uicompos, org.shape)
    # draw.draw_bounding_box(org, uicompos, show=show, name='merged compo', write_path=pjoin(ip_root, name + '_org.jpg'),
    #                        wait_key=wai_key)

    # *** Step 7 *** save detection result
    img_org = cv2.imread(input_img_path)  # 原图
    h_org, w_org = img_org.shape[:2]
    h_resized, w_resized = org.shape[:2]

    scale_x = w_org / w_resized
    scale_y = h_org / h_resized
    # 把检测到的每个组件映射回原图尺寸
    for compo in uicompos:
        compo.bbox.col_min = int(compo.bbox.col_min) - PAD
        compo.bbox.col_max = int(compo.bbox.col_max) - PAD
        compo.bbox.row_min = int(compo.bbox.row_min) - PAD
        compo.bbox.row_max = int(compo.bbox.row_max) - PAD
        compo.bbox.width = compo.bbox.col_max - compo.bbox.col_min
        compo.bbox.height = compo.bbox.row_max - compo.bbox.row_min

        compo.bbox.box_area = compo.bbox.width * compo.bbox.height
        compo.width = compo.bbox.width
        compo.height = compo.bbox.height
        compo.area = compo.width * compo.height
        compo.bbox_area = compo.bbox.box_area
        compo.image_shape = img_org.shape


    #Compo.compos_update(uicompos, img_org.shape)
    file.save_corners_json(pjoin(ip_root, name + '.json'), uicompos)
    draw.draw_bounding_box(img_org, uicompos, show=show, name='merged compo', write_path=pjoin(ip_root, name + '.jpg'),
                           wait_key=wai_key)

    print("[Compo Detection Completed in %.3f s] Input: %s Output: %s" % (time.perf_counter() - start, input_img_path, pjoin(ip_root, name + '.json')))
