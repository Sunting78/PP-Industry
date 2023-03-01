import os
import json
import argparse

import numpy as np
import pandas as pd
import seaborn as sns
import cv2
import matplotlib.pyplot as plt


def check_dir(check_path, show=True):
    if os.path.isdir(check_path):
        check_directory = check_path
    else:
        check_directory = os.path.dirname(check_path)
    if len(check_directory) > 0 and not os.path.exists(check_directory):
        os.makedirs(check_directory)
        if show:
            print('make dir:', check_directory)

def square(bbox, size):
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1 + 1, y2 - y1 + 1
    if w < h:
        pad = (h - w) // 2
        x1 = max(0, x1 - pad)
        x2 = min(size[1], x2 + pad)
    else:
        pad = (w - h) // 2
        y1 = max(0, y1 - pad)
        y2 = min(size[0], y2 + pad)
    return x1, y1, x2, y2


def pad(bbox, img_size, pad_scale=0.0):
    """pad bbox with scale
    Args:
        bbox (list):[x1, y1, x2, y2]
        img_size (tuple): (height, width)
        pad_scale (float): scale for padding
    Return:
        bbox (list)
    """
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1 + 1, y2 - y1 + 1
    dw = int(w * pad_scale)
    dh = int(h * pad_scale)
    x1 = max(0, x1 - dw)
    x2 = min(img_size[1], x2 + dw)
    y1 = max(0, y1 - dh)
    y2 = min(img_size[0], y2 + dh)
    return x1, y1, x2, y2


def adjust_bbox(bbox, img_shape, pad_scale=0.0):
    bbox = square(bbox, img_shape)
    bbox = pad(bbox, img_shape, pad_scale)
    return bbox

def generate_mask_RoI(img_to_anno):
    RoI_bboxes = []
    for img_path, annos in img_to_anno.items():
        img = cv2.imread(img_path)

        for anno in annos:
            bbox = anno['bbox']
            bbox = adjust_bbox(bbox)






def read_json(json_path): # read json
    with open(json_path, "r") as f:
        data = json.load(f)
    return data

def group_images_annotations(data, output_path, class_id=None):
    id_to_img = {}
    for img in data["images"]:
        id_to_img[img["id"]] = img['image_path']

    img_to_anno = {}
    for anno in data["annotations"]:
        image_path = id_to_img[anno["image_id"]]
        if class_id is not None:
            class_name = anno["category_id"]
            if class_name not in class_id:
                continue
        
        if image_path  not in img_to_anno.keys():
            img_to_anno[image_path] = [{"bbox": anno["bbox"], 
                                        "segm": anno["segmentation"],
                                         "category": anno["category_id"]}]
        else:
            img_to_anno[image_path].append({"bbox": anno["bbox"], 
                                            "segm": anno["segmentation"], 
                                            "category": anno["category_id"]})
    return img_to_anno



def get_args():
    parser = argparse.ArgumentParser(
        description='Json Format convert to RoI Mask ')

    # Parameters
    parser.add_argument(
        '--json_path',
        type=str,
        help='json path to statistic images information')
    parser.add_argument(
        '--image_path',
        type=str,
        default=None,
        help='images path, default None, do not save'
    )
    parser.add_argument(
        '--seg_classid',
        type=str,
        default=None,
        help='images path, default None, do not save'
    )
    parser.add_argument(
        '--output_path',
        type=str,
        default=None,
        help='save path to save images and mask, default None, do not save'
    )

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = get_args()
    data = read_json(args.json_path)
    img_to_annos = group_images_annotations(data, args.seg_classid)


