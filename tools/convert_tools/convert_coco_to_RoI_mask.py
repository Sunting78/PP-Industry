# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import json
import os
import os.path as osp

import cv2
import numpy as np
import pycocotools.mask as mask_util


def _mkdir_p(path):
    """Make the path exists"""
    if not osp.exists(path):
        os.makedirs(path)

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
    return int(x1), int(y1), int(x2), int(y2)


def adjust_bbox(bbox, img_shape, pad_scale=0.0):
    bbox = square(bbox, img_shape)
    bbox = pad(bbox, img_shape, pad_scale)
    return bbox

def polygons_to_bitmask(polygons, height, width):
    if len(polygons) == 0:
        # COCOAPI does not support empty polygons
        return np.zeros((height, width)).astype(int)

    rles = mask_util.frPyObjects(polygons, height, width)
    rle = mask_util.merge(rles)

    return mask_util.decode(rle).astype(int)

def generate_mask_RoI(img_to_anno, image_root, output_path, suffix, pad_scale=0.5):
    output_image_path = osp.join(output_path, 'images')
    _mkdir_p(output_image_path)
    output_anno_path = osp.join(output_path, 'anno')
    _mkdir_p(output_anno_path)

    for img_path, annos in img_to_anno.items():
        img = cv2.imread(osp.join(image_root, img_path))
        base_name = os.path.basename(img_path).split('.')[0]
        polygons = []
        for anno in annos:
            polygons.extend(anno['segmentation'])
        mask = polygons_to_bitmask(polygons, img.shape[0], img.shape[1])

        for idx, anno in enumerate(annos):
            bbox = anno['bbox']
            bbox = adjust_bbox([int(bbox[0]), int(bbox[1]), int(bbox[0]+bbox[2]), int(bbox[1]+bbox[3])], img.shape[:2], pad_scale=pad_scale)
            crop_img = img[bbox[1]:bbox[3], bbox[0]:bbox[2], :]
            crop_mask = mask[bbox[1]:bbox[3], bbox[0]:bbox[2]]
            cv2.imwrite(osp.join(output_image_path, f'{base_name}_{idx}.png'), crop_img)
            cv2.imwrite(osp.join(output_anno_path, f'{base_name}_{idx}{suffix}.png'), crop_mask)

    print('task done!')

def read_json(json_path): 
    """
    read json from given path 
    """
    with open(json_path, "r") as f:
        data = json.load(f)
    return data

def group_images_annotations(data, class_id=None):
    id_to_img = {}
    for img in data["images"]:
        id_to_img[img["id"]] = img['file_name']

    img_to_annos = {}
    for anno in data["annotations"]:
        if anno.get('iscrowd', 1):
            continue
        if class_id is not None:
            if anno["category_id"] not in class_id:
                continue
        image_path = id_to_img[anno["image_id"]]
        if image_path not in img_to_annos.keys():
            img_to_annos[image_path] = [{"bbox": anno["bbox"], 
                                        "segmentation": anno["segmentation"],
                                         "category_id": anno["category_id"]}]
        else:
            img_to_annos[image_path].append({"bbox": anno["bbox"], 
                                            "segmentation": anno["segmentation"], 
                                             "category_id": anno["category_id"]})
    return img_to_annos



def get_args():
    parser = argparse.ArgumentParser(
        description='Json Format convert to RoI binary Mask ')

    # Parameters
    parser.add_argument(
        '--json_path',
        type=str,
        help='json path to statistic images information')
    parser.add_argument(
        '--image_path',
        type=str,
        default=None,
        help='images root path, default None if img path in json is absolute path'
    )
    parser.add_argument(
        '--seg_classid',
        type=int,
        nargs='+',
        default=None,
        help='classid to segment, default None means all classes'
    )
    parser.add_argument(
        '--pad_scale',
        type=float,
        default=0.5,
        help='the pad scale of crop img'
    )
    parser.add_argument(
        '--suffix',
        type=str,
        default='',
        help='the gt suffix of img when save them'
    )
    parser.add_argument(
        '--output_path',
        type=str,
        default='./output/',
        help='save path to save images and mask, default None, do not save'
    )
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = get_args()
    data = read_json(args.json_path)
    img_to_annos = group_images_annotations(data, args.seg_classid)
    generate_mask_RoI(img_to_annos, args.image_path, args.output_path, args.suffix, args.pad_scale)
