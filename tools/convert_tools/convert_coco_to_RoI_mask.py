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
import os
import os.path as osp
import sys
parent_path = os.path.abspath(os.path.join(__file__, *(['..'] * 3)))
sys.path.insert(0, parent_path)

import cv2
import numpy as np
from pycocotools.coco import COCO
import pycocotools.mask as mask_util

from ppindustry.utils.bbox_utils import adjust_bbox


def _mkdir_p(path):
    """Make the path exists"""
    if not osp.exists(path):
        os.makedirs(path)

def polygons_to_bitmask(polygons, height, width):
    if len(polygons) == 0:
        # COCOAPI does not support empty polygons
        return np.zeros((height, width)).astype(int)
    rles = mask_util.frPyObjects(polygons, height, width)
    rle = mask_util.merge(rles)
    return mask_util.decode(rle).astype(int)

def generate_mask_RoI(data, image_root, output_path, suffix, class_id=None, pad_scale=0.5):
    output_image_path = osp.join(output_path, 'images')
    _mkdir_p(output_image_path)
    output_anno_path = osp.join(output_path, 'anno')
    _mkdir_p(output_anno_path)

    img_ids = list(sorted(data.imgs.keys()))
    for img_id in img_ids:
        im_info = data.loadImgs(img_id)[0]
        ann_ids = data.getAnnIds(imgIds=img_id, catIds=class_id, iscrowd=False)
        annos = data.loadAnns(ann_ids)
        if len(annos) == 0:
            continue
        img = cv2.imread(osp.join(image_root, im_info['file_name']))
        base_name = os.path.basename(im_info['file_name']).split('.')[0]
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

def parse_args():
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
    args = parse_args()
    data = COCO(args.json_path)
    generate_mask_RoI(data, args.image_path, args.output_path, args.suffix, args.seg_classid, args.pad_scale)
