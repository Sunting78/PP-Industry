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


def get_args():
    """
    Parses input arguments. 
    """
    parser = argparse.ArgumentParser(
        description='Mask Format convert to Json for detection')
    # Parameters
    parser.add_argument(
        '--image_path',
        type=str,
        required=True,
        help='image path to provide images information')
    parser.add_argument(
        '--anno_path',
        type=str,
        required=True,
        help='mask ground truth path'
    )
    parser.add_argument(
        '--class_num',
        type=int,
        required=True,
        help='number of classes'
    )
    parser.add_argument(
        '--suffix',
        type=str,
        default='.png',
        help='the gt suffix to img'
    )
    parser.add_argument(
        '--output_name',
        type=str,
        default='coco.json',
        help='output path to save coco format json'
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    image_path = args.image_path
    anno_path = args.anno_path
    output_name = args.output_name

    classid_list = list(range(1, args.class_num + 1))
    images = []
    annotations = []
    image_id = 0
    anno_id = 0
    
    for img_name in os.listdir(image_path):
        image_id += 1
        file_name = osp.join(image_path, img_name)
        img = cv2.imread(file_name)
        image_info = {'file_name': file_name,
                    'id': image_id,
                    'width': img.shape[1],
                    'height': img.shape[0],
        }
        images.append(image_info)
        basename = osp.splitext(img_name)[0]
        anno_name = osp.join(anno_path, basename + args.suffix )
        mask = cv2.imread(anno_name, -1)
        if mask.sum() == 0: 
            continue
        
        for cls_id in classid_list:  
            class_map = np.equal(mask, cls_id).astype(np.uint8)
            if class_map.sum()==0:
                continue   
            ret, labels, stats, centroid = cv2.connectedComponentsWithStats(class_map)
            for i, stat in enumerate(stats):
                if i == 0:
                    continue # skip background
                anno = {}
                polygon = []
                contours, _ = cv2.findContours((labels == i).astype(np.uint8), 
                                                cv2.RETR_LIST,
                                                cv2.CHAIN_APPROX_SIMPLE)
                polygon.append(contours[0].flatten().astype(np.uint8).tolist())
                anno_id += 1
                anno = {
                    'id': anno_id,
                    'image_id': image_id,
                    'category_id': cls_id,
                    "segmentation": polygon,
                    "area": int(stat[2])*int(stat[3]), 
                    "bbox": [int(stat[0]), int(stat[1]), int(stat[2]), int(stat[3])],
                    "iscrowd": 0,     
                }
                annotations.append(anno)
                
    categories = [{'supercategory': 'defect',
                    'id': cls_id, 
                    'name': str(cls_id)}  for cls_id in classid_list]

    json_data = {'images': images, 'annotations': annotations, 'categories': categories}
    with open(output_name, "w") as f:
        json.dump(json_data, f, indent=2)
