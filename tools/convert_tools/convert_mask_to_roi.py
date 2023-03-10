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

import numpy as np
import cv2

def _mkdir_p(path):
    """Make the path exists"""
    if not osp.exists(path):
        os.makedirs(path)
    
def square(bbox, size):
    """Convert a `Boxes` into a square tensor."""
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


def pad(bbox, img_size, pad_scale=0.5):
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


def adjust_bbox(bbox, img_shape, pad_scale=0.5):
    """
    adjust box according to img_shape and pad_scale 
    """
    bbox = square(bbox, img_shape)
    bbox = pad(bbox, img_shape, pad_scale)
    return bbox


def get_args():
    parser = argparse.ArgumentParser(
        description='Mask Format convert to Json for detection ')

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
        '--to_binary',
        action='store_true',
        help='conver to binary masks'
    )
    parser.add_argument(
        '--suffix',
        type=str,
        default='',
        help='the gt suffix to img'
    )
    parser.add_argument(
        '--pad_scale',
        type=float,
        default=0.5,
        help='the pad scale of crop img'
    )
    parser.add_argument(
        '--output_path',
        type=str,
        default='./output/',
        help='save path to save RoI, Mask and txt list, output_path/images/, output_path/anno/ and output_path/RoI.txt'
    )

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = get_args()
    image_path = args.image_path
    anno_path = args.anno_path
    class_num = args.class_num
    suffix = args.suffix
    output_path = args.output_path
    pad_scale = args.pad_scale
    to_binary = args.to_binary

    
    output_image_path = osp.join(output_path, 'images')
    _mkdir_p(output_image_path)
    output_anno_path = osp.join(output_path, 'anno')
    _mkdir_p(output_anno_path)


    classid_list = list(range(1, class_num + 1)) 
    file_list = osp.join(output_path, 'RoI.txt')
    f = open(file_list, "w")

    for img_name in os.listdir(image_path):
        file_name = osp.join(image_path, img_name)
        img = cv2.imread(file_name)

        base_name = img_name.split('.')[0]
        anno_name = osp.join(anno_path, base_name + suffix + '.png')
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
                    continue
                bbox = adjust_bbox([int(stat[0]), int(stat[1]), int(stat[0]+stat[2]), int(stat[1]+stat[3])], img.shape[:2], pad_scale=pad_scale)
                img_crop = img[bbox[1]:bbox[3], bbox[0]:bbox[2], :]
                mask_crop = mask[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                if to_binary:
                    mask_crop[mask_crop>0] = 1
                # save crop img, mask and write to txt file
                img_save_path = osp.join(output_image_path, f'{base_name}_{cls_id}_{i}.png')
                anno_save_path = osp.join(output_anno_path, f'{base_name}_{cls_id}_{i}{suffix}.png')
                cv2.imwrite(img_save_path, img_crop)
                cv2.imwrite(anno_save_path, mask_crop)

                line = img_save_path + " " + anno_save_path + '\n'
                f.write(line)
    f.close()
