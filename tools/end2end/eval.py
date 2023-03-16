import argparse
import json
import os
import os.path as osp

import cv2
import numpy as np
import sys

parent_path = os.path.abspath(os.path.join(__file__, *(['..'] * 3)))
sys.path.insert(0, parent_path)
import prettytable as pt
import pycocotools.mask as mask_util

from ppindustry.cvlib.configs import ConfigParser
from ppindustry.cvlib.framework import Builder
from tools.convert_tools.convert_coco_to_RoI_mask import read_json, group_images_annotations
from ppindustry.utils.logger import setup_logger

logger = setup_logger('Eval')

def get_args():
    parser = argparse.ArgumentParser(
        description='Json Format convert to RoI binary Mask ')

    # Parameters
    parser.add_argument(
        '--input_path',
        type=str,
        help='json path or txt path for evaluation, both of them must have isNG information',
        required=True)
    parser.add_argument(
        '--image_root',
        type=str,
        default='',
        help='image root path ')
    parser.add_argument(
        '--pred_path',
        type=str,
        help='the prediction results, json format same as the output json of predict.py',
        required=True)
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='config path, you can set conf_file by yourself')
    parser.add_argument(
        '--rules_eval',
        action='store_true',
        help='update rules params to eval')
    parser.add_argument(
        '--instance_eval',
        action='store_true',
        help='instance-wise evaluation, only support in the json format input with instance information')
    parser.add_argument(
        '--output_path',
        type=str,
        default='./output/',
        help='save path to save images and mask, default None, do not save'
    )
    args = parser.parse_args()

    return args


def evaluation(gt_data, preds_data, post_modules=None, image_root='', instance_level=None):
    
    ng_in_ok_num, ok_in_ng_num, ng_gt, ok_gt  = 0, 0, 0, 0
    import pdb;pdb.set_trace()
    for img_path, img_preds in preds_data.items():
        img_gt_anno =  gt_data[img_path] 
        if len(img_gt_anno) == 0: 
            ok_gt+=1
            if img_preds['isNG']:
                ng_in_ok_num+=1
        else:  
            ng_gt+=1
            if not img_preds['isNG']:
                ok_in_ng_num+=1
    
    column_names = ["Eval", "OK", "NG"]
    table = pt.PrettyTable(column_names)
    table.add_row([
        "pred OK", ok_gt - ng_in_ok_num, ok_in_ng_num
    ])
    table.add_row(["pred NG", ng_in_ok_num, ng_gt - ok_in_ng_num])

    table.add_row([
        "Overkill", "{:.2f}%".format((ng_in_ok_num / ok_gt) * 100) if ok_gt > 0 else 0,
       " "
    ])
    table.add_row([
        "Escape", "{:.2f}%".format((ok_in_ng_num / ng_gt) * 100) if ng_gt > 0 else 0,
       " "
    ])
    print(table)






    




if __name__ == '__main__':
    args = get_args()
    data = read_json(args.input_path)
    img_to_gt_annos = group_images_annotations(data)
    img_to_pred_annos = read_json(args.pred_path)
    import pdb;pdb.set_trace()
    if args.rules_eval:
        config = ConfigParser(args)
        postprocess = config.parse()[0][-1]
        post_modules = Builder(postprocess['PostProcess'])
    
    evaluation(img_to_gt_annos, img_to_pred_annos, post_modules, args.image_root, args.instance_eval)



        





