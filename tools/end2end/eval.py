import argparse
import json
import os
import os.path as osp
from collections import defaultdict

import cv2
import numpy as np
import sys
parent_path = os.path.abspath(os.path.join(__file__, *(['..'] * 3)))
sys.path.insert(0, parent_path)

import prettytable as pt
from pycocotools.coco import COCO

from ppindustry.cvlib.configs import ConfigParser
from ppindustry.ops.postprocess import PostProcess
from ppindustry.utils.logger import setup_logger
from ppindustry.utils.data_dict import post_process_image_info
from ppindustry.utils.bbox_utils import iou_one_to_multiple 

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
        '--instance_level',
        action='store_false',
        help='instance-wise evaluation, only support in the json format input with instance information')
    parser.add_argument(
        '--iou_theshold',
        type=float,
        default=0.1,
        help='iou threshold, only support in the json format input with instance information')
    parser.add_argument(
        '--badcase',
        action='store_false',
        help='show badcase')
    parser.add_argument(
        '--output_path',
        type=str,
        default='./output/',
        help='save path to save images and mask, default None, do not save'
    )
    args = parser.parse_args()

    return args

def read_json(json_path): 
    """
    read json from given path 
    """
    with open(json_path, "r") as f:
        data = json.load(f)
    return data

def set_class_to_set(data_dict):
    for cls in data_dict.keys():
        data_dict[cls] = set(data_dict[cls])
    return data_dict

def eval_ng(gt_data, gt_ng_ids, preds_data, image_root='', instance_level=True, iou_theshold=0.1):
    ok_image_in_ng_num = 0
    class_list = GT_data.getCatIds()
    clsid_to_name =  GT_data.cats
    class_name = [clsid_to_name[i]['name'] for i in class_list]
    ok_instance_in_ng_num = 0
    ng_instance_num = 0
    ng_class_num = defaultdict(int)
    ok_in_ng_class_num = defaultdict(int)
    escape_info_image = []
    escape_info_image_instance = defaultdict(list)
    for ng_id in gt_ng_ids:
        im_name = gt_data.loadImgs(ng_id)[0]['file_name']
        preds_info = preds_data[osp.join(image_root, im_name[2:])]
        if not preds_info['isNG']:
            ok_image_in_ng_num += 1
            escape_info_image.append(ng_id)
        if instance_level:
            gt_anno_id = gt_data.getAnnIds(imgIds=ng_id, iscrowd=False)
            gt_annos = gt_data.loadAnns(gt_anno_id)
            if len(gt_annos) == 0:
                continue
            ng_instance_num += len(gt_annos)
            if not preds_info['isNG']:
                ok_instance_in_ng_num += len(gt_annos)
                for gt in gt_annos:
                    ok_in_ng_class_num[gt['category_id']] += 1
                    ng_class_num[gt['category_id']] += 1
                    escape_info_image_instance[gt['category_id']].append((ng_id, gt['id']))
            else:
                pred_bbox = []
                for pred in preds_info['pred']:
                    if pred['isNG']:
                        pred_bbox.append(pred['bbox'])       

                for gt in gt_annos:
                    ng_class_num[gt['category_id']] += 1
                    max_iou = max(iou_one_to_multiple(gt['bbox'], pred_bbox))
                    if max_iou < iou_theshold:
                        ok_instance_in_ng_num += 1
                        ok_in_ng_class_num[gt['category_id']] += 1
                        escape_info_image_instance[gt['category_id']].append((ng_id, gt['id']))

    ng_num = len(gt_ng_ids)
    column_names = ["Image Level", "Total", "NG", "OK", "Escape"]
    table = pt.PrettyTable(column_names)
    table.add_row([
        "Lucky Result",
        ng_num, ng_num - ok_image_in_ng_num, ok_image_in_ng_num, "{:.2f}%".format(ok_image_in_ng_num/ng_num* 100) if ng_num > 0 else 0,
    ])
    logger.info("Result of Image-Level NG Evaluation")
    print(table)

    if instance_level:        
        ng_class_num = [ng_class_num[cat] for cat in class_list]
        ok_in_ng_class_num = [ok_in_ng_class_num[cat] for cat in class_list]
        column_names = ["NG", "ALL", *class_name]
        table = pt.PrettyTable(column_names)
        table.add_row([
            "Total",
            ng_instance_num,
            *[x for x in ng_class_num]
        ])
        table.add_row([
            "NG",
            ng_instance_num - ok_instance_in_ng_num,
            *[ng_class_num[i] - ok_in_ng_class_num[i] for i in range(len(ng_class_num))]
        ])
        table.add_row([
            "OK",
            ok_instance_in_ng_num,
            *[x for x in ok_in_ng_class_num]
        ])
        table.add_row([
            " Escape ",
            "{:.2f}%".format(ok_instance_in_ng_num/ng_instance_num* 100) if ng_instance_num> 0 else 0,
            *["{:.2f}%".format(ok_in_ng_class_num[i]/ng_class_num[i]* 100) for i in range(len(ng_class_num))]
        ])
        logger.info("Result of Instance-Level NG Evaluation")
        print(table)

    return escape_info_image, escape_info_image_instance

def eval_ok(gt_data, gt_ok_ids, preds_data, image_root=''):
    ng_in_ok_num = 0
    ng_info = defaultdict(list)
    class_list = GT_data.getCatIds()
    clsid_to_name =  GT_data.cats
    class_name = [clsid_to_name[i]['name'] for i in class_list]

    for ok_id in gt_ok_ids:
        im_name = gt_data.loadImgs(ok_id)[0]['file_name']
        preds_info = preds_data[osp.join(image_root, im_name[2:])]
        if preds_info['isNG']:
            ng_in_ok_num += 1
            for pred in preds_info['pred']:
                if pred['isNG']:
                    ng_info[pred['category_id']].append(ok_id)

    ng_info = set_class_to_set(ng_info)
    ng_class_nums = [len(ng_info[cat]) for cat in class_list]
    ok_num = max(len(gt_ok_ids), 1)

    column_names = ["OK", "ALL", *class_name]
    table = pt.PrettyTable(column_names)
    table.add_row([
        "Total",
        len(gt_ok_ids), *([len(gt_ok_ids)] * len(class_name))
    ])
    table.add_row([
        "OK",
        len(gt_ok_ids) - ng_in_ok_num,
        *[len(gt_ok_ids) - x for x in ng_class_nums]
    ])
    table.add_row(["NG", ng_in_ok_num, *ng_class_nums])
    table.add_row([
        "Overkill", "{:.2f}%".format((ng_in_ok_num / ok_num) * 100),
        *["{:.2f}%".format(x / ok_num * 100) for x in ng_class_nums]
    ])
    logger.info("Result of OK Evaluation")
    print(table)

    return ng_info


def evaluation(gt_data, preds_data, post_modules=None, image_root='', instance_level=True, iou_theshold=0.1):
    if post_modules:
        for img_path, img_info in preds_data.items():
            preds = img_info['pred']
            img_info.pop('isNG')
            for pred in preds:
                pred.pop('isNG')
        preds_data = post_modules(preds_data)
        post_process_image_info(preds_data)

    img_ids = list(sorted(gt_data.imgs.keys()))
    gt_ok_ids = []
    gt_ng_ids = []
    for img_id in img_ids:
        im_info = gt_data.loadImgs(img_id)[0]
        ann_ids = gt_data.getAnnIds(imgIds=img_id, iscrowd=False)
        annos = gt_data.loadAnns(ann_ids)
        if len(annos) == 0:
            gt_ok_ids.append(img_id)
        else:
            gt_ng_ids.append(img_id)

    overkill_info = eval_ok(gt_data, gt_ok_ids, preds_data, image_root) # overkill_info = {class_id: set(img_id)}
    escape_info_image, escape_info_image_instance = eval_ng(gt_data, gt_ng_ids, preds_data, image_root, 
                                                            instance_level=instance_level,
                                                            iou_theshold=iou_theshold)
    
    return overkill_info, escape_info_image, escape_info_image_instance

def show_badcase(overkill_info, escape_info_image, escape_info_image_instance, output_dir):
    overkill_path = osp.join(output_dir, 'overkill')
    escape_path = osp.join(output_dir, 'escape')
    import pdb;pdb.set_trace()




if __name__ == '__main__':
    args = get_args()
    GT_data = COCO(args.input_path)
    img_to_pred_annos = read_json(args.pred_path)
    post_modules = None
    if args.rules_eval:
        config = ConfigParser(args)
        postprocess = config.parse()[0][-1]
        post_modules = PostProcess(postprocess['PostProcess'])
    
    overkill_info, escape_info_image, escape_info_image_instance = evaluation(GT_data, img_to_pred_annos, post_modules, 
                                                                        args.image_root, args.instance_level, args.iou_theshold)
    if args.badcase:
        show_badcase(GT_data, img_to_pred_annos, overkill_info, 
                     escape_info_image, escape_info_image_instance,
                     args.output_path)




        
        





        





