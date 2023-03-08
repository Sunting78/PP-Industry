
import os
import math

import cv2
import numpy as np
import paddle

from paddleseg import utils
from paddleseg.core import infer
from paddleseg.cvlibs import Config
from paddleseg.utils import progbar
from ppindustry.utils.logger import setup_logger
logger = setup_logger('Predictor')
from paddleseg.transforms import Compose
from paddleseg.utils import progbar, visualize
from paddleseg.core.predict import partition_list, preprocess, mkdir

class SegPredictor(object):
    def __init__(self, seg_config, seg_model):
        self.seg_config = seg_config
        self.model = seg_config.model
        self.model_path = seg_model
        self.visualize = False
        self.transforms = Compose(seg_config.val_transforms)
        utils.utils.load_entire_model(self.model, self.model_path)
        self.model.eval()

    def predict(self,
                image_list,
                image_dir=None,
                save_dir='output',
                aug_pred=False,
                scales=1.0,
                flip_horizontal=True,
                flip_vertical=False,
                is_slide=False,
                stride=None,
                crop_size=None,
                custom_color=None):

        results = []
        nranks = paddle.distributed.get_world_size()
        local_rank = paddle.distributed.get_rank()
        if nranks > 1:
            img_lists = partition_list(image_list, nranks)
        else:
            img_lists = [image_list]

        added_saved_dir = os.path.join(save_dir, 'added_prediction')
        pred_saved_dir = os.path.join(save_dir, 'pseudo_color_prediction')

        logger.info("Start to predict...")
        progbar_pred = progbar.Progbar(target=len(img_lists[0]), verbose=1)
        color_map = visualize.get_color_map_list(256, custom_color=custom_color)
        with paddle.no_grad():
            for i, im_path in enumerate(img_lists[local_rank]):
                data = preprocess(im_path, self.transforms)
                import pdb;pdb.set_trace()
                if aug_pred:
                    pred, _ = infer.aug_inference(
                        self.model,
                        data['img'],
                        trans_info=data['trans_info'],
                        scales=scales,
                        flip_horizontal=flip_horizontal,
                        flip_vertical=flip_vertical,
                        is_slide=is_slide,
                        stride=stride,
                        crop_size=crop_size)
                else:
                    pred, _ = infer.inference(
                        self.model,
                        data['img'],
                        trans_info=data['trans_info'],
                        is_slide=is_slide,
                        stride=stride,
                        crop_size=crop_size)
                pred = paddle.squeeze(pred)
                pred = pred.numpy().astype('uint8')
                results.append({'mask': pred, 
                                'img_path': im_path})


                # save added image
                if self.visualize:
                    # get the saved name
                    if image_dir is not None:
                        im_file = im_path.replace(image_dir, '')
                    else:
                        im_file = os.path.basename(im_path)
                    if im_file[0] == '/' or im_file[0] == '\\':
                        im_file = im_file[1:]

                    added_image = utils.visualize.visualize(
                        im_path, pred, color_map, weight=0.6)
                    added_image_path = os.path.join(added_saved_dir, im_file)
                    mkdir(added_image_path)
                    cv2.imwrite(added_image_path, added_image)

                    # save pseudo color prediction
                    pred_mask = utils.visualize.get_pseudo_color_map(pred, color_map)
                    pred_saved_path = os.path.join(
                        pred_saved_dir, os.path.splitext(im_file)[0] + ".png")
                    mkdir(pred_saved_path)
                    pred_mask.save(pred_saved_path)

                progbar_pred.update(i + 1)
        return results