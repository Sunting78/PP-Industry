import random
import math

import cv2
import numpy as np
from PIL import Image
from ppdet.core.workspace import serializable
from ppdet.data.transform.operators import BaseOperator, register_op
from ppdet.core.workspace import register

__all__ = [
    'RandomCopyPaste'
]

@register_op
class RandomCopyPaste(BaseOperator):
    """
    Rotate an image randomly with padding.

    Args:
        max_rotation (float, optional): The maximum rotation degree. Default: 15.
        im_padding_value (float, optional): The padding value of raw image. Default: 127.5.
        label_padding_value (int, optional): The padding value of annotation image. Default: 255.
    """

    def __init__(self,
                 choice_classes='all',
                 max_mem_size=20,
                 max_paste_number_per_img=3,
                 copy_prob=0.5,
                 copy_shape='rectangle', # or polygon
                 ):

        self.choice_classes = choice_classes
        self.copy_prob = copy_prob
        self.max_mem_size = max_mem_size
        self.max_paste_number_per_img = max_paste_number_per_img
        self.copy_shape = copy_shape
        self.memory_bank = []

    def update_mem(self, data):
        pass
        
    def copy_and_paste(self,data):
        return data

    def __call__(self, data):
        data_new = data.deep_copy()

        if np.random.random() <= self.copy_prob and len(self.memory_bank)>0:
            data = self.copy_and_paste(data)

        self.update_mem(data)

        return data

