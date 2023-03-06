import random
import math
import copy

import cv2
import numpy as np
from PIL import Image
from shapely.geometry import Polygon

from paddleseg.cvlibs import manager
from paddleseg.transforms import functional
from paddleseg.utils import logger

@manager.TRANSFORMS.add_component
class CopyPaste:
    """
    Rotate an image randomly with padding.

    Args:
        max_rotation (float, optional): The maximum rotation degree. Default: 15.
        im_padding_value (float, optional): The padding value of raw image. Default: 127.5.
        label_padding_value (int, optional): The padding value of annotation image. Default: 255.
    """

    def __init__(self,
                 prob=0.5,
                 choice_classes='all',
                 max_mem_size=20,
                 max_paste_number_per_img=3,
                 copy_shape='rectangle', # or polygon
                 ):

        self.choice_classes = choice_classes
        self.copy_prob = prob
        self.max_mem_size = max_mem_size
        self.max_paste_number_per_img = max_paste_number_per_img
        self.copy_shape = copy_shape
        self.memory_bank = []

    def update_mem(self, data):
        pass

    def intersection(g, p):
        """
        Intersection.
        """

        g = g[:8].reshape((4, 2))
        p = p[:8].reshape((4, 2))

        a = g
        b = p

        use_filter = True
        if use_filter:
            # step1:
            inter_x1 = np.maximum(np.min(a[:, 0]), np.min(b[:, 0]))
            inter_x2 = np.minimum(np.max(a[:, 0]), np.max(b[:, 0]))
            inter_y1 = np.maximum(np.min(a[:, 1]), np.min(b[:, 1]))
            inter_y2 = np.minimum(np.max(a[:, 1]), np.max(b[:, 1]))
            if inter_x1 >= inter_x2 or inter_y1 >= inter_y2:
                return 0.
            x1 = np.minimum(np.min(a[:, 0]), np.min(b[:, 0]))
            x2 = np.maximum(np.max(a[:, 0]), np.max(b[:, 0]))
            y1 = np.minimum(np.min(a[:, 1]), np.min(b[:, 1]))
            y2 = np.maximum(np.max(a[:, 1]), np.max(b[:, 1]))
            if x1 >= x2 or y1 >= y2 or (x2 - x1) < 2 or (y2 - y1) < 2:
                return 0.

        g = Polygon(g)
        p = Polygon(p)
        if not g.is_valid or not p.is_valid:
            return 0

        inter = Polygon(g).intersection(Polygon(p)).area
        union = g.area + p.area - inter
        if union == 0:
            return 0
        else:
            return inter / union

    def copy_and_paste(self, data):

        return data

    def __call__(self, data):
        data_new = copy.deep_copy(data)

        if np.random.random() <= self.copy_prob and len(self.memory_bank) > 0:
            data = self.copy_and_paste(data)

        self.update_mem(data)

        return data


