from functools import reduce
import os
import numpy as np
import math
import paddle

import importlib

from .detection import Detection
from .postprocess import PostProcess
from .segmentation import BaseSegmentation, CropSegmentation

__all__ = ['Detection','PostProcess','BaseSegmentation', 'CropSegmentation']