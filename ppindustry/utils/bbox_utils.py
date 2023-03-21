import numpy as np

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
    """
    adjust box according to img_shape and pad_scale 
    """
    bbox = square(bbox, img_shape)
    bbox = pad(bbox, img_shape, pad_scale)
    return bbox

def iou_one_to_multiple(one_box, key_boxes):
    key_boxes = np.array(key_boxes, dtype=np.float32).reshape([-1, 4])
    ixmin = np.maximum(key_boxes[:, 0], one_box[0])
    iymin = np.maximum(key_boxes[:, 1], one_box[1])
    ixmax = np.minimum(key_boxes[:, 0] + key_boxes[:, 2], one_box[0] + one_box[2])
    iymax = np.minimum(key_boxes[:, 1] + key_boxes[:, 3], one_box[1] + one_box[3])
    w = np.maximum(ixmax - ixmin, 0.0)
    h = np.maximum(iymax - iymin, 0.0)
    inters = w * h
    # union
    uni = one_box[2] * one_box[3] + key_boxes[:, 2] * key_boxes[:, 3] - inters
    overlaps = inters / uni
    return overlaps