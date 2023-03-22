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


def iou_one_to_multiple(box, boxes):
    """
    Calculate the Intersection over Union (IoU) of a bounding box with a batch of bounding boxes.

    Args:
        box (list of 4 floats): [xmin, ymin, xmax, ymax]
        boxes (list of N lists of 4 floats): [[xmin, ymin, xmax, ymax], [xmin, ymin, xmax, ymax], ...]

    Returns:
        list of N floats: the IoU of the box with each of the boxes in the batch
    """
    # Calculate the intersection area
    boxes = np.array(boxes, dtype=np.float32).reshape([-1, 4])
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[0] + box[2], boxes[:, 0] + boxes[:, 2])
    y2 = np.minimum(box[1] + box[3], boxes[:, 1] + boxes[:, 3])
    intersection_area = np.maximum(0, x2 - x1 + 1) * np.maximum(0, y2 - y1 + 1)

    # Calculate the union area
    box_area = box[2] * box[3]
    boxes_area = boxes[:, 2] * boxes[:, 3]
    union_area = box_area + boxes_area - intersection_area

    # Calculate the IoU
    iou = intersection_area / union_area

    return iou