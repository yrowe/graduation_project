import numpy as np
from ipdb import set_trace


def bbox_iou(box1, box2):
    ax1, ay1, ax2, ay2 = box1
    bx1, by1, bx2, by2 = box2

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_area = (inter_x2 - inter_x1)*(inter_y2 - inter_y1)

    a_area = (ax2 - ax1)*(ay2 - ay1)
    b_area = (bx2 - bx1)*(by2 - by1)

    iou = inter_area /(a_area + b_area - inter_area)
    return iou

a = [0,0,200,200]
b = [100,100,300,300]


print(bbox_iou(a,b))