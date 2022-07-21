import numpy as np


def convert_bbox2labels(bboxes):
    """
    :param bboxes:(N,5)的bbox信息列表
    :return:(30,7,7)的yolov1格式的label,需要将(cls_index,dx,dy,dw,dh)转换成(cx,cy,dw,dh,confidence,cx,cy,dw,dh,confidence,....)

    tips:(30,7,7) = (info_dim,x,y)
    """
    grid_size = 1 / 7.0
    label = np.zeros(shape=(30, 7, 7))
    # 遍历每一个bbox，把它放入该放的地方
    for i in range(len(bboxes)):
        bbox = bboxes[i]
        # 获取中心点所在-下标-
        grid_x = int(bbox[1] // grid_size)
        grid_y = int(bbox[2] // grid_size)
        cx = bbox[1] / grid_size - grid_x
        cy = bbox[2] / grid_size - grid_y
        label[0:5, grid_x, grid_y] = np.array([cx, cy, bbox[3], bbox[4], 1])
        label[5:10, grid_x, grid_y] = np.array([cx, cy, bbox[3], bbox[4], 1])
        label[10 + int(bbox[0]), grid_x, grid_y] = 1
    return label