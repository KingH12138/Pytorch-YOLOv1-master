from torch.nn import Module
import torch
from utils.bbox_utils.calculate_iou import calculate_iou


class YOLO_loss(Module):
    def __init__(self, noobj_weight=0.5):
        super(YOLO_loss, self).__init__()
        self.noobj_weight = noobj_weight
        self.obj_weight = 1 - self.noobj_weight

    def forward(self, pred, label):
        """
        :param pred:YOLO_resnet输出的(batch_size,30,7,7)
        :param label:Dataset返回的的(batch_size,30,7,7)大小的label
        :return:yolo_loss(noobj_loss+obj_loss+classes_loss+confidence_loss)
        """
        # 开始遍历每一个grid
        grid_size = 1 / 7.0
        batch_size = pred.shape[0]
        noobj_loss = 0.
        obj_loss = 0.
        classes_loss = 0.
        confidence_loss = 0.
        for n in range(batch_size):
            for i in range(7):  # x方向遍历
                for j in range(7):  # y方向遍历
                    pred_grid = pred[n, :, i, j]
                    label_grid = label[n, :, i, j]
                    if label_grid[4] == 1:  # 如果对应label包含物体
                        # 现在要将得到两个预测的bbox和label的bbox的极坐标
                        # 用这个三个极坐标可以分别算出两个预测的bbox与标签的
                        # 极坐标的iou，取大的作为我们计算对象。
                        bbox1_pred = (
                            (pred_grid[0] + i) * grid_size - pred_grid[2] / 2,
                            (pred_grid[1] + j) * grid_size - pred_grid[3] / 2,
                            (pred_grid[0] + i) * grid_size + pred_grid[2] / 2,
                            (pred_grid[1] + j) * grid_size + pred_grid[3] / 2,
                        )
                        bbox2_pred = (
                            (pred_grid[5] + i) * grid_size - pred_grid[7] / 2,
                            (pred_grid[6] + j) * grid_size - pred_grid[8] / 2,
                            (pred_grid[5] + i) * grid_size + pred_grid[7] / 2,
                            (pred_grid[6] + j) * grid_size + pred_grid[8] / 2,
                        )
                        bbox_label = (
                            (label_grid[0] + i) * grid_size - label_grid[2] / 2,
                            (label_grid[1] + j) * grid_size - label_grid[3] / 2,
                            (label_grid[0] + i) * grid_size - label_grid[2] / 2,
                            (label_grid[1] + j) * grid_size - label_grid[3] / 2,
                        )
                        iou1 = calculate_iou(bbox1_pred, bbox_label)
                        iou2 = calculate_iou(bbox2_pred, bbox_label)

                        if iou1 > iou2:
                            obj_loss = obj_loss + (torch.sum((pred_grid[0:2] - label_grid[0:2]) ** 2) +
                                         torch.sum((pred_grid[2:4].sqrt() - label_grid[2:4].sqrt()) ** 2))
                            confidence_loss = confidence_loss + (pred_grid[4] - 1) ** 2
                            noobj_loss = noobj_loss + ((pred_grid[9] - 0) ** 2)
                        else:
                            obj_loss = obj_loss + (torch.sum((pred_grid[5:7] - label_grid[5:7]) ** 2) +
                                         torch.sum((pred_grid[7:9].sqrt() - label_grid[7:9].sqrt()) ** 2))
                            confidence_loss = confidence_loss + (pred_grid[9] - 1) ** 2
                            noobj_loss = noobj_loss + ((pred_grid[4] - 0) ** 2)
                        classes_loss = classes_loss + torch.sum((pred_grid[10:] - label_grid[10:]) ** 2)
                    else:
                        noobj_loss = noobj_loss + (pred_grid[4] ** 2 + pred_grid[9] ** 2)

        loss = (obj_loss * self.obj_weight + noobj_loss * self.noobj_weight + classes_loss + confidence_loss) / batch_size

        return loss








