import numpy as np
import torch
import argparse
from tools.getcls import class_get
from PIL import Image
from torchvision.transforms import *
from model.yolov1 import Yolov1
from utils.file_utils.label2bbox import labels2bbox
from utils.bbox_utils.bbox_draw import drawbbox


class YOLOV1Detector(object):
    def __init__(self, image_path:str,weight_path:str,cls_path:str,resize:tuple):
        super(YOLOV1Detector, self).__init__()
        self.img = Image.open(image_path)
        print("Loading prediction:{}.".format(image_path))
        self.w = self.img.width
        self.h = self.img.height
        self.resize = resize
        self.transformer = Compose([
            Resize(self.resize),
            ToTensor(),
        ])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torch.load(weight_path)
        self.model = self.model.to(device=self.device)
        print("Loading checkpoint from:{}.\n Using device:{}".format(weight_path,self.device))
        self.classes = class_get(cls_path)

    def predict(self):
        inputs = self.transformer(self.img).to(self.device)
        inputs = inputs.unsqueeze(0)
        outputs = self.model(inputs).permute(0,2,3,1)
        outputs = outputs.cpu().detach().clone().numpy().squeeze()
        outputs = labels2bbox(outputs)
        # 逆归一化位置信息
        outputs[:, 0] *= detector.w
        outputs[:, 1] *= detector.h
        outputs[:, 2] *= detector.w
        outputs[:, 3] *= detector.h
        # outputs转为list
        outputs = outputs.tolist()
        prediction = []
        for i in range(len(outputs)):
            location = outputs[i][0:4]
            kind = int(outputs[i][-1])
            objects = [self.classes[kind]]+location
            prediction.append(objects)
        return prediction


def args_parse():
    parser = argparse.ArgumentParser(description='This is a training parser of a object localization.')
    parser.add_argument(
        '-imagep',
        type=str,
        default=r'D:\PythonCodes\Pytorch-ObjectDetection-master\img1.jpg',
        help="The prediction image's path",
    )
    parser.add_argument(
        '-checkpoint',
        type=str,
        default=r'D:\PythonCodes\Pytorch-ObjectDetection-master\workdir\voc2007\exp-Pytorch-ObjectLocalization-master_2022_6_16_11_29\Pytorch-ObjectLocalization-master_2022_6_16_11_29.pth',
        help="weight_path",
    )
    parser.add_argument(
        '-clsp',
        type=str,
        default=r'D:\PythonCodes\Pytorch-ObjectDetection-master\data\voc\classes.txt',
        help="The dataset classes' information file's path",
    )
    parser.add_argument(
        '-rs',
        type=tuple,
        default=(448, 448),
        help="regular size of images",
    )
    return parser.parse_args()


model = Yolov1()
args = args_parse()
detector = YOLOV1Detector(
    args.imagep,
    args.checkpoint,
    args.clsp,
    args.rs,
)
prediction = detector.predict()
drawbbox(args.imagep, prediction)


