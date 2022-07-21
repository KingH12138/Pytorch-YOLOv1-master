"""

该脚本文件对JPEGImage和Annotation中的jpg、xml文件进行读取，并且生产存储位置信息(x,y,x,y)的numpy及其文件夹:

txt文件中的信息:
    1.根据原图像size归一化
    2.获取classes从而label数值化

"""
import os
import pandas as pd
from class_get_generator import class_get, class_generator
from utils.file_utils.readvocxml import readvocxml
import numpy as np
from tqdm import tqdm


def generator(image_dir, xml_dir, txt_dir, cls_path,refer_path):
    if os.path.exists(cls_path)==False: # 如果没有classes.txt，就生成一个
        class_generator(xml_dir, cls_path)
    if os.path.exists(txt_dir)==False:  # 没有npy文件夹就创建一个
        os.makedirs(txt_dir)
    classes = class_get(cls_path)
    info_dict = {"img_name":[],"img_path":[],"object_path":[]}
    for xml_name in tqdm(os.listdir(xml_dir)):
        xml_path = xml_dir + '/' + xml_name
        img_name, img_path, _, height, width, objects_info = readvocxml(xml_path, image_dir)
        for i in range(len(objects_info)):
            objects_info[i][0] = classes.index(objects_info[i][0])
            objects_info[i][1] /= width
            objects_info[i][2] /= height
            objects_info[i][3] /= width
            objects_info[i][4] /= height
        npy_path = txt_dir + '/' + img_name[:-4] + '.npy'
        objects_info = np.array(objects_info)
        np.save(npy_path,objects_info)
        info_dict['img_name'].append(img_name)
        info_dict['img_path'].append(img_path)
        info_dict['object_path'].append(npy_path)
    df = pd.DataFrame(info_dict)
    df.to_csv(refer_path,encoding='utf-8')

generator(
    r"F:\VOCdevkit\VOC2007\JPEGImages",
    r"F:\VOCdevkit\VOC2007\Annotations",
    r"F:\VOCdevkit\VOC2007\Annotations_array",
    r"D:\PythonCode\Pytorch-ObjectDetection-master\data\voc\classes.txt",
    r"D:\PythonCode\Pytorch-ObjectDetection-master\data\voc\refer.csv"
)
