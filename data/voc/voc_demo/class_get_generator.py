import os
from xml.dom.minidom import parse
from tqdm import tqdm


def class_get(cls_txt):
    """
    遍历xml读取kind
    txt file -> buffer
    """
    with open(cls_txt,'r') as f:
        content = f.read()
        content = content.split('\n')[:-1]
        return content


def class_generator(xml_dir, cls_txt):
    """
    buffer -> txt file
    """
    classes = []
    for xml_name in tqdm(os.listdir(xml_dir)):
        xml_path = xml_dir + '/' + xml_name
        tree = parse(xml_path)
        rootnode = tree.documentElement
        objects = rootnode.getElementsByTagName('object')
        for object in objects:
            label = object.getElementsByTagName('name')[0].childNodes[0].data
            classes.append(label)
    classes = list(set(classes))
    strs = ""
    for name in classes:
        strs = strs + "{}\n".format(name)
    with open(cls_txt, 'w') as f:
        f.write(strs)
