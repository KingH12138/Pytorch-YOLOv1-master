from xml.dom.minidom import parse


def readvocxml(xml_path, image_dir):
    """

    The function can read single xml file and transform information of xml file into a list containing:
    the filename of the xml indicates(str),
    the filepath of image that xml indicates(a str.you need to give the dir which this image located in.Aka,the second parameter.)
    the depth,height,width of the Image(three int data.channel first),
    the annotated objects' infomation.(
        a 2D int list:
        [
            row1:[label_1,xmin_1,ymin_1,xmax_1,ymax_1]
            row2:[label_2,xmin_2,ymin_2,xmax_2,ymax_2]
            ....
            row_i[label_i,xmin_i,ymin_i,xmax_i,ymax_i]
        ]
    )

    Args:

    xml_path:singal xml file's path.

    image_dir:the image's location dir that xml file indicates.


    """
    tree = parse(xml_path)
    rootnode = tree.documentElement
    sizenode = rootnode.getElementsByTagName('size')[0]
    width = int(sizenode.getElementsByTagName('width')[0].childNodes[0].data)
    height = int(sizenode.getElementsByTagName('height')[0].childNodes[0].data)
    depth = int(sizenode.getElementsByTagName('depth')[0].childNodes[0].data)

    name_node = rootnode.getElementsByTagName('filename')[0]
    filename = name_node.childNodes[0].data

    path = image_dir + '\\' + filename

    objects = rootnode.getElementsByTagName('object')
    objects_info = []
    for object in objects:
        label = object.getElementsByTagName('name')[0].childNodes[0].data
        xmin = int(object.getElementsByTagName('xmin')[0].childNodes[0].data)
        ymin = int(object.getElementsByTagName('ymin')[0].childNodes[0].data)
        xmax = int(object.getElementsByTagName('xmax')[0].childNodes[0].data)
        ymax = int(object.getElementsByTagName('ymax')[0].childNodes[0].data)
        info = []
        info.append(label)
        info.append(xmin)
        info.append(ymin)
        info.append(xmax)
        info.append(ymax)
        objects_info.append(info)

    return [filename, path, depth, height, width, objects_info]