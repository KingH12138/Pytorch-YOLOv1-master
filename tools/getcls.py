def class_get(cls_txt):
    """
    遍历xml读取kind
    txt file -> buffer
    """
    with open(cls_txt,'r') as f:
        content = f.read()
        content = content.split('\n')[:-1]
        return content