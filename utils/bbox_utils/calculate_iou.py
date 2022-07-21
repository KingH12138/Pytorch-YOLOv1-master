def calculate_iou(pred, label):
    x1, a1 = pred[0], label[0]
    y1, b1 = pred[1], label[1]
    x2, a2 = pred[2], label[2]
    y2, b2 = pred[3], label[3]
    ax = max(x1, a1)  # 相交区域左上角横坐标
    ay = max(y1, b1)  # 相交区域左上角纵坐标
    bx = min(x2, a2)  # 相交区域右下角横坐标
    by = min(y2, b2)  # 相交区域右下角纵坐标

    area_N = (x2 - x1) * (y2 - y1)
    area_M = (a2 - a1) * (b2 - b1)

    w = bx - ax
    h = by - ay
    # 假设相交，那么按道理算出来的相交区域
    # 的w和h如果只要有一个是小于0的，那么
    # 就不成立(反证法)
    if w <= 0 or h <= 0:
        return 0
    area_X = w * h
    return area_X / (area_N + area_M - area_X)
