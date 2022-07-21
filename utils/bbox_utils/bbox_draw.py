import cv2


def drawbbox(img_path, bboxes,
             colors=None, classes=None, plot_mode=1, grid_size=None,
             text_font_size=0.5, text_font_weight=1, text_font_family=cv2.FONT_HERSHEY_SIMPLEX, text_color=(0, 0, 0),
             linewidth=4, show=True, image_save_path=False):
    """
    :param img_path:图片路径
    :param bboxes:2D数组/列表
    eg:
        [[cls,X,X,X,X],[cls,X,X,X,X],[cls,X,X,X,X]......]
    :param colors:rgb列表表示的调色板
    :param classes:颜色列表
    :param plot_mode:绘制模式，1为极坐标，其他的为中心坐标绘制
    :param linewidth:bbox的线条粗细
    :param show:是否展示
    :param image_save_path:是否保存，为路径
    :param grid_size:为一个二元元组或者列表，表示网格模式的网格大小
    :param text_font_size:字体大小
    :param text_font_weight:字体加粗，为一个整形
    :param text_font_family:字体类型
    :param text_color:
    :return:None
    """
    # 以不改变图片模式的方式读入图片为numpy数组(height,width,channels)
    img_arr = cv2.imread(img_path)
    if type(bboxes[0][1]) == float:
        for i in range(len(bboxes)):
            for j in range(1, 5):
                bboxes[i][j] = int(bboxes[i][j])
    for bbox in bboxes:
        if plot_mode == 1:  # 极坐标形式绘图
            if colors == None or classes == None:  # 如果画色板没有给出，就默认(0,0,255)_red_为color
                cv2.rectangle(img_arr, bbox[1:3], bbox[3:5], (0, 0, 255), linewidth)
                cv2.putText(img_arr, bbox[0], (bbox[1], bbox[2] - linewidth), text_font_family, text_font_size,
                            text_color, text_font_weight)
            else:  # 若给出画色板，按照画色板进行绘制
                cv2.rectangle(img_arr, bbox[1:3], bbox[3:5], colors[classes.index[bbox[0]]], linewidth)
                cv2.putText(img_arr, bbox[0], (bbox[1], bbox[2] - linewidth), text_font_family, text_font_size,
                            text_color, text_font_weight)
        else:  # 中心坐标模式
            if colors == None or classes == None:
                cv2.rectangle(img_arr, [bbox[1] - bbox[3] // 2, bbox[2] - bbox[4] // 2],
                              [bbox[1] + bbox[3] // 2, bbox[2] + bbox[4] // 2], (0, 0, 255), linewidth)
                cv2.putText(img_arr, bbox[0], (bbox[1] - bbox[3] // 2, bbox[2] - bbox[4] // 2 - linewidth),
                            text_font_family, text_font_size,
                            text_color, text_font_weight)
            else:
                cv2.rectangle(img_arr, [bbox[1] - bbox[3] // 2, bbox[2] - bbox[4] // 2],
                              [bbox[1] + bbox[3] // 2, bbox[2] + bbox[4] // 2], colors[classes.index[bbox[0]]],
                              linewidth)
                cv2.putText(img_arr, bbox[0], (bbox[1] - bbox[3] // 2, bbox[2] - bbox[4] // 2 - linewidth),
                            text_font_family, text_font_size,
                            text_color, text_font_weight)
    if grid_size:  # shape=(h,w)
        x_gap = img_arr.shape[1] // grid_size[1]
        y_gap = img_arr.shape[0] // grid_size[0]
        for i in range(grid_size[0]):  # 高度变化——画横线
            cv2.line(img_arr, [0, y_gap * i], [img_arr.shape[1], y_gap * i], (0, 0, 0), 4)
        for j in range(grid_size[1]):  # 宽度变化——画竖线
            cv2.line(img_arr, [x_gap * j, 0], [x_gap * j, img_arr.shape[0]], (0, 0, 0), 4)
    if show:
        cv2.imshow("Display", img_arr)
        cv2.waitKey(0)
    if image_save_path:
        cv2.imwrite(image_save_path, img_arr)
        print("Save output into path:{}.".format(image_save_path))
