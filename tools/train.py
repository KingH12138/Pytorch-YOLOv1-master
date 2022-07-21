import argparse
from datetime import datetime
from tqdm import tqdm
from log_generator import log_generator
import torch
from prettytable import PrettyTable
from sklearn.metrics import accuracy_score, precision_score, roc_auc_score, average_precision_score, recall_score,\
    f1_score
from dataset.voc_dataload import get_dataloader
from model.yolov1 import Yolov1
from torch.optim import Adam
from loss.yolov1loss import YOLO_loss
from getcls import class_get
from torchsummary import summary
from utils.file_utils.label2bbox import labels2bbox


def args_parse():
    parser = argparse.ArgumentParser(description='This is a training parser of a object localization.')
    parser.add_argument(
        '-t',
        type=str,
        default='Pytorch-ObjectLocalization-master',
        help='the theme of your training task',
    )
    parser.add_argument(
        '-imaged',
        type=str,
        default=r'F:\VOCdevkit\VOC2007\JPEGImages',
        help="The JPEGImage's directory",
    )
    parser.add_argument(
        '-infop',
        type=str,
        default=r'D:\PythonCode\Pytorch-ObjectDetection-master\data\voc\refer.csv',
        help="The dataset information csv file's path",
    )
    parser.add_argument(
        '-clsp',
        type=str,
        default=r'D:\PythonCode\Pytorch-ObjectDetection-master\data\voc\classes.txt',
        help="The dataset classes' information file's path",
    )
    parser.add_argument(
        '-rs',
        type=tuple,
        default=(448, 448),
        help="regular size of images",
    )
    parser.add_argument(
        '-nw',
        type=int,
        default=6,
        help="number of workers",
    )
    parser.add_argument(
        '-wd',
        type=str,
        default=r'D:\PythonCodes\Pytorch-ObjectDetection-master\workdir\voc2007',
        help="the directory of log's saving path",
    )
    parser.add_argument(
        '-bs',
        type=int,
        default=32,
        help="batch size",
    )
    parser.add_argument(
        '-tp',
        type=float,
        default=0.9,
        help="train percent",
    )
    parser.add_argument(
        '-lr',
        type=float,
        default=1e-4,
        help="learning rate",
    )
    parser.add_argument(
        '-e',
        type=int,
        default=50,
        help="epoch",
    )
    return parser.parse_args()


if __name__=='__main__':
    args = args_parse()
# ----------------------------------------------------------------------------------------------------------------------
    print("Training device information:")
    # 训练设备信息
    device_table = ""
    if torch.cuda.is_available():
        device_table = PrettyTable(['number of gpu', 'applied gpu index', 'applied gpu name'], min_table_width=80)
        gpu_num = torch.cuda.device_count()
        gpu_index = torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name()
        device_table.add_row([str(gpu_num), str(gpu_index), str(gpu_name)])
        print('{}\n'.format(device_table))
    else:
        print("Using cpu......")
        device_table = 'CPU'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# ----------------------------------------------------------------------------------------------------------------------
    # 数据集信息
    print("Use dataset information file:{}\nLoading dataset from path: {}......".format(args.infop, args.imaged))
    train_dl, valid_dl, samples_num, train_num, valid_num = get_dataloader(args.imaged, args.infop, args.rs, args.bs,
                                                                           args.nw, args.tp)
    dataset_table = PrettyTable(['number of samples', 'train number', 'valid number', 'percent',"num_workers"], min_table_width=80)
    dataset_table.add_row([samples_num, train_num, valid_num, args.tp, args.nw])
    print("{}\n".format(dataset_table))
# ----------------------------------------------------------------------------------------------------------------------
    # 训练组件配置
    print("Classes information:")
    classes = class_get(args.clsp)
    classes_table = PrettyTable(classes, min_table_width=80)
    classes_table.add_row(range(len(classes)))
    print("{}\n".format(classes_table))
    print("Train information:")
    model = Yolov1().to(device)     ##########################################
    summary(model, (3,*args.rs), args.bs)
    optimizer = Adam(params=model.parameters(), lr=args.lr)
    loss_fn = YOLO_loss()   ##########################################
    train_table = PrettyTable(['theme', 'resize', 'batch size', 'epoch', 'learning rate', 'directory of log'],
                              min_table_width=150)
    train_table.add_row([args.t, args.rs, args.bs, args.e, args.lr, args.wd])
    print('{}\n\n'.format(train_table))
# ----------------------------------------------------------------------------------------------------------------------
    # 开始训练
    losses = []
    accuracies = []
    precisions = []
    recalls = []
    f1s = []
    aucs = []
    maps = []
    best_checkpoint = 0.

    st = datetime.now()
    for epoch in range(args.e):

        prediction = []
        label = []
        score = []

        model.train()
        train_bar = tqdm(iter(train_dl), ncols=150, colour='red')
        train_loss = 0.
        i = 0
        for train_data in train_bar:
            x_train, y_train = train_data
            x_train = x_train.to(device)
            y_train = y_train.to(device)
            output = model(x_train)
            loss = loss_fn(output, y_train)
            optimizer.zero_grad()
            # clone().detach()：可以仅仅复制一个tensor的数值而不影响tensor# 原内存和计算图
            train_loss += loss.clone().detach().cpu().numpy()
            loss.backward()
            optimizer.step()
            # 显示每一批次的loss
            train_bar.set_description("Epoch:{}/{} Step:{}/{}".format(epoch + 1, args.e, i + 1, len(train_dl)))
            train_bar.set_postfix({"train loss": "%.3f" % loss.data})
            i += 1
        train_loss = train_loss / i
        # 最后得到的i是一次迭代中的样本数批数
        losses.append(train_loss)

        # with torch.no_grad:
        #     model.eval()
        #     valid_bar = tqdm(iter(valid_dl), ncols=150, colour='red')
        #     valid_acc = 0.
        #     valid_pre = 0.
        #     valid_recall = 0.
        #     valid_f1 = 0.
        #     valid_auc = 0.
        #     valid_ap = 0.
        #     i = 0
        #     for valid_data in valid_bar:
        #         x_valid, y_valid = valid_data
        #         x_valid = x_valid.to(device)
        #         y_valid_ = y_valid.clone().detach().numpy().tolist()  # y_valid就不必放到gpu上训练了
        #         output = model(x_valid)  # shape:(N*cls_n)
        #         bbox = labels2bbox(output)
        #         # 显示每一批次的acc/precision/recall/f1
        #         valid_bar.set_description("Epoch:{}/{} Step:{}/{}".format(epoch + 1, args.e, i + 1, len(valid_dl)))
        #         prediction = prediction + pred_
        #         label = label + y_valid_
        #         score = score + output_
        #         i += 1
        #     # 最后得到的i是一次迭代中的样本数批数
        # # 每一次epoch计算一次
        # valid_acc = accuracy_score(y_true=label, y_pred=prediction)
        # valid_pre = precision_score(y_true=label, y_pred=prediction, average='weighted')
        # valid_recall = recall_score(y_true=label, y_pred=prediction, average='weighted')
        # valid_f1 = f1_score(y_true=label, y_pred=prediction, average='weighted')
        # # valid_auc = roc_auc_score(y_true=label, y_score=score, average='weighted', multi_class="ovr")
        # # valid_ap = average_precision_score(y_true=label, y_score=score)
        #
        # accuracies.append(valid_acc)
        # precisions.append(valid_pre)
        # recalls.append(valid_recall)
        # f1s.append(valid_f1)
        # # aucs.append(valid_auc)
        # # maps.append(valid_ap)
        #
        # if valid_f1 >= max(f1s):    # 如果本次epoch的f1大于了存储f1列表的最大值，那么最好的checkpoint赋值为model
        #     best_checkpoint = model
        #
        # indicator_table = PrettyTable(["Epoch",'Accuracy', 'Precision', 'Recall', 'F1'], )
        # indicator_table.add_row([epoch+1, valid_acc, valid_pre, valid_recall, valid_f1])
        # print('\n{}\n'.format(indicator_table))
    et = datetime.now()

    log_generator(args.t, et - st, dataset_table, classes_table, device_table, train_table, optimizer, model, args.e,
                  [losses], args.wd)
# ----------------------------------------------------------------------------------------------------------------------


