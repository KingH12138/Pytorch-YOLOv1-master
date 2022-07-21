import os
from datetime import datetime

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import torch


def df_generator(epoches, tags, save_path=None):
    keys = ['epoch', 'losses', 'accuracy', 'precision', 'recall', 'f1']
    values = [range(1, epoches + 1)] + tags
    data = dict(zip(keys, values))
    df = pd.DataFrame(data=data)
    if save_path:
        df.to_csv(save_path, encoding='utf-8')
    return df


def indicators_plot(epoches, tags, save_fig_path=None, csv_save_path=None):
    """
    :param epoches:迭代次数
    :param tags:[loss,acc,precision,recall,f1]
    :param save_fig_path:保存路径
    """
    df = df_generator(epoches, tags, save_path=csv_save_path)
    sns.set_style("darkgrid")
    plt.figure(figsize=(8.,8.))
    plt.subplot(2, 2, 1)
    sns.lineplot(x='epoch',y='losses',data=df)
    # sns.lineplot(x='epoch',y='accuracy',data=df)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    # plt.ylabel('Loss/Accuracy')
    # plt.title('Loss/Accuracy-Epoch')
    # plt.legend(['losses','accuracy'])
    # plt.subplot(2, 2, 2)
    # sns.lineplot(x='epoch', y='precision', data=df)
    # plt.xlabel('Epoch')
    # plt.ylabel("Precision")
    # plt.title('Precision-Epoch')
    # plt.legend(['precision'])
    # plt.subplot(2, 2, 3)
    # sns.lineplot(x='epoch', y='recall', data=df)
    # plt.xlabel('Epoch')
    # plt.ylabel('Recall')
    # plt.title('Recall-Epoch')
    # plt.legend(['recall'])
    # plt.subplot(2, 2, 4)
    # sns.lineplot(x='epoch', y='f1', data=df)
    # plt.xlabel('Epoch')
    # plt.ylabel('F1')
    # plt.title('F1-Epoch')
    # plt.legend(['F1'])
    plt.subplots_adjust(wspace=0.3, hspace=0.35)
    if save_fig_path:
        plt.savefig(save_fig_path)


def log_generator(train_theme_name, duration,
                  dataset_info_table, classes_info_table,
                  training_device_table, training_info_table,
                  optimizer, model, epoches,
                  tags, log_save_dir):
    nowtime = datetime.now()
    year = str(nowtime.year)
    month = str(nowtime.month)
    day = str(nowtime.day)
    hour = str(nowtime.hour)
    minute = str(nowtime.minute)
    second = str(nowtime.second)
    nowtime_strings = year + '/' + month + '/' + day + '/' + hour + ':' + minute + ':' + second
    workplace_path = os.getcwd()
    content = """
Theme:{}\n
Date:{}\n
Time used:{}\n
workplace:{}\n
folder information:\n{}\n
classes:\n{}\n
training device:\n{}\n
training basic configuration:\n{}\n
Optimizer:\n{}\n
Model:\n{}\n,
    """.format(
        train_theme_name,
        nowtime_strings,
        duration,
        workplace_path,
        dataset_info_table,
        classes_info_table,
        training_device_table,
        training_info_table,
        str(optimizer),
        str(model)
    )
    exp_name = 'exp-{}_{}_{}_{}_{}_{}'.format(
        train_theme_name,
        year, month, day,
        hour, minute, second)
    exp_path = log_save_dir + '/' + exp_name
    if os.path.exists(exp_path) == 0:
        os.makedirs(exp_path)
    log_name = '{}_{}_{}_{}_{}_{}.log'.format(
        train_theme_name,
        year, month, day,
        hour, minute, second)
    file = open(exp_path + '/' + log_name, 'w', encoding='utf-8')
    file.write(content)
    file.close()
    torch.save(model, exp_path + '/' + '{}_{}_{}_{}_{}_{}.pth'.format(
        train_theme_name,
        year, month, day, hour,
        minute, second
    ))
    indicators_plot(epoches, tags, save_fig_path=exp_path + '/indicators.jpg', csv_save_path=exp_path +'/indicators.csv')
    print("Training log has been saved to path:{}".format(exp_path))
