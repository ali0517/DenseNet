# -*- coding:utf-8 -*-
# @time :2020.03.06
# @IDE : pycharm
# @author : aligoo

# #数据集的类别
NUM_CLASSES = 10

# 训练时batch的大小
BATCH_SIZE = 16

# 网络默认输入图像的大小
INPUT_SIZE = 224

# #预训练模型的存放位置
# 下载地址：https://download.pytorch.org/models/densenet169-b2777c0a.pth
PRETRAINED_MODEL = './densenet169.pth'  # densenet121.pth

# 训练完成，权重文件的保存路径,默认保存在trained_model下  验证测试时读取
TRAINED_MODEL = './trained_model/dense169_110.pth' 

# 数据集的存放位置
TRAIN_DATASET_DIR = './datasets/NWPU_RESISC45/'
VAL_DATASET_DIR = './datasets/NWPU_RESISC45VAL/'

# labels
labels_to_classes = {
    '0': 'airplane',
    '1': 'airport',
    '2': 'baseball_diamond',
    '3': 'basketball_court',
    '4': 'beach',
    '5': 'bridge',
    '6': 'chaparral',
    '7': 'church',
    '8': 'circular_farmland',
    '9': 'cloud'
}
