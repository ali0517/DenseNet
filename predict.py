# -*- coding:utf-8 -*-
# @time :2019.09.07
# @IDE : pycharm
# @author :aligoo

import torch
import cv2
from torchvision import transforms
from models.densenet import densenet169
import cfg
import time
from collections import OrderedDict
# 开始时间
start = time.time()
# #定义模型的框架
model = densenet169(num_classes=cfg.NUM_CLASSES)

# #将模型放置在gpu上运行
# if torch.cuda.is_available():
#    model.cuda()

# 读取输出结果类别的键值对
labels2classes = cfg.labels_to_classes

# ##读取网络模型的键值对
trained_model = cfg.TRAINED_MODEL
state_dict = torch.load(trained_model)

# create new OrderedDict that does not contain `module.`
# #由于之前的模型是在多gpu上训练的，因而保存的模型参数，键前边有‘module’，需要去掉，和训练模型一样构建新的字典
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    head = k[:7]
    if head == 'module.':
        name = k[7:]  # remove `module.`
    else:
        name = k
    new_state_dict[name] = v
model.load_state_dict(new_state_dict)

# #进行模型测试时，eval（）会固定下BN与Dropout的参数
model.eval()

# 读入一张图像
img = cv2.imread('./datasets/test/bridge_512.jpg')  # cloud_515  bridge_512 circular_farmland_533
# 然后将numpy转换为tensor格式
tran = transforms.ToTensor()
img_tensor = tran(img)
# 将其变换为四维格式
img_tensor = img_tensor.unsqueeze(0)

# 模型预测 图像分类结果
out = model(img_tensor)
print(out)
# out = model(img_tensor.cuda())

# 选择置信度最高的类别
prediction = torch.max(out, 1)[1]

# 得到的prediction为cuda的tensor格式，需要转换为cpu格式，然后将tensor转换为numpy
aligo = prediction.cpu().numpy()
print(aligo[0])
print(labels2classes[str(aligo[0])])

# 计算运行时间
end = time.time()
print("运行时间:%.2f秒" % (end-start))
