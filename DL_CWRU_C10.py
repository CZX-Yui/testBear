import logging
import time
import os
import numpy as np
from tqdm import tqdm
import pandas as pd
import torch
from torch import optim
from torch import nn
from sklearn.model_selection import train_test_split

from utils import data_load, dataset, data_transforms, setlogger
from models import CNN_1D

# 全局变量定义
PATH_ROOT_DATA = '/home/CZX/Data/Bearing/CWRU/raw_mat_data/'     # TODO 实验中需要更改的值
PATH_OUT_LOG = '/home/CZX/FedAI/TestBearing/logs'

NUM_CLASS = 10
FAULT_LABEL = [1, 2, 3, 4, 5, 6, 7, 8, 9]
SIGNAL_SIZE = 128   # 时间窗，对应样本长度
BATCH_SIZE = 64
LR = 0.001
WEIGHT_DECAY = 1e-5
MOMENTUM = 0.9
MAX_EPOCH = 2

# 使用GPU
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    raise NotImplementedError

# 设置日志读取器
time_now = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(time.time()))
save_dir = os.path.join(PATH_OUT_LOG, "test_{}".format(time_now))
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
setlogger(os.path.join(save_dir, 'training.log'))
for ll in range(5):
    logging.info('\n')

# 模型保存位置
PATH_MODEL = os.path.join("/home/CZX/FedAI/TestBearing/logs/test_2022-09-14-16_48_58", 'latest_model.pth')   # TODO
#------------------------------------------------------------------------------------ 1. 加载数据集
# 仅考虑1797rpm工况下的数据集
datasetname = ["12k Drive End Bearing Fault Data", "Normal Baseline Data"]
normalname = ["97.mat"]  # 1个正常类
dataname1 = ["105.mat", "118.mat", "130.mat", "169.mat", "185.mat", "197.mat", "209.mat", "222.mat", "234.mat"] # 9个故障类
# 将10个类的数据集混合加载，构建样本
PATH_ROOT_DATA_FAULT = os.path.join(PATH_ROOT_DATA, datasetname[0])
PATH_ROOT_DATA_NORMAL = os.path.join(PATH_ROOT_DATA, datasetname[1])
path1 = os.path.join(PATH_ROOT_DATA_NORMAL, normalname[0])  
data, label = data_load(path1, axisname=normalname[0], label=0, signal_size=SIGNAL_SIZE) 
for i in tqdm(range(len(dataname1))):
    path2 = os.path.join(PATH_ROOT_DATA_FAULT, dataname1[i])
    data1, lab1 = data_load(path2, axisname=dataname1[i], label=FAULT_LABEL[i], signal_size=SIGNAL_SIZE)
    data += data1
    label += lab1
# 打印样本信息
check_sum = np.unique(label, return_counts=True)
print("Data load done ------ \n class: {}\n count: {}".format(check_sum[0], check_sum[1]))
print("Each input sample shape: {}".format(data[0].shape))
# 数据预处理，构建dataset，训练阶段使用train_dataset和val_dataset，测试阶段使用test_dataset
data_pd = pd.DataFrame({"data": data, "label": label})
train_pd, val_pd = train_test_split(data_pd, test_size=0.20, random_state=40, stratify=data_pd["label"])
train_dataset = dataset(list_data=train_pd, transform=data_transforms('train','0-1'))
val_dataset = dataset(list_data=val_pd, transform=data_transforms('val','0-1'))
test_dataset = dataset(list_data=data_pd, transform=data_transforms('test','0-1'))
# 按batch加载dataset，构建dataloader
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4,
                                                pin_memory=(True if DEVICE == 'cuda' else False))
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4,
                                                pin_memory=(True if DEVICE == 'cuda' else False))
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4,
                                                pin_memory=(True if DEVICE == 'cuda' else False))

#---------------------------------------------------------------------------------- 2. 初始化模型等
# 初始化模型
model = CNN_1D(in_channel=1, out_channel=NUM_CLASS).to(DEVICE)
# 初始化优化器
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
# optimizer = optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
# 初始化损失函数
criterion = nn.CrossEntropyLoss()

#---------------------------------------------------------------------------------- 3. 执行模型训练与保存

for epoch in range(0, MAX_EPOCH):
    logging.info('-'*5 + 'Epoch {}/{}'.format(epoch+1, MAX_EPOCH) + '-'*5)
    # train_epoch
    model.train()

    epoch_acc = 0
    epoch_loss = 0.0

    for batch_idx, (inputs, labels) in enumerate(train_dataloader):
        inputs = inputs.to(DEVICE)
        labels = labels.to(DEVICE)

        with torch.set_grad_enabled(True):
            logits = model(inputs)
            loss = criterion(logits, labels)
            pred = logits.argmax(dim=1)
            correct = torch.eq(pred, labels).float().sum().item()
            loss_temp = loss.item() * inputs.size(0)
            epoch_loss += loss_temp
            epoch_acc += correct        

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    epoch_loss_all = epoch_loss / len(train_dataloader.dataset)
    epoch_acc_all = epoch_acc / len(train_dataloader.dataset)
    logging.info('Epoch: {} train-Loss: {:.4f} train-Acc: {:.4f}'.format(
        epoch, epoch_loss_all, epoch_acc_all
    ))

    # val_epoch
    model.eval()
    epoch_acc_val = 0
    epoch_loss_val = 0
    for batch_idx, (inputs, labels) in enumerate(val_dataloader):
        inputs = inputs.to(DEVICE)
        labels = labels.to(DEVICE)

        with torch.set_grad_enabled(False):
            # forward
            logits = model(inputs)
            loss = criterion(logits, labels)
            pred = logits.argmax(dim=1)
            correct = torch.eq(pred, labels).float().sum().item()
            loss_temp = loss.item() * inputs.size(0)
            epoch_loss_val += loss_temp
            epoch_acc_val += correct
    epoch_loss_val_all = epoch_loss_val / len(val_dataloader.dataset)
    epoch_acc_val_all = epoch_acc_val / len(val_dataloader.dataset)
    logging.info('Epoch: {} val-Loss: {:.4f} val-Acc: {:.4f}'.format(
            epoch, epoch_loss_val_all, epoch_acc_val_all
        ))

model_state_dic = model.state_dict()
# 保存模型默认保存latest，先不做成保存best
logging.info("----- save latest model ----- ")

torch.save(model_state_dic, PATH_MODEL)

#----------------------------------------------------------------------------------- 4. 加载模型与测试
pretrained_model = CNN_1D(in_channel=1, out_channel=NUM_CLASS)    # 测试阶段都在CPU上执行
pretrained_param = torch.load(PATH_MODEL)
pretrained_model.load_state_dict(pretrained_param)
acc_test = 0
pretrained_model.eval()
for batch_idx, (inputs, labels) in enumerate(test_dataloader):
    with torch.set_grad_enabled(False):
        logits = pretrained_model(inputs)
        pred = logits.argmax(dim=1)
        correct = torch.eq(pred, labels).float().sum().item()
        acc_test += correct

acc_test_all = acc_test / len(test_dataloader.dataset)
logging.info('-----test-----\ntest-Acc: {:.4f}'.format(acc_test_all))

print("DONE")