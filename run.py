from BMANet.dataset import *
from BMANet.mobile import *
from BMANet.modules import *
import os
import torch
import numpy as np
import torch.utils.data as data
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import time
os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1' #本次运行【train.py】文件使用的是第一块和第二块显卡
gpu_devices = list(np.arange(torch.cuda.device_count())) #返回GPU排列
#torch.cuda.device_count() 返回GPU数量 np.arange返回一个排列（从0开始）
output_folder = r'../Outputs/pred/MSCNet/ORSI-4199/Test'
# output_folder = r'../Outputs/pred/MSCNet/ORSSD/Test'
ckpt_folder = r'../Checkpoints/ORSI-4199'
dataset_root = r'../Dataset/ORSI-4199'
# dataset_root = r'../Dataset/ORSSD'

batch_size = 6

#IOU损失
def iou(pred, mask):
    inter = (pred * mask) .sum(dim=(2, 3))
    union = (pred + mask) .sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return wiou.mean()
#BCELoss+IOULoss
class BCEloss(nn.Module):
    def __init__(self):
        super(BCEloss, self).__init__()
        self.bce = nn.BCELoss()
        #自己加入-改进二
        # self.floss = FLoss()
        #自己加入

    def forward(self, sm,label):
        mask_loss =self.bce(sm,label)+0.6*iou(sm,label)
        total_loss = mask_loss
        return [total_loss, mask_loss, mask_loss]


def count_param(model):
    param_count = 0
    for param in model.parameters():
        param_count += param.view(-1).size()[0]
    return param_count

class Run:
    def __init__(self):
        self.train_set = EORSSD(dataset_root, 'train', aug=True) # 打开训练集，即train.txt  return image, label, edge
        #加载训练数据集                                                                                              训练的时候剩下不是一个batch就被扔掉了
        self.train_loader = data.DataLoader(self.train_set, shuffle=True, batch_size=batch_size, num_workers=4, drop_last=True)  #true
        # 加载测试数据集
        self.test_set = EORSSD(dataset_root, 'test', aug=False) #打开训练集，即test.txt  return image, label,name
        self.test_loader = data.DataLoader(self.test_set, shuffle=False, batch_size=1, num_workers=4, drop_last=False)

        self.init_lr = 1e-4
        self.min_lr = 1e-7
        self.train_epoch = 40

        self.net = MobileNetV2()         #../Checkpoints\trained\best.pth
        # self.net.load_state_dict(torch.load(os.path.join(ckpt_folder, 'trained', 'trained.pth'))) #将训练好的模型参数重新加载至网络模型中
        self.loss=BCEloss() #计算损失 返回总损失，显著损失，边缘损失

    def train(self):
        self.net.train().cuda() #训练 加载到GPU上
        max_F=0.86
        base, head = [], []
        for name, param in self.net.named_parameters(): #输出每层的参数名称（如mbv0.weight/bias：XXX）以及数值大小
            if 'mbv' in name:
                base.append(param) #编码器的权重与误差
            else:
                head.append(param) #解码器的权重与误差
        #构造优化器,调整每个参数的学习速率
        optimizer = optim.Adam([{'params': base}, {'params': head}], lr=self.init_lr, weight_decay=5e-4)
        #余弦退火调整学习速率                                         T_max 个 epoch 之后重新设置学习率  最小学习速率
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.train_epoch,eta_min=self.min_lr)
        for epc in range(1, self.train_epoch + 1): #遍历epoch1-40
            records = [0] * 3 #输出为[0,0,0]  记录损失
            N = 0
            #动态修改学习率  param_groups[0]长度为6的字典，里面有6个参数
            optimizer.param_groups[0]['lr'] = 0.5 * optimizer.param_groups[1]['lr']  # for backbone
            optimizer.param_groups[1]['lr'] = optimizer.param_groups[1]['lr']
            for image, label, edge in tqdm(self.train_loader): #tqdm 进度条
                # prepare input data\n",
                image, label, edge = image.cuda(), label.cuda(), edge.cuda()
                B = image.size(0) #获取图像的高度
                # forward\n",
                optimizer.zero_grad() # 梯度初始化为零，把loss关于weight的导数变成0
                sm = self.net(image)  # forward：将数据传入模型，前向传播求出预测的值
                losses_list = self.loss(sm,label) #计算损失 返回总损失，显著损失，边缘损失



                # compute loss\n",
                total_loss = losses_list[0].mean() #求总损失均值
                # record loss\n",
                N += B
                for i in range(len(records)): #0-2
                    records[i] += losses_list[i].mean().item() * B #item()返回的是一个浮点型数据，精度更高
                # backward\n",
                total_loss.backward() # backward：反向传播求梯度
                optimizer.step()      # optimizer：更新所有参数(模型会更新)
            # update learning rate\n",
            scheduler.step()  #调整学习速率
            F=self.test(epc)
            #保存训练好的模型参数
            if F>max_F: #max_F=0.86
                cache_model(self.net, os.path.join(ckpt_folder, 'trained', 'trained.pth')) #缓存
                max_F=F
            if epc==self.train_epoch:
                cache_model(self.net, os.path.join(ckpt_folder, 'trained', 'last.pth'))
            # print training information\n",
            records = proc_loss(records, N, 4)
            print('epoch: {} || total loss: {} || mask loss: {} || edge loss: {}'
                  .format(epc, records[0], records[1], records[2])) #输出epoch，总损失，显著损失，边缘损失
        print('finish training.'+'maxF:',max_F)

    def test(self,ep):
        self.net.eval().cuda() #测试 加载到GPU上
        #print("params:",(count_param(self.net)/1e6))
        num_test = 0
        mae = 0.0
        F_value=0.0

        for image, label, prefix in self.test_loader:
            num_test += 1
            with torch.no_grad(): #在该模块下，requires_grad设置为False,反向传播时就不会自动求导了，因此大大节约了内存
                image, label = image.cuda(), label.cuda()
                B=image.size(0) #获取图像的高度

                smap= self.net(image) # 将数据传入模型，前向传播求出测试的值
                mae += Eval_mae(smap, label) #计算MAE
                F_value += Eval_fmeasure(smap, label) #计算F-Measure
                if ep%4==0: #Test文件中保存边缘灰度图
                    for b in range(B): #0-243
                        path = os.path.join(output_folder, prefix[b] + '.png') #边缘图路径
                        save_smap(smap[b, ...], path) #转换成边缘灰度图，保存其路径
        maxF=(F_value/num_test).max().item()
        meanF = (F_value / num_test).mean().item()
        mae=(mae/num_test)
        print('finish testing.', 'F—value : {:.4f}\t'.format(maxF),'mF—value : {:.4f}\t'.format(meanF),'MAE : {:.4f}\t'.format(mae))
        return maxF
#MAE
def Eval_mae(pred,gt):
    pred=pred.cuda()
    gt=gt.cuda()
    with torch.no_grad():
        mae = torch.abs(pred - gt).mean() #abs求绝对值
        if mae == mae:  # for Nan
            return mae.item() #item()返回的是一个浮点型数据，精度更高
#F-Measure
def Eval_fmeasure(pred,gt):
    beta2 = 0.3
    pred=pred.cuda()
    gt=gt.cuda()
    with torch.no_grad():
        pred = (pred - torch.min(pred)) / (torch.max(pred) -
                                           torch.min(pred) + 1e-20)
        prec, recall = _eval_pr(pred, gt, 255)
        f_score = (1 + beta2) * prec * recall / (beta2 * prec + recall)
        f_score[f_score != f_score] = 0  # for Nan
    return f_score#.max().item()

#求精度和召回率
def _eval_pr(y_pred, y, num):
    if y_pred.sum() == 0: # a negative sample
        y_pred = 1 - y_pred
        y = 1 - y

    prec, recall = torch.zeros(num).cuda(), torch.zeros(num).cuda()
    thlist = torch.linspace(0, 1 - 1e-10, num).cuda()

    for i in range(num):
        y_temp = (y_pred >= thlist[i]).float()
        tp = (y_temp * y).sum()
        prec[i], recall[i] = tp / (y_temp.sum() + 1e-20), tp / (y.sum() + 1e-20)
    return prec, recall


if __name__=='__main__':
    run=Run()
    run.train()
    # run.test(8)
