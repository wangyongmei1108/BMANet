from torch import nn
from torch import Tensor
from BMANet.modules import *
from BMANet.attention import *
import torch
import os
import math

"""
IMPORTANT:
To adapt it to the SOD task, weremove the global average pooling layer and the last fully-connected layer from the backbon
"""

__all__ = ["MobileNetV2"]

model_urls = {
    "mobilenet_v2": "https://download.pytorch.org/models/mobilenet_v2-b0353104.pth",
}

#MSCE+APFA
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.d1=SEME(64,80)
        self.d2 = SEME(64, 88)
        self.d3 = SEME(64, 96)
        self.d4 = SEME(64, 416)

        self.agg=PA()

        #自己加入-改进三
        # Semantic Knowledge Compression(SKC) unit, k3 and k4
        self.conv5 = DSConv3x3(320, 64, stride=1)  #压缩通道

        self.pool = nn.AdaptiveAvgPool2d(5)              #压缩分辨率

    def forward(self, F1,F2,F3,F4,F5):
        kernel_conv5 = self.pool(self.conv5(F5))  # 64*5*5

        P4 = torch.cat([F4, US2(F5)], dim=1) #F52倍上采样，与F4通道连接 P4：14*14*416
        P4 = self.d4(P4,kernel_conv5)                #用MSCE融合 P4:14*14*64

        P3 = torch.cat([F3, US2(P4)], dim=1)  #P3：28*28*96
        P3 = self.d3(P3,kernel_conv5)                    #P3:28*28*64

        P2 = torch.cat([F2, US2(P3)], dim=1) #P2:56*56*88
        P2 = self.d2(P2,kernel_conv5)                       #P2:56*56*64

        P1 = torch.cat([F1, US2(P2)], dim=1) #P1:112*112*80
        P1 = self.d1(P1,kernel_conv5)                      #112*112*64

        #APFA
        S=self.agg(P1,P2,P3,P4)
        return S

mob_conv1_2 = mob_conv2_2 = mob_conv3_3 = mob_conv4_3 = mob_conv5_3 = None
#hook可以修改input和output，但是不会影响forward的结果，根据该层的输入，提取模型的某一层（不是最后一层）的输出特征
def conv_1_2_hook(module, input, output):
    global mob_conv1_2
    mob_conv1_2 = output
    return None


def conv_2_2_hook(module, input, output):
    global mob_conv2_2
    mob_conv2_2 = output
    return None


def conv_3_3_hook(module, input, output):
    global mob_conv3_3
    mob_conv3_3 = output
    return None


def conv_4_3_hook(module, input, output):
    global mob_conv4_3
    mob_conv4_3 = output
    return None


def conv_5_3_hook(module, input, output):
    global mob_conv5_3
    mob_conv5_3 = output
    return None

#提取特征
class MobileNet(nn.Module):
    def __init__(self):
        super(MobileNet,self).__init__()
        self.mbv = models.mobilenet_v2(pretrained=True).features
        #fetures里有19个元素，取前18个，去掉了特征提取阶段的Conv2d1*1
        #mbv[1]:112*112*32 mbv[3]:56*56*24 mbv[6]:28*28*32 mbv[13]:14*14*96 mbv[17]:7*7*320
        self.mbv[1].register_forward_hook(conv_1_2_hook) #获取mbv的中间结果，即获取mbv[1]的结果mod_conv1_2
        self.mbv[3].register_forward_hook(conv_2_2_hook)
        self.mbv[6].register_forward_hook(conv_3_3_hook)
        self.mbv[13].register_forward_hook(conv_4_3_hook)
        self.mbv[17].register_forward_hook(conv_5_3_hook)

    def forward(self, x: Tensor) -> Tensor:
        global mob_conv1_2, mob_conv2_2, mob_conv3_3, mob_conv4_3, mob_conv5_3#global是在函数内部使用，在函数内部给一个在外部定义的变量赋值时就要用global先声明一下
        self.mbv(x)

        return mob_conv1_2, mob_conv2_2, mob_conv3_3, mob_conv4_3, mob_conv5_3
#               112*112*32  56*56*24      28*28*32     14*14*96     7*7*320
#MSCNet
class MobileNetV2(nn.Module):
    def __init__(self):

        super(MobileNetV2, self).__init__()
        #self.mbv = MobileNet()

        self.encoder=MobileNet()
        self.decoder = Decoder()
        self.head = nn.ModuleList([])
        for i in range(1):
            self.head.append(SalHead(64,3))


    def forward(self, x):
        #f1,f2,f3,f4,f5 = self.mbv(x)

        f1, f2, f3, f4, f5 = self.encoder(x)# f1:112*112*32  f2:56*56*24 f3:28*28*32 f4:14*14*96 f5:7*7*320
        S= self.decoder(f1,f2,f3,f4,f5)  # S:112*112*64
        sm = self.head[0](US2(S))        #sm:224*224*64

        return sm



