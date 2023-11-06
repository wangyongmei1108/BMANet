import math
import scipy.stats as st
from BMANet.utils import *
from BMANet.modules import *
import torch.nn.functional as F

class SpatialAttention(nn.Module):
    def __init__(self,k=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=k, padding=k//2,bias=False) # infer a one-channel attention map
    def forward(self, ftr):
        # ftr: [B, C, H, W]
        ftr_avg = torch.mean(ftr, dim=1, keepdim=True) # [B, 1, H, W], average  keepdim=True保持维度不变
        ftr_max, _ = torch.max(ftr, dim=1, keepdim=True) # [B, 1, H, W], max,返回最大值以及最大值对应的索引
        ftr_cat = torch.cat([ftr_avg, ftr_max], dim=1) # [B, 2, H, W]
        att_map = F.sigmoid(self.conv(ftr_cat)) # [B, 1, H, W] Conv3*3*1--单通道特征图，Sigmoid
        out=att_map*ftr
        return out


class ChannelAttention(nn.Module):# 64，64，4
    def __init__(self, in_planes,g):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1) #输出为1*1
        self.max_pool = nn.AdaptiveMaxPool2d(1) #输出为1*1   FC前后大小不变
        # 1*1组卷机，尺寸不变
        self.fc1 = nn.Conv2d(in_planes, in_planes, 1, groups=g, bias=False)
        self.relu1 = nn.LeakyReLU()
        self.fc2 = nn.Conv2d(in_planes, in_planes, 1, groups=g, bias=False)

        self.sigmoid = nn.Sigmoid()   # 输出不变：b*64--b*16--ReLu--b*64

    def forward(self, x):
        ym = self.max_pool(x)  # 1*1*64
        ya = self.avg_pool(x)  # 1*1*64
        # 一维组卷机
        avg_out = self.fc2(self.relu1(self.fc1(ya)))  # 1*1*64
        max_out = self.fc2(self.relu1(self.fc1(ym)))  # 1*1*64
        out = self.sigmoid(avg_out + max_out)  # 1*1*64
        return out*x


class BRAM(nn.Module):
    def __init__(self, in_planes,g):
        super(BRAM, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1) #输出为1*1
        self.max_pool = nn.AdaptiveMaxPool2d(1) #输出为1*1   FC前后大小不变
        # 1*1组卷机，尺寸不变
        self.fc1 = nn.Conv2d(in_planes, in_planes, 1, groups=g, bias=False)
        self.relu1 = nn.LeakyReLU()
        self.fc2 = nn.Conv2d(in_planes, in_planes, 1, groups=g, bias=False)
        self.sigmoid = nn.Sigmoid()   # 输出不变：b*64--b*16--ReLu--b*64

        self.conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, padding=3,bias=False)
        self.la = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False), nn.Sigmoid())
        self.c3 = nn.Sequential(
            nn.Conv2d(in_planes, in_planes, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_planes),
            nn.LeakyReLU(inplace=True)

        )
        self.c5 = nn.Sequential(
            nn.Conv2d(in_planes, in_planes, 5, padding=2, bias=False),
            nn.BatchNorm2d(in_planes),
            nn.LeakyReLU(inplace=True)

        )
        self.resb=ResidualsBlock(64)
        self.ca=ChannelAttention(64,8)
        self.sa=SpatialAttention(k=7)
        self.eca=eca_layer(64)
        self.convchannel = nn.Sequential(
            nn.Conv2d(in_planes, 1, 1,  bias=False),
            nn.BatchNorm2d(1),
            nn.LeakyReLU(inplace=True)

        )
        self.convchannel3 = nn.Sequential(
            nn.Conv2d(in_planes, 1, 3,stride=1,padding=1, bias=False),
            nn.BatchNorm2d(1),
            nn.LeakyReLU(inplace=True)

        )
        self.gsconv=nn.Sequential(
            GSConv(in_planes, in_planes),
            nn.BatchNorm2d(in_planes),
            nn.LeakyReLU(inplace=True)
        )
        # self.resblock=ResBlock(128,64)
        self.gamma = nn.Parameter(torch.zeros(1))


    def forward(self, x):
        #改进十五--二--5
        x1=self.c3(x)
        x2=self.c5(x)
        ym1 = self.max_pool(x1)  # 1*1*64
        ya1 = self.avg_pool(x1)  # 1*1*64
        # 一维组卷机
        avg_out1 = self.fc2(self.relu1(self.fc1(ya1)))  # 1*1*64
        max_out1= self.fc2(self.relu1(self.fc1(ym1)))  # 1*1*64
        out1_3 = self.sigmoid(avg_out1 + max_out1)
        ym2 = self.max_pool(x2)  # 1*1*64
        ya2 = self.avg_pool(x2)  # 1*1*64
        # 一维组卷机
        avg_out2 = self.fc2(self.relu1(self.fc1(ya2)))  # 1*1*64
        max_out2 = self.fc2(self.relu1(self.fc1(ym2)))  # 1*1*64
        out1_5 = self.sigmoid(avg_out2 + max_out2)
        out1=self.resb(out1_3*out1_5)
        #over

        x_avg = torch.mean(x, dim=1, keepdim=True)  # [B, 1, H, W], average  keepdim=True保持维度不变
        x_max, _ = torch.max(x, dim=1, keepdim=True) # [B, 1, H, W], max,返回最大值以及最大值对应的索引
        # x_avg =self.gsconv(x)
        # x_max=self.gsconv(x)
        x_cat = torch.cat([x_avg, x_max], dim=1) # [B, 2, H, W]
        out2 = F.sigmoid(self.conv(x_cat))

        out = out1*out2+x
        out=self.la(out)*x

        return out



class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


# 映射函数
class MappingModule(nn.Module):
    def __init__(self, out_c):  # out_c=64
        super(MappingModule, self).__init__()


        self.cv1_3 = nn.Sequential(
            nn.Conv2d(out_c, out_c, kernel_size=3, stride=1,
                      padding=1, dilation=1),
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU(inplace=True)
        )
        self.cv1_1 = nn.Sequential(
            nn.Conv2d(out_c, out_c, kernel_size=1),
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, out1, out2):#out1:高级特征，尺寸大---3*3
        out1_size = out1.size()[2:]
        out2 = F.interpolate(out2, size=out1_size, mode="bilinear", align_corners=True)
        o2_1 = self.cv1_3(out1)  # Conv3*3+BN+ReLu
        o2_2 = self.cv1_1(out2)  # Conv1*1+BN+ReLu
        out = o2_1 + o2_2
        return out

class ResidualsBlock(nn.Module):
    def __init__(self, in_channel):
        super(ResidualsBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(in_channel),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channel, in_channel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channel)
        )


    def forward(self, x):
        out = self.block(x)
        return F.relu(out + x)



#通道为128,64
class ResBlock(nn.Module):
    def __init__(self, channel,in_channel):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channel, in_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(in_channel),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channel, in_channel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channel)
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(channel, in_channel, kernel_size=1),
            nn.BatchNorm2d(in_channel),
            nn.LeakyReLU(inplace=True)
        )
        # self.Ghost=GhostModule(128,64)
    def forward(self, x):
        # x1=self.Ghost(x)
        out = self.block(x)
        return F.relu(out + self.conv1(x))


#残差细化模块
class RRM(nn.Module):
    def __init__(self, in_channel):
        super(RRM, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(in_channel),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channel, in_channel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channel)
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, kernel_size=1),
            nn.BatchNorm2d(in_channel),
            nn.LeakyReLU(inplace=True)
        )


    def forward(self, x):
        x1=self.conv1(x)
        out = self.block(x)
        return F.relu(out + x1)

def gkern(kernlen=16, nsig=3):
    interval = (2*nsig+1.)/kernlen
    x = np.linspace(-nsig-interval/2., nsig+interval/2., kernlen+1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw/kernel_raw.sum()
    return kernel

def min_max_norm(in_):
    max_ = in_.max(3)[0].max(2)[0].unsqueeze(2).unsqueeze(3).expand_as(in_)
    min_ = in_.min(3)[0].min(2)[0].unsqueeze(2).unsqueeze(3).expand_as(in_)
    in_ = in_ - min_
    return in_.div(max_-min_+1e-8)


#改进二十一的PA
class PA(nn.Module):
    def __init__(self):
        super(PA, self).__init__()
        self.conv1=convbnrelu(128, 64)
        self.conv2=convbnrelu(128, 64)
        self.conv3=convbnrelu(128, 64)
        self.conv4=convbnrelu(128, 64)
        self.conv5=convbnrelu(128, 64)
        self.conv6=convbnrelu(128, 64)
        self.bram=BRAM(64,8)
        self.resblock=ResBlock(128,64)
        self.c3 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True)

        )
        self.c1 = nn.Sequential(
            nn.Conv2d(128, 64, 1, padding=0, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True)

        )
        self.resblock1 = ResBlock(64, 64)
        self.cpool = nn.Sequential(
            nn.Conv2d(128, 64, 1, stride=2,padding=0, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True)

        )
    def forward(self, x1,x2,x3,x4):
        #原版本
        p1=torch.cat([x1,US2(x2)],dim=1)
        p1=self.c1(p1)

        #中间层
        p2 = torch.cat([x2, US2(x3)], dim=1)
        p2=self.c1(p2)

        #深层操作 CA
        p3 = torch.cat([x3, US2(x4)], dim=1)
        fea_size = p2.size()[2:]
        p3 = F.interpolate(p3, size=fea_size, mode="bilinear", align_corners=True)
        p3=self.c1(p3)

        p23=p2*p3+p3
        p123=self.resblock1(p23)
        fea_size1 = p1.size()[2:]
        p123 = F.interpolate(p123, size=fea_size1, mode="bilinear", align_corners=True)
        p123=p123*p1+p123
        fuse=self.resblock1(p123) #resblock1
        out=self.bram(fuse)
        return out
