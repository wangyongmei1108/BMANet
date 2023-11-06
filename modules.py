from BMANet.utils import *
import torch.nn as nn
import torch


class ConvBlock(nn.Module):
    def __init__(self, ic, oc, ks, use_bn, nl):
        # ic: input channels
        # oc: output channels
        # ks: kernel size
        # use_bn: True or False
        # nl: type of non-linearity, 'Non' or 'ReLU' or 'Sigmoid'
        super(ConvBlock, self).__init__()
        assert ks in [1, 3, 5, 7]
        assert isinstance(use_bn, bool)
        assert nl in ['None', 'ReLU', 'Sigmoid']
        self.use_bn = use_bn
        self.nl = nl
        if ks == 1:
            self.conv = nn.Conv2d(ic, oc, kernel_size=1, bias=False)
        else:
            self.conv = nn.Conv2d(ic, oc, kernel_size=ks, padding=(ks - 1) // 2, bias=False)
        if self.use_bn == True:
            self.bn = nn.BatchNorm2d(oc)
        if self.nl == 'ReLU':
            self.ac = nn.LeakyReLU(inplace=True)
        if self.nl == 'Sigmoid':
            self.ac = nn.Sigmoid()

    def forward(self, x):
        y = self.conv(x)
        if self.use_bn == True:
            y = self.bn(y)
        if self.nl != 'None':
            y = self.ac(y)
        return y


class SalHead(nn.Module):
    def __init__(self, in_channels, inter_ks):
        super(SalHead, self).__init__()
        self.conv_1 = ConvBlock(in_channels, in_channels // 2, inter_ks, True, 'ReLU')
        self.conv_2 = ConvBlock(in_channels // 2, in_channels // 2, 3, True, 'ReLU')
        self.conv_3 = ConvBlock(in_channels // 2, in_channels // 8, 3, True, 'ReLU')
        self.conv_4 = ConvBlock(in_channels // 8, 1, 1, False, 'Sigmoid')

    def forward(self, dec_ftr):
        dec_ftr_ups = dec_ftr
        outputs = self.conv_4(self.conv_3(self.conv_2(self.conv_1(dec_ftr_ups))))
        return outputs


# Conv1*1+BN+ReLu            卷积核个数：64，
class convbnrelu(nn.Module):
    def __init__(self, in_channel, out_channel, k=1, s=1, p=0, g=1, d=1, bias=False, bn=True, relu=True):
        super(convbnrelu, self).__init__()
        conv = [nn.Conv2d(in_channel, out_channel, k, s, p, dilation=d, groups=g, bias=bias)]
        if bn:
            conv.append(nn.BatchNorm2d(out_channel))
        if relu:
            conv.append(nn.LeakyReLU(inplace=True))
        self.conv = nn.Sequential(*conv)

    def forward(self, x):
        return self.conv(x)


class DSConv3x3(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1, dilation=1, relu=True):
        super(DSConv3x3, self).__init__()
        self.conv = nn.Sequential(
            convbnrelu(in_channel, in_channel, k=3, s=stride, p=dilation, d=dilation, g=in_channel),
            convbnrelu(in_channel, out_channel, k=1, s=1, p=0, relu=relu)
        )

    def forward(self, x):
        return self.conv(x)


# GSConv
def autopad(k, p=None, d=1):  # kernel, padding, dilation
    # Pad to 'same' shape outputs
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    # Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        # self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class GSConv(nn.Module):
    # GSConv https://github.com/AlanLi1997/slim-neck-by-gsconv
    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):
        super().__init__()
        c_ = c2 // 2
        self.cv1 = Conv(c1, c_, k, s, None, g, act)
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, act)

    def forward(self, x):
        x1 = self.cv1(x)
        x2 = torch.cat((x1, self.cv2(x1)), 1)
        # shuffle
        b, n, h, w = x2.data.size()
        b_n = b * n // 2
        y = x2.reshape(b_n, 2, h * w)
        y = y.permute(1, 0, 2)
        y = y.reshape(2, -1, n // 2, h, w)
        return torch.cat((y[0], y[1]), 1)

# 改进十一--七--20
class SEME(nn.Module):
    def __init__(self, inc1, inc2, size=0):  # 64,  416，96，88，80
        super(SEME, self).__init__()
        self.s = size
        self.inc1 = inc1
        self.inc2 = inc2
        self.conv1 = convbnrelu(inc2, inc1)
        self.conv2 = convbnrelu(inc2, inc1)
        self.conv3 = convbnrelu(inc2, inc1)
        self.c0 = nn.Sequential(
            nn.Conv2d(inc1, inc1, 3, padding=1, bias=False),
            nn.BatchNorm2d(inc1),
            nn.LeakyReLU(inplace=True)

        )
        #   416   64
        self.conv = nn.Conv2d(inc2, inc1, kernel_size=1)
        self.conv0 = nn.Sequential(convbnrelu(2 * inc1, inc1), convbnrelu(inc1, inc1, k=3, p=1))  # Conv1*1+Conv3*3
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.maxpool = nn.MaxPool2d(1)

        self.conv0 = nn.Sequential(convbnrelu(2 * inc1, inc1), convbnrelu(inc1, inc1, k=3, p=1))
        self.attconv = convbnrelu(inc1, inc1, k=3, p=1)
        self.c05 = nn.Sequential(
            nn.Conv2d(inc1, inc1, 5, padding=2, bias=False),
            nn.BatchNorm2d(inc1),
            nn.LeakyReLU(inplace=True)

        )
        self.attconv5 = convbnrelu(inc1, inc1, k=5, p=2)
        self.c01 = nn.Sequential(
            nn.Conv2d(inc1, inc1, 1, padding=0, bias=False),
            nn.BatchNorm2d(inc1),
            nn.LeakyReLU(inplace=True)

        )
        # self.SEM = eca_layer(64)
        self.SEM = SELayer(64)



    def forward(self, x1, k):
        if self.s == 1:  # s=0
            x1 = F.interpolate(x1, scale_factor=0.5, mode='bilinear', align_corners=True)
        # 改进十一--七--20

        fea_size = x1.size()[2:]  # 提取输入特征尺寸
        kse = F.interpolate(k, size=fea_size, mode="bilinear", align_corners=True)

        va = self.avg(x1)  # 全局平均池化,输出为1*1
        vm = self.maxpool(x1)  # 最大池化，池化窗口为1*1，输出不变 14*14
        fuseadd = self.conv(va) + self.conv(vm)

        fuseadd = fuseadd * kse + fuseadd

        att = torch.sigmoid(fuseadd)

        eadd = self.c0(fuseadd)
        A1 = torch.sigmoid(self.attconv(eadd))

        att = self.SEM(att)

        f1 = self.conv1(x1)  # Conv1*1+BN+ReLu

        f2 = self.conv2(x1)  # Conv1*1+BN+ReLu
        out1 = self.c0(f1)  # Conv3*3+BN+ReLu

        out1 = out1 * kse + out1

        A2 = torch.sigmoid(self.attconv(out1))

        n_b, n_c, n_b, n_w = f2.shape
        B_i, C_i, H_i, W_i = k.size()
        out2 = f2.clone()
        for i in range(1, B_i):
            kernel_i = k[i, :, :, :]  # 大小为64*5*5
            kernel4 = kernel_i.view(C_i, 1, H_i, W_i)  # [64,1,5,5]  64种1通道的5*5卷积核
            # DDconv          输入样本：一张416通道的14*14的图像
            f2_r1 = F.conv2d(f2[i, :, :, :].view(1, C_i, n_b, n_w), kernel4, stride=1, padding=2, dilation=1,
                             groups=C_i)  # 14*14*64
            f2_r2 = F.conv2d(f2[i, :, :, :].view(1, C_i, n_b, n_w), kernel4, stride=1, padding=4, dilation=2,
                             groups=C_i)  # 14*14*64
            f2_r3 = F.conv2d(f2[i, :, :, :].view(1, C_i, n_b, n_w), kernel4, stride=1, padding=6, dilation=3,
                             groups=C_i)  # 14*14*64
            out2[i, :, :, :] = f2_r1 + f2_r2 + f2_r3  # 14*14*64

        out2 = out2 * kse + out2

        out2conv = self.c05(out2)
        # 交叉相乘
        fuseout2 = A1 * A2 * out2conv
        finout2 = self.c05(fuseout2)
        eout2 = finout2 + out2conv + f2
        out2 = self.c05(eout2)

        out = torch.cat([out1, out2], dim=1)
        out = self.conv0(out)
        x1 = att * out * A1
        return x1


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


# CWA
class WeightedBlock(nn.Module):
    """Weighted Block
    """

    def __init__(self, in_channels, out_channels=64):  # 都为64
        super().__init__()
        self.input_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=1, bias=False),  # 1x1
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
        )
        self.weight = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),  # 输出尺寸为1*1
            nn.Conv2d(out_channels, out_channels,
                      kernel_size=1, bias=False),  # 1x1
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels,
                      kernel_size=1, bias=False),  # 1x1
            nn.BatchNorm2d(out_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        input_conv = self.input_conv(x)
        return input_conv * self.weight(input_conv)


# ECA
class eca_layer(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """

    def __init__(self, channel, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)
        x = x * y.expand_as(x)

        return x * y.expand_as(x)
