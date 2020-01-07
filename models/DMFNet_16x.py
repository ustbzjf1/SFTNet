import torch.nn as nn
import torch.nn.functional as F
import torch
import logging
import warnings
try:
    from .sync_batchnorm import SynchronizedBatchNorm3d
except:
    pass

# import functools


def normalization(planes, norm='bn'):
    if norm == 'bn':
        m = nn.BatchNorm3d(planes)
    elif norm == 'gn':
        m = nn.GroupNorm(4, planes)
    elif norm == 'in':
        m = nn.InstanceNorm3d(planes)
    elif norm == 'sync_bn':
        m = SynchronizedBatchNorm3d(planes)
    else:
        raise ValueError('normalization type {} is not supported'.format(norm))
    return m


class Conv3d_Block(nn.Module):
    def __init__(self,num_in,num_out,kernel_size=1,stride=1,g=1,padding=None,norm=None):
        super(Conv3d_Block, self).__init__()
        if padding == None:
            padding = (kernel_size - 1) // 2
        self.bn = normalization(num_in,norm=norm)
        self.act_fn = nn.ReLU(inplace=True)
        # self.act_fn = nn.PReLU() # report error : out of CUDA
        # self.act_fn = nn.ELU(inplace=True)
        self.conv = nn.Conv3d(num_in, num_out, kernel_size=kernel_size, padding=padding,stride=stride, groups=g, bias=False)

    def forward(self, x):# BN + Relu + Conv
        h = self.act_fn(self.bn(x))
        h = self.conv(h)
        return h


class DilatedConv3DBlock(nn.Module):
    # 3D Dilated in Spatial, reject dilating in Depth!
    def __init__(self, num_in, num_out, kernel_size=(1,1,1), stride=1, g=1, d=(1,1,1), norm=None):
        super(DilatedConv3DBlock, self).__init__()
        assert isinstance(kernel_size,tuple) and isinstance(d,tuple)

        padding = tuple(
            [(ks-1)//2 *dd for ks, dd in zip(kernel_size, d)]
        )

        self.bn = normalization(num_in, norm=norm)
        self.act_fn = nn.ReLU(inplace=True)
        # self.act_fn = nn.PReLU(num_in)
        # self.act_fn = nn.ELU(inplace=True)
        self.conv = nn.Conv3d(num_in,num_out,kernel_size=kernel_size,padding=padding,stride=stride,groups=g,dilation=d,bias=False)

    def forward(self, x):
        h = self.act_fn(self.bn(x))
        h = self.conv(h)
        return h


class MFUnit_A(nn.Module):
    def __init__(self, num_in, num_out, g=1, stride=1, d=(1,1),norm=None):
        """  The second 3x3x1 group conv is replaced by 3x3x3.
        :param num_in: number of input channels
        :param num_out: number of output channels
        :param g: groups of group conv.
        :param stride: 1 or 2
        :param d: tuple, d[0] for the first 3x3x3 conv while d[1] for the 3x3x1 conv
        :param norm: Batch Normalization
        """
        super(MFUnit_A, self).__init__()
        num_mid = num_in if num_in <= num_out else num_out
        self.conv1x1x1_in1 = Conv3d_Block(num_in,num_in//4,kernel_size=1,stride=1,norm=norm)
        self.conv1x1x1_in2 = Conv3d_Block(num_in//4,num_mid,kernel_size=1,stride=1,norm=norm)
        self.conv3x3x3_m1 = DilatedConv3DBlock(num_mid,num_out,kernel_size=(3,3,3),stride=stride,g=g,d=(d[0],d[0],d[0]),norm=norm) # dilated
        self.conv3x3x3_m2 = DilatedConv3DBlock(num_out,num_out,kernel_size=(3,3,1),stride=1,g=g,d=(d[1],d[1],1),norm=norm)
        # self.conv3x3x3_m2 = DilatedConv3DBlock(num_out,num_out,kernel_size=(1,3,3),stride=1,g=g,d=(1,d[1],d[1]),norm=norm)

        # skip connection
        if num_in != num_out or stride != 1:
            if stride == 1:
                self.conv1x1x1_shortcut = Conv3d_Block(num_in, num_out, kernel_size=1, stride=1, padding=0,norm=norm)
            if stride == 2:
                # if MF block with stride=2, 2x2x2
                self.conv2x2x2_shortcut = Conv3d_Block(num_in, num_out, kernel_size=2, stride=2,padding=0, norm=norm) # params

    def forward(self, x):
        x1 = self.conv1x1x1_in1(x)
        x2 = self.conv1x1x1_in2(x1)
        x3 = self.conv3x3x3_m1(x2)
        x4 = self.conv3x3x3_m2(x3)

        shortcut = x

        if hasattr(self,'conv1x1x1_shortcut'):
            shortcut = self.conv1x1x1_shortcut(shortcut)
        if hasattr(self,'conv2x2x2_shortcut'):
            shortcut = self.conv2x2x2_shortcut(shortcut)

        return x4 + shortcut


class MFUnit_add1(nn.Module):
    def __init__(self, num_in, num_out, g=1, stride=1,norm=None,dilation=None):

        super(MFUnit_add1, self).__init__()
        num_mid = num_in if num_in <= num_out else num_out

        self.conv1x1x1_in1 = Conv3d_Block(num_in, num_in // 4, kernel_size=1, stride=1, norm=norm)
        self.conv1x1x1_in2 = Conv3d_Block(num_in // 4,num_mid,kernel_size=1, stride=1, norm=norm)

        self.conv3x3x3_m1 = nn.ModuleList()
        if dilation == None:
            dilation = [1,2,3]
        for i in range(3):
            self.conv3x3x3_m1.append(
                DilatedConv3DBlock(num_mid,num_out, kernel_size=(3, 3, 3), stride=stride, g=g, d=(dilation[i],dilation[i], dilation[i]),norm=norm)
            )

        # It has not Dilated operation
        self.conv3x3x3_m2 = DilatedConv3DBlock(num_out, num_out, kernel_size=(3, 3, 1), stride=(1,1,1), g=g,d=(1,1,1), norm=norm)
        # self.conv3x3x3_m2 = DilatedConv3DBlock(num_out, num_out, kernel_size=(1, 3, 3), stride=(1,1,1), g=g,d=(1,1,1), norm=norm)

        # skip connection
        if num_in != num_out or stride != 1:
            if stride == 1:
                self.conv1x1x1_shortcut = Conv3d_Block(num_in, num_out, kernel_size=1, stride=1, padding=0, norm=norm)
            if stride == 2:
                self.conv2x2x2_shortcut = Conv3d_Block(num_in, num_out, kernel_size=2, stride=2, padding=0, norm=norm)

    def forward(self, x):
        x1 = self.conv1x1x1_in1(x)
        x2 = self.conv1x1x1_in2(x1)
        x3 = self.conv3x3x3_m1[0](x2)
        x3 += self.conv3x3x3_m1[1](x2)
        x3 += self.conv3x3x3_m1[2](x2)
        x4 = self.conv3x3x3_m2(x3)
        shortcut = x
        if hasattr(self, 'conv1x1x1_shortcut'):
            shortcut = self.conv1x1x1_shortcut(shortcut)
        if hasattr(self, 'conv2x2x2_shortcut'):
            shortcut = self.conv2x2x2_shortcut(shortcut)
        return x4 + shortcut

class MFUnit_add2(MFUnit_add1):
    # weighred add
    def __init__(self, num_in, num_out, g=1, stride=1,norm=None,dilation=None,layerlogs=None):
        super(MFUnit_add2, self).__init__(num_in, num_out, g, stride,norm,dilation)
        self.layerlogs = layerlogs
        self.weight1 = nn.Parameter(torch.ones(1))
        self.weight2 = nn.Parameter(torch.ones(1))
        self.weight3 = nn.Parameter(torch.ones(1))

    def forward(self, x):
        x1 = self.conv1x1x1_in1(x)
        x2 = self.conv1x1x1_in2(x1)
        x3 = self.weight1*self.conv3x3x3_m1[0](x2) + self.weight2*self.conv3x3x3_m1[1](x2) + self.weight3*self.conv3x3x3_m1[2](x2)
        # print(self.weight1.get_device())
        # if self.weight1.get_device() == 0 and self.layerlogs is not None:
        #     self.layerlogs.write('{}\t{}\t{}\n'.format(self.weight1.tolist()[0], self.weight2.tolist()[0], self.weight3.tolist()[0]))
        #     self.layerlogs.flush()
            # print(self.weight1.data[0],self.weight2.data[0],self.weight3.data[0])
            # self.layerlogs.info(self.weight1.data[0],self.weight2.data[0],self.weight3.data[0])
            # print(self.weight1.data,self.weight2.data,self.weight3.data)

        x4 = self.conv3x3x3_m2(x3)
        shortcut = x
        if hasattr(self, 'conv1x1x1_shortcut'):
            shortcut = self.conv1x1x1_shortcut(shortcut)
        if hasattr(self, 'conv2x2x2_shortcut'):
            shortcut = self.conv2x2x2_shortcut(shortcut)
        return x4 + shortcut

# (1) channels are set to 96 (2) The MF unit is all 3x3x3 without 1x3x3.
class BaseModel(nn.Module): # softmax
    # [96]   Flops:  13.361G  &  Params: 1.81M
    # [112]  Flops:  16.759G  &  Params: 2.46M
    # [128]  Flops:  20.611G  &  Params: 3.19M

    def __init__(self, c=4,n=32,channels=128,groups = 16,norm='bn', num_classes=4,output_func='sigmoid'):
        super(BaseModel, self).__init__()

        # Entry flow
        self.encoder_block1 = nn.Conv3d( c, n, kernel_size=3, padding=1, stride=2, bias=False)# H//2
        self.encoder_block2 = nn.Sequential(
            MFUnit_A(n, channels, g=groups, stride=2, norm=norm),# H//4 down
            (MFUnit_A(channels, channels, g=groups, stride=1, norm=norm)), # Dilated Conv 3
            (MFUnit_A(channels, channels, g=groups, stride=1, norm=norm))
        )
        #
        self.encoder_block3 = nn.Sequential(
            (MFUnit_A(channels, channels*2, g=groups, stride=2, norm=norm)), # H//8
            (MFUnit_A(channels * 2, channels * 2, g=groups, stride=1, norm=norm)),# Dilated Conv 3
            (MFUnit_A(channels * 2, channels * 2, g=groups, stride=1, norm=norm))
        )
        #  feature maps : 8x8x8 without dilated conv.
        self.encoder_block4 = nn.Sequential(# H//8,channels*4
            (MFUnit_A(channels*2, channels*3, g=groups, stride=2, norm=norm)), # H//16
            (MFUnit_A(channels*3, channels*3, g=groups, stride=1, norm=norm)),
            (MFUnit_A(channels*3, channels*2, g=groups, stride=1, norm=norm)), # before 2x upsample, we 2x reduce the channels respondingly.
        )

        self.upsample1 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False) # H//8
        self.decoder_block1 = MFUnit_A(channels*2+channels*2, channels*2, g=groups, stride=1, norm=norm)# before 2x upsample, we 2x reduce the channels respondingly.

        self.upsample2 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False) # H//4
        self.decoder_block2 = MFUnit_A(channels*2 + channels, channels, g=groups, stride=1, norm=norm) # before 2x upsample, we 2x reduce the channels respondingly.

        self.upsample3 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False) # H//2
        self.decoder_block3 = MFUnit_A(channels + n, n, g=groups, stride=1, norm=norm)
        self.upsample4 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False) # H
        self.seg = nn.Conv3d(n, num_classes, kernel_size=1, padding=0,stride=1,bias=False)

        output_func = output_func.lower()
        if output_func == 'sigmoid':
            self.output_func = nn.Sigmoid()
        elif output_func == 'softmax':
            self.output_func = nn.Softmax(dim=1)
        elif output_func == 'logsoftmax':
            self.output_func = nn.LogSoftmax(dim=1)
        else:
            raise ValueError

        # Initialization
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                torch.nn.init.torch.nn.init.kaiming_normal_(m.weight) #
                # torch.nn.init.torch.nn.init.xavier_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.GroupNorm) or isinstance(m, SynchronizedBatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Encoder
        x1 = self.encoder_block1(x)# H//2 down
        x2 = self.encoder_block2(x1)# H//4 down
        x3 = self.encoder_block3(x2)# H//8 down
        x4 = self.encoder_block4(x3) # H//16
        # Decoder
        y1 = self.upsample1(x4)# H//8
        y1 = torch.cat([x3,y1],dim=1)
        y1 = self.decoder_block1(y1)

        y2 = self.upsample2(y1)# H//4
        y2 = torch.cat([x2,y2],dim=1)
        y2 = self.decoder_block2(y2)

        y3 = self.upsample3(y2)# H//2
        y3 = torch.cat([x1,y3],dim=1)
        y3 = self.decoder_block3(y3)
        y4 = self.upsample4(y3)
        y4 = self.seg(y4)
        if hasattr(self,'output_func') and self.output_func is not None:
            y4 = self.output_func(y4)
        return y4


class DilatedMFNet(BaseModel): # softmax
    # [96]   Flops:  17.091G  &  Params: 2.2M
    # [112]  Flops:  21.749G  &  Params: 2.98M
    # [128]  Flops:  27.045G  &  Params: 3.88M
    def __init__(self, c=4,n=32,channels=128, groups=16,norm='bn', num_classes=4,output_func='sigmoid'):
        super(DilatedMFNet, self).__init__(c,n,channels,groups, norm, num_classes,output_func)

        self.logger_layer1 = None
        self.logger_layer2 = None
        self.logger_layer3 = None
        self.logger_layer4 = None
        self.logger_layer5 = None
        self.logger_layer6 = None

        # self.logger_layer1 =  open( r'./layerlogger/logger_layer1.txt','w')
        # self.logger_layer2 =  open( r'./layerlogger/logger_layer2.txt','w')
        # self.logger_layer3 =  open( r'./layerlogger/logger_layer3.txt','w')
        # self.logger_layer4 =  open( r'./layerlogger/logger_layer4.txt','w')
        # self.logger_layer5 =  open( r'./layerlogger/logger_layer5.txt','w')
        # self.logger_layer6 =  open( r'./layerlogger/logger_layer6.txt','w')

        self.encoder_block2 = nn.Sequential(
            MFUnit_add2(n, channels, g=groups, stride=2, norm=norm,dilation=[1,2,3],layerlogs=self.logger_layer1),# H//4 down
            # after the block C, the channels would be 3x changing (Deprecated).
            MFUnit_add2(channels, channels, g=groups, stride=1, norm=norm,dilation=[1,2,3],layerlogs=self.logger_layer2), # Dilated Conv 3
            MFUnit_add2(channels, channels, g=groups, stride=1, norm=norm,dilation=[1,2,3],layerlogs=self.logger_layer3)
        )

        self.encoder_block3 = nn.Sequential(
            MFUnit_add2(channels, channels*2, g=groups, stride=2, norm=norm,dilation=[1,2,3],layerlogs=self.logger_layer4), # H//8
            MFUnit_add2(channels * 2, channels * 2, g=groups, stride=1, norm=norm,dilation=[1,2,3],layerlogs=self.logger_layer5),# Dilated Conv 3
            MFUnit_add2(channels * 2, channels * 2, g=groups, stride=1, norm=norm,dilation=[1,2,3],layerlogs=self.logger_layer6)
        )


if __name__ == '__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = torch.device('cuda:0')
    x = torch.rand((1,4,32,32,32),device=device) # [bsize,channels,H,W,Depth] [bsize,channels,H,W,D]
    # model = MF_VNet_16x_Dilated_A(c=4, groups=16, norm='bn', num_classes=4)
    model = DilatedMFNet(c=4, groups=16, norm='bn', num_classes=4)
    model.cuda(device)
    y = model(x)
    print(y.shape)
