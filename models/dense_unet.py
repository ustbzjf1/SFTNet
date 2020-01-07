import torch.nn as nn
import torch.nn.functional as F
import torch
import logging
import warnings
try:
    from sync_batchnorm import SynchronizedBatchNorm3d
except:
    from .sync_batchnorm import SynchronizedBatchNorm3d

# import functools
def normalization(planes, norm='sync_bn'):
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

class Transition_down(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.bn = normalization(c_in)
        self.relu = nn.ReLU(True)
        self.conv = nn.Conv3d(c_in, c_out, 3, 1, 1)
        self.drop = nn.Dropout3d(0.5, True)
        self.maxpool = nn.MaxPool3d(2)

    def forward(self, x):
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv(x)
        x = self.drop(x)
        out = self.maxpool(x)

        return out


class Convolution_layer(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.bn = normalization(c_in)
        self.relu = nn.ReLU(True)
        self.conv = nn.Conv3d(c_in, c_out, 3, 1, 1)
        self.drop = nn.Dropout3d(0.5, True)

    def forward(self, x):
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv(x)
        out = self.drop(x)

        return out


class Dense_block(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.layer1 = Convolution_layer(c_in, c_out)
        self.layer2 = Convolution_layer(c_out, c_out)
        self.layer3 = Convolution_layer(c_out, c_out)
        self.layer4 = Convolution_layer(c_out, c_out)

    def forward(self, x):
        layer1 = self.layer1(x)
        layer1_add = x+layer1
        layer2 = self.layer2(layer1_add)
        layer2_add = layer1_add + layer2
        layer3 = self.layer3(layer2_add)
        layer3_add = layer2_add+layer3
        layer4 = self.layer4(layer3_add)
        layer4_add = layer3_add+layer4
        out = layer4_add+layer1+layer2+layer3

        return out


class Transition_up(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.bn = normalization(c_in)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.ConvTranspose3d(c_in, c_out, 2, 2, 0)
        self.drop = nn.Dropout3d(0.5, True)

    def forward(self, x):
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv(x)
        out = self.drop(x)

        return out


class Dense_unet(nn.Module):
    def __init__(self, c_in, num_classes):
        super().__init__()

        self.encoder1_1 = Convolution_layer(c_in, 64)
        self.encoder1_2 = Dense_block(64, 64)

        self.encoder2_1 = Transition_down(64, 128)
        self.encoder2_2 = Dense_block(128, 128)

        self.encoder3_1 = Transition_down(128, 256)
        self.encoder3_2 = Dense_block(256, 256)

        self.encoder4_1 = Transition_down(256, 512)
        self.encoder4_2 = Dense_block(512, 512)

        self.decoder1_1 = Transition_up(512, 256)
        self.decoder1_2 = Dense_block(256, 256)

        self.decoder2_1 = Transition_up(256, 128)
        self.decoder2_2 = Dense_block(128, 128)

        self.decoder3_1 = Transition_up(128, 64)
        self.decoder3_2 = Dense_block(64, 64)

        self.decoder4 = Convolution_layer(64, num_classes)

        self.activation = nn.Softmax(1)

    def forward(self, x):
        out1 = self.encoder1_1(x)
        out1_1 = self.encoder1_2(out1)
        out_add1 = out1+out1_1

        out2 = self.encoder2_1(out_add1)
        out2_1 = self.encoder2_2(out2)
        out_add2 = out2+out2_1

        out3 = self.encoder3_1(out_add2)
        out3_1 = self.encoder3_2(out3)
        out_add3 = out3+out3_1

        out4 = self.encoder4_1(out_add3)
        out4 = self.encoder4_2(out4)
        out4 = self.decoder1_1(out4)

        decoder1 = self.decoder1_2(out_add3+out4)
        decoder1 = self.decoder2_1(decoder1)

        decoder2 = self.decoder2_2(decoder1+out_add2)
        decoder2 = self.decoder3_1(decoder2)

        decoder3 = self.decoder3_2(decoder2+out_add1)
        decoder4 = self.decoder4(decoder3)

        out = self.activation(decoder4)

        return out


if __name__ == '__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = torch.device('cuda:0')
    x = torch.rand((1, 4, 128, 128, 128), device=device) # [bsize,channels,H,W,Depth] [bsize,channels,H,W,D]
    # model = MF_VNet_16x_Dilated_A(c=4, groups=16, norm='bn', num_classes=4)
    model = Dense_unet(4, 4)
    model.cuda(device)
    y = model(x)
    print(y.shape)
