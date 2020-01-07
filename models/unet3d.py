import torch.nn as nn
import torch.nn.functional as F
import torch as t
from functools import partial
import logging
import warnings
try:
    from .sync_batchnorm import SynchronizedBatchNorm3d
except:
    from sync_batchnorm import SynchronizedBatchNorm3d

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
class GCN(nn.Module):
    """ Graph convolution unit (single layer)
    """

    def __init__(self, num_state, num_node, bias=False):
        super(GCN, self).__init__()
        self.conv1 = nn.Conv1d(num_node, num_node, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(num_state, num_state, kernel_size=1, bias=bias)

    def forward(self, x):
        # (n, num_state, num_node) -> (n, num_node, num_state)
        #                          -> (n, num_state, num_node)
        h = self.conv1(x.permute(0, 2, 1).contiguous()).permute(0, 2, 1)
        h = h + x
        # (n, num_state, num_node) -> (n, num_state, num_node)
        h = self.conv2(self.relu(h))
        return h

class SFT_adaptive(nn.Module):
    '''
    The spatial feature temporal unit
    input: b, c, h, w, d
    '''
    def __init__(self, in_c, in_h, in_t, ConvND = nn.Conv3d, normalize=None, mode=None):
        super().__init__()
        self.node_f = in_c  #32
        self.state_f = in_c  #32
        self.node_t = 128
        self.state_t = 128
        self.in_h = self.in_w = in_h
        self.normalize = normalize
        self.mode = mode

        self.S_project =ConvND(in_c, in_c, kernel_size=3, padding=1, stride=2, groups=in_c)
        self.phi_s = ConvND(in_channels=self.in_h//2, out_channels=self.in_h//2, kernel_size=1)
        self.v = ConvND(in_channels=self.in_w//2, out_channels=self.in_w//2, kernel_size=1)
        self.delta = ConvND(in_c, in_c, 1)
        self.Ws = ConvND(in_c, in_c, 1)
        self.xi = ConvND(in_c, in_c, 1)
        self.sigmoid = nn.Sigmoid()

        self.phi_f = ConvND(in_channels=in_c, out_channels=self.state_f, kernel_size=1)
        self.theta_f = ConvND(in_channels=in_c, out_channels=self.node_f, kernel_size=1)

        self.phi_t = ConvND(in_channels=in_t, out_channels=self.state_t, kernel_size=1)
        self.theta_t = ConvND(in_channels=in_t, out_channels=self.node_t, kernel_size=1)

        self.GCN_f = GCN(num_state=self.state_f, num_node=self.node_f)
        self.GCN_t = GCN(num_state=self.state_t, num_node=self.node_t)

        self.extend_f = nn.Sequential(ConvND(self.state_f, in_c, kernel_size=1),
                                      normalization(in_c))
        self.extend_t = nn.Sequential(ConvND(self.state_t, in_t, kernel_size=1),
                                      normalization(in_t))

        self.weight1 = nn.Parameter(t.ones(1))
        self.weight2 = nn.Parameter(t.ones(1))
        self.weight3 = nn.Parameter(t.ones(1))

    def forward(self, x):
        b, c, h, w, d = x.size()
        s_in = f_in = x  #(b, c, h, w, d)
        t_in = x.permute(0, 4, 1, 2, 3).contiguous()  #(b, d, c, h, w)

        

        '''the feature branch'''
        phi_f = self.phi_f(f_in).view(b, self.state_f, -1)  #(b, state_f, d*h*w)
        theta_f = self.theta_f(f_in).view(b, self.node_f, -1)  #(b, node_f, d*h*w)
        graph_f = t.matmul(phi_f, theta_f.permute(0, 2, 1))  #(b, state_f, node_f)
        if self.normalize:
            graph_f = graph_f * (1. / graph_f.size(2))
        out_f = self.GCN_f(graph_f)  #(b, state_f, node_f)
        out_f = t.matmul(out_f, theta_f).view(b, self.state_f, *x.size()[2:])  #(b, state_f, h, w, d)
        out_f = self.extend_f(out_f)  #(b, c, h, w, d)

        '''the temporal branch'''
        phi_t = self.phi_t(t_in).view(b, self.state_t, -1)  # (b, state_t, c*h*w)
        theta_t = self.theta_t(t_in).view(b, self.node_t, -1)  # (b, node_t, c*h*w)
        graph_t = t.matmul(phi_t, theta_t.permute(0, 2, 1))  # (b, state_t, node_t)
        if self.normalize:
            graph_t = graph_t * (1. / graph_t.size(2))
        out_t = self.GCN_t(graph_t)  # (b, state_t, node_t)
        out_t = t.matmul(out_t, theta_t).view(b, self.state_t, *x.size()[1:4])  # (b, state_t, c, h, w)
        out_t = self.extend_t(out_t).permute(0, 2, 3, 4, 1)  # (b, d, c, h, w)-->(b, c, h, w, d)
        
        '''the spatial branch'''
        if 's' in self.mode.lower():
            Hs = self.S_project(s_in)  #(b, c, h//2, w//2, d//2)
            phi_s = self.phi_s(Hs.permute(0, 2, 1, 3, 4)).permute(0, 1, 3, 2, 4).contiguous().view(b, h*w//4, -1)
            #(b, h, c, w, d)-->(b, h, w, c, d)-->(b, hw//4, cd//2)
            v = self.v(Hs.permute(0, 3, 1, 2, 4)).permute(0, 1, 3, 2, 4).contiguous().view(b, h*w//4, -1)
            #(b, w, c, h, d)-->(b, w, h, c, d)-->(b, hw//4, cd//2)
            A = self.sigmoid(t.matmul(v, phi_s.permute(0, 2, 1)))  #(b, hw//4, hw//4)
            delta = self.delta(Hs).view(b, c*d//2, -1)  #(b, c, h//2, w//2, d//2)-->(b, cd//2, hw//4)
            AVs = t.matmul(delta, A).view(b, c, d//2, h//2, w//2).permute(0, 1, 3, 4, 2).contiguous()  #(b, cd//2, hw//4)-->(b, c, h//2, w//2, d//2)
            Ws = self.Ws(AVs) #(b, c, h, w, d)
            Ws = t.nn.functional.interpolate(Ws, scale_factor=2, mode='nearest')
            out_s = self.xi(Ws)
            return x + self.weight1*out_s + self.weight2*out_f + self.weight3*out_t


        return x + self.weight2*out_f + self.weight3*out_t

class Conv3D(nn.Module):
    def __init__(self, c, o, kernel=3, stride=1, padding=1, g=1):
        super(Conv3D, self).__init__()
        if padding == None:
            padding = (kernel - 1) // 2
        self.conv = nn.Conv3d(c, o, kernel_size=kernel, stride=stride, padding=padding, groups=g)
        self.bn = nn.BatchNorm3d(o)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        h = self.conv(x)
        h = self.bn(h)
        h = self.relu(h)

        return h

class SFT_unit(nn.Module):
    '''
    The spatial feature temporal unit
    input: b, c, h, w, d
    '''
    def __init__(self, in_c, in_h, in_t, ConvND = nn.Conv3d, normalize=None, mode=None, adaptive = True):
        super().__init__()
        self.node_f = in_c  #32
        self.state_f = in_c  #32
        self.node_t = 128
        self.state_t = 128
        self.in_h = self.in_w = in_h
        self.normalize = normalize
        self.mode = mode
        self.adaptive = adaptive

        self.S_project =ConvND(in_c, in_c, kernel_size=3, padding=1, stride=2, groups=in_c)
        self.phi_s = ConvND(in_channels=self.in_h//2, out_channels=self.in_h//2, kernel_size=1)
        self.v = ConvND(in_channels=self.in_w//2, out_channels=self.in_w//2, kernel_size=1)
        self.delta = ConvND(in_c, in_c, 1)
        self.Ws = ConvND(in_c, in_c, 1)
        self.xi = ConvND(in_c, in_c, 1)
        self.sigmoid = nn.Sigmoid()

        self.phi_f = ConvND(in_channels=in_c, out_channels=self.state_f, kernel_size=1)
        self.theta_f = ConvND(in_channels=in_c, out_channels=self.node_f, kernel_size=1)

        self.phi_t = ConvND(in_channels=in_t, out_channels=self.state_t, kernel_size=1)
        self.theta_t = ConvND(in_channels=in_t, out_channels=self.node_t, kernel_size=1)

        self.GCN_f = GCN(num_state=self.state_f, num_node=self.node_f)
        self.GCN_t = GCN(num_state=self.state_t, num_node=self.node_t)

        self.extend_f = nn.Sequential(ConvND(self.state_f, in_c, kernel_size=1),
                                      normalization(in_c))
        self.extend_t = nn.Sequential(ConvND(self.state_t, in_t, kernel_size=1),
                                      normalization(in_t))

        self.weight1 = nn.Parameter(t.ones(1))
        self.weight2 = nn.Parameter(t.ones(1))
        self.weight3 = nn.Parameter(t.ones(1))

    def forward(self, x):
        b, c, h, w, d = x.size()
        s_in = f_in = x  #(b, c, h, w, d)
        t_in = x.permute(0, 4, 1, 2, 3).contiguous()  #(b, d, c, h, w)

        

        '''the feature branch'''
        phi_f = self.phi_f(f_in).view(b, self.state_f, -1)  #(b, state_f, d*h*w)
        theta_f = self.theta_f(f_in).view(b, self.node_f, -1)  #(b, node_f, d*h*w)
        graph_f = t.matmul(phi_f, theta_f.permute(0, 2, 1))  #(b, state_f, node_f)
        if self.normalize:
            graph_f = graph_f * (1. / graph_f.size(2))
        out_f = self.GCN_f(graph_f)  #(b, state_f, node_f)
        out_f = t.matmul(out_f, theta_f).view(b, self.state_f, *x.size()[2:])  #(b, state_f, h, w, d)
        out_f = self.extend_f(out_f)  #(b, c, h, w, d)

        '''the temporal branch'''
        phi_t = self.phi_t(t_in).view(b, self.state_t, -1)  # (b, state_t, c*h*w)
        theta_t = self.theta_t(t_in).view(b, self.node_t, -1)  # (b, node_t, c*h*w)
        graph_t = t.matmul(phi_t, theta_t.permute(0, 2, 1))  # (b, state_t, node_t)
        if self.normalize:
            graph_t = graph_t * (1. / graph_t.size(2))
        out_t = self.GCN_t(graph_t)  # (b, state_t, node_t)
        out_t = t.matmul(out_t, theta_t).view(b, self.state_t, *x.size()[1:4])  # (b, state_t, c, h, w)
        out_t = self.extend_t(out_t).permute(0, 2, 3, 4, 1)  # (b, d, c, h, w)-->(b, c, h, w, d)
        
        '''the spatial branch'''
        
        Hs = self.S_project(s_in)  #(b, c, h//2, w//2, d//2)
        phi_s = self.phi_s(Hs.permute(0, 2, 1, 3, 4)).permute(0, 1, 3, 2, 4).contiguous().view(b, h*w//4, -1)
        #(b, h, c, w, d)-->(b, h, w, c, d)-->(b, hw//4, cd//2)
        v = self.v(Hs.permute(0, 3, 1, 2, 4)).permute(0, 1, 3, 2, 4).contiguous().view(b, h*w//4, -1)
        #(b, w, c, h, d)-->(b, w, h, c, d)-->(b, hw//4, cd//2)
        A = self.sigmoid(t.matmul(v, phi_s.permute(0, 2, 1)))  #(b, hw//4, hw//4)
        delta = self.delta(Hs).view(b, c*d//2, -1)  #(b, c, h//2, w//2, d//2)-->(b, cd//2, hw//4)
        AVs = t.matmul(delta, A).view(b, c, d//2, h//2, w//2).permute(0, 1, 3, 4, 2).contiguous()  #(b, cd//2, hw//4)-->(b, c, h//2, w//2, d//2)
        Ws = self.Ws(AVs) #(b, c, h, w, d)
        Ws = t.nn.functional.interpolate(Ws, scale_factor=2, mode='nearest')
        out_s = self.xi(Ws)
        
        if 'sft' == self.mode.lower():
            if self.adaptive:
                return x + self.weight1*out_s + self.weight2*out_f + self.weight3*out_t
            else:
                return x + out_s + out_f + out_t
        elif 's' == self.mode.lower():
            if self.adaptive:
                return x + self.weight1*out_s
            else:
                return x + out_s
        elif 'f' == self.mode.lower():
            if self.adaptive:
                return x + self.weight2*out_f
            else:
                return x + out_f
        elif 't' == self.mode.lower():
            if self.adaptive:
                return x + self.weight3*out_t
            else:
                return x + out_t
        elif 'ft' == self.mode.lower():
            if self.adaptive:
                return x + self.weight2*out_f + self.weight3*out_t
            else:
                return x + out_f + out_t


        return x  #self.weight2*out_f + self.weight3*out_t


class Unet_3D(nn.Module):
    def __init__(self, c_in=4, f=16, num_classes=4):
        super().__init__()
        self.norm_GCN = False

        self.layer1 = nn.Sequential(
            Conv3D(4, f)
        )

        self.layer2 = nn.Sequential(
            Conv3D(f, 2*f),
            Conv3D(2*f, 2*f)
        )

        self.layer3 = nn.Sequential(
            Conv3D(2*f, 4*f),
            Conv3D(4*f, 4*f),
            Conv3D(4*f, 4*f)
        )
        
        self.SFT3 = SFT_unit(64, 16, 16, normalize=self.norm_GCN, mode='ft')

        self.layer4 = nn.Sequential(
            Conv3D(4*f, 8*f),
            Conv3D(8*f, 8*f),
            Conv3D(8*f, 8*f)
        )
        
        self.SFT4 = SFT_unit(128, 8, 8, normalize=self.norm_GCN, mode='sft')

        self.layer5 = nn.Sequential(
            Conv3D(8*f, 16*f),
            Conv3D(16*f, 16*f),
            Conv3D(16*f, 16*f)
        )

        self.downsample = nn.MaxPool3d(kernel_size=2, stride=2)

        self.upsample11 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear'),
            Conv3D(16*f, 8*f)
        )

        self.upsample12 = nn.Sequential(
            Conv3D(16 * f, 8 * f),
            Conv3D(8 * f, 8 * f),
            Conv3D(8 * f, 8 * f)
        )

        self.upsample21 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear'),
            Conv3D(8*f, 4*f)
        )

        self.upsample22 = nn.Sequential(
            Conv3D(8 * f, 4 * f),
            Conv3D(4 * f, 4 * f),
            Conv3D(4 * f, 4 * f)
        )

        self.upsample31 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear'),
            Conv3D(4 * f, 2 * f)
        )

        self.upsample32 = nn.Sequential(
            Conv3D(4 * f, 2 * f),
            Conv3D(2 * f, 2 * f),
            Conv3D(2 * f, 2 * f)
        )

        self.upsample41 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear'),
            Conv3D(2 * f, f)
        )

        self.upsample42 = nn.Sequential(
            Conv3D(2 * f, f),
            Conv3D(f, f),
            Conv3D(f, 4)
        )



        self.softmax = nn.Softmax(dim=1)
        #self.dropout = nn.Dropout3d(p=0.5)

    def forward(self, x):
        x1 = self.layer1(x)  #16, 128, 128, 128
        down1 = self.downsample(x1)  #16, 64, 64. 64

        x2 = self.layer2(down1)  #32, 64, 64, 64
        down2 = self.downsample(x2)  #32, 32, 32, 32

        x3 = self.layer3(down2)  #64, 32, 32, 32
        down3 = self.downsample(x3)  #64, 16, 16, 16
        

        x4 = self.layer4(down3)  #128, 16, 16, 16
        #x4 = self.dropout(x4)
        down4 = self.downsample(x4)  #128, 8, 8, 8
        

        x5 = self.layer5(down4) #256, 8, 8, 8
        #x5 = self.dropout(x5)

        up1 = self.upsample11(x5)  #128, 16, 16, 16
        up1 = t.cat([x4, up1], dim=1)
        up1 = self.upsample12(up1)  #128, 16, 16, 16

        up2 = self.upsample21(up1)  #64, 32, 32, 32
        up2 = t.cat([x3, up2], dim=1)
        up2 = self.upsample22(up2)  #64, 32, 32, 32

        up3 = self.upsample31(up2)  #32, 64, 64, 64
        up3 = t.cat([x2, up3], dim=1)
        up3 = self.upsample32(up3)  #32, 64, 64, 64



        up4 = self.upsample41(up3)  #16, 128, 128, 128
        up4 = t.cat([x1, up4], dim=1)
        up4 = self.upsample42(up4)  #4, 128, 128, 128

        out = self.softmax(up4)  #4, 128, 128, 128

        return out
        
class ConvD(nn.Module):
    def __init__(self, inplanes, planes, dropout=0.0, norm='gn', first=False):
        super(ConvD, self).__init__()

        self.first = first
        self.maxpool = nn.MaxPool3d(2, 2)

        self.dropout = dropout
        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Conv3d(inplanes, planes, 3, 1, 1, bias=False)
        self.bn1   = normalization(planes, norm)

        self.conv2 = nn.Conv3d(planes, planes, 3, 1, 1, bias=False)
        self.bn2   = normalization(planes, norm)

        self.conv3 = nn.Conv3d(planes, planes, 3, 1, 1, bias=False)
        self.bn3   = normalization(planes, norm)

    def forward(self, x):
        if not self.first:
            x = self.maxpool(x)
        x = self.bn1(self.conv1(x))
        y = self.relu(self.bn2(self.conv2(x)))
        if self.dropout > 0:
            y = F.dropout3d(y, self.dropout)
        y = self.bn3(self.conv3(x))
        return self.relu(x + y)


class ConvU(nn.Module):
    def __init__(self, planes, norm='gn', first=False):
        super(ConvU, self).__init__()

        self.first = first

        if not self.first:
            self.conv1 = nn.Conv3d(2*planes, planes, 3, 1, 1, bias=False)
            self.bn1   = normalization(planes, norm)

        self.conv2 = nn.Conv3d(planes, planes//2, 1, 1, 0, bias=False)
        self.bn2   = normalization(planes//2, norm)

        self.conv3 = nn.Conv3d(planes, planes, 3, 1, 1, bias=False)
        self.bn3   = normalization(planes, norm)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, prev):
        # final output is the localization layer
        if not self.first:
            x = self.relu(self.bn1(self.conv1(x)))

        y = F.upsample(x, scale_factor=2, mode='trilinear', align_corners=False)
        y = self.relu(self.bn2(self.conv2(y)))

        y = torch.cat([prev, y], 1)
        y = self.relu(self.bn3(self.conv3(y)))

        return y


class Unet(nn.Module):
    def __init__(self, c=4, n=16, dropout=0.5, norm='gn', num_classes=4):
        super(Unet, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2,
                mode='trilinear', align_corners=False)

        self.convd1 = ConvD(c,     n, dropout, norm, first=True)
        self.convd2 = ConvD(n,   2*n, dropout, norm)
        self.convd3 = ConvD(2*n, 4*n, dropout, norm)
        self.convd4 = ConvD(4*n, 8*n, dropout, norm)
        self.convd5 = ConvD(8*n,16*n, dropout, norm)

        self.convu4 = ConvU(16*n, norm, True)
        self.convu3 = ConvU(8*n, norm)
        self.convu2 = ConvU(4*n, norm)
        self.convu1 = ConvU(2*n, norm)

        self.seg3 = nn.Conv3d(8*n, num_classes, 1)
        self.seg2 = nn.Conv3d(4*n, num_classes, 1)
        self.seg1 = nn.Conv3d(2*n, num_classes, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x1 = self.convd1(x)
        x2 = self.convd2(x1)
        x3 = self.convd3(x2)
        x4 = self.convd4(x3)
        x5 = self.convd5(x4)

        y4 = self.convu4(x5, x4)
        y3 = self.convu3(y4, x3)
        y2 = self.convu2(y3, x2)
        y1 = self.convu1(y2, x1)

        y3 = self.seg3(y3)
        y2 = self.seg2(y2) + self.upsample(y3)
        y1 = self.seg1(y1) + self.upsample(y2)

        return y1


if __name__ == '__main__':
    x = t.randn((1, 4, 128, 128, 128))
    model = Unet_3D()
    x = model(x)
    print(x.shape)