import torch as t
import torch.nn as nn
import numpy as np

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
            padding = (kernel_size - 1) // 2
        self.conv = nn.Conv3d(c, o, kernel_size=kernel, stride=stride, padding=padding, groups=g, )
        # self.bn = nn.BatchNorm3d(o)
        self.bn = nn.InstanceNorm3d(o)
        self.relu = nn.LeakyReLU(negative_slope=1e-2, inplace=True)

    def forward(self, x):
        h = self.conv(x)
        h = self.bn(h)
        h = self.relu(h)

        return h



class No_new_net(nn.Module):
    def __init__(self, c_in=4, num_classes=4):
        super().__init__()
        self.norm_GCN = False

        self.layer1 = nn.Sequential(
            Conv3D(c_in, 30),
            Conv3D(30, 30)
        )

        self.layer2 = nn.Sequential(
            Conv3D(30, 60),
            Conv3D(60, 60)
        )

        self.layer3 = nn.Sequential(
            Conv3D(60, 120),
            Conv3D(120, 120)
        )

        self.layer4 = nn.Sequential(
            Conv3D(120, 240),
            Conv3D(240, 240)
        )
        
        self.SFT3 = SFT_unit(120, 16, 16, normalize=self.norm_GCN, mode='ft')
        self.SFT4 = SFT_unit(240, 8, 8, normalize=self.norm_GCN, mode='sft')

        self.layer5 = nn.Sequential(
            Conv3D(240, 480),
            Conv3D(480, 480),
            Conv3D(480, 240)
        )

        self.downsample = nn.MaxPool3d(kernel_size=2, stride=2)
        self.upsample = nn.Upsample(scale_factor=2, mode='trilinear')
        self.conv1 = Conv3D(240, 120)
        self.conv2 = Conv3D(120, 60)
        self.conv3 = Conv3D(60, 30)
        self.conv4 = Conv3D(30, num_classes, kernel=1, padding=0)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x1 = self.layer1(x)  #30, 128, 128, 128
        down1 = self.downsample(x1)  #30, 64, 64. 64

        x2 = self.layer2(down1)  #60, 64, 64, 64
        down2 = self.downsample(x2)  #60, 32, 32, 32

        x3 = self.layer3(down2)  #120, 32, 32, 32
        down3 = self.downsample(x3)  #120, 16, 16, 16
        down3 = self.SFT3(down3)

        x4 = self.layer4(down3)  #240, 16, 16, 16
        down4 = self.downsample(x4)  #240, 8, 8, 8
        down4 = self.SFT4(down4)
        

        x5 = self.layer5(down4)  #240-480-240, 240, 8, 8, 8
        up1 = self.upsample(x5)  #240, 16, 16, 16
        up1 = up1 + x4
        up1 = self.conv1(up1)  #120, 16, 16, 16

        up2 = self.upsample(up1)  #120, 32, 32, 32
        up2 = up2 + x3
        up2 = self.conv2(up2)  #60, 32, 32, 32

        up3 = self.upsample(up2)  #60, 64, 64, 64
        up3 = up3 + x2
        up3 = self.conv3(up3)  #30, 64, 64, 64

        up4 = self.upsample(up3)  #30, 128, 128, 128
        up4 = up4 + x1

        out = self.softmax(self.conv4(up4))  #4, 128, 128, 128

        return out



if __name__ == '__main__':
    x = t.randn((1, 4, 128, 128, 128))
    model = No_new_net()
    x = model(x)
    print(x.shape)