import torch.nn as nn
import torch.nn.functional as F
import torch
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

def conv3x3x3(in_planes, out_planes, stride=1, dilation=1):
    # 3x3x3 convolution with padding
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        dilation=dilation,
        stride=stride,
        padding=dilation,
        bias=False)

def downsample_basic_block(x, planes, stride, no_cuda=False):
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)
    zero_pads = torch.Tensor(
        out.size(0), planes - out.size(1), out.size(2), out.size(3),
        out.size(4)).zero_()
    if not no_cuda:
        if isinstance(out.data, torch.cuda.FloatTensor):
            zero_pads = zero_pads.cuda()

    out = Variable(torch.cat([out.data, zero_pads], dim=1))

    return out

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3x3(inplanes, planes, stride=stride, dilation=dilation)
        self.bn1 = normalization(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes, dilation=dilation)
        self.bn2 = normalization(planes)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = normalization(planes)
        self.conv2 = nn.Conv3d(
            planes, planes, kernel_size=3, stride=stride, dilation=dilation, padding=dilation, bias=False)
        self.bn2 = normalization(planes)
        self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = normalization(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNet(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 sample_input_D,
                 sample_input_H,
                 sample_input_W,
                 num_seg_classes,
                 shortcut_type='B',
                 no_cuda=False):
        self.inplanes = 64
        self.no_cuda = no_cuda
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv3d(
            4,
            64,
            kernel_size=7,
            stride=(2, 2, 2),
            padding=(3, 3, 3),
            bias=False)

        self.bn1 = normalization(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], shortcut_type)
        self.layer2 = self._make_layer(
            block, 128, layers[1], shortcut_type, stride=2)
        self.layer3 = self._make_layer(
            block, 256, layers[2], shortcut_type, stride=1, dilation=2)
        self.layer4 = self._make_layer(
            block, 512, layers[3], shortcut_type, stride=1, dilation=4)
        self.SFT = SFT_unit(512, 30, 20)
        self.softmax = nn.Softmax(1)

        self.conv_seg = nn.Sequential(
            nn.ConvTranspose3d(
                512 * block.expansion,
                32,
                2,
                stride=2
            ),
            normalization(32),
            nn.ReLU(inplace=True),
            nn.Conv3d(
                32,
                32,
                kernel_size=3,
                stride=(1, 1, 1),
                padding=(1, 1, 1),
                bias=False),
            normalization(32),
            nn.ReLU(inplace=True),
            nn.Conv3d(
                32,
                num_seg_classes,
                kernel_size=1,
                stride=(1, 1, 1),
                bias=False)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
            # elif isinstance(m, nn.BatchNorm3d):
            #     m.weight.data.fill_(1)
            #     m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(
                    downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride,
                    no_cuda=self.no_cuda)
            else:
                downsample = nn.Sequential(
                    nn.Conv3d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=False), normalization(planes * block.expansion))

        layers = []
        layers.append(block(self.inplanes, planes, stride=stride, dilation=dilation, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.SFT(x)
        x = self.conv_seg(x)
        x = nn.functional.interpolate(x, scale_factor=4, mode='nearest')
        x = self.softmax(x)

        return x

def resnet10(**kwargs):
    """Constructs a ResNet-10 model.
    """
    model = ResNet(BasicBlock, [1, 1, 1, 1], **kwargs)
    return model

def resnet18(**kwargs):
    """Constructs a ResNet-18 model.
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model

def resnet34(**kwargs):
    """Constructs a ResNet-34 model.
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model

def resnet50(**kwargs):
    """Constructs a ResNet-50 model.
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model

def resnet101(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    return model

def resnet152(**kwargs):
    """Constructs a ResNet-152 model.
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    return model

def resnet200(**kwargs):
    """Constructs a ResNet-200 model.
    """
    model = ResNet(Bottleneck, [3, 24, 36, 3], **kwargs)
    return model

def generate_model(opt):
    assert opt.model in [
        'resnet'
    ]

    if opt.model == 'resnet':
        assert opt.model_depth in [10, 18, 34, 50, 101, 152, 200]

        if opt.model_depth == 10:
            model = resnet10(
                sample_input_W=opt.input_W,
                sample_input_H=opt.input_H,
                sample_input_D=opt.input_D,
                shortcut_type=opt.resnet_shortcut,
                no_cuda=opt.no_cuda,
                num_seg_classes=opt.num_class)
        elif opt.model_depth == 18:
            model = resnet18(
                sample_input_W=opt.input_W,
                sample_input_H=opt.input_H,
                sample_input_D=opt.input_D,
                shortcut_type=opt.resnet_shortcut,
                no_cuda=opt.no_cuda,
                num_seg_classes=opt.num_class)
        elif opt.model_depth == 34:
            model = resnet34(
                sample_input_W=opt.input_W,
                sample_input_H=opt.input_H,
                sample_input_D=opt.input_D,
                shortcut_type=opt.resnet_shortcut,
                no_cuda=opt.no_cuda,
                num_seg_classes=opt.num_class)
        elif opt.model_depth == 50:
            model = resnet50(
                sample_input_W=opt.input_W,
                sample_input_H=opt.input_H,
                sample_input_D=opt.input_D,
                shortcut_type=opt.resnet_shortcut,
                no_cuda=opt.no_cuda,
                num_seg_classes=opt.num_class)
        elif opt.model_depth == 101:
            model = resnet101(
                sample_input_W=opt.input_W,
                sample_input_H=opt.input_H,
                sample_input_D=opt.input_D,
                shortcut_type=opt.resnet_shortcut,
                no_cuda=opt.no_cuda,
                num_seg_classes=opt.num_class)
        elif opt.model_depth == 152:
            model = resnet152(
                sample_input_W=opt.input_W,
                sample_input_H=opt.input_H,
                sample_input_D=opt.input_D,
                shortcut_type=opt.resnet_shortcut,
                no_cuda=opt.no_cuda,
                num_seg_classes=opt.num_class)
        elif opt.model_depth == 200:
            model = resnet200(
                sample_input_W=opt.input_W,
                sample_input_H=opt.input_H,
                sample_input_D=opt.input_D,
                shortcut_type=opt.resnet_shortcut,
                no_cuda=opt.no_cuda,
                num_seg_classes=opt.num_class)

    return model

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
    def __init__(self, in_c, in_h, in_t, ConvND = nn.Conv3d, normalize=None, downsample=1):
        super().__init__()
        self.node_f = in_c  #32
        self.state_f = in_c  #32
        self.node_t = 128
        self.state_t = 128
        self.in_h = self.in_w = in_h
        self.normalize = normalize

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

    def forward(self, x):
        b, c, h, w, d = x.size()
        s_in = f_in = x  #(b, c, h, w, d)
        t_in = x.permute(0, 4, 1, 2, 3).contiguous()  #(b, d, c, h, w)

        '''the spatial branch'''
        # Hs = self.S_project(s_in)  #(b, c, h//2, w//2, d//2)
        # phi_s = self.phi_s(Hs.permute(0, 2, 1, 3, 4)).permute(0, 1, 3, 2, 4).contiguous().view(b, h*w//4, -1)
        # #(b, h, c, w, d)-->(b, h, w, c, d)-->(b, hw//4, cd//2)
        # v = self.v(Hs.permute(0, 3, 1, 2, 4)).permute(0, 1, 3, 2, 4).contiguous().view(b, h*w//4, -1)
        # #(b, w, c, h, d)-->(b, w, h, c, d)-->(b, hw//4, cd//2)
        # A = self.sigmoid(torch.matmul(v, phi_s.permute(0, 2, 1)))  #(b, hw//4, hw//4)
        # delta = self.delta(Hs).view(b, c*d//2, -1)  #(b, c, h//2, w//2, d//2)-->(b, cd//2, hw//4)
        # AVs = torch.matmul(delta, A).view(b, c, d//2, h//2, w//2).permute(0, 1, 3, 4, 2).contiguous()  #(b, cd//2, hw//4)-->(b, c, h//2, w//2, d//2)
        # Ws = self.Ws(AVs) #(b, c, h, w, d)
        # Ws = torch.nn.functional.interpolate(Ws, scale_factor=2, mode='nearest')
        # out_s = self.xi(Ws)

        '''the feature branch'''
        # phi_f = self.phi_f(f_in).view(b, self.state_f, -1)  #(b, state_f, d*h*w)
        # theta_f = self.theta_f(f_in).view(b, self.node_f, -1)  #(b, node_f, d*h*w)
        # graph_f = torch.matmul(phi_f, theta_f.permute(0, 2, 1))  #(b, state_f, node_f)
        # if self.normalize:
        #     graph_f = graph_f * (1. / graph_f.size(2))
        # out_f = self.GCN_f(graph_f)  #(b, state_f, node_f)
        # out_f = torch.matmul(out_f, theta_f).view(b, self.state_f, *x.size()[2:])  #(b, state_f, h, w, d)
        # out_f = self.extend_f(out_f)  #(b, c, h, w, d)

        '''the temporal branch'''
        phi_t = self.phi_t(t_in).view(b, self.state_t, -1)  # (b, state_t, c*h*w)
        theta_t = self.theta_t(t_in).view(b, self.node_t, -1)  # (b, node_t, c*h*w)
        graph_t = torch.matmul(phi_t, theta_t.permute(0, 2, 1))  # (b, state_t, node_t)
        if self.normalize:
            graph_t = graph_t * (1. / graph_t.size(2))
        out_t = self.GCN_t(graph_t)  # (b, state_t, node_t)
        out_t = torch.matmul(out_t, theta_t).view(b, self.state_t, *x.size()[1:4])  # (b, state_t, c, h, w)
        out_t = self.extend_t(out_t).permute(0, 2, 3, 4, 1)  # (b, d, c, h, w)-->(b, c, h, w, d)
        # + out_s + out_f
        return x + out_t

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

    def __init__(self, c=4, n=32, channels=128, groups=16, norm='sync_bn', num_classes=4, output_func='softmax', norm_GCN=False):
        super(BaseModel, self).__init__()

        # Entry flow
        self.encoder_block1 = nn.Conv3d(c, n, kernel_size=3, padding=1, stride=2, bias=False)  # H//2
        self.encoder_block2 = nn.Sequential(
            MFUnit_A(n, channels, g=groups, stride=2, norm=norm),  # H//4 down
            (MFUnit_A(channels, channels, g=groups, stride=1, norm=norm)),  # Dilated Conv 3
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
        self.norm_GCN=norm_GCN
        self.SFT1 = SFT_unit(32, 64, 64, normalize=self.norm_GCN)
        self.SFT2 = SFT_unit(128, 32, 32, normalize=self.norm_GCN)
        self.SFT3 = SFT_unit(256, 16, 16, normalize=self.norm_GCN)
        self.SFT4 = SFT_unit(256, 8, 8, normalize=self.norm_GCN)

        output_func = output_func.lower()
        if output_func == 'sigmoid':
            self.output_func = nn.Sigmoid()
        elif output_func == 'softmax':
            self.output_func = self.output_func = nn.Softmax(1)
            # warnings.warn('[!] The softmax is added automaticlly during training, but you need to add the softmax function during testing manually.')
            # self.output_func = nn.Softmax(dim=1)
        elif output_func == 'logsoftmax':
            self.output_func = nn.LogSoftmax(dim=1)
        else:
            raise ValueError

        # Initialization
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                torch.nn.init.torch.nn.init.kaiming_normal_(m.weight) #
            elif isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.GroupNorm) or isinstance(m, SynchronizedBatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Encoder
        x1 = self.encoder_block1(x)  # H//2 down
        x1 = self.SFT1(x1)
        x2 = self.encoder_block2(x1)  # H//4 down
        x2 = self.SFT2(x2)
        x3 = self.encoder_block3(x2)  # H//8 down
        x3 = self.SFT3(x3)
        x4 = self.encoder_block4(x3)  # H//16
        # x4 = self.SFT4(x4)

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


class DilatedMFNet_3T(BaseModel): # softmax
    # [96]   Flops:  17.091G  &  Params: 2.2M
    # [112]  Flops:  21.749G  &  Params: 2.98M
    # [128]  Flops:  27.045G  &  Params: 3.88M
    def __init__(self, c=4, n=32, channels=128, groups=16, norm='sync_bn', num_classes=4, output_func='softmax', norm_GCN=False):
        super().__init__(c, n, channels, groups, norm, num_classes, output_func)

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
            MFUnit_add2(n, channels, g=groups, stride=2, norm=norm, dilation=[1, 2, 3], layerlogs=self.logger_layer1),# H//4 down
            # after the block C, the channels would be 3x changing (Deprecated).
            MFUnit_add2(channels, channels, g=groups, stride=1, norm=norm, dilation=[1, 2, 3], layerlogs=self.logger_layer2), # Dilated Conv 3
            MFUnit_add2(channels, channels, g=groups, stride=1, norm=norm, dilation=[1, 2, 3], layerlogs=self.logger_layer3)
        )

        self.encoder_block3 = nn.Sequential(
            MFUnit_add2(channels, channels*2, g=groups, stride=2, norm=norm, dilation=[1, 2, 3], layerlogs=self.logger_layer4), # H//8
            MFUnit_add2(channels * 2, channels * 2, g=groups, stride=1, norm=norm, dilation=[1, 2, 3], layerlogs=self.logger_layer5),# Dilated Conv 3
            MFUnit_add2(channels * 2, channels * 2, g=groups, stride=1, norm=norm, dilation=[1, 2, 3], layerlogs=self.logger_layer6)
        )


if __name__ == '__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = torch.device('cuda:0')
    x = torch.rand((2, 4, 128, 128, 128), device=device)  # [bsize,channels,H,W,Depth] [bsize,channels,H,W,D]
    # model = MF_VNet_16x_Dilated_A(c=4, groups=16, norm='bn', num_classes=4)
    model = DilatedMFNet_T(c=4, num_classes=4)
    model.cuda(device)
    y = model(x)
    print(y.shape)
