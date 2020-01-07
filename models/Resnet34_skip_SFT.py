import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from functools import partial
from config import opts
from .sync_batchnorm import SynchronizedBatchNorm3d

config = opts()

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

class Conv3d_Block(nn.Module):
    def __init__(self, num_in, num_out, kernel_size=3, stride=1, g=1, padding=1):
        super(Conv3d_Block, self).__init__()
        if padding == None:
            padding = (kernel_size - 1) // 2
        self.bn = normalization(num_in)
        self.act_fn = nn.ReLU(inplace=True)
        # self.act_fn = nn.PReLU() # report error : out of CUDA
        # self.act_fn = nn.ELU(inplace=True)
        self.conv = nn.Conv3d(num_in, num_out, kernel_size=kernel_size, padding=padding,stride=stride, groups=g, bias=False)

    def forward(self, x):# BN + Relu + Conv
        h = self.act_fn(self.bn(x))
        h = self.conv(h)
        return h

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

        if config.model_depth == 34:
            self.SFT = SFT_unit(512, config.crop_H//8, config.crop_D//8)  #resnet34
        else:
            self.SFT = SFT_unit(2048, config.crop_H//8, config.crop_D//8)  #resnet50
        self.softmax = nn.Softmax(1)

        self.decoder1 = Conv3d_Block(512 * block.expansion, 256 * block.expansion)

        self.decoder2 = Conv3d_Block(512 * block.expansion, 128 * block.expansion)  #1024+1024-->512

        self.decoder3 = Conv3d_Block(256 * block.expansion, 64 * block.expansion)  #512+512-->256

        self.decoder4 = Conv3d_Block(128 * block.expansion, 64 * block.expansion)  #256+256-->64

        self.decoder5 = Conv3d_Block(128 * block.expansion, num_seg_classes)  # 64+64-->4


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
        x1 = self.relu(x)
        x2 = self.maxpool(x1)
        x3 = self.layer1(x2)
        x4 = self.layer2(x3)
        x5 = self.layer3(x4)
        x6 = self.layer4(x5)
        x6 = self.SFT(x6)
        x7 = self.decoder1(x6)
        x7 = torch.cat([x5, x7], dim=1)
        x8 = self.decoder2(x7)
        x8 = torch.cat([x4, x8], dim=1)
        x9 = self.decoder3(x8)
        x9 = nn.functional.interpolate(x9, scale_factor=2, mode='trilinear')
        x9 = torch.cat([x3, x9], dim=1)
        x10 = self.decoder4(x9)
        x10 = nn.functional.interpolate(x10, scale_factor=2, mode='trilinear')
        x10 = torch.cat([x1, x10], dim=1)
        x10 = nn.functional.interpolate(x10, scale_factor=2, mode='trilinear')
        output = self.decoder5(x10)
        output = self.softmax(output)

        return output

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
                sample_input_W=opt.crop_W,
                sample_input_H=opt.crop_H,
                sample_input_D=opt.crop_D,
                shortcut_type=opt.resnet_shortcut,
                no_cuda=opt.no_cuda,
                num_seg_classes=opt.num_class)
        elif opt.model_depth == 18:
            model = resnet18(
                sample_input_W=opt.crop_W,
                sample_input_H=opt.crop_H,
                sample_input_D=opt.crop_D,
                shortcut_type=opt.resnet_shortcut,
                no_cuda=opt.no_cuda,
                num_seg_classes=opt.num_class)
        elif opt.model_depth == 34:
            model = resnet34(
                sample_input_W=opt.crop_W,
                sample_input_H=opt.crop_H,
                sample_input_D=opt.crop_D,
                shortcut_type=opt.resnet_shortcut,
                no_cuda=opt.no_cuda,
                num_seg_classes=opt.num_class)
        elif opt.model_depth == 50:
            model = resnet50(
                sample_input_W=opt.crop_W,
                sample_input_H=opt.crop_H,
                sample_input_D=opt.crop_D,
                shortcut_type=opt.resnet_shortcut,
                no_cuda=opt.no_cuda,
                num_seg_classes=opt.num_class)
        elif opt.model_depth == 101:
            model = resnet101(
                sample_input_W=opt.crop_W,
                sample_input_H=opt.crop_H,
                sample_input_D=opt.crop_D,
                shortcut_type=opt.resnet_shortcut,
                no_cuda=opt.no_cuda,
                num_seg_classes=opt.num_class)
        elif opt.model_depth == 152:
            model = resnet152(
                sample_input_W=opt.crop_W,
                sample_input_H=opt.crop_H,
                sample_input_D=opt.crop_D,
                shortcut_type=opt.resnet_shortcut,
                no_cuda=opt.no_cuda,
                num_seg_classes=opt.num_class)
        elif opt.model_depth == 200:
            model = resnet200(
                sample_input_W=opt.crop_W,
                sample_input_H=opt.crop_H,
                sample_input_D=opt.crop_D,
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
        self.node_f = in_c//2
        self.state_f = in_c
        self.node_t = in_t//2
        self.state_t = in_t
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
        Hs = self.S_project(s_in)  #(b, c, h//2, w//2, d//2)
        phi_s = self.phi_s(Hs.permute(0, 2, 1, 3, 4)).permute(0, 1, 3, 2, 4).contiguous().view(b, h*w//4, -1)
        #(b, h, c, w, d)-->(b, h, w, c, d)-->(b, hw//4, cd//2)
        v = self.v(Hs.permute(0, 3, 1, 2, 4)).permute(0, 1, 3, 2, 4).contiguous().view(b, h*w//4, -1)
        #(b, w, c, h, d)-->(b, w, h, c, d)-->(b, hw//4, cd//2)
        A = self.sigmoid(torch.matmul(v, phi_s.permute(0, 2, 1)))  #(b, hw//4, hw//4)
        delta = self.delta(Hs).view(b, c*d//2, -1)  #(b, c, h//2, w//2, d//2)-->(b, cd//2, hw//4)
        AVs = torch.matmul(delta, A).view(b, c, d//2, h//2, w//2).permute(0, 1, 3, 4, 2).contiguous()  #(b, cd//2, hw//4)-->(b, c, h//2, w//2, d//2)
        Ws = self.Ws(AVs) #(b, c, h, w, d)
        Ws = torch.nn.functional.interpolate(Ws, scale_factor=2, mode='nearest')
        out_s = self.xi(Ws)

        '''the feature branch'''
        phi_f = self.phi_f(f_in).view(b, self.state_f, -1)  #(b, state_f, d*h*w)
        theta_f = self.theta_f(f_in).view(b, self.node_f, -1)  #(b, node_f, d*h*w)
        graph_f = torch.matmul(phi_f, theta_f.permute(0, 2, 1))  #(b, state_f, node_f)
        if self.normalize:
            graph_f = graph_f * (1. / graph_f.size(2))
        out_f = self.GCN_f(graph_f)  #(b, state_f, node_f)
        out_f = torch.matmul(out_f, theta_f).view(b, self.state_f, *x.size()[2:])  #(b, state_f, h, w, d)
        out_f = self.extend_f(out_f)  #(b, c, h, w, d)

        '''the temporal branch'''
        phi_t = self.phi_t(t_in).view(b, self.state_t, -1)  # (b, state_t, c*h*w)
        theta_t = self.theta_t(t_in).view(b, self.node_t, -1)  # (b, node_t, c*h*w)
        graph_t = torch.matmul(phi_t, theta_t.permute(0, 2, 1))  # (b, state_t, node_t)
        if self.normalize:
            graph_t = graph_t * (1. / graph_t.size(2))
        out_t = self.GCN_t(graph_t)  # (b, state_t, node_t)
        out_t = torch.matmul(out_t, theta_t).view(b, self.state_t, *x.size()[1:4])  # (b, state_t, c, h, w)
        out_t = self.extend_t(out_t).permute(0, 2, 3, 4, 1)  # (b, d, c, h, w)-->(b, c, h, w, d)

        return x + out_s + out_f + out_t

if __name__ == '__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = torch.device('cuda:0')
    x = torch.rand((1, 4, 128, 128, 128), device=device)
    model = generate_model(config).cuda()
    y = model(x)
    print(y.shape)