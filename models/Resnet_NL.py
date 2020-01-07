import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from functools import partial
from config import opts
from models.sync_batchnorm import SynchronizedBatchNorm3d

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
    zero_pads = torch.Tensor(out.size(0), planes - out.size(1), out.size(2), out.size(3), out.size(4)).zero_()
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
                 no_cuda=False,
                 non_layers=[0, 1, 1, 1]):
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

        non_idx = 0
        self.layer1 = self._make_layer(block, 64, layers[0], shortcut_type)
        self.NL_1 = nn.ModuleList([NonLocalBlock(self.inplanes, self.inplanes // 2, sub_sample=True) for i in range(non_layers[non_idx])])
        self.NL_1_idx = sorted([layers[0] - (i + 1) for i in range(non_layers[non_idx])])

        non_idx += 1
        self.layer2 = self._make_layer(block, 128, layers[1], shortcut_type, stride=2)
        self.NL_2 = nn.ModuleList([NonLocalBlock(self.inplanes, self.inplanes // 2) for i in range(non_layers[non_idx])])
        self.NL_2_idx = sorted([layers[1] - (i + 1) for i in range(non_layers[non_idx])])

        non_idx += 1
        self.layer3 = self._make_layer(block, 256, layers[2], shortcut_type, stride=1, dilation=2)
        self.NL_3 = nn.ModuleList([NonLocalBlock(self.inplanes, self.inplanes // 2) for i in range(non_layers[non_idx])])
        self.NL_3_idx = sorted([layers[2] - (i + 1) for i in range(non_layers[non_idx])])

        non_idx += 1
        self.layer4 = self._make_layer(block, 512, layers[3], shortcut_type, stride=1, dilation=4)
        self.NL_4 = nn.ModuleList([NonLocalBlock(self.inplanes, self.inplanes // 2) for i in range(non_layers[non_idx])])
        self.NL_4_idx = sorted([layers[3] - (i + 1) for i in range(non_layers[non_idx])])

        self.softmax = nn.Softmax(1)

        self.conv_seg = nn.Sequential(nn.ConvTranspose3d(512 * block.expansion, 32, 2, stride=2),
                                      normalization(32),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 32, kernel_size=3, stride=(1, 1, 1), padding=(1, 1, 1), bias=False),
                                      normalization(32),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, num_seg_classes, kernel_size=1, stride=(1, 1, 1), bias=False)
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
        # x's shape (B,C,H,W,D)
        # Down Sample
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Layer 1
        NL1_counter = 0
        if len(self.NL_1_idx) == 0:
            self.NL_1_idx = [-1]
        for i in range(len(self.layer1)):
            x = self.layer1[i](x)
            if i == self.NL_1_idx[NL1_counter]:
                x = x.permute(0, 1, 4, 2, 3)
                x = self.NL_1[NL1_counter](x)
                x = x.permute(0, 1, 3, 4, 2)
                NL1_counter += 1

        # Layer 1
        NL2_counter = 0
        for i in range(len(self.layer2)):
            x = self.layer2[i](x)
            if i == self.NL_2_idx[NL2_counter]:
                x = x.permute(0, 1, 4, 2, 3)
                x = self.NL_2[NL2_counter](x)
                x = x.permute(0, 1, 3, 4, 2)
                NL2_counter+=1

        # Layer 3
        NL3_counter = 0
        for i in range(len(self.layer3)):
            x = self.layer3[i](x)
            if i == self.NL_3_idx[NL3_counter]:
                x = x.permute(0, 1, 4, 2, 3)
                x = self.NL_3[NL3_counter](x)
                x = x.permute(0, 1, 3, 4, 2)
                NL3_counter+=1

        # Layer 4
        NL4_counter = 0
        for i in range(len(self.layer4)):
            x = self.layer4[i](x)
            if i == self.NL_4_idx[NL4_counter]:
                x = x.permute(0, 1, 4, 2, 3)
                x = self.NL_4[NL4_counter](x)
                x = x.permute(0, 1, 3, 4, 2)
                NL4_counter += 1

        # Up Sample
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


class NonLocalBlock(nn.Module):
    def __init__(self, in_channels, inter_channels=None, sub_sample=False, bn_layer=True, instance='soft'):
        super(NonLocalBlock, self).__init__()
        self.sub_sample = sub_sample
        self.instance = instance
        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        conv_nd = nn.Conv3d
        max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
        bn = nn.BatchNorm3d

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)
        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)
            )
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)
        self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)
        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool_layer)
            self.phi = nn.Sequential(self.phi, max_pool_layer)

    def forward(self, x):
        '''
        :param x: (b, c, t, h, w)
        :return:
        '''
        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)  # (b, c, thw)
        g_x = g_x.permute(0, 2, 1)  # (b, thw, c)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)  # (b, c, thw)
        theta_x = theta_x.permute(0, 2, 1)  # (b, thw, c)

        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)  # (b, c, thw)

        f = torch.matmul(theta_x, phi_x)  # (b, thw, thw)

        if self.instance == 'soft':
            f_div_C = F.softmax(f, dim=-1)
        elif self.instance == 'dot':
            f_div_C = f / f.shape[1]

        y = torch.matmul(f_div_C, g_x)  # (b, thw, c)
        y = y.permute(0, 2, 1).contiguous()  # (b, c, thw)
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])  # (b, c, t, h, w)
        W_y = self.W(y)
        z = W_y + x

        return z


if __name__ == "__main__":
    import torch
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = torch.device('cuda:0')
    model = generate_model(config).cuda()
    # print(model)
    x = torch.rand((2, 4, 128, 128, 128), device=device)
    y = model(x)
    print(y.shape)