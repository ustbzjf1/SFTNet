import torch.nn as nn
import torch.nn.functional as F
import torch
import logging
import warnings
try:
    from .sync_batchnorm import SynchronizedBatchNorm3d
except:
    from sync_batchnorm import SynchronizedBatchNorm3d
# from utils import get_model_complexity_info
# from utils import add_flops_counting_methods, flops_to_string, get_model_parameters_number
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


class ConvBlock_Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock_Down, self).__init__()
        layers = [
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            normalization(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            normalization(out_channels),
            nn.ReLU(inplace=True),
        ]
        self.encode = nn.Sequential(*layers)

    def forward(self, x):
        y = self.encode(x)
        y = y + x
        return y


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EncoderBlock, self).__init__()
        layers1 = [
            nn.Conv3d(in_channels, out_channels, kernel_size=2, stride=2, padding=0),
            normalization(out_channels),
            nn.ReLU(inplace=True),
        ]
        layers2 = [
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            normalization(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            normalization(out_channels),
            nn.ReLU(inplace=True),
        ]
        self.encode1 = nn.Sequential(*layers1)
        self.encode2 = nn.Sequential(*layers2)

    def forward(self, x):
        x1 = self.encode1(x)
        y = self.encode2(x1)
        y = y + x1
        return y


class DecoderBlock2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock2, self).__init__()
        self.decode = nn.Sequential(
            # nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2),
            normalization(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.decode(x)
        # x = nn.functional.interpolate(x, scale_factor=2, mode='trilinear')
        return x


class ConvBlock_Up(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=False):
        super(ConvBlock_Up, self).__init__()
        layers = [
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            normalization(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            normalization(out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout == True:
            layers.insert(3, nn.Dropout(0.5))
            layers.insert(0, nn.Dropout(0.5))
        self.encode = nn.Sequential(*layers)

    def forward(self, x):
        y = self.encode(x)
        return y

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

class PSPModule(nn.Module):
    # (1, 2, 3, 6)
    def __init__(self, sizes, dimension=3):
        super(PSPModule, self).__init__()
        self.stages = nn.ModuleList([self._make_stage(size, dimension) for size in sizes])

    def _make_stage(self, size, dimension=3):
        if dimension == 1:
            prior = nn.AdaptiveAvgPool1d(output_size=size)
        elif dimension == 2:
            prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        elif dimension == 3:
            prior = nn.AdaptiveAvgPool3d(output_size=(size, size, size))
        return prior

    def forward(self, feats):
        b, c, _, _, _ = feats.size()
        priors = [stage(feats).view(b, c, -1) for stage in self.stages]
        center = torch.cat(priors, -1)
        return center

class SelfAttentionBlock3D(nn.Module):
    '''
    The basic implementation for self-attention block/non-local block
    Input:
        N X C X H X W
    Parameters:
        in_channels       : the dimension of the input feature map
        key_channels      : the dimension after the key/query transform
        value_channels    : the dimension after the value transform
        scale             : choose the scale to downsample the input feature maps (save memory cost)
    Return:
        N X C X H X W
        position-aware context features.(w/o concate or add with the input)
    '''

    def __init__(self, in_channels, key_channels, value_channels, out_channels, norm, psp_size, scale=1):
        super(SelfAttentionBlock3D, self).__init__()
        self.scale = scale
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.key_channels = key_channels
        self.value_channels = value_channels
        if out_channels == None:
            self.out_channels = in_channels
        self.pool = nn.MaxPool3d(kernel_size=(scale, scale, scale))
        self.f_key = nn.Sequential(
            nn.Conv3d(in_channels=self.in_channels, out_channels=self.key_channels,
                      kernel_size=1, stride=1, padding=0),
            # ModuleHelper.BNReLU(self.key_channels, norm_type=norm_type),
            normalization(self.key_channels, norm=norm),
            nn.ReLU(inplace=True)
        )
        self.f_query = self.f_key
        self.f_value = nn.Conv3d(in_channels=self.in_channels, out_channels=self.value_channels,
                                 kernel_size=1, stride=1, padding=0)
        self.W = nn.Conv3d(in_channels=self.value_channels, out_channels=self.out_channels,
                           kernel_size=1, stride=1, padding=0)

        self.psp = PSPModule(psp_size)
        nn.init.constant_(self.W.weight, 0)
        nn.init.constant_(self.W.bias, 0)

    def forward(self, x):
        # b, c, h, w, d = x.size(0), x.size(1), x.size(2), x.size(3), x.size(4)
        b, c, h, w, d = x.shape
        if self.scale > 1:
            x = self.pool(x)

        value = self.f_value(x)
        value = self.psp(value)
        value = value.permute(0, 2, 1)
        # print('value.shape:', value.shape)

        query = self.f_query(x)
        query = query.view(b, self.key_channels, -1)
        query = query.permute(0, 2, 1)

        key = self.f_key(x)
        key = self.psp(key)

        sim_map = torch.matmul(query, key)
        sim_map = (self.key_channels ** -.5) * sim_map
        sim_map = F.softmax(sim_map, dim=-1)

        context = torch.matmul(sim_map, value)
        context = context.permute(0, 2, 1).contiguous()
        context = context.view(b, self.value_channels, *x.size()[2:])
        context = self.W(context)

        return context

class APNB(nn.Module):
    """
    Parameters:
        in_features / out_features: the channels of the input / output feature maps.
        dropout: we choose 0.05 as the default value.
        size: you can apply multiple sizes. Here we only use one size.
    Return:
        features fused with Object context information.
    """

    def __init__(self, in_channels, out_channels, key_channels, value_channels, dropout, norm, psp_size, sizes=([1])):
        super(APNB, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.key_channels = key_channels
        self.value_channels = value_channels
        self.stages = []
        self.norm = norm
        self.psp_size = psp_size
        self.stages = nn.ModuleList(
            [self._make_stage(in_channels, out_channels, key_channels, value_channels, size) for size in sizes])
        self.conv_bn_dropout = nn.Sequential(
            nn.Conv3d(2 * in_channels, out_channels, kernel_size=1, padding=0),
            # ModuleHelper.BNReLU(out_channels, norm_type=norm_type),
            normalization(self.in_channels, norm=norm),
            nn.ReLU(inplace=True),
        nn.Dropout3d(dropout)
        )

    def _make_stage(self, in_channels, output_channels, key_channels, value_channels, size):
        return SelfAttentionBlock3D(in_channels=self.in_channels,
                                    key_channels=self.key_channels,
                                    value_channels=self.value_channels,
                                    out_channels=self.out_channels,
                                    norm=self.norm,
                                    psp_size=self.psp_size)

    def forward(self, feats):
        priors = [stage(feats) for stage in self.stages]
        context = priors[0]
        for i in range(1, len(priors)):
            context += priors[i]
        # print('context.shape:', context.shape)
        # print('teats.shape:', feats.shape)
        output = torch.cat([context, feats], 1)
        output = self.conv_bn_dropout(output)
        return output

class Nature_APNB(nn.Module):
    def __init__(self, c, num_classes, base_channel=64):
        super(Nature_APNB, self).__init__()
        self.enc1 = ConvBlock_Down(c, base_channel*1)
        self.enc2 = EncoderBlock(base_channel*1, base_channel*2)
        self.enc3 = EncoderBlock(base_channel*2, base_channel*4)
        self.enc4 = EncoderBlock(base_channel*4, base_channel*8)

        # self.non_local = NonLocalBlock(in_channels=base_channel*4, inter_channels=base_channel*4)
        self.APNB3 = APNB(in_channels=base_channel*4, out_channels=base_channel*4, key_channels=base_channel*2, value_channels=base_channel*2,
                          dropout=0.05, sizes=([1]), norm='sync_bn', psp_size=(1, 3, 6, 8))

        self.dec3_1 = DecoderBlock2(base_channel*8, base_channel*4)
        self.dec3_2 = ConvBlock_Up(base_channel*8, base_channel*4, dropout=True)

        self.dec2_1 = DecoderBlock2(base_channel*4, base_channel*2)
        self.dec2_2 = ConvBlock_Up(base_channel*4, base_channel*2, dropout=True)

        self.dec1_1 = DecoderBlock2(base_channel * 2, base_channel)
        self.dec1_2 = ConvBlock_Up(base_channel*2, base_channel, dropout=True)

        self.final = nn.Conv3d(base_channel, num_classes, kernel_size=1)
        self.softmax = nn.Softmax(1)
        # initialize_weights(self)

    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        # print('x3:', x3.shape)
        # x3n = self.APNB3(x3)
        x4 = self.enc4(x3)
        # print('x4:', x4.shape)

        y3 = self.dec3_1(x4)
        # print('y3:', y3.shape)
        y3_1 = torch.cat((x3n, y3), 1)

        y3_1 = self.dec3_2(y3_1)
        y3 = y3_1 + y3

        y2 = self.dec2_1(y3)
        y2_1 = torch.cat((x2, y2), 1)
        y2_1 = self.dec2_2(y2_1)
        y2 = y2_1 + y2

        y1 = self.dec1_1(y2)
        y1_1 = torch.cat((x1, y1), 1)
        y1_1 = self.dec1_2(y1_1)
        y1 = y1_1 + y1

        y = self.final(y1)
        y = self.softmax(y)

        return y


if __name__ == '__main__':
    with torch.no_grad():
        import os
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        model = Nature_APNB(c=4, num_classes=3, base_channel=16)
        device = torch.device('cuda:0')

        model = model.cuda()
        x = torch.rand(1, 4, 128, 128, 128, device=device)
        # x = torch.rand(1, 1, 512, 512, 48, device=device)
        y = model(x)
        print(y.shape)

        # # summary(model, (4, 128, 128, 128))
        #
        # flops, params = get_model_complexity_info(model, (1, 384, 384, 48), as_strings=True, print_per_layer_stat=False)
        # print('Flops:  ' + flops)
        # print('Params: ' + params)
















