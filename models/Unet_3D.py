import torch.nn as nn
import torch.nn.functional as F
import torch
import ipdb
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

    def __init__(self, in_c, in_h, in_t, ConvND=nn.Conv3d, normalize=None, mode=None):
        super().__init__()
        self.node_f = in_c  # 32
        self.state_f = in_c  # 32
        self.node_t = 128
        self.state_t = 128
        self.in_h = self.in_w = in_h
        self.normalize = normalize
        self.mode = mode

        self.S_project = ConvND(in_c, in_c, kernel_size=3, padding=1, stride=2, groups=in_c)
        self.phi_s = ConvND(in_channels=self.in_h // 2, out_channels=self.in_h // 2, kernel_size=1)
        self.v = ConvND(in_channels=self.in_w // 2, out_channels=self.in_w // 2, kernel_size=1)
        self.delta = ConvND(in_c, in_c, 1)
        self.Ws = ConvND(in_c, in_c, 1)
        self.xi = ConvND(in_c, in_c, 1)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(-1)

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

        self.weight1 = nn.Parameter(torch.ones(1))
        self.weight2 = nn.Parameter(torch.ones(1))
        self.weight3 = nn.Parameter(torch.ones(1))

    def forward(self, x):
        b, c, h, w, d = x.size()
        s_in = f_in = x  # (b, c, h, w, d)
        t_in = x.permute(0, 4, 1, 2, 3).contiguous()  # (b, d, c, h, w)

        '''the feature branch'''
        phi_f = self.phi_f(f_in).view(b, self.state_f, -1)  # (b, state_f, d*h*w)
        theta_f = self.theta_f(f_in).view(b, self.node_f, -1)  # (b, node_f, d*h*w)
        graph_f = torch.matmul(phi_f, theta_f.permute(0, 2, 1))  # (b, state_f, node_f)
        if self.normalize:
            graph_f = graph_f * (1. / graph_f.size(2))
        out_f = self.GCN_f(graph_f)  # (b, state_f, node_f)
        out_f = torch.matmul(out_f, theta_f).view(b, self.state_f, *x.size()[2:])  # (b, state_f, h, w, d)
        out_f = self.extend_f(out_f)  # (b, c, h, w, d)

        '''the temporal branch'''
        phi_t = self.phi_t(t_in).view(b, self.state_t, -1)  # (b, state_t, c*h*w)
        theta_t = self.theta_t(t_in).view(b, self.node_t, -1)  # (b, node_t, c*h*w)
        graph_t = torch.matmul(phi_t, theta_t.permute(0, 2, 1))  # (b, state_t, node_t)
        if self.normalize:
            graph_t = graph_t * (1. / graph_torch.size(2))
        out_t = self.GCN_t(graph_t)  # (b, state_t, node_t)
        out_t = torch.matmul(out_t, theta_t).view(b, self.state_t, *x.size()[1:4])  # (b, state_t, c, h, w)
        out_t = self.extend_t(out_t).permute(0, 2, 3, 4, 1)  # (b, d, c, h, w)-->(b, c, h, w, d)

        '''the spatial branch'''
        if 's' in self.mode.lower():
            Hs = self.S_project(s_in)  # (b, c, h//2, w//2, d//2)
            phi_s = self.phi_s(Hs.permute(0, 2, 1, 3, 4)).permute(0, 1, 3, 2, 4).contiguous().view(b, h * w // 4, -1)
            # (b, h, c, w, d)-->(b, h, w, c, d)-->(b, hw//4, cd//2)
            v = self.v(Hs.permute(0, 3, 1, 2, 4)).permute(0, 1, 3, 2, 4).contiguous().view(b, h * w // 4, -1)
            # (b, w, c, h, d)-->(b, w, h, c, d)-->(b, hw//4, cd//2)
            A = self.softmax(torch.matmul(v, phi_s.permute(0, 2, 1)))  # (b, hw//4, hw//4)
            delta = self.delta(Hs).view(b, c * d // 2, -1)  # (b, c, h//2, w//2, d//2)-->(b, cd//2, hw//4)
            AVs = torch.matmul(delta, A).view(b, c, d // 2, h // 2, w // 2).permute(0, 1, 3, 4,
                                                                                2).contiguous()  # (b, cd//2, hw//4)-->(b, c, h//2, w//2, d//2)
            Ws = self.Ws(AVs)  # (b, c, h, w, d)
            Ws = torch.nn.functional.interpolate(Ws, scale_factor=2, mode='nearest')
            out_s = self.xi(Ws)
            return x + self.weight1 * out_s + self.weight2 * out_f + self.weight3 * out_t

        return x + self.weight2 * out_f + self.weight3 * out_t

class Conv3D(nn.Module):
    def __init__(self, c_in, c_out, kernel_size=3, stride=1, padding=None, norm='sync_bn'):
        super().__init__()

        if padding == None:
            padding = (kernel_size - 1) // 2
        self.bn = normalization(c_in, norm=norm)
        # self.act_fn = nn.ReLU(inplace=True)
        # self.act_fn = nn.PReLU() # report error : out of CUDA
        # self.act_fn = nn.ELU(inplace=True)
        self.act_fn = nn.ReLU()
        self.conv = nn.Conv3d(c_in, c_out, kernel_size, stride, padding, bias=False)

    def forward(self, x):
        return self.conv(self.act_fn(self.bn(x)))

class Conv_down(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.bn = normalization(c_in)
        self.act_fn = nn.ReLU()
        self.conv = nn.Conv3d(c_in, c_out, kernel_size=3, stride=2, padding=1, bias=False)

    def forward(self, x):
        return self.conv(self.act_fn(self.bn(x)))

class Residual_block(nn.Module):
    def __init__(self, c_in):
        super().__init__()
        self.conv1 = Conv3D(c_in, c_in)
        self.conv2 = Conv3D(c_in, c_in)

    def forward(self, x):
        return self.conv2(self.conv1(x))+x


class Unet_3D(nn.Module):
    def __init__(self, c_in=4, num_classes=4, channels=16):
        super().__init__()

        self.layer1 = Conv3D(c_in, channels)

        self.encoder1 = nn.Conv3d(channels, channels, kernel_size=3, stride=2, padding=1, bias=False)

        self.encoder2 = nn.Sequential(
            Conv_down(channels, 2*channels),
            Residual_block(2*channels),
            Residual_block(2*channels)
        )

        self.encoder3 = nn.Sequential(
            Conv_down(2*channels, 4 * channels),
            Residual_block(4 * channels),
            Residual_block(4 * channels)
        )

        self.encoder4 = nn.Sequential(
            Conv_down(4 * channels, 8 * channels),
            Residual_block(8 * channels),
            Residual_block(8 * channels)
        )

        self.encoder5 = nn.Sequential(
            Conv_down(8 * channels, 16 * channels),
            Residual_block(16 * channels),
            Conv3D(16*channels, 8*channels)
        )

        self.decoder4 = nn.Sequential(
            Conv3D(16*channels, 8*channels),
            Conv3D(8*channels, 4*channels)
        )

        self.decoder3 = nn.Sequential(
            Conv3D(8 * channels, 4 * channels),
            Conv3D(4*channels, 2*channels)
        )

        self.decoder2 = nn.Sequential(
            Conv3D(4 * channels, 2 * channels),
            Conv3D(2*channels, channels)
        )

        self.decoder1 = nn.Sequential(
            Conv3D(2*channels, channels)
        )

        self.decoder = nn.Sequential(
            Residual_block(2*channels),
            Conv3D(2*channels, num_classes)
        )

        self.softmax = nn.Softmax(1)

    def forward(self, x):
        x = self.layer1(x)
        x1 = self.encoder1(x)
        x2 = self.encoder2(x1)
        x3 = self.encoder3(x2)
        x4 = self.encoder4(x3)
        x5 = self.encoder5(x4)
        x5 = F.interpolate(x5, scale_factor=2, mode='trilinear')
        x4 = torch.cat([x4, x5], dim=1)
        x4 = self.decoder4(x4)
        x4 = F.interpolate(x4, scale_factor=2, mode='trilinear')
        x3 = torch.cat([x3, x4], dim=1)
        x3 = self.decoder3(x3)
        x3 = F.interpolate(x3, scale_factor=2, mode='trilinear')
        x2 = torch.cat([x2, x3], dim=1)
        x2 = self.decoder2(x2)
        x2 = F.interpolate(x2, scale_factor=2, mode='trilinear')
        x1 = torch.cat([x1, x2], dim=1)
        x1 = self.decoder1(x1)
        x1 = F.interpolate(x1, scale_factor=2, mode='trilinear')
        x = torch.cat([x, x1], dim=1)
        x = self.decoder(x)
        out = self.softmax(x)

        return out

class Unet_3D_SFT(nn.Module):
    def __init__(self, c_in=4, num_classes=4, channels=16):
        super().__init__()

        self.layer1 = Conv3D(c_in, channels)

        self.encoder1 = nn.Conv3d(channels, channels, kernel_size=3, stride=2, padding=1, bias=False)

        self.encoder2 = nn.Sequential(
            Conv_down(channels, 2*channels),
            Residual_block(2*channels),
            Residual_block(2*channels)
        )

        self.encoder3 = nn.Sequential(
            Conv_down(2*channels, 4 * channels),
            Residual_block(4 * channels),
            Residual_block(4 * channels)
        )

        self.down4 = Conv_down(4 * channels, 8 * channels)
        self.SFT4 = SFT_adaptive(128, 8, 8, mode='sft')
        self.encoder4 = nn.Sequential(
            Residual_block(8 * channels),
            Residual_block(8 * channels)
        )

        self.down5 = Conv_down(8 * channels, 16 * channels)
        self.SFT5 = SFT_adaptive(256, 4, 4, mode='sft')
        self.encoder5 = nn.Sequential(
            Residual_block(16 * channels),
            Residual_block(16 * channels),
            Conv3D(16*channels, 8*channels)
        )

        self.decoder4 = nn.Sequential(
            Conv3D(16*channels, 8*channels),
            Conv3D(8*channels, 4*channels)
        )

        self.decoder3 = nn.Sequential(
            Conv3D(8 * channels, 4 * channels),
            Conv3D(4*channels, 2*channels)
        )

        self.decoder2 = nn.Sequential(
            Conv3D(4 * channels, 2 * channels),
            Conv3D(2*channels, channels)
        )

        self.decoder1 = nn.Sequential(
            Conv3D(2*channels, channels)
        )

        self.decoder = nn.Sequential(
            Residual_block(2*channels),
            Conv3D(2*channels, num_classes)
        )

        self.softmax = nn.Softmax(1)

    def forward(self, x):
        x = self.layer1(x)
        x1 = self.encoder1(x)
        x2 = self.encoder2(x1)
        x3 = self.encoder3(x2)
        x4 = self.down4(x3)
        x4 = self.SFT4(x4)
        x4 = self.encoder4(x4)
        x5 = self.down5(x4)
        x5 = self.SFT5(x5)
        x5 = self.encoder5(x5)
        x5 = F.interpolate(x5, scale_factor=2, mode='trilinear')
        x4 = torch.cat([x4, x5], dim=1)
        x4 = self.decoder4(x4)
        x4 = F.interpolate(x4, scale_factor=2, mode='trilinear')
        x3 = torch.cat([x3, x4], dim=1)
        x3 = self.decoder3(x3)
        x3 = F.interpolate(x3, scale_factor=2, mode='trilinear')
        x2 = torch.cat([x2, x3], dim=1)
        x2 = self.decoder2(x2)
        x2 = F.interpolate(x2, scale_factor=2, mode='trilinear')
        x1 = torch.cat([x1, x2], dim=1)
        x1 = self.decoder1(x1)
        x1 = F.interpolate(x1, scale_factor=2, mode='trilinear')
        x = torch.cat([x, x1], dim=1)
        x = self.decoder(x)
        out = self.softmax(x)

        return out



if __name__ == '__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = torch.device('cuda:0')
    x = torch.rand((1, 4, 128, 128, 128), device=device)

    model = Unet_3D_SFT()
    model.cuda(device)
    y = model(x)
    print(y.shape)

