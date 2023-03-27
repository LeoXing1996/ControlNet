import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D convolution module.
    """
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def avg_pool_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D average pooling module.
    """
    if dims == 1:
        return nn.AvgPool1d(*args, **kwargs)
    elif dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    elif dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self,
                 channels,
                 use_conv,
                 dims=2,
                 out_channels=None,
                 padding=1):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(dims,
                              self.channels,
                              self.out_channels,
                              3,
                              stride=stride,
                              padding=padding)
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


class ResnetBlock(nn.Module):

    def __init__(self, in_c, out_c, down, ksize=3, sk=False, use_conv=True):
        super().__init__()
        ps = ksize // 2
        if in_c != out_c or sk == False:
            self.in_conv = nn.Conv2d(in_c, out_c, ksize, 1, ps)
        else:
            # print('n_in')
            self.in_conv = None
        self.block1 = nn.Conv2d(out_c, out_c, 3, 1, 1)
        self.act = nn.ReLU()
        self.block2 = nn.Conv2d(out_c, out_c, ksize, 1, ps)
        if sk == False:
            self.skep = nn.Conv2d(in_c, out_c, ksize, 1, ps)
        else:
            self.skep = None

        self.down = down
        if self.down == True:
            self.down_opt = Downsample(in_c, use_conv=use_conv)

    def forward(self, x):
        import ipdb
        ipdb.set_trace()
        if self.skep is not None:
            shoutcut = self.skep(x)

        if self.down == True:
            x = self.down_opt(x)
        if self.in_conv is not None:  # edit
            x = self.in_conv(x)

        h = self.block1(x)
        h = self.act(h)
        h = self.block2(h)
        if self.skep is not None:
            # return h + self.skep(x)
            return h + shoutcut
        else:
            return h + x


class MiddleAdapter(nn.Module):

    def __init__(self,
                 in_dim,
                 cond_dim=3,
                 channel_list=[320, 320],
                 nums_rb=3) -> None:
        super().__init__()
        self.unshuffle = nn.PixelUnshuffle(8)
        # self.channels = channel_list
        self.nums_rb = nums_rb
        self.body = []
        ksize = 3
        sk = False
        use_conv = True

        self.cond_conv = ResnetBlock(
            cond_dim,
            in_dim,
            down=False,
            ksize=ksize,
            sk=sk,
        )

        blocks = []
        ch_in = in_dim * 2  # feature + cond
        for ch_out in channel_list + [in_dim]:
            blocks.append(
                ResnetBlock(ch_in,
                            ch_out,
                            down=False,
                            ksize=ksize,
                            sk=sk,
                            use_conv=use_conv))
            ch_in = ch_out
        self.net = nn.Sequential(*blocks)

    def forward(self, x_in, cond):
        cond_in = F.interpolate(cond, x_in.shape[-2:], mode='bilinear')
        cond_feat = self.cond_conv(cond_in)

        feat_in = torch.cat([cond_feat, x_in], dim=1)
        feat_adapted = self.net(feat_in)
        x_out = x_in + feat_adapted
        return x_out
