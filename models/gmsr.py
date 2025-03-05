import torch
import torch.nn as nn
from torch.nn import functional as F

from .rep import CBAReParam, EDBB


class GMSR(nn.Module):
    def __init__(
        self,
        scale=1,
        num_input_channels=4,
        channel=32,
        df_num=10,
    ):
        super(GMSR, self).__init__()
        self.nbit_w = 8
        self.nbit_b = 32
        self.nbit_a = 8
        self.tail_act_bits = 8

        self.sf = Conv(
            in_channels=num_input_channels,
            out_channels=channel,
            kernel_size=3,
            nbit_w=self.nbit_w,
            nbit_b=self.nbit_b,
            nbit_a=self.nbit_a,
            act="Tanh",
        )
        self.df = []
        for _ in range(df_num):
            self.df.append(
                Conv(
                    in_channels=channel,
                    out_channels=channel,
                    kernel_size=3,
                    nbit_w=self.nbit_w,
                    nbit_b=self.nbit_b,
                    nbit_a=self.nbit_a,
                    act="ReLU",
                )
            )
        self.df = nn.Sequential(*self.df)

        self.transition = Conv(
            in_channels=channel * 2,
            out_channels=channel,
            kernel_size=3,
            nbit_w=self.nbit_w,
            nbit_b=self.nbit_b,
            nbit_a=self.nbit_a,
            act="None",
        )

        self.last_conv = Conv(
            in_channels=channel + num_input_channels,
            out_channels=scale * scale * num_input_channels,
            kernel_size=1,
            nbit_w=self.nbit_w,
            nbit_b=self.nbit_b,
            nbit_a=self.tail_act_bits,
            act="ReLU",
            padding=0,
        )

    def forward(self, x):
        x = quant_a(x, self.nbit_a, "ReLU")
        x = torch.nn.functional.pixel_unshuffle(x, 2)
        img = x

        sf_feat = self.sf(x)

        feat = self.df(sf_feat)

        feat = torch.cat([feat, sf_feat], dim=1)
        feat = quant_a(feat, self.nbit_a, "Tanh")

        feat = self.transition(feat)

        feat = torch.cat([feat, img], dim=1)
        feat = quant_a(feat, self.nbit_a, "Tanh")

        feat = self.last_conv(feat)
        feat = torch.clamp(feat, 0.0, 1.0)
        out = torch.nn.functional.pixel_shuffle(feat, 2)

        return out


class GMSR_FP32(nn.Module):
    def __init__(
        self,
        scale=1,
        num_input_channels=4,
        channel=64,
        df_num=12,
    ):
        super(GMSR_FP32, self).__init__()

        self.sf = nn.Conv2d(
            in_channels=num_input_channels,
            out_channels=channel,
            kernel_size=3,
            padding=1,
        )

        self.df = []
        for _ in range(df_num):
            self.df.append(
                nn.Conv2d(
                    in_channels=channel,
                    out_channels=channel,
                    kernel_size=3,
                    padding=1,
                )
            )
            self.df.append(nn.ReLU())
        self.df = nn.Sequential(*self.df)

        self.transition = nn.Conv2d(
            in_channels=channel * 2,
            out_channels=channel,
            kernel_size=3,
            padding=1,
        )

        self.last_conv = nn.Conv2d(
            in_channels=channel + num_input_channels,
            out_channels=4 * 1,
            kernel_size=1,
            padding=0,
        )

    def forward(self, x):
        x = torch.nn.functional.pixel_unshuffle(x, 2)
        img = x

        sf_feat = self.sf(x)
        sf_feat = F.hardtanh(sf_feat, -1, 1)

        feat = self.df(sf_feat)

        feat = torch.cat([feat, sf_feat], dim=1)

        feat = self.transition(feat)
        feat = F.hardtanh(feat, -1, 1)

        feat = torch.cat([feat, img], dim=1)

        feat = self.last_conv(feat)
        feat = F.relu(feat)
        feat = torch.clamp(feat, 0.0, 1.0)
        out = torch.nn.functional.pixel_shuffle(feat, 2)

        return out


class GMSR_ECB(nn.Module):
    def __init__(
        self,
        scale=2,
        num_input_channels=4,
        channel=128,
        df_num=10,
    ):
        super(GMSR_ECB, self).__init__()

        self.sf = CBAReParam(
            num_input_channels, channel, 3, 1, 1, act="LReLU", bn=False, type="ecb"
        )

        self.df = []
        for _ in range(df_num):
            self.df.append(
                CBAReParam(channel, channel, 3, 1, 1, act="LReLU", bn=False, type="ecb")
            )
            self.df.append(nn.ReLU())
        self.df = nn.Sequential(*self.df)

        self.transition = CBAReParam(
            channel * 2, channel, 3, 1, 1, act="LReLU", bn=False, type="ecb"
        )

        self.last_conv = CBAReParam(
            channel + num_input_channels, 64, 3, 1, 1, act="ReLU", bn=False, type="ecb"
        )

    def forward(self, input):
        yuv = RGB2YCbCr(input)
        x, uv = yuv[:, :1, :, :], yuv[:, 1:, :, :]
        x = torch.nn.functional.pixel_unshuffle(x, 2)
        img = x

        sf_feat = self.sf(x)

        feat = self.df(sf_feat)

        feat = torch.cat([feat, sf_feat], dim=1)

        feat = self.transition(feat)

        feat = torch.cat([feat, img], dim=1)

        feat = self.last_conv(feat)
        feat = torch.clamp(feat, 0.0, 1.0)
        out = torch.nn.functional.pixel_shuffle(feat, 8)

        uv = F.interpolate(uv, scale_factor=4, mode="bicubic", align_corners=False)
        yuv = torch.cat([out, uv], dim=1)
        out = YCbCr2RGB(yuv)

        return out


class Sequential(nn.Sequential):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x):
        for layer in self:
            x = layer(x)
        return x


class SlimUNetModelConv(nn.Module):
    """
    The full UNet model with attention and timestep embedding.
    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
    :param resblock_updown: use residual blocks for up/downsampling.
    """

    def __init__(
        self,
        in_channels=3,
        model_channels=96,
        out_channels=3 * 16,
        num_res_blocks=[1, 1, 1],
        channel_mult=[1, 1, 1],
    ):
        super().__init__()

        if isinstance(num_res_blocks, int):
            num_res_blocks = [
                num_res_blocks,
            ] * len(channel_mult)
        else:
            assert len(num_res_blocks) == len(channel_mult)
        self.num_res_blocks = num_res_blocks

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.channel_mult = channel_mult

        init_in_ch = in_channels
        ch = input_ch = int(channel_mult[-1] * model_channels)
        self.input_blocks = nn.ModuleList(
            [
                Sequential(
                    CBAReParam(
                        init_in_ch, ch, 3, 1, 1, act="LReLU", bn=False, type="ecb"
                    )
                )
            ]
        )
        input_block_chans = []
        for level, mult in enumerate(channel_mult):
            layers = []
            for _ in range(num_res_blocks[level]):
                layers.append(
                    CBAReParam(
                        ch,
                        int(mult * model_channels),
                        3,
                        1,
                        1,
                        act="LReLU",
                        bn=False,
                        type="ecb",
                    )
                )
                ch = int(mult * model_channels)
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                layers.append(
                    Sequential(
                        CBAReParam(
                            ch, ch, k=3, s=2, p=1, act="LReLU", bn=False, type="basic"
                        )
                    )
                )
                ch = out_ch
            self.input_blocks.append(Sequential(*layers))

        self.middle_block = Sequential(
            CBAReParam(ch, ch, 3, 1, 1, act="LReLU", bn=False, type="ecb"),
            CBAReParam(ch, ch, 3, 1, 1, act="LReLU", bn=False, type="ecb"),
        )

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult)):
            layers = []
            ich = input_block_chans.pop()
            for i in range(num_res_blocks[level]):
                layers.append(
                    CBAReParam(
                        ch + ich,
                        int(model_channels * mult),
                        3,
                        1,
                        1,
                        act="LReLU",
                        bn=False,
                        type="ecb",
                    ),
                )
                ch = int(model_channels * mult)
                if level and i == num_res_blocks[level] - 1:
                    out_ch = ch
                    layers.append(
                        CBAReParam(
                            ch, out_ch, 3, 1, 1, act="LReLU", bn=False, type="ecb"
                        ),
                    )
                    layers.append(nn.Upsample(scale_factor=2, mode="nearest"))
            self.output_blocks.append(Sequential(*layers))

        self.out = nn.Sequential(
            CBAReParam(
                input_ch, out_channels, 3, 1, 1, act="LReLU", bn=False, type="ecb"
            ),
        )

    def forward(self, x):
        """
        Apply the model to an input batch.
        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param lq: an [N x C x ...] Tensor of low quality iamge.
        :return: an [N x C x ...] Tensor of outputs.
        """
        hs = []
        h = x
        for i, module in enumerate(self.input_blocks):
            h = module(h)
            hs.append(h)
        h = self.middle_block(h)
        cnt = 0
        for module in self.output_blocks:
            pop = hs.pop()
            h = h[:, :, : pop.shape[2], : pop.shape[3]]
            h = torch.cat([h, pop], dim=1)
            h = module(h)
            cnt += 1
        h = h.type(x.dtype)
        out = self.out(h)
        out = F.pixel_shuffle(out, 4)
        return out[:, :, : x.shape[2] * 4, : x.shape[3] * 4]


def RGB2YCbCr(rgb):
    # BT601
    ycbcr = rgb.clone()
    ycbcr[:, :1, :, :] = (
        0.298828 * rgb[:, :1, :, :]
        + 0.586914 * rgb[:, 1:2, :, :]
        + 0.114258 * rgb[:, 2:, :, :]
    )
    ycbcr[:, 1:2, :, :] = (
        -0.167969 * rgb[:, :1, :, :]
        - 0.330078 * rgb[:, 1:2, :, :]
        + 0.498047 * rgb[:, 2:, :, :]
    )
    ycbcr[:, 2:, :, :] = (
        0.498047 * rgb[:, :1, :, :]
        - 0.416992 * rgb[:, 1:2, :, :]
        - 0.081055 * rgb[:, 2:, :, :]
    )
    return ycbcr


def YCbCr2RGB(ycbcr):
    # BT601
    rgb = ycbcr.clone()
    rgb[:, :1, :, :] = ycbcr[:, :1, :, :] + 1.407739 * ycbcr[:, 2:, :, :]
    rgb[:, 1:2, :, :] = (
        ycbcr[:, :1, :, :]
        - 0.3460532 * ycbcr[:, 1:2, :, :]
        - 0.716708 * ycbcr[:, 2:, :, :]
    )
    rgb[:, 2:, :, :] = ycbcr[:, :1, :, :] + 1.778394 * ycbcr[:, 1:2, :, :]
    return rgb


class GMSR_EDBB(nn.Module):
    def __init__(
        self,
        scale=2,
        num_input_channels=4,
        channel=128,
        df_num=10,
    ):
        super(GMSR_EDBB, self).__init__()

        self.sf = EDBB(num_input_channels, channel)

        self.df = []
        for _ in range(df_num):
            self.df.append(EDBB(channel, channel))
        self.df = nn.Sequential(*self.df)

        self.transition = EDBB(channel * 2, channel)

        self.last_conv = EDBB(channel + num_input_channels, 64, act_type="relu")

    def forward(self, input):
        yuv = RGB2YCbCr(input)
        x, uv = yuv[:, :1, :, :], yuv[:, 1:, :, :]
        x = torch.nn.functional.pixel_unshuffle(x, 2)
        img = x

        sf_feat = self.sf(x)

        feat = self.df(sf_feat)

        feat = torch.cat([feat, sf_feat], dim=1)

        feat = self.transition(feat)

        feat = torch.cat([feat, img], dim=1)

        feat = self.last_conv(feat)
        feat = torch.clamp(feat, 0.0, 1.0)
        out = torch.nn.functional.pixel_shuffle(feat, 8)

        uv = F.interpolate(uv, scale_factor=4, mode="bicubic", align_corners=False)
        yuv = torch.cat([out, uv], dim=1)
        out = YCbCr2RGB(yuv)

        return out


class GMVSR(nn.Module):
    def __init__(
        self,
        scale=2,
        num_input_channels=16,
        channel=64,
        df_num=10,
    ):
        super(GMVSR, self).__init__()
        self.nbit_w = 8
        self.nbit_b = 32
        self.nbit_a = 8
        self.tail_act_bits = 8
        self.channel = channel

        self.sf = Conv(
            in_channels=num_input_channels,
            out_channels=channel,
            kernel_size=3,
            nbit_w=self.nbit_w,
            nbit_b=self.nbit_b,
            nbit_a=self.nbit_a,
            act="ReLU",
        )
        middle = df_num // 2 if df_num % 2 == 0 else df_num // 2 + 1
        self.df_1 = []
        for _ in range(middle):
            self.df_1.append(
                Conv(
                    in_channels=channel,
                    out_channels=channel,
                    kernel_size=3,
                    nbit_w=self.nbit_w,
                    nbit_b=self.nbit_b,
                    nbit_a=self.nbit_a,
                    act="ReLU",
                )
            )
        self.df_1 = nn.Sequential(*self.df_1)

        self.hidden_transition = Conv(
            in_channels=channel,
            out_channels=channel,
            kernel_size=1,
            nbit_w=self.nbit_w,
            nbit_b=self.nbit_b,
            nbit_a=self.nbit_a,
            act="ReLU",
            padding=0,
        )

        self.after_concat = Conv(
            in_channels=channel * 2,
            out_channels=channel,
            kernel_size=3,
            nbit_w=self.nbit_w,
            nbit_b=self.nbit_b,
            nbit_a=self.nbit_a,
            act="ReLU",
        )

        self.df_2 = []
        for _ in range(df_num - middle - 1):
            self.df_2.append(
                Conv(
                    in_channels=channel,
                    out_channels=channel,
                    kernel_size=3,
                    nbit_w=self.nbit_w,
                    nbit_b=self.nbit_b,
                    nbit_a=self.nbit_a,
                    act="ReLU",
                )
            )
        self.df_2 = nn.Sequential(*self.df_2)

        self.transition = Conv(
            in_channels=channel,
            out_channels=channel,
            kernel_size=3,
            nbit_w=self.nbit_w,
            nbit_b=self.nbit_b,
            nbit_a=self.nbit_a,
            act="ReLU",
        )

        self.last_conv = Conv(
            in_channels=channel + num_input_channels,
            out_channels=16,
            kernel_size=1,
            nbit_w=self.nbit_w,
            nbit_b=self.nbit_b,
            nbit_a=self.tail_act_bits,
            act="ReLU",
            padding=0,
        )

    def forward(self, input):
        B, N, _, H, W = input.shape
        hidden = torch.zeros((B, self.channel, H // 2, W // 2), device=input.device)
        output = []
        for i in range(N):
            x = input[:, i, :, :, :]
            x = torch.nn.functional.pixel_unshuffle(x, 2)
            x = quant_a(x, self.nbit_a, "ReLU")
            img = x

            sf_feat = self.sf(x)

            df1_feat = self.df_1(sf_feat)

            hidden = quant_a(hidden, self.nbit_a, "ReLU")
            hidden = self.hidden_transition(hidden)

            feat = torch.cat([hidden, df1_feat], dim=1)
            feat = quant_a(feat, self.nbit_a, "ReLU")
            feat = hidden = self.after_concat(feat)

            feat = self.df_2(feat)

            feat = feat + sf_feat
            feat = quant_a(feat, self.nbit_a, "ReLU")

            feat = self.transition(feat)

            feat = torch.cat([feat, img], dim=1)
            feat = quant_a(feat, self.nbit_a, "ReLU")

            feat = self.last_conv(feat)
            feat = torch.clamp(feat, 0.0, 1.0)
            out = torch.nn.functional.pixel_shuffle(feat, 4)

            output.append(out)

        return torch.stack(output, dim=1)


class GMVSR_FP32(nn.Module):
    def __init__(
        self,
        scale=1,
        num_input_channels=4,
        channel=64,
        df_num=12,
        luma_only=False,
        uv_channel=32,
        uv_df_num=7,
    ):
        super(GMVSR_FP32, self).__init__()
        self.channel = channel
        self.luma_only = luma_only

        self.sf = nn.Conv2d(
            in_channels=num_input_channels,
            out_channels=channel,
            kernel_size=3,
            padding=1,
        )
        middle = df_num // 2 if df_num % 2 == 0 else df_num // 2 + 1
        self.df_1 = []
        for _ in range(middle):
            self.df_1.append(
                nn.Conv2d(
                    in_channels=channel,
                    out_channels=channel,
                    kernel_size=3,
                    padding=1,
                )
            )
        self.df_1 = nn.Sequential(*self.df_1)

        self.hidden_transition = nn.Conv2d(
            in_channels=channel,
            out_channels=channel,
            kernel_size=1,
            padding=0,
        )

        self.after_concat = nn.Conv2d(
            in_channels=channel * 2,
            out_channels=channel,
            kernel_size=3,
            padding=1,
        )

        self.df_2 = []
        for _ in range(df_num - middle - 1):
            self.df_2.append(
                nn.Conv2d(
                    in_channels=channel,
                    out_channels=channel,
                    kernel_size=3,
                    padding=1,
                )
            )
        self.df_2 = nn.Sequential(*self.df_2)

        self.transition = nn.Conv2d(
            in_channels=channel,
            out_channels=channel,
            kernel_size=3,
            padding=1,
        )

        self.last_conv = nn.Conv2d(
            in_channels=channel + num_input_channels,
            out_channels=4,
            kernel_size=1,
            padding=0,
        )

        if not self.luma_only:
            self.uv_df_num = uv_df_num
            self.uv_channel = uv_channel
            self.init_chroma()

    def forward(self, input):
        if self.luma_only:
            return self.forward_luma(input)
        else:
            return self.forward_yuv(input)

    def forward_yuv(self, input):
        luma, chroma = input[0], input[1]
        B, N, _, H, W = luma.shape
        hidden = torch.zeros((B, self.channel, H // 2, W // 2), device=luma.device)
        output = []
        output_uv = []
        for i in range(N):
            x = luma[:, i, :, :, :]
            x = torch.nn.functional.pixel_unshuffle(x, 2)
            img = x

            sf_feat = self.sf(x)
            sf_feat = feat = F.relu(sf_feat)

            for block in self.df_1:
                feat = block(feat)
                feat = F.relu(feat)
            df1_feat = feat

            hidden = self.hidden_transition(hidden)
            hidden = F.relu(hidden)
            feat = torch.cat([hidden, df1_feat], dim=1)
            feat = self.after_concat(feat)
            feat = hidden = F.relu(feat)

            for block in self.df_2:
                feat = block(feat)
                feat = F.relu(feat)

            feat = feat + sf_feat

            feat = self.transition(feat)
            feat = F.relu(feat)

            feat = torch.cat([feat, img], dim=1)

            feat = self.last_conv(feat)
            feat = F.relu(feat)
            feat = torch.clamp(feat, 0.0, 1.0)
            out = torch.nn.functional.pixel_shuffle(feat, 2)
            output.append(out)

            uv = chroma[:, i, :, :, :] + 0.5
            uv = self.forward_chroma(uv)
            uv = uv - 0.5
            output_uv.append(uv)

        return torch.stack(output, dim=1), torch.stack(output_uv, dim=1)

    def forward_luma(self, input):
        B, N, _, H, W = input.shape
        hidden = torch.zeros((B, self.channel, H // 2, W // 2), device=input.device)
        output = []

        for i in range(N):
            x = input[:, i, :, :, :]
            x = torch.nn.functional.pixel_unshuffle(x, 2)
            img = x

            sf_feat = self.sf(x)
            sf_feat = feat = F.relu(sf_feat)

            for block in self.df_1:
                feat = block(feat)
                feat = F.relu(feat)
            df1_feat = feat

            hidden = self.hidden_transition(hidden)
            hidden = F.relu(hidden)
            feat = torch.cat([hidden, df1_feat], dim=1)
            feat = self.after_concat(feat)
            feat = hidden = F.relu(feat)

            for block in self.df_2:
                feat = block(feat)
                feat = F.relu(feat)

            feat = feat + sf_feat

            feat = self.transition(feat)
            feat = F.relu(feat)

            feat = torch.cat([feat, img], dim=1)

            feat = self.last_conv(feat)
            feat = F.relu(feat)
            feat = torch.clamp(feat, 0.0, 1.0)
            out = torch.nn.functional.pixel_shuffle(feat, 2)

            output.append(out)

        return torch.stack(output, dim=1)

    def init_chroma(self):
        self.sf_uv = nn.Conv2d(
            in_channels=2,
            out_channels=self.uv_channel,
            kernel_size=3,
            padding=1,
        )

        self.df_uv = []
        for _ in range(self.uv_df_num):
            self.df_uv.append(
                nn.Conv2d(
                    in_channels=self.uv_channel,
                    out_channels=self.uv_channel,
                    kernel_size=3,
                    padding=1,
                )
            )
            self.df_uv.append(nn.ReLU())
        self.df_uv = nn.Sequential(*self.df_uv)

        self.transition_uv = nn.Conv2d(
            in_channels=self.uv_channel * 2,
            out_channels=self.uv_channel,
            kernel_size=3,
            padding=1,
        )

        self.last_conv_uv = nn.Conv2d(
            in_channels=self.uv_channel + 2,
            out_channels=2,
            kernel_size=1,
            padding=0,
        )

    def forward_chroma(self, input):
        sf_uv_feat = self.sf_uv(input)
        sf_uv_feat = F.relu(sf_uv_feat)

        feat = self.df_uv(sf_uv_feat)

        feat = torch.cat([feat, sf_uv_feat], dim=1)

        feat = self.transition_uv(feat)
        feat = F.relu(feat)

        feat = torch.cat([feat, input], dim=1)

        feat = self.last_conv_uv(feat)
        feat = F.relu(feat)
        out = torch.clamp(feat, 0.0, 1.0)

        return out


class GMVSR_ECB(nn.Module):
    def __init__(
        self,
        scale=2,
        num_input_channels=4,
        channel=32,
        df_num=7,
    ):
        super(GMVSR_ECB, self).__init__()
        self.channel = channel

        self.sf = CBAReParam(
            num_input_channels, channel, 3, 1, 1, act="LReLU", bn=False, type="basic"
        )

        middle = df_num // 2 if df_num % 2 == 0 else df_num // 2 + 1
        self.df_1 = []
        for _ in range(middle):
            self.df_1.append(
                CBAReParam(channel, channel, 3, 1, 1, act="LReLU", bn=False, type="ecb")
            )
        self.df_1 = nn.Sequential(*self.df_1)

        self.hidden_transition = CBAReParam(
            channel, channel, 3, 1, 1, act="LReLU", bn=False, type="ecb"
        )

        self.after_concat = CBAReParam(
            channel * 2, channel, 3, 1, 1, act="LReLU", bn=False, type="ecb"
        )

        self.df_2 = []
        for _ in range(df_num - middle - 1):
            self.df_2.append(
                CBAReParam(channel, channel, 3, 1, 1, act="LReLU", bn=False, type="ecb")
            )
        self.df_2 = nn.Sequential(*self.df_2)

        self.transition = CBAReParam(
            channel, channel, 3, 1, 1, act="LReLU", bn=False, type="ecb"
        )

        self.last_conv = CBAReParam(
            channel + num_input_channels, 4, 3, 1, 1, act="LReLU", bn=False, type="ecb"
        )

    def forward(self, input):
        B, N, _, H, W = input.shape
        hidden = torch.zeros((B, self.channel, H // 2, W // 2), device=input.device)
        output = []
        for i in range(N):
            x = input[:, i, :, :, :]
            # x = F.interpolate(x, scale_factor=0.5, mode="bilinear")
            x = torch.nn.functional.pixel_unshuffle(x, 2)
            img = x

            sf_feat = self.sf(x)
            sf_feat = feat = F.relu(sf_feat)

            for block in self.df_1:
                feat = block(feat)
                feat = F.relu(feat)
            df1_feat = feat

            hidden = self.hidden_transition(hidden)
            hidden = F.relu(hidden)

            feat = torch.cat([hidden, df1_feat], dim=1)
            feat = self.after_concat(feat)
            feat = hidden = F.relu(feat)

            for block in self.df_2:
                feat = block(feat)
                feat = F.relu(feat)

            feat = feat + sf_feat

            feat = self.transition(feat)
            feat = F.relu(feat)

            feat = torch.cat([feat, img], dim=1)

            feat = self.last_conv(feat)
            feat = F.relu(feat)
            feat = torch.clamp(feat, 0.0, 1.0)
            out = torch.nn.functional.pixel_shuffle(feat, 2)

            output.append(out)

        return torch.stack(output, dim=1)


class GMSR_FP32_VIDEO(nn.Module):
    def __init__(
        self,
        scale=1,
        num_input_channels=4,
        channel=64,
        df_num=12,
    ):
        super(GMSR_FP32_VIDEO, self).__init__()
        self.model = GMSR_FP32(scale, num_input_channels, channel, df_num)

    def forward(self, input):
        B, N, _, H, W = input.shape
        output = []
        for i in range(N):
            x = input[:, i, :, :, :]
            out = self.model(x)

            output.append(out)

        return torch.stack(output, dim=1)


class GMSR_ECB_VIDEO(nn.Module):
    def __init__(
        self,
        scale=2,
        num_input_channels=4,
        channel=32,
        df_num=10,
    ):
        super(GMSR_ECB_VIDEO, self).__init__()
        self.model = GMSR_ECB(scale, num_input_channels, channel, df_num)

    def forward(self, input):
        B, N, _, H, W = input.shape
        output = []
        for i in range(N):
            x = input[:, i, :, :, :]
            out = self.model(x)

            output.append(out)

        return torch.stack(output, dim=1)


class RealESRGANVideo(nn.Module):
    def __init__(
        self,
    ):
        super(RealESRGANVideo, self).__init__()
        self.model = RRDBNet(
            num_in_ch=1,
            num_out_ch=1,
            scale=1,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
        )

    def forward(self, input):
        B, N, _, H, W = input.shape
        output = []
        for i in range(N):
            x = input[:, i, :, :, :]
            out = self.model(x)

            output.append(out)

        return torch.stack(output, dim=1)
