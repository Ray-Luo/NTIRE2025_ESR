import torch
import torch.nn as nn
import torch.nn.functional as F


class SeqConv3x3(nn.Module):
    def __init__(self, seq_type, inp_planes, out_planes, depth_multiplier, bias=True):
        super(SeqConv3x3, self).__init__()

        self.type = seq_type
        self.inp_planes = inp_planes
        self.out_planes = out_planes
        self._bias = bias

        if self.type == "conv1x1-conv3x3":
            self.mid_planes = int(out_planes * depth_multiplier)
            conv0 = torch.nn.Conv2d(
                self.inp_planes, self.mid_planes, kernel_size=1, padding=0, bias=bias
            )
            self.k0 = conv0.weight
            self.b0 = conv0.bias

            conv1 = torch.nn.Conv2d(
                self.mid_planes, self.out_planes, kernel_size=3, bias=bias
            )
            self.k1 = conv1.weight
            self.b1 = conv1.bias

        elif self.type == "conv1x1-sobelx":
            conv0 = torch.nn.Conv2d(
                self.inp_planes, self.out_planes, kernel_size=1, padding=0, bias=bias
            )
            self.k0 = conv0.weight
            self.b0 = conv0.bias

            # init scale & bias
            scale = torch.randn(size=(self.out_planes, 1, 1, 1)) * 1e-3
            self.scale = nn.Parameter(scale)
            bias = torch.randn(self.out_planes) * 1e-3
            bias = torch.reshape(bias, (self.out_planes,))
            self.bias = nn.Parameter(bias)
            # init mask
            self.mask = torch.zeros((self.out_planes, 1, 3, 3), dtype=torch.float32)
            for i in range(self.out_planes):
                self.mask[i, 0, 0, 0] = 1.0
                self.mask[i, 0, 1, 0] = 2.0
                self.mask[i, 0, 2, 0] = 1.0
                self.mask[i, 0, 0, 2] = -1.0
                self.mask[i, 0, 1, 2] = -2.0
                self.mask[i, 0, 2, 2] = -1.0
            self.mask = nn.Parameter(data=self.mask, requires_grad=False)

        elif self.type == "conv1x1-sobely":
            conv0 = torch.nn.Conv2d(
                self.inp_planes, self.out_planes, kernel_size=1, padding=0, bias=bias
            )
            self.k0 = conv0.weight
            self.b0 = conv0.bias

            # init scale & bias
            scale = torch.randn(size=(self.out_planes, 1, 1, 1)) * 1e-3
            self.scale = nn.Parameter(torch.FloatTensor(scale))
            bias = torch.randn(self.out_planes) * 1e-3
            bias = torch.reshape(bias, (self.out_planes,))
            self.bias = nn.Parameter(torch.FloatTensor(bias))
            # init mask
            self.mask = torch.zeros((self.out_planes, 1, 3, 3), dtype=torch.float32)
            for i in range(self.out_planes):
                self.mask[i, 0, 0, 0] = 1.0
                self.mask[i, 0, 0, 1] = 2.0
                self.mask[i, 0, 0, 2] = 1.0
                self.mask[i, 0, 2, 0] = -1.0
                self.mask[i, 0, 2, 1] = -2.0
                self.mask[i, 0, 2, 2] = -1.0
            self.mask = nn.Parameter(data=self.mask, requires_grad=False)

        elif self.type == "conv1x1-laplacian":
            conv0 = torch.nn.Conv2d(
                self.inp_planes, self.out_planes, kernel_size=1, padding=0, bias=bias
            )
            self.k0 = conv0.weight
            self.b0 = conv0.bias

            # init scale & bias
            scale = torch.randn(size=(self.out_planes, 1, 1, 1)) * 1e-3
            self.scale = nn.Parameter(torch.FloatTensor(scale))
            bias = torch.randn(self.out_planes) * 1e-3
            bias = torch.reshape(bias, (self.out_planes,))
            self.bias = nn.Parameter(torch.FloatTensor(bias))
            # init mask
            self.mask = torch.zeros((self.out_planes, 1, 3, 3), dtype=torch.float32)
            for i in range(self.out_planes):
                self.mask[i, 0, 0, 1] = 1.0
                self.mask[i, 0, 1, 0] = 1.0
                self.mask[i, 0, 1, 2] = 1.0
                self.mask[i, 0, 2, 1] = 1.0
                self.mask[i, 0, 1, 1] = -4.0
            self.mask = nn.Parameter(data=self.mask, requires_grad=False)
        elif self.type == "conv1x1-laplacian4":
            conv0 = torch.nn.Conv2d(
                self.inp_planes, self.out_planes, kernel_size=1, padding=0, bias=bias
            )
            self.k0 = conv0.weight
            self.b0 = conv0.bias

            # init scale & bias
            scale = torch.randn(size=(self.out_planes, 1, 1, 1)) * 1e-3
            self.scale = nn.Parameter(torch.FloatTensor(scale))
            bias = torch.randn(self.out_planes) * 1e-3
            bias = torch.reshape(bias, (self.out_planes,))
            self.bias = nn.Parameter(torch.FloatTensor(bias))
            # init mask
            self.mask = torch.zeros((self.out_planes, 1, 3, 3), dtype=torch.float32)
            for i in range(self.out_planes):
                self.mask[i, 0, 0, 1] = 1.0
                self.mask[i, 0, 1, 0] = 1.0
                self.mask[i, 0, 1, 2] = 1.0
                self.mask[i, 0, 2, 1] = 1.0
                self.mask[i, 0, 1, 1] = -4.0
            self.mask = nn.Parameter(data=self.mask, requires_grad=False)
        elif self.type == "conv1x1-laplacian8":
            conv0 = torch.nn.Conv2d(
                self.inp_planes, self.out_planes, kernel_size=1, padding=0, bias=bias
            )
            self.k0 = conv0.weight
            self.b0 = conv0.bias

            # init scale & bias
            scale = torch.randn(size=(self.out_planes, 1, 1, 1)) * 1e-3
            self.scale = nn.Parameter(torch.FloatTensor(scale))
            bias = torch.randn(self.out_planes) * 1e-3
            bias = torch.reshape(bias, (self.out_planes,))
            self.bias = nn.Parameter(torch.FloatTensor(bias))
            # init mask
            self.mask = torch.ones((self.out_planes, 1, 3, 3), dtype=torch.float32)
            for i in range(self.out_planes):
                self.mask[i, 0, 0, 1] = 1.0
                self.mask[i, 0, 1, 0] = 1.0
                self.mask[i, 0, 1, 2] = 1.0
                self.mask[i, 0, 2, 1] = 1.0
                self.mask[i, 0, 1, 1] = -8.0
            self.mask = nn.Parameter(data=self.mask, requires_grad=False)
        elif self.type == "conv1x1-prewittx":
            conv0 = torch.nn.Conv2d(
                self.inp_planes, self.out_planes, kernel_size=1, padding=0, bias=bias
            )
            self.k0 = conv0.weight
            self.b0 = conv0.bias

            # init scale & bias
            scale = torch.randn(size=(self.out_planes, 1, 1, 1)) * 1e-3
            self.scale = nn.Parameter(torch.FloatTensor(scale))
            bias = torch.randn(self.out_planes) * 1e-3
            bias = torch.reshape(bias, (self.out_planes,))
            self.bias = nn.Parameter(torch.FloatTensor(bias))
            # init mask
            self.mask = torch.zeros((self.out_planes, 1, 3, 3), dtype=torch.float32)
            for i in range(self.out_planes):
                self.mask[i, 0, 0, 0] = 1.0
                self.mask[i, 0, 1, 0] = 1.0
                self.mask[i, 0, 2, 0] = 1.0
                self.mask[i, 0, 0, 2] = -1.0
                self.mask[i, 0, 1, 2] = -1.0
                self.mask[i, 0, 2, 2] = -1.0
            self.mask = nn.Parameter(data=self.mask, requires_grad=False)
        elif self.type == "conv1x1-prewitty":
            conv0 = torch.nn.Conv2d(
                self.inp_planes, self.out_planes, kernel_size=1, padding=0, bias=bias
            )
            self.k0 = conv0.weight
            self.b0 = conv0.bias

            # init scale & bias
            scale = torch.randn(size=(self.out_planes, 1, 1, 1)) * 1e-3
            self.scale = nn.Parameter(torch.FloatTensor(scale))
            bias = torch.randn(self.out_planes) * 1e-3
            bias = torch.reshape(bias, (self.out_planes,))
            self.bias = nn.Parameter(torch.FloatTensor(bias))
            # init mask
            self.mask = torch.zeros((self.out_planes, 1, 3, 3), dtype=torch.float32)
            for i in range(self.out_planes):
                self.mask[i, 0, 0, 0] = 1.0
                self.mask[i, 0, 0, 1] = 1.0
                self.mask[i, 0, 0, 2] = 1.0
                self.mask[i, 0, 2, 0] = -1.0
                self.mask[i, 0, 2, 1] = -1.0
                self.mask[i, 0, 2, 2] = -1.0
            self.mask = nn.Parameter(data=self.mask, requires_grad=False)
        elif self.type == "conv1x1-usm":
            conv0 = torch.nn.Conv2d(
                self.inp_planes, self.out_planes, kernel_size=1, padding=0, bias=bias
            )
            self.k0 = conv0.weight
            self.b0 = conv0.bias

            # init scale & bias
            scale = torch.randn(size=(self.out_planes, 1, 1, 1)) * 1e-3
            self.scale = nn.Parameter(torch.FloatTensor(scale))
            bias = torch.randn(self.out_planes) * 1e-3
            bias = torch.reshape(bias, (self.out_planes,))
            self.bias = nn.Parameter(torch.FloatTensor(bias))
            # init mask
            level = 1 / 0.25
            self.mask = torch.ones((self.out_planes, 1, 3, 3), dtype=torch.float32)
            for i in range(self.out_planes):
                self.mask[i, 0, 0, 0] = -1.0
                self.mask[i, 0, 0, 1] = -2.0
                self.mask[i, 0, 0, 2] = -1.0
                self.mask[i, 0, 1, 0] = -2.0
                self.mask[i, 0, 1, 1] = 12 + level
                self.mask[i, 0, 1, 2] = -2.0
                self.mask[i, 0, 2, 0] = -1.0
                self.mask[i, 0, 2, 1] = -2.0
                self.mask[i, 0, 2, 2] = -1.0
            self.mask = self.mask / level

            self.mask = nn.Parameter(data=self.mask, requires_grad=False)

        else:
            raise ValueError("the type of seqconv is not supported!")

        if not self._bias:
            self.bias = None  #  置所有bias为0

    def forward(self, x):
        if self.type == "conv1x1-conv3x3":
            # conv-1x1
            y0 = F.conv2d(input=x, weight=self.k0, bias=self.b0, stride=1)
            # explicitly padding with bias
            y0 = F.pad(y0, (1, 1, 1, 1), "constant", 0)
            if self._bias:
                b0_pad = self.b0.view(1, -1, 1, 1)
                y0[:, :, 0:1, :] = b0_pad
                y0[:, :, -1:, :] = b0_pad
                y0[:, :, :, 0:1] = b0_pad
                y0[:, :, :, -1:] = b0_pad
            # conv-3x3
            y1 = F.conv2d(input=y0, weight=self.k1, bias=self.b1, stride=1)
        else:
            y0 = F.conv2d(input=x, weight=self.k0, bias=self.b0, stride=1)
            # explicitly padding with bias
            y0 = F.pad(y0, (1, 1, 1, 1), "constant", 0)
            if self._bias:
                b0_pad = self.b0.view(1, -1, 1, 1)
                y0[:, :, 0:1, :] = b0_pad
                y0[:, :, -1:, :] = b0_pad
                y0[:, :, :, 0:1] = b0_pad
                y0[:, :, :, -1:] = b0_pad
            # conv-3x3
            y1 = F.conv2d(
                input=y0,
                weight=self.scale * self.mask,
                bias=self.bias,
                stride=1,
                groups=self.out_planes,
            )
        return y1

    def rep_params(self):
        device = self.k0.get_device()
        if device < 0:
            device = None

        if self.type == "conv1x1-conv3x3":
            # re-param conv kernel
            RK = F.conv2d(input=self.k1, weight=self.k0.permute(1, 0, 2, 3))
            # re-param conv bias
            if self._bias:
                RB = torch.ones(1, self.mid_planes, 3, 3, device=device) * self.b0.view(
                    1, -1, 1, 1
                )
                RB = (
                    F.conv2d(input=RB, weight=self.k1).view(
                        -1,
                    )
                    + self.b1
                )
            else:
                RB = None
        else:
            tmp = self.scale * self.mask
            k1 = torch.zeros((self.out_planes, self.out_planes, 3, 3), device=device)
            for i in range(self.out_planes):
                k1[i, i, :, :] = tmp[i, 0, :, :]
            b1 = self.bias
            # re-param conv kernel
            RK = F.conv2d(input=k1, weight=self.k0.permute(1, 0, 2, 3))
            # re-param conv bias
            if self._bias:
                RB = torch.ones(1, self.out_planes, 3, 3, device=device) * self.b0.view(
                    1, -1, 1, 1
                )
                RB = (
                    F.conv2d(input=RB, weight=k1).view(
                        -1,
                    )
                    + b1
                )
            else:
                RB = None
        return RK, RB


class ECB(nn.Module):
    def __init__(
        self,
        inp_planes,
        out_planes,
        depth_multiplier=2.0,
        act_type="linear",
        with_idt=False,
        bias=True,
    ):
        super(ECB, self).__init__()

        self.depth_multiplier = depth_multiplier
        self.inp_planes = inp_planes
        self.out_planes = out_planes
        self.act_type = act_type
        self._bias = bias

        if with_idt and (self.inp_planes == self.out_planes):
            self.with_idt = True
        else:
            self.with_idt = False

        self.conv3x3 = torch.nn.Conv2d(
            self.inp_planes, self.out_planes, kernel_size=3, padding=1, bias=bias
        )
        self.conv1x1_3x3 = SeqConv3x3(
            "conv1x1-conv3x3",
            self.inp_planes,
            self.out_planes,
            self.depth_multiplier,
            bias=bias,
        )
        self.conv1x1_sbx = SeqConv3x3(
            "conv1x1-sobelx", self.inp_planes, self.out_planes, -1, bias=bias
        )
        self.conv1x1_sby = SeqConv3x3(
            "conv1x1-sobely", self.inp_planes, self.out_planes, -1, bias=bias
        )
        self.conv1x1_lpl = SeqConv3x3(
            "conv1x1-laplacian", self.inp_planes, self.out_planes, -1, bias=bias
        )

        if self.act_type == "prelu":
            self.act = nn.PReLU(num_parameters=self.out_planes)
        elif self.act_type == "relu":
            self.act = nn.ReLU(inplace=True)
        elif self.act_type == "rrelu":
            self.act = nn.RReLU(lower=-0.05, upper=0.05)
        elif self.act_type == "softplus":
            self.act = nn.Softplus()
        elif self.act_type == "linear":
            pass
        else:
            raise ValueError("The type of activation if not support!")

    def forward(self, x):
        if self.training:
            y = (
                self.conv3x3(x)
                + self.conv1x1_3x3(x)
                + self.conv1x1_sbx(x)
                + self.conv1x1_sby(x)
                + self.conv1x1_lpl(x)
            )
            if self.with_idt:
                y += x
        else:
            RK, RB = self.rep_params()
            y = F.conv2d(input=x, weight=RK, bias=RB, stride=1, padding=1)
        # if self.act_type != 'linear':
        #     y = self.act(y)
        return y

    def rep_params(self):
        K0, B0 = self.conv3x3.weight, self.conv3x3.bias
        K1, B1 = self.conv1x1_3x3.rep_params()
        K2, B2 = self.conv1x1_sbx.rep_params()
        K3, B3 = self.conv1x1_sby.rep_params()
        K4, B4 = self.conv1x1_lpl.rep_params()
        RK = K0 + K1 + K2 + K3 + K4
        if self._bias:
            RB = B0 + B1 + B2 + B3 + B4
        else:
            RB = None

        if self.with_idt:
            device = RK.get_device()
            if device < 0:
                device = None
            K_idt = torch.zeros(self.out_planes, self.out_planes, 3, 3, device=device)
            for i in range(self.out_planes):
                K_idt[i, i, 1, 1] = 1.0
            B_idt = 0.0
            RK = RK + K_idt
            # RK, RB = RK + K_idt, RB + B_idt

            if self._bias:
                RB = RB + B_idt
            else:
                RB = None

        return RK, RB


class CBAReParam(nn.Module):
    """
    basic conv + bn + relu
    """

    def __init__(
        self,
        inp,
        oup,
        k=3,
        s=1,
        p=1,
        d=1,
        act=None,
        bn=False,
        bias=False,
        type="basic",
        doubleconv=False,
        deploy=False,
        expand=1,
        res1x1=False,
        frozen=False,
    ):
        """
        inp/oup: input/output channel
        k: kernel size
        p: padding
        s: stride
        d: dilation
        act: None:, ReLU: ReLU, LReLU: leaky ReLU
        """
        super(CBAReParam, self).__init__()

        self._doubleconv = doubleconv
        self._deploy = deploy
        self._bias = bias
        self._type = type
        self._k = k
        self._s = s
        self._p = p

        self._res1x1 = res1x1  #  simulate usm

        layers = []

        if doubleconv:
            mid_c = inp * expand
            layers.append(nn.Conv2d(inp, mid_c, 1, 1, 0, bias=bias))
            inp = mid_c

        if type == "basic":
            layers.append(nn.Conv2d(inp, oup, k, s, p, d, bias=bias))
        elif type == "ecb":
            layers.append(ECB(inp, oup, bias=bias))

        if act == "ReLU":
            # layers.append(nn.ReLU())
            self._act = nn.ReLU()
        elif act == "LReLU":
            # layers.append(nn.LeakyReLU(inplace=True, negative_slope=0.1))
            self._act = nn.LeakyReLU(inplace=True, negative_slope=0.1)
        else:
            self._act = nn.Identity()

        self._cbm = nn.Sequential(*layers)

        if frozen:
            self._set_requires_grad(self, False)

        if self._deploy:
            self.switch_to_deploy()

    def _set_requires_grad(self, model, require_grad=True):
        for i in model.parameters():
            i.requires_grad = require_grad
        model.eval()
        return model

    def switch_to_deploy(self):
        if self._deploy:
            print("op is on deploy mode..")
            return

        RK, RB = self.rep_params()
        inp = RK.size(1)
        oup = RK.size(0)
        self._cbm = nn.Conv2d(inp, oup, self._k, self._s, self._p, bias=self._bias)
        self._cbm.weight.data = RK
        if self._bias:
            self._cbm.bias.data = RB

        self._deploy = True

    def rep_params(self):
        # max_layer_id = 0
        if self._doubleconv:
            in_conv = self._cbm[0]
            in_k = in_conv.weight
            in_b = in_conv.bias

            #  merge conv block
            conv_block = self._cbm[1]
            if self._type == "basic":
                conv_RK = conv_block.weight
                conv_RB = conv_block.bias
            else:
                conv_RK, conv_RB = conv_block.rep_params()
            # return conv_RK, conv_RB

            #  merge
            RK = F.conv2d(input=conv_RK, weight=in_k.permute(1, 0, 2, 3))
            #  TODO bias meger buggy
            if self._bias:
                RB = torch.ones(
                    1, in_k.size(0), self._k, self._k, device=in_k.device
                ) * in_b.view(1, -1, 1, 1)
                RB = (
                    F.conv2d(input=RB, weight=conv_RK).view(
                        -1,
                    )
                    + conv_RB
                )
            else:
                RB = None

            # max_layer_id = 2
        else:
            conv_block = self._cbm[0]
            if self._type == "basic":
                RK = conv_block.weight
                RB = conv_block.bias
            else:
                RK, RB = conv_block.rep_params()

            # max_layer_id = 1

        if self._res1x1:
            # res1x1_conv = self._cbm[max_layer_id]
            res1x1_conv = self._res1x1_conv
            k_res, b_res = res1x1_conv.re_params()

            RK = F.conv2d(input=k_res, weight=RK.permute(1, 0, 2, 3))
            #  TODO bias meger buggy
            if self._bias:
                RB = torch.ones(
                    1, RB.size(0), self._k, self._k, device=RB.device
                ) * RB.view(1, -1, 1, 1)
                RB = (
                    F.conv2d(input=RB, weight=k_res).view(
                        -1,
                    )
                    + b_res
                )
            else:
                RB = None

        return RK, RB

    def forward(self, x):
        out = self._act(self._cbm(x))
        return out


def multiscale(kernel, target_kernel_size):
    H_pixels_to_pad = (target_kernel_size - kernel.size(2)) // 2
    W_pixels_to_pad = (target_kernel_size - kernel.size(3)) // 2
    return F.pad(
        kernel, [H_pixels_to_pad, H_pixels_to_pad, W_pixels_to_pad, W_pixels_to_pad]
    )


class EDBB(nn.Module):
    def __init__(
        self,
        inp_planes,
        out_planes,
        depth_multiplier=2.0,
        act_type="prelu",
        with_idt=True,
        deploy=False,
        with_13=True,
        gv=False,
    ):
        super(EDBB, self).__init__()

        self.deploy = deploy
        self.act_type = act_type

        self.inp_planes = inp_planes
        self.out_planes = out_planes

        self.gv = gv

        if depth_multiplier is None:
            self.depth_multiplier = 1.0
        else:
            self.depth_multiplier = depth_multiplier  # For mobilenet, it is better to have 2X internal channels

        if deploy:
            self.rep_conv = nn.Conv2d(
                in_channels=inp_planes,
                out_channels=out_planes,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True,
            )
        else:
            self.with_13 = with_13
            if with_idt and (self.inp_planes == self.out_planes):
                self.with_idt = True
            else:
                self.with_idt = False

            self.rep_conv = nn.Conv2d(
                self.inp_planes, self.out_planes, kernel_size=3, padding=1
            )
            self.conv1x1 = nn.Conv2d(
                self.inp_planes, self.out_planes, kernel_size=1, padding=0
            )
            self.conv1x1_3x3 = SeqConv3x3(
                "conv1x1-conv3x3",
                self.inp_planes,
                self.out_planes,
                self.depth_multiplier,
            )
            self.conv1x1_sbx = SeqConv3x3(
                "conv1x1-sobelx", self.inp_planes, self.out_planes, -1
            )
            self.conv1x1_sby = SeqConv3x3(
                "conv1x1-sobely", self.inp_planes, self.out_planes, -1
            )
            self.conv1x1_lpl = SeqConv3x3(
                "conv1x1-laplacian", self.inp_planes, self.out_planes, -1
            )

        if self.act_type == "prelu":
            self.act = nn.PReLU(num_parameters=self.out_planes)
        elif self.act_type == "relu":
            self.act = nn.ReLU(inplace=True)
        elif self.act_type == "rrelu":
            self.act = nn.RReLU(lower=-0.05, upper=0.05)
        elif self.act_type == "softplus":
            self.act = nn.Softplus()
        elif self.act_type == "linear":
            pass
        else:
            raise ValueError("The type of activation if not support!")

    def forward(self, x):
        if self.deploy:
            y = self.rep_conv(x)
        elif self.gv:
            y = (
                self.rep_conv(x)
                + self.conv1x1_sbx(x)
                + self.conv1x1_sby(x)
                + self.conv1x1_lpl(x)
                + x
            )
        else:
            y = (
                self.rep_conv(x)
                + self.conv1x1(x)
                + self.conv1x1_sbx(x)
                + self.conv1x1_sby(x)
                + self.conv1x1_lpl(x)
            )
            # self.conv1x1_3x3(x) + \
            if self.with_idt:
                y += x
            if self.with_13:
                y += self.conv1x1_3x3(x)

        if self.act_type != "linear":
            y = self.act(y)
        return y

    def switch_to_gv(self):
        if self.gv:
            return
        self.gv = True

        K0, B0 = self.rep_conv.weight, self.rep_conv.bias
        K1, B1 = self.conv1x1_3x3.rep_params()
        K5, B5 = multiscale(self.conv1x1.weight, 3), self.conv1x1.bias
        RK, RB = (K0 + K5), (B0 + B5)
        if self.with_13:
            RK, RB = RK + K1, RB + B1

        self.rep_conv.weight.data = RK
        self.rep_conv.bias.data = RB

        for para in self.parameters():
            para.detach_()

    def switch_to_deploy(self):
        if self.deploy:
            return
        self.deploy = True

        K0, B0 = self.rep_conv.weight, self.rep_conv.bias
        K1, B1 = self.conv1x1_3x3.rep_params()
        K2, B2 = self.conv1x1_sbx.rep_params()
        K3, B3 = self.conv1x1_sby.rep_params()
        K4, B4 = self.conv1x1_lpl.rep_params()
        K5, B5 = multiscale(self.conv1x1.weight, 3), self.conv1x1.bias
        if self.gv:
            RK, RB = (K0 + K2 + K3 + K4), (B0 + B2 + B3 + B4)
        else:
            RK, RB = (K0 + K2 + K3 + K4 + K5), (B0 + B2 + B3 + B4 + B5)
            if self.with_13:
                RK, RB = RK + K1, RB + B1
        if self.with_idt:
            device = RK.get_device()
            if device < 0:
                device = None
            K_idt = torch.zeros(self.out_planes, self.out_planes, 3, 3, device=device)
            for i in range(self.out_planes):
                K_idt[i, i, 1, 1] = 1.0
            B_idt = 0.0
            RK, RB = RK + K_idt, RB + B_idt

        self.rep_conv = nn.Conv2d(
            in_channels=self.inp_planes,
            out_channels=self.out_planes,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
        )
        self.rep_conv.weight.data = RK
        self.rep_conv.bias.data = RB

        for para in self.parameters():
            para.detach_()

        # self.__delattr__('conv3x3')
        self.__delattr__("conv1x1_3x3")
        self.__delattr__("conv1x1")
        self.__delattr__("conv1x1_sbx")
        self.__delattr__("conv1x1_sby")
        self.__delattr__("conv1x1_lpl")


class EDBB_deploy(nn.Module):
    def __init__(self, inp_planes, out_planes):
        super(EDBB_deploy, self).__init__()

        self.rep_conv = nn.Conv2d(
            in_channels=inp_planes,
            out_channels=out_planes,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
        )

        self.act = nn.PReLU(num_parameters=out_planes)

    def forward(self, x):
        y = self.rep_conv(x)
        y = self.act(y)

        return y
