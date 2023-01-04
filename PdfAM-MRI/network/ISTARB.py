from utils import *

class ISTARB(torch.nn.Module):
    def __init__(self, LayerNo, rb_num=2):
        super(ISTARB, self).__init__()
        onelayer = []
        self.LayerNo = LayerNo

        for i in range(LayerNo):
            onelayer.append(ISTARB_BasicBlock(rb_num))

        self.fcs = nn.ModuleList(onelayer)

    def forward(self, PhiTb, mask):

        x = PhiTb

        for i in range(self.LayerNo):
            x = self.fcs[i](x, PhiTb, mask)

        x_final = x

        return x_final

class ISTARB_BasicBlock(torch.nn.Module):
    def __init__(self, rb_num):
        super(ISTARB_BasicBlock, self).__init__()

        self.lambda_step = nn.Parameter(torch.Tensor([0.5]))
        self.soft_thr = nn.Parameter(torch.Tensor([0.01]))

        kernel_size = 3
        bias = True
        n_feat = 32

        self.conv_D = nn.Conv2d(1, n_feat, kernel_size, padding=(kernel_size//2), bias=bias)

        modules_body = [Residual_Block(n_feat, n_feat, 3, bias=True, res_scale=1) for _ in range(rb_num)]

        self.body = nn.Sequential(*modules_body)

        self.conv_G = nn.Conv2d(n_feat, 1, kernel_size, padding=(kernel_size//2), bias=bias)

    def forward(self, x, PhiTb, mask):
        x = x - self.lambda_step * zero_filled(x, mask)
        x = x + self.lambda_step * PhiTb
        x_input = x

        x_D = self.conv_D(x_input)

        x_backward = self.body(x_D)

        x_G = self.conv_G(x_backward)

        x_pred = x_input + x_G

        return x_pred

class Residual_Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias=True, res_scale=1):

        super(Residual_Block, self).__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size//2), bias=bias)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=(kernel_size//2), bias=bias)
        self.act1 = nn.ReLU(inplace=True)

    def forward(self, x):
        input = x
        x = self.conv1(x)
        x = self.act1(x)
        x = self.conv2(x)
        res = x
        x = res * self.res_scale + input
        return x
