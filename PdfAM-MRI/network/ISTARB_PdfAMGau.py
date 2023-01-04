from utils import *

class ISTARB_PdfAMGau(torch.nn.Module):
    def __init__(self, LayerNo, rb_num=2):
        super(ISTARB_PdfAMGau, self).__init__()
        onelayer = []
        self.LayerNo = LayerNo

        for i in range(LayerNo):
            onelayer.append(ISTARB_PdfAMGau_BasicBlock(rb_num))

        self.fcs = nn.ModuleList(onelayer)

    def forward(self, PhiTb, mask):

        x = PhiTb

        for i in range(self.LayerNo):
            x = self.fcs[i](x, PhiTb, mask)

        x_final = x

        return x_final

class ISTARB_PdfAMGau_BasicBlock(torch.nn.Module):
    def __init__(self, rb_num):
        super(ISTARB_PdfAMGau_BasicBlock, self).__init__()

        self.lambda_step = nn.Parameter(torch.Tensor([0.5]))
        self.soft_thr = nn.Parameter(torch.Tensor([0.01]))

        kernel_size = 3
        bias = True
        n_feat = 32

        self.conv_D = nn.Conv2d(1, n_feat, kernel_size, padding=(kernel_size//2), bias=bias)

        modules_body = [Residual_Block_PdfAMGau(n_feat, n_feat, 3, bias=True, res_scale=1) for _ in range(rb_num)]

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

class Residual_Block_PdfAMGau(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias=True, res_scale=1):

        super(Residual_Block_PdfAMGau, self).__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size//2), bias=bias)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=(kernel_size//2), bias=bias)
        self.act1 = nn.ReLU(inplace=True)
        self.att = pdfam_gau_module(channels=out_channels)

    def forward(self, x):
        input = x
        x = self.conv1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.att(x)
        res = x
        x = res * self.res_scale + input
        return x

class pdfam_gau_module(torch.nn.Module):
    def __init__(self, channels=32):
        super(pdfam_gau_module, self).__init__()
        self.learn_eps = nn.Parameter(torch.zeros([1, channels, 1, 1]))
        self.learn_coef = nn.Parameter(torch.ones([1, channels, 1, 1]))
        self.learn_width = nn.Parameter(torch.ones([1, channels, 1, 1]))
        self.act = nn.Softplus()

    def forward(self, x):
        b, c, h, w = x.size()
        n = w * h - 1
        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        sigma_square = x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n
        pi = torch.tensor(math.pi)
        learn_width_limit = self.act(self.learn_width)
        gau_pdf = (-x_minus_mu_square / (2 * sigma_square) * learn_width_limit).exp() / (2 * pi * sigma_square).sqrt()
        learn_eps_limit = self.act(self.learn_eps)
        learn_coef_limit = self.act(self.learn_coef)
        return x * (1 / (learn_eps_limit + gau_pdf * learn_coef_limit + 1))
