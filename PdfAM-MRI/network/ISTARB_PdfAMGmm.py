from utils import *
from network.gmm_utils import cal_gmm_widthC

class ISTARB_PdfAMGmm(torch.nn.Module):
    def __init__(self, LayerNo, rb_num=2, num_T=3, num_K=2):
        super(ISTARB_PdfAMGmm, self).__init__()
        onelayer = []
        self.LayerNo = LayerNo

        for i in range(LayerNo):
            onelayer.append(ISTARB_PdfAMGmm_BasicBlock(rb_num, num_T, num_K))

        self.fcs = nn.ModuleList(onelayer)

    def forward(self, PhiTb, mask):

        x = PhiTb

        for i in range(self.LayerNo):
            x = self.fcs[i](x, PhiTb, mask)

        x_final = x

        return x_final

class ISTARB_PdfAMGmm_BasicBlock(torch.nn.Module):
    def __init__(self, rb_num, num_T, num_K):
        super(ISTARB_PdfAMGmm_BasicBlock, self).__init__()

        self.lambda_step = nn.Parameter(torch.Tensor([0.5]))
        self.soft_thr = nn.Parameter(torch.Tensor([0.01]))

        kernel_size = 3
        bias = True
        n_feat = 32

        self.conv_D = nn.Conv2d(1, n_feat, kernel_size, padding=(kernel_size//2), bias=bias)

        modules_body = [Residual_Block_PdfAMGmm(n_feat, n_feat, 3, bias=True, res_scale=1, num_T=num_T, num_K=num_K) for _ in range(rb_num)]

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

class Residual_Block_PdfAMGmm(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias=True, res_scale=1, num_T=3, num_K=2):

        super(Residual_Block_PdfAMGmm, self).__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size//2), bias=bias)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=(kernel_size//2), bias=bias)
        self.act1 = nn.ReLU(inplace=True)
        self.att = pdfam_gmm_module(channels=out_channels, num_T=num_T, num_K=num_K)

    def forward(self, x):
        input = x
        x = self.conv1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.att(x)
        res = x
        x = res * self.res_scale + input
        return x

class pdfam_gmm_module(torch.nn.Module):
    def __init__(self, channels=32, num_T=3, num_K=2):
        super(pdfam_gmm_module, self).__init__()
        self.learn_eps = nn.Parameter(torch.zeros([1, channels, 1, 1]))
        self.learn_coef = nn.Parameter(torch.ones([1, channels, 1, 1]))
        self.learn_width = nn.Parameter(torch.ones([1, channels, 1, 1]))
        self.act = nn.Softplus()
        self.num_T = num_T
        self.num_K = num_K

    def forward(self, x):
        b, c, h, w = x.size()
        X = x.view(b, c, -1)
        learn_width_limit = self.act(self.learn_width)
        gmm_pdf = cal_gmm_widthC(X, K=self.num_K, times=self.num_T, widthC=learn_width_limit).view(b, c, h, w)
        learn_eps_limit = self.act(self.learn_eps)
        learn_coef_limit = self.act(self.learn_coef)
        return x * (1 / (learn_eps_limit + gmm_pdf * learn_coef_limit + 1))
