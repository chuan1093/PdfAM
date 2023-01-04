import torch
import torch.nn as nn
from networks.attentions.gmm_utils import cal_gmm_widthC

class pdfam_gmm_module(torch.nn.Module):
    def __init__(self, channels=None, num_T=3, num_K=2):
        super(pdfam_gmm_module, self).__init__()
        self.learn_eps = nn.Parameter(torch.zeros([1, channels, 1, 1]))
        self.learn_coef = nn.Parameter(torch.ones([1, channels, 1, 1]))
        self.learn_width = nn.Parameter(torch.ones([1, channels, 1, 1]))
        self.act = nn.Softplus()
        self.num_T = int(num_T)
        self.num_K = int(num_K)

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += ('num_T=%d, num_K=%d)' % (self.num_T, self.num_K))
        return s

    @staticmethod
    def get_module_name():
        return "pdfam_gmm"

    def forward(self, x):
        b, c, h, w = x.size()
        X = x.view(b, c, -1)
        learn_width_limit = self.act(self.learn_width)
        gmm_pdf = cal_gmm_widthC(X, K=self.num_K, times=self.num_T, widthC=learn_width_limit).view(b, c, h, w)
        learn_eps_limit = self.act(self.learn_eps)
        learn_coef_limit = self.act(self.learn_coef)
        return x * (1 / (learn_eps_limit + gmm_pdf * learn_coef_limit + 1))
