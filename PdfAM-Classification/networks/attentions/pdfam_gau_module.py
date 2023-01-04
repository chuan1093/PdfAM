import torch
import torch.nn as nn
import math

class pdfam_gau_module(torch.nn.Module):
    def __init__(self, channels=None):
        super(pdfam_gau_module, self).__init__()
        self.learn_eps = nn.Parameter(torch.zeros([1, channels, 1, 1]))
        self.learn_coef = nn.Parameter(torch.ones([1, channels, 1, 1]))
        self.learn_width = nn.Parameter(torch.ones([1, channels, 1, 1]))
        self.act = nn.Softplus()

    @staticmethod
    def get_module_name():
        return "pdfam_gau"

    def forward(self, x):
        b, c, h, w = x.size()
        n = w * h - 1
        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        sigma_square = x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + 1e-4
        pi = torch.tensor(math.pi)
        learn_width_limit = self.act(self.learn_width)
        gau_pdf = (-x_minus_mu_square / (2 * sigma_square) * learn_width_limit).exp() / (2 * pi * sigma_square).sqrt()
        learn_eps_limit = self.act(self.learn_eps)
        learn_coef_limit = self.act(self.learn_coef)
        return x * (1 / (learn_eps_limit + gau_pdf * learn_coef_limit + 1))
