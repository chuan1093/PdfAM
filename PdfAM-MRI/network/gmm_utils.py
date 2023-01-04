import torch.nn as nn
import torch
import torch.nn.functional as F
import math

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def cal_gmm_widthC(Y, K, times, widthC):
    mu, cov, alpha = GMM_EM(Y, K, times)
    _, prob_sum = getExpectation_widthC(Y, mu, cov, alpha, K, widthC)
    ret_gmm = prob_sum.squeeze(-1)
    return ret_gmm

def GMM_EM(Y, K, times):
    mu, cov, alpha = init_params(Y.shape, K)
    for i in range(times):
        gamma, _ = getExpectation(Y, mu, cov, alpha, K)
        mu, cov, alpha = maximize(Y, gamma, K)
    return mu, cov, alpha

def init_params(shape, K):
    B, C, N = shape
    mu = torch.rand(B, C, 1, K).to(device)
    cov = torch.ones(B, C, 1, K).to(device)
    alpha = (torch.ones(B, C, 1, K) / K).to(device)
    return mu, cov, alpha

def getExpectation(Y, mu, cov, alpha, K):
    Y_repeatK = Y.unsqueeze(-1).repeat(1,1,1,K)
    prob = gau_pdf(Y_repeatK, mu=mu, var=cov) * alpha
    prob_sum = torch.sum(prob, dim=-1, keepdim=True)
    gamma = prob / (prob_sum + 1e-8)
    return gamma, prob_sum

def getExpectation_widthC(Y, mu, cov, alpha, K, widthC):
    Y_repeatK = Y.unsqueeze(-1).repeat(1,1,1,K)
    prob = gau_pdf_widthC(Y_repeatK, mu=mu, var=cov, widthC=widthC) * alpha
    prob_sum = torch.sum(prob, dim=-1, keepdim=True)
    gamma = prob / (prob_sum + 1e-8)
    return gamma, prob_sum

def gau_pdf(x, mu, var):
    x_minus_mu_square = (x - mu).pow(2)
    pi = torch.tensor(math.pi)
    var = var + 1e-4
    gau_pdf = (-x_minus_mu_square / (2 * var)).exp() / (2 * pi * var).sqrt()
    return gau_pdf

def gau_pdf_widthC(x, mu, var, widthC):
    x_minus_mu_square = (x - mu).pow(2)
    pi = torch.tensor(math.pi)
    var = var + 1e-4
    gau_pdf = (-x_minus_mu_square / (2 * var) * widthC).exp() / (2 * pi * var).sqrt()
    return gau_pdf

def maximize(Y, gamma, K):
    B, C, N = Y.shape
    Y_repeatK = Y.unsqueeze(-1).repeat(1,1,1,K)
    Nk = torch.sum(gamma, dim=2, keepdim=True)
    mu = torch.sum(Y_repeatK * gamma, dim=2, keepdim=True) / Nk
    cov = torch.sum((Y_repeatK - mu).pow(2) * gamma, dim=2, keepdim=True) / Nk
    alpha = Nk / N
    return mu, cov, alpha

