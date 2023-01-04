from network.ISTARB import *
from network.ISTARB_PdfAMGau import *
from network.ISTARB_PdfAMGmm import *


def chooseModel(args):
    model_type = args.model_type
    layer_num = args.layer_num
    rb_num = args.rb_num
    if model_type == 'ISTARB':
        model = ISTARB(LayerNo=layer_num, rb_num=rb_num)
    elif model_type == 'ISTARB_PdfAMGau':
        model = ISTARB_PdfAMGau(LayerNo=layer_num, rb_num=rb_num)
    elif model_type == 'ISTARB_PdfAMGmm':
        model = ISTARB_PdfAMGmm(LayerNo=layer_num, rb_num=rb_num, num_T=args.gmm_num_T, num_K=args.gmm_num_K)
    else:
        assert False, "not support such model_type"
    return model