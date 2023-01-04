Less Is More Important: A Probability Density Function Guided Attention Module for Convolutional Neural Networks

This is an anonymous code repository for a submission to ACM MM 2022 under review.




# Environments

- OS: Ubuntu 18.04.3 LTS
- CUDA: 10.2
- Python: 3.6.9
- Toolkit: PyTorch 1.4.0
- GPU:  NVIDIA Tesla V100



# Experiments

The following commands show how to train or evaluate a model：

```shell
# Training from scratch

python train.py --model_type ISTARB --mark_str Brain_Radial_ISTARB 

python train.py --model_type ISTARB_PdfAMGau --mark_str Brain_Radial_ISTARB_PdfAMGau 

python train.py --model_type ISTARB_PdfAMGmmT3K2 --mark_str Brain_Radial_ISTARB_PdfAMGmm --gmm_num_T 3 --gmm_num_K 2


# Evaluating the trained model

python train.py --model_type ISTARB_PdfAMGau --mark_str Brain_Radial_ISTARB_PdfAMGau --epoch_num 530
```

We provide our trained best model for ISTARB with PdfAM-Gau under ratio 5% in the `model` directory.

