# Experiments

## CIFAR classification

The following commands show how to train a model for CIFAR10 or CIFAR100：

```shell
python main_cifar.py --attention_type pdfam_gau --other_mark Trial01

python main_cifar.py --attention_type pdfam_gau --other_mark Trial01 --dataset cifar100

python main_cifar.py --attention_type pdfam_gmm --attention_param 3 --attention_param2 2 --other_mark Trial01

python main_cifar.py --attention_type pdfam_gmm --attention_param 3 --attention_param2 2 --other_mark Trial01 --dataset cifar100
```

## ImageNet classification

The following commands show how to train or evaluate a model：

```shell
# Training from scratch

python main_imagenet.py {the path of ImageNet dataset} --gpu 0,1,2,3 --epochs 100 -j 20 -a resnet18 

python main_imagenet.py {the path of ImageNet dataset} --gpu 0,1,2,3 --epochs 100 -j 20 -a resnet18 --attention_type pdfam_gau 

python main_imagenet.py {the path of ImageNet dataset} --gpu 0,1,2,3 --epochs 100 -j 20 -a resnet18 --attention_type pdfam_gmm --attention_param 3 --attention_param2 2 


# Evaluating the trained model

python main_imagenet.py {the path of ImageNet} --gpu 0,1,2,3 -j 20 -a resnet18 -e --resume {the path of pretrained .pth}
```

We provide our trained model for resnet18 with PdfAM-Gau here: https://drive.google.com/file/d/1iKc3SlZSYb5pFZ2KMEUgEcPmcmmxzofK/view?usp=sharing

