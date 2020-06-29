# Burst Denoising of Dark Images

#### This repo is an unofficial Pytorch implementation of Burst Denoising of Dark Images ([Link](https://arxiv.org/abs/2003.07823)) by Ahmet Serdar Karadeniz, Erkut Erdem, Aykut Erdem


## Requirement
```
* pytorch
* rawpy
* tqdm
* torchsummary
* numpy
* skimage
* tensorboard
* opencv-python 
```

## Dataset (SID)
* Please check out cchen156's [repository](https://github.com/cchen156/Learning-to-See-in-the-Dark) to download dataset and put them in dataset folder
* We only use Sony part of SID dataset for training the models so if you want to try with Fuji, model's input need to be modified.

## Link to pretrained model (will be updated)
## How to train
### 1. Single frame model
#### 1.1. Train Coarse model 
```
python train_coarse.py train -e <experiment_name> 
```

#### 1.2. Train Fine model
If we understand correctly the author's idea, fine model reuses trained weights from coarse model, so please adjust the code of train_fine.py to correct location of pretrained coarse model (gotten from step 1.1)

```
python train_fine.py train -e <experiment_name>
```

### 2. Set-based model 
* currently not supported

## How to test
Testing trained model on dataset/Sony_test_list.txt
```
python test.py -c <coarse_checkpoint_dir> -f <fine_checkpoint_dir> -s <saved_folder>
```

## Sample inference (will be updated)



