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
- SID dataset contains two type of RAW images, one from Sony camera and another from Fuji camera, this repo uses Sony part of SID dataset for training the models. 
- Link: [Sony](https://drive.google.com/open?id=1G6VruemZtpOyHjOC5N8Ww3ftVXOydSXx)(25GB), [Fuji](https://drive.google.com/open?id=1C7GeZ3Y23k1B8reRL79SqnZbRBc4uizH)(52GB) [2]
- Download and extract to folder dataset/
- Remember to modify input channel of two models if you want to train model for Fuji subset. 

## Link to pretrained model: 
[Link](https://drive.google.com/drive/folders/10fmRnTSTs0zIVYdpJPsyH6G6s11jrYDy?usp=sharing)

## Train
### 1. Single frame model
#### 1.1. Train Coarse model 
```
python train_coarse.py train -e <experiment_name> 
```

#### 1.2. Train Fine model
- If we understand correctly the author's idea, fine model reuses trained weights from coarse model, so please adjust the code of train_fine.py to correct location of pretrained coarse model (gotten from step 1.1)

- In order to save time for preprocessing while training, this repo uses preprocessed ground truth images [Sony GT](https://drive.google.com/file/d/1wfkWVkauAsGvXtDJWX0IFDuDl5ozz2PM/view?usp=sharing). Download and extract it to folder dataset/gt/

```
python train_fine.py train -e <experiment_name> -c <coarse_checkpoint>
```

### 2. Set-based model 
* currently not supported

## Test
Testing trained model on dataset/Sony_test_list.txt
```
python test.py -c <coarse_checkpoint_dir> -f <fine_checkpoint_dir> -s <saved_folder>
```

## Sample inference
check out notebook file: inference_sample.ipynb


## References: 
- [1] Karadeniz, A.S., Erdem, E. and Erdem, A., 2020. Burst Denoising of Dark Images. arXiv preprint arXiv:2003.07823. https://arxiv.org/abs/2003.07823v1
- [2] https://github.com/cchen156/Learning-to-See-in-the-Dark
- [3] https://github.com/cydonia999/Learning_to_See_in_the_Dark_PyTorch

