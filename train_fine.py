import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable

import torch.nn as nn
import torch.optim as optim
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import cv2
from glob import glob
import os
import utils
import numpy as np
import rawpy
import pdb
import time
import copy
from tqdm import tqdm
from torchsummary import summary

from coarsenet import CoarseNet
from finenet import FineNet
import argparse
import scipy
import scipy.misc
from PIL import Image


torch.manual_seed(777)
np.random.seed(777)

from torch.multiprocessing import Pool, Process, set_start_method

try:
    set_start_method('spawn')
except RuntimeError:
    pass

writer = SummaryWriter("runs/finenet_coarse300")
SONY_TRAIN_PATH = "dataset/Sony_train_list.txt"
SONY_VAL_PATH = "dataset/Sony_val_list.txt"

class SonyDataset(Dataset):
    def __init__(self, coarseNet, device, path="dataset/Sony_train_list.txt"):
        root = "dataset"
        self.coarseNet = coarseNet
        self.device = device

        with open(path, 'r')  as f:
            self.paths = f.readlines()

        # format: ./Sony/short/00001_00_0.04s.ARW ./Sony/long/00001_00_10s.ARW ISO200 F8
        self.input_paths = [os.path.join(root, l.split(' ')[0]) for l in self.paths]
        self.gt_paths = [os.path.join(root, l.split(' ')[1]) for l in self.paths]
        
        input_exposure = [float(os.path.split(input_file)[-1][9:-5]) for input_file in self.input_paths]
        gt_exposure = [float(os.path.split(gt_file)[-1][9:-5]) for gt_file in self.gt_paths]
        
        self.ratios = np.array(gt_exposure)/np.array(input_exposure)
        
    def buildInput(self, x):
        x = torch.unsqueeze(x, 0)
        with torch.no_grad():
            coarse_x = self.coarseNet(x)
            
        noise_x = x - coarse_x
        inp = torch.cat([x, coarse_x, noise_x], dim=1)
        inp = torch.squeeze(inp, 0)
        return inp

    def __getitem__(self, idx):
        raw = rawpy.imread(self.input_paths[idx])
        raw_img = utils.pack_raw(raw) * self.ratios[idx]
        raw_img = np.clip(raw_img, 0, 1)
        
        # use random patch of size 512x512 from input 
        H, W = raw_img.shape[0], raw_img.shape[1]
        
        rand_x = np.random.randint(0, W - 512)
        rand_y = np.random.randint(0, H - 512)
        
        img = raw_img[rand_y: rand_y + 512, rand_x: rand_x + 512, :]
        
        img = torch.from_numpy(img.copy())
        
        img = img.to(self.device)
        img = img.permute(2, 0, 1)
       
        img = self.buildInput(img)

        preprocess_gt_path = self.gt_paths[idx].split('/')[-1].split('.')[0]
        gt_img = cv2.imread('sony_gt/gt/' + preprocess_gt_path + '.png')
        gt_img = cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB)
        gt_img = gt_img/255.

        xx = rand_x*2
        yy = rand_y*2
        
        gt_img = gt_img[yy: yy+ 512*2, xx: xx + 512*2, :]
    
        gt_img = torch.from_numpy(gt_img.copy())
        gt_img = gt_img.permute(2, 0, 1)
        return img, gt_img

    def __len__(self):
        return len(self.paths)


def train(experiment_name, model, train_loader, optimizer, criterion, epoch, save_freq, device):
    epoch_loss = 0.0
    psnr = 0.0
    ssim = 0.0
    total_cnt = 0
    for idx, (inps, targets) in tqdm(enumerate(train_loader), total=len(train_loader),
                                   desc='{} epoch={}'.format('train', epoch), ncols=80, leave=False):

        
        inps = inps.to(device)
        targets = targets.to(device)
        
        outs = model(inps)
        outs = torch.clamp(outs, 0, 1)

        # write image to tensorboard
        img_grid = torchvision.utils.make_grid(outs)

        optimizer.zero_grad()

        loss = criterion(targets, outs)
        
        for (out, target) in zip(outs, targets):
            out    = out.detach().cpu().numpy().transpose(1, 2, 0) * 255
            target = target.detach().cpu().numpy().transpose(1, 2, 0) * 255
            
            psnr += peak_signal_noise_ratio(target, out, data_range=255)
            ssim += structural_similarity(target, out, data_range=255, gaussian_weights=True, use_sample_covariance=False, multichannel=True)
            
            saved_dir = 'result/{}/{:04d}'.format(experiment_name, epoch)
            if not os.path.exists(saved_dir):
                os.makedirs(saved_dir, exist_ok=True)

            fname = saved_dir + '/{:04d}.jpg'.format(idx)
            
            temp = np.concatenate((target[:, :, :], out[:, :, :]), axis=1)
            im = Image.fromarray(np.uint8(temp))
            im.save(fname)

            total_cnt += 1
        
        epoch_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # end epoch 
    
    return epoch_loss / len(train_loader), psnr/total_cnt, ssim/total_cnt


def validate(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    psnr = 0.0
    ssim = 0.0
    total_cnt = 0
    for idx, (inps, targets) in tqdm(enumerate(val_loader), total=len(val_loader),
                                   desc='{}'.format('val'), ncols=80, leave=False):
        
        inps = inps.to(device)
        targets = targets.to(device)

        with torch.no_grad():
            outs = model(inps)
            
            outs = torch.clamp(outs, 0, 1)
            
            loss = criterion(targets, outs)
            val_loss += loss
            
            for (out, target) in zip(outs, targets):
                out    = out.detach().cpu().numpy().transpose(1, 2, 0) * 255
                target = target.detach().cpu().numpy().transpose(1, 2, 0) * 255
                
                psnr += peak_signal_noise_ratio(target, out, data_range=255)
                ssim += structural_similarity(target, out, data_range=255, gaussian_weights=True, use_sample_covariance=False, multichannel=True)
                total_cnt += 1

    return val_loss / len(val_loader), psnr/ total_cnt, ssim/total_cnt


def main():
    parser = argparse.ArgumentParser("Burst denoising of dark images")
    parser.add_argument('--epochs', type=int, default=4000, help='number of training epoch')
    parser.add_argument('--batch-size', type=int, default=4, help='batch size')
    parser.add_argument('--optimizer', '-o', type=str, default='adam', choices=['adam'], help='optimizer')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--patch-size', type=int, default=512, choices=[512, 256], help='patch size')
    parser.add_argument('--save-freq', type=int, default=20, help='saving model frequency')
    parser.add_argument('--val-freq', type=int, default=1, help='saving model frequency')
    parser.add_argument('--resume', default="",  help='resume model from a checkpoint')
    parser.add_argument('--experiment-name', '-e', type=str, default='trial', help='experiment name')
    parser.add_argument('--coarse', '-c', type=str, help='location of coarse checkpoint')

    args = parser.parse_args()
    torch.backends.cudnn.benchmark = True
    
    cuda = torch.cuda.is_available()
    if cuda:
        torch.cuda.manual_seed(777)
        
    # load Coarse Net model to create input for Fine Net 
    coarseModel = CoarseNet()
    coarseModel.load_state_dict(torch.load('checkpoints_coarse/exp1_amp_ratio/coarse_e300.pth', map_location = 'cuda')) 
    coarseModel.eval()
    
    fineModel = FineNet()
    
    copy_state_dict = [c for c in list(fineModel.state_dict().keys()) if c in list(coarseModel.state_dict().keys())]
    copy_state_dict = [c for c in copy_state_dict if (('conv1_1' not in c ) and ('conv10' not in c))]

    for k in copy_state_dict:
        print(f'copying weight from {k}')
        fineModel.state_dict()[k].copy_(coarseModel.state_dict()[k])
    
    # reuse weights from all layers of coarse model except conv1_1 and conv10
    fineModel.conv1_1 = nn.Conv2d(12, 32, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True)
    fineModel.conv10 = nn.Conv2d(32, 12, kernel_size=1, stride=1, padding=0, bias=True) 
    
    if args.resume.strip():
        print("loading Fine Net from checkpoint {args.resume}")
        checkpoint = torch.load(args.resume)
        fineModel.load_state_dict(checkpoint['model_sd'])
        
        optimizer = optim.Adam(fineModel.parameters())
        optimizer.load_state_dict(checkpoint['optimizer_sd'])
      
    device =  torch.device("cuda" if cuda else "cpu")
    fineModel = fineModel.to(device)
    coarseModel = coarseModel.to(device)


    criterion = nn.L1Loss()
    
    if args.optimizer == 'adam':
        optimizer = optim.Adam(fineModel.parameters(), lr=args.lr)
    else:
        raise NotImplementedError(f"{args.optimizer} optimizer is not supported")

    sonyTrainDataset = SonyDataset(coarseModel, device, SONY_TRAIN_PATH)
    sonyValDataset = SonyDataset(coarseModel, device, SONY_VAL_PATH)

    train_loader = DataLoader(sonyTrainDataset, args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(sonyValDataset, args.batch_size, num_workers=4)

    fineModel.train()
    current_val_loss = 0.0
    no_improve_cnt = 0
    best_epoch = 0

    for epoch in range(args.epochs):
        start_time = time.time()

        # no improvement in val loss in 50 epochs then stop training 
        if no_improve_cnt == 50:
            print(
                f'no improvement in 50 epochs, best epoch {best_epoch}, best val loss: {np.round(current_val_loss.cpu().numpy(), 2)}')
            fineModel.eval()
            torch.save(fineModel.state_dict(), 'checkpoints/fine_e.{}_val.{}_best.pth'.format(best_epoch, np.round(
                current_val_loss.cpu().numpy(), 2)))
            break
            
        if (args.optimizer == 'adam') & (epoch == 100):
            print("reduce learning rate to 0.00001")
            print("")
            optimizer = optim.Adam(fineModel.parameters(), lr=0.00001)
            
        train_loss, train_psnr, train_ssim = train(args.experiment_name, fineModel, train_loader, optimizer, criterion, epoch, args.save_freq, device)

        print(f'epoch: {epoch}, train loss: {train_loss}, psnr: {train_psnr}, ssim: {train_ssim}')

        if epoch % args.val_freq == 0:
            val_loss, val_psnr, val_ssim  = validate(fineModel, val_loader, criterion, device)

            print(f'epoch: {epoch}, val_loss: {val_loss}, psnr: {val_psnr}, ssim: {val_ssim}')

            if (epoch == 0):
                current_val_loss = val_loss

            if val_loss < current_val_loss:
                print(f'val_loss ({val_loss}) < best val loss ({current_val_loss}), saved model at epoch {epoch} successuflly!')
                best_epoch = epoch
                current_val_loss = val_loss

                # save best model for resume later
                saved_dir = os.path.join('checkpoints', args.experiment_name)
                if not os.path.exists(saved_dir):
                    os.makedirs(saved_dir, exist_ok=True)
                fineModel.eval()    
                torch.save({'optimizer_sd': optimizer.state_dict(), 'model_sd': fineModel.state_dict(), 'loss': val_loss}, saved_dir + '/fine_e{}.pth.tar'.format(epoch))
                
                # save best model for inference
                if not os.path.exists(f"checkpoints/{args.experiment_name}"):
                    os.makedirs(f"checkpoints/{args.experiment_name}")

                torch.save(fineModel.state_dict(), f'checkpoints/{args.experiment_name}/fine_e{epoch}.pth')
                print(f'validation score improved, saved model at epoch {epoch} successuflly!')
                print()

                # reset no_improve_cnt variable when val loss shows improvement 
                no_improve_cnt = 0
            else:
                no_improve_cnt += 1

            writer.add_scalar('Fine Loss/train', train_loss, epoch)
            writer.add_scalar('Fine Loss/val', val_loss, epoch)
            
            writer.add_scalar('Fine PSNR/train', train_psnr, epoch)
            writer.add_scalar('Fine PSNR/val', val_psnr, epoch)
            
            writer.add_scalar('Fine SSIM/train', train_ssim, epoch)
            writer.add_scalar('Fine SSIM/val', val_ssim, epoch)
        

        time_per_epoch = time.time() - start_time
        print(f'time per epoch: {time_per_epoch}')
        print()

    writer.close()

if __name__ == "__main__":
    main()
