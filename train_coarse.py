import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim

import cv2
from glob import glob
import os
import utils
import numpy as np
import rawpy
import pdb
import time
from tqdm import tqdm
from torchsummary import summary

from models.coarsenet import CoarseNet
import argparse

torch.manual_seed(777)
np.random.seed(777)

SONY_TRAIN_PATH = "dataset/Sony_train_list_trial.txt"
SONY_VAL_PATH = "dataset/Sony_val_list_trial.txt"

writer = SummaryWriter()



class SonyDataset(Dataset):
    def __init__(self, path="dataset/Sony_train_list.txt"):
        root = "dataset"
        with open(path, 'r')  as f:
            self.paths = f.readlines()

        # format: ./Sony/short/00001_00_0.04s.ARW ./Sony/long/00001_00_10s.ARW ISO200 F8
        self.input_paths = [os.path.join(root, l.split(' ')[0]) for l in self.paths]
        self.gt_paths = [os.path.join(root, l.split(' ')[1]) for l in self.paths]
        
        input_exposure = [float(os.path.split(input_file)[-1][9:-5]) for input_file in self.input_paths] # 0.04
        gt_exposure = [float(os.path.split(gt_file)[-1][9:-5]) for gt_file in self.gt_paths]
        
        self.ratios = np.array(gt_exposure)/np.array(input_exposure)

    def __getitem__(self, idx):
        raw = rawpy.imread(self.input_paths[idx])
        raw_img = utils.pack_raw(raw) * self.ratios[idx]
        raw_img = np.clip(raw_img, 0, 1)
        
        
        H, W = raw_img.shape[0], raw_img.shape[1]

        rand_x = np.random.randint(0, W - 512)
        rand_y = np.random.randint(0, H - 512)

        # use random patch of size 512x512 from input
        img = raw_img[rand_y: rand_y + 512, rand_x: rand_x + 512, :]
        
        img = cv2.resize(img, (256, 256), cv2.INTER_LINEAR)
        img = img.transpose((2, 0, 1))
        img = torch.from_numpy(img)

        # use random patch of size 512x512 from gt
        raw_gt = rawpy.imread(self.gt_paths[idx])
        gt_img = utils.pack_raw(raw_gt)
        
        gt_img = gt_img[rand_y: rand_y + 512, rand_x: rand_x + 512,  :]
        gt_img = cv2.resize(gt_img, (256, 256), cv2.INTER_LINEAR)
        gt_img = gt_img.transpose((2, 0, 1))
        gt_img = torch.from_numpy(gt_img)

        return img, gt_img

    def __len__(self):
        return len(self.paths)


def train(model, train_loader, optimizer, criterion, epoch, save_freq, device):

    epoch_loss = 0.0
    for idx, (inp, target) in tqdm(enumerate(train_loader), total=len(train_loader),
                                   desc='{} epoch={}'.format('train', epoch), ncols=80, leave=False):

        
        inp = inp.to(device)
        target = target.to(device)

        out = model(inp)

        # write image to tensorboard
        img_grid = torchvision.utils.make_grid(out)
        writer.add_image('output', img_grid)

        optimizer.zero_grad()

        loss = criterion(target, out)
        epoch_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # end epoch

    return epoch_loss / len(train_loader)


def validate(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0

    for idx, (inp, target) in tqdm(enumerate(val_loader), total=len(val_loader),
                                   desc='{}'.format('val'), ncols=80, leave=False):
        
        inp = inp.to(device)
        target = target.to(device)

        with torch.no_grad():
            out = model(inp)
            loss = criterion(target, out)
            val_loss += loss

    return val_loss / len(val_loader)


def main():
    parser = argparse.ArgumentParser("Burst denoising of dark images")
    parser.add_argument('--epochs', type=int, default=4000, help='number of training epoch')
    parser.add_argument('--batch-size', type=int, default=64, help='batch size')
    parser.add_argument('--optimizer', '-o', type=str, default='adam', choices=['adam'], help='optimizer')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--save-freq', type=int, default=20, help='saving model frequency')
    parser.add_argument('--val-freq', type=int, default=1, help='validation frequency')
    parser.add_argument('--resume', type=str, default='', help='resume model from a checkpoint')
    parser.add_argument('--experiment-name', '-e', type=str, default='trial', help='experiment name')

    args = parser.parse_args()

    cuda = torch.cuda.is_available()
    model = CoarseNet()

    device =  torch.device("cuda" if cuda else "cpu")

   
    model = model.to(device)
        
    criterion = nn.L1Loss()

    if args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    else:
        raise NotImplementedError(f"{args.optimizer} optimizer is not supported")

    sonyTrainDataset = SonyDataset(SONY_TRAIN_PATH)
    sonyValDataset = SonyDataset(SONY_VAL_PATH)

    train_loader = DataLoader(sonyTrainDataset, args.batch_size, shuffle=True)
    val_loader = DataLoader(sonyValDataset, args.batch_size)

    
    model.train()

    current_val_loss = 0.0
    no_improve_cnt = 0
    best_epoch = 0

    for epoch in range(args.epochs):
        start_time = time.time()

        # no improvement in val loss in 50 epochs then stop training
        if no_improve_cnt == 50:
            print(
                f'no improvement in 50 epochs, best epoch {best_epoch}, best val loss: {current_val_loss.cpu().numpy()}')
            torch.save(model.state_dict(), 'checkpoints/{}/coarse_e{}_val{}_best.pth'.format(args.experiment_name, best_epoch, np.round(
                current_val_loss.cpu().numpy(), 2)))
            break

        if (args.optimizer == 'adam') & (epoch == 2):
            print("reduce learning rate to 0.00001")
            print("")
            optimizer = optim.Adam(model.parameters(), lr=0.00001)

        train_loss = train(model, train_loader, optimizer, criterion, epoch, args.save_freq, device)

        print(f'epoch: {epoch}, train loss: {train_loss}')

        if epoch % args.val_freq == 0:
            val_loss = validate(model, val_loader, criterion, device)
            if (epoch == 0):
                current_val_loss = val_loss

            print(f'epoch: {epoch}, val loss: {val_loss}')
            if val_loss < current_val_loss:
                best_epoch = epoch
                current_val_loss = val_loss

                # save best model
                torch.save(model.state_dict(), 'checkpoints_coarse/{}/coarse_e{}.pth'.format(args.experiment_name, epoch))
                print(f'saved model at epoch {epoch} successuflly!')

                # reset variable when val loss show improvement
                no_improve_cnt = 0
            else:
                no_improve_cnt += 1

            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/val', val_loss, epoch)

        time_per_epoch = time.time() - start_time
        print(f'time per epoch: {time_per_epoch}')
        print()

        if epoch % args.save_freq == 0:
            if not os.path.exists(f"checkpoints_coarse/{args.experiment_name}"):
                os.makedirs(f"checkpoints_coarse/{args.experiment_name}")

            torch.save(model.state_dict(), 'checkpoints_coarse/{}/coarse_e{}.pth'.format(args.experiment_name, epoch))
            print(f'saved model at epoch {epoch} successuflly!')
            print()
        writer.close()


if __name__ == "__main__":
    main()

