from models.coarsenet import CoarseNet
from models.finenet import FineNet
import matplotlib.pyplot as plt
import torch 
import rawpy
import cv2
import utils
import torch.nn as nn
import numpy as np
import os
import argparse
import pdb
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

torch.manual_seed(777)
np.random.seed(777)


    
def read_input(path, ratio = 100):
    inp_raw = rawpy.imread(path)
    inp = utils.pack_raw(inp_raw)
    
    inp = inp * ratio 
    inp = np.clip(inp, 0, 1)
    inp = inp.transpose(2, 0, 1)
    inp_tensor = torch.Tensor(inp)
    inp_tensor = inp_tensor.unsqueeze(0)
    return inp_tensor

def read_target(path):
    raw_gt = rawpy.imread(path)
    target = raw_gt.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
    target = (target/65535.0) * 255.
    return target

def read_target_disk(path):
    preprocess_gt_path = path.split('/')[-1].split('.')[0]
    gt_img = cv2.imread('sony_gt/gt/' + preprocess_gt_path + '.png')
    gt_img = cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB)
    return gt_img

def bgr2rgb(img):
    b,g,r = img[:,:,0].copy(), img[:,:,1].copy(), img[:,:,2].copy()
    img[:,:,0] = r
    img[:,:,1] = g
    img[:,:,2] = b
    return img

def evaluate(target, output):
    
    psnr = peak_signal_noise_ratio(target, output, data_range=255)
    ssim = structural_similarity(target, output, data_range=255, gaussian_weights=True, use_sample_covariance=False, multichannel=True)
    
    print(f'psnr: {psnr}')
    print(f'ssim: {ssim}')
    return psnr, ssim

def build_input(inp_tensor, coarseModel):
    with torch.no_grad():
        coarse_x = coarseModel(inp_tensor)
        noise_x = inp_tensor - coarse_x
        return torch.cat([inp_tensor , coarse_x, noise_x], dim = 1)
    
    
def main():
    parser = argparse.ArgumentParser("Test class for Burst Denoising")
    parser.add_argument('--coarse_net', '-c', type=str, default= '', help = 'location of coarsenet model')
    parser.add_argument('--fine_net', '-f', type=str, default='', help= 'location of finenet model')
    parser.add_argument('--dataset', '-data', type=str, default='dataset/Sony_test_list.txt', help = 'location of test file in txt format')
    parser.add_argument('--save_dir', '-s', type=str, default = 'test', help= 'save result or not')
    
    args = parser.parse_args()
    
    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    
    print(f'device: {device}')
    if cuda: 
        torch.cuda.manual_seed(777)
    
    coarseModel = CoarseNet()
    coarseModel.load_state_dict(torch.load(args.coarse_net))
    coarseModel.eval()
    
    fineModel = FineNet()
    fineModel.load_state_dict(torch.load(args.fine_net))
    fineModel.eval()
    
    fineModel = fineModel.to(device)
    coarseModel = coarseModel.to(device)
    
    psnr_all = []
    ssim_all = []
    
    
    with open(args.dataset, 'r') as f:
        paths = f.readlines()
        
    input_paths = [os.path.join('dataset', l.split(' ')[0]) for l in paths]
    gt_paths = [os.path.join('dataset', l.split(' ')[1]) for l in paths]
        
    input_exposure = [float(os.path.split(input_file)[-1][9:-5]) for input_file in input_paths] 
    gt_exposure = [float(os.path.split(gt_file)[-1][9:-5]) for gt_file in gt_paths]
    
    # ratio between GT exposure time and input exposure time
    ratios = np.array(gt_exposure)/np.array(input_exposure)
    
    if args.save_dir:
        saved_dir = 'test/' + args.save_dir
        os.makedirs(saved_dir, exist_ok=True)
        os.makedirs(saved_dir + '/images', exist_ok=True)
    for inp_path, gt_path, ratio in zip(input_paths, gt_paths, ratios):
        print(f"image: {inp_path}")
        inp = read_input(inp_path, ratio)
        inp = inp.to(device)
        
        gt = read_target_disk(gt_path)

        inp = build_input(inp, coarseModel)
        
        with torch.no_grad():
            out = fineModel(inp)
            out = torch.clamp(out, 0, 1) 

            out = out[0].cpu().numpy().transpose(1, 2, 0) * 255
            
            if args.save_dir:
                save_name = ''.join(inp_path.split('/')[-1].split('.')[:-1])
                cv2.imwrite(f"{saved_dir}/images/{save_name}.jpg", out)
            
            out = bgr2rgb(out)    
            psnr, ssim = evaluate(gt, out)

            psnr_all.append(psnr)
            ssim_all.append(ssim)
        print("")
            
    print("")
    print('===============')
    print('testing summary')
    print(f'total testing files: {len(paths)}')
    print(f'psnr: {np.mean(psnr_all)}')
    print(f'ssim: {np.mean(ssim_all)}')
    
    if args.save_dir:
        np.save(saved_dir + '/psnr.npy', psnr_all)
        np.save(saved_dir + '/ssim.npy', ssim_all)
    
    
if __name__ == "__main__":
    main()