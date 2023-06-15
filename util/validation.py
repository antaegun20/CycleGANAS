import os
import numpy as np
from cleanfid import fid
#import matplotlib.pyplot as plt
from imageio import imsave, imread
#from cv2 import imread, imwrite

import torch
import torch.nn as nn
from torchvision.utils import make_grid

def get_fids(opt, save_dir, dataset, model): # just pass model
    # path: .ckpt/exp_name/fid_images/
    netG_A = model.netG_A.eval()
    netG_B = model.netG_B.eval()
    
    # clean first
    if os.path.exists(save_dir):
        os.system("rm -r {}".format(save_dir))
    
    os.makedirs(save_dir, exist_ok=True)
    A_B_dir = os.path.join(save_dir, 'A->B/') # A->B
    A_B_A_dir = os.path.join(save_dir, 'A->B->A/') # A->B->A
    B_A_dir = os.path.join(save_dir, 'B->A/') # B->A
    B_A_B_dir = os.path.join(save_dir, 'B->A->B/') # B->A->B
    os.makedirs(A_B_dir, exist_ok=True)
    os.makedirs(A_B_A_dir, exist_ok=True)
    os.makedirs(B_A_dir, exist_ok=True)
    os.makedirs(B_A_B_dir, exist_ok=True)
    
    A_B_img_list, A_B_A_img_list, B_A_img_list, B_A_B_img_list = list(), list(), list(), list()
    AtoB = opt.direction == 'AtoB'
    device = next(netG_A.parameters()).device
    for iter_idx, data in enumerate(dataset):
        real_A = data['A' if AtoB else 'B'].to(device)
        real_B = data['B' if AtoB else 'A'].to(device)
        
        fake_B = netG_A(real_A)
        recon_A = netG_B(fake_B)
        A_B_img_list.extend(list(fake_B.detach().cpu()))
        A_B_A_img_list.extend(list(recon_A.detach().cpu()))
        
        fake_A = netG_B(real_B)
        recon_B = netG_A(fake_A)
        B_A_img_list.extend(list(fake_A.detach().cpu()))
        B_A_B_img_list.extend(list(recon_B.detach().cpu()))
        
        if iter_idx == 0:
            ra = real_A.mul_(127.5).add_(127.5).clamp_(0.0, 255.0).permute(0, 2, 3, 1).to("cpu", torch.uint8).numpy()
            rb = real_B.mul_(127.5).add_(127.5).clamp_(0.0, 255.0).permute(0, 2, 3, 1).to("cpu", torch.uint8).numpy()
            imsave(os.path.join(save_dir, 'A.png'), ra[0])
            imsave(os.path.join(save_dir, 'B.png'), rb[0])
        
    A_B_img_list = torch.stack(A_B_img_list).mul_(127.5).add_(127.5).clamp_(0.0, 255.0).permute(0, 2, 3, 1).to("cpu", torch.uint8)
    A_B_A_img_list = torch.stack(A_B_A_img_list).mul_(127.5).add_(127.5).clamp_(0.0, 255.0).permute(0, 2, 3, 1).to("cpu", torch.uint8)
    B_A_img_list = torch.stack(B_A_img_list).mul_(127.5).add_(127.5).clamp_(0.0, 255.0).permute(0, 2, 3, 1).to("cpu", torch.uint8)
    B_A_B_img_list = torch.stack(B_A_B_img_list).mul_(127.5).add_(127.5).clamp_(0.0, 255.0).permute(0, 2, 3, 1).to("cpu", torch.uint8)
    
    for img_idx, (AB, ABA, BA, BAB) in enumerate(zip(A_B_img_list, A_B_A_img_list, B_A_img_list, B_A_B_img_list)):
        imsave(os.path.join(A_B_dir, '{}.png'.format(img_idx)), AB.numpy())
        imsave(os.path.join(A_B_A_dir, '{}.png'.format(img_idx)), ABA.numpy())
        imsave(os.path.join(B_A_dir, '{}.png'.format(img_idx)), BA.numpy())
        imsave(os.path.join(B_A_B_dir, '{}.png'.format(img_idx)), BAB.numpy())
        
    # clean fid
    A_B_clean_fid = fid.compute_fid(A_B_dir, os.path.join(opt.dataroot, 'testB'),
                                    verbose=False, device=device, use_dataparallel=False)
    A_B_A_clean_fid = fid.compute_fid(A_B_A_dir, os.path.join(opt.dataroot, 'testA'),
                                      verbose=False, device=device, use_dataparallel=False)
    B_A_clean_fid = fid.compute_fid(B_A_dir, os.path.join(opt.dataroot, 'testA'),
                                    verbose=False, device=device, use_dataparallel=False)
    B_A_B_clean_fid = fid.compute_fid(B_A_B_dir, os.path.join(opt.dataroot, 'testB'),
                                      verbose=False, device=device, use_dataparallel=False)    
    netG_A = netG_A.train()
    netG_B = netG_B.train()
    
    return A_B_clean_fid, A_B_A_clean_fid, B_A_clean_fid, B_A_B_clean_fid


def get_grid_image(images_dir):
    imgs = []
    imgs += [torch.tensor(imread(os.path.join(images_dir, 'A.png')))]
    imgs += [torch.tensor(imread(os.path.join(images_dir, 'A->B', '0.png')))]
    imgs += [torch.tensor(imread(os.path.join(images_dir, 'A->B->A', '0.png')))]
    imgs += [torch.tensor(imread(os.path.join(images_dir, 'B.png')))]
    imgs += [torch.tensor(imread(os.path.join(images_dir, 'B->A', '0.png')))]
    imgs += [torch.tensor(imread(os.path.join(images_dir, 'B->A->B', '0.png')))]
    for i in range(len(imgs)):
        imgs[i] = imgs[i].permute(2, 0, 1).to("cpu", torch.uint8)
        
    grid_img = make_grid(imgs, nrow=3, ncol=2)
    return grid_img
    

def get_arch_diffs(model):
    def get_discrete_arch(ws):
        discrete_arch = []
        for w in ws:
            discrete_arch.append(torch.argmax(nn.Softmax(dim=-1)(w)).item())
        return np.array(discrete_arch)
    
    G_A_diff = np.sum(abs(get_discrete_arch(model.netG_A.arch_parameters).astype(np.int64)))
    G_B_diff = np.sum(abs(get_discrete_arch(model.netG_B.arch_parameters).astype(np.int64)))
    D_A_diff = np.sum(abs(get_discrete_arch(model.netD_A.arch_parameters).astype(np.int64)))
    D_B_diff = np.sum(abs(get_discrete_arch(model.netD_B.arch_parameters).astype(np.int64)))
    
    return G_A_diff, G_B_diff, D_A_diff, D_B_diff
    
def get_G_arch_diffs(model):
    def get_discrete_arch(ws):
        discrete_arch = []
        for w in ws:
            discrete_arch.append(torch.argmax(nn.Softmax(dim=-1)(w)).item())
        return np.array(discrete_arch)
    
    G_A_diff = np.sum(abs(get_discrete_arch(model.netG_A.arch_parameters).astype(np.int64)))
    G_B_diff = np.sum(abs(get_discrete_arch(model.netG_B.arch_parameters).astype(np.int64)))
    
    return G_A_diff, G_B_diff

def save_arch(model, save_dir, epoch):
    G_A_arch, G_B_arch, D_A_arch, D_B_arch = [], [], [], []
    for p in model.netG_A.arch_parameters:
        G_A_arch.append(torch.argmax(nn.Softmax(dim=-1)(p)).item())
            
    for p in model.netG_B.arch_parameters:
        G_B_arch.append(torch.argmax(nn.Softmax(dim=-1)(p)).item())
            
    for p in model.netD_A.arch_parameters:
        D_A_arch.append(torch.argmax(nn.Softmax(dim=-1)(p)).item())
            
    for p in model.netD_B.arch_parameters:
        D_B_arch.append(torch.argmax(nn.Softmax(dim=-1)(p)).item())
            
    np.save(save_dir + '/{}_G_A'.format(epoch), G_A_arch)
    np.save(save_dir + '/{}_G_B'.format(epoch), G_B_arch)
    np.save(save_dir + '/{}_D_A'.format(epoch), D_A_arch)
    np.save(save_dir + '/{}_D_B'.format(epoch), D_B_arch)
    
def save_G_archs(model, save_dir, epoch):
    G_A_arch, G_B_arch, D_A_arch, D_B_arch = [], [], [], []
    for p in model.netG_A.arch_parameters:
        G_A_arch.append(torch.argmax(nn.Softmax(dim=-1)(p)).item())
            
    for p in model.netG_B.arch_parameters:
        G_B_arch.append(torch.argmax(nn.Softmax(dim=-1)(p)).item())
            
    np.save(save_dir + '/{}_G_A'.format(epoch), G_A_arch)
    np.save(save_dir + '/{}_G_B'.format(epoch), G_B_arch)
    
    
    
    
    
    
    
    
    