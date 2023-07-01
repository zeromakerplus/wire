#!/usr/bin/env python

import os
import sys
import glob
import tqdm
import importlib
import time
import pdb
import copy

import numpy as np
from scipy import io
from scipy import ndimage
import cv2

import torch
from torch.optim.lr_scheduler import LambdaLR

import matplotlib.pyplot as plt
plt.gray()

from modules import models
from modules import utils
from modules import volutils


def get_spectral_proj(img):
    [H,W,T,L] = img.shape
    ind = np.argmax((np.sum(img, axis = 3)), axis = 2)
    spectral_img = np.zeros([H,W,L])
    for i in range(H):
        for j in range(W):
            spectral_img[i,j,:] = img[i,j,ind[i,j],:]
    return spectral_img

if __name__ == '__main__':
    nonlin = 'wire' # type of nonlinearity, 'wire', 'siren', 'mfn', 'relu', 'posenc', 'gauss'
    niters = 20000               # Number of SGD iterations
    learning_rate = 5e-3        # Learning rate 
    expname = 'thai_statue'     # Volume to load
    scale = 1.0                 # Run at lower scales to testing, default 1.0
    mcubes_thres = 0.5          # Threshold for marching cubes
    
    # Gabor filter constants
    # These settings work best for 3D occupancies
    omega0 = 10.0          # Frequency of sinusoid
    sigma0 = 40.0          # Sigma of Gaussian
    
    # Network constants
    hidden_layers = 2       # Number of hidden layers in the mlp
    hidden_features = 256   # Number of hidden units per layer
    maxpoints = int(2e5)    # Batch size, default 2e5
    
    if expname == 'thai_statue':
        occupancy = True
    else:
        occupancy = False
    
    # Load image and scale
    # im = io.loadmat('data/%s.mat'%expname)['hypercube'].astype(np.float32)

    depth = cv2.imread('./data/depth.hdr', flags=cv2.IMREAD_ANYDEPTH)
    depth = depth[:,:,0]
    depth = depth / np.max(depth) * 2 - 1 + 0.1


    spectral_img = io.loadmat('data/chrac_spectral.mat')['img'].astype(np.float32)
    spectral_img = spectral_img[:,:,[26,16,6]]
    depth = ndimage.zoom(depth, [0.1,0.1], order=0, mode='nearest')
    spectral_img = ndimage.zoom(spectral_img, [0.1,0.1,1.0], order=0, mode='nearest')

    N = depth.shape[0]
    L = spectral_img.shape[2]
    im = np.zeros([N,N,N,L]).astype(np.float32())
    x = np.linspace(-1,1,N)
    y = np.linspace(-1,1,N)
    z = np.linspace(-1,1,N)
    for i in range(N):
        for j in range(N):
            ind = int((depth[i,j] - z[0]) / (z[1] - z[0]))
            ind = ind if ind < N else N-1
            ind = ind if ind > 0 else 0
            if depth[i,j] < 1.0:
                im[i,j,ind] = spectral_img[i,j,:]


    im = ndimage.zoom(im/im.max(), [scale, scale, scale, 1.0], order=0,mode='nearest')

    
    # If the volume is an occupancy, clip to tightest bounding box
    if occupancy:
        hidx, widx, tidx = np.where(np.mean(im, axis = 3) > 0.00001)
        im = im[hidx.min()-1:hidx.max()+1,
                widx.min()-1:widx.max()+1,
                tidx.min()-1:tidx.max()+1, :]
    
    print(im.shape)
    H, W, T, L = im.shape

    x = np.linspace(-1,1,H)
    y = np.linspace(-1,1,W)
    z = np.linspace(-1,1,T)
    colors = np.reshape(im,[H*W*T,L])
    inds = np.mean(colors, axis = 1) > 0.0002
    out_inds = np.sum(colors, axis = 1) <= 0.0002
    colors = colors[inds,:]

    [X,Y,Z] = np.meshgrid(x,y,z,indexing='ij')
    points = np.stack((X,Y,Z), axis = 3)
    points = np.reshape(points, [H*W*T, 3])
    points = points[inds,:]

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(points[:,0], points[:,1], points[:,2], c=colors[:,[26,16,6]], marker='o')
    # # ax.set_xlim(-0.5,0.5)
    # # ax.set_ylim(-0.5,0.5)
    # # ax.set_zlim(0,1)
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    # plt.show()


    im = im / np.mean(colors) * 1

    # im = im.mean(axis = 3)[...,None]
    # # im[im > 0.00005] = 1
    # L = 1
    
    maxpoints = min(H*W*T, maxpoints)
        
    imten = torch.tensor(im).cuda().reshape(H*W*T, L)
    
    if nonlin == 'posenc':
        nonlin = 'relu'
        posencode = True
    else:
        posencode = False
    
    # Create model
    model = models.get_INR(
                    nonlin=nonlin,
                    in_features=3,
                    out_features=L, 
                    hidden_features=hidden_features,
                    hidden_layers=hidden_layers,
                    first_omega_0=omega0,
                    hidden_omega_0=omega0,
                    scale=sigma0,
                    pos_encode=posencode,
                    sidelength=max(H, W, T)).cuda()
    
    # Optimizer
    optim = torch.optim.Adam(lr=learning_rate, params=model.parameters())
    
    # Schedule to 0.1 times the initial rate
    scheduler = LambdaLR(optim, lambda x: 0.2**min(x/niters, 1))

    criterion = torch.nn.MSELoss()
    
    # Create inputs
    coords = utils.get_coords(H, W, T)
    
    mse_array = np.zeros(niters)
    time_array = np.zeros(niters)
    best_mse = float('inf')
    best_img = None

    tbar = tqdm.tqdm(range(niters))
    
    im_estim = torch.zeros((H*W*T, L), device='cuda')
    im_mask = torch.rand(H*W*T,L, device='cuda') > 0.9
    im_mask[:,:] = True
    
    tic = time.time()
    print('Running %s nonlinearity'%nonlin)
    for idx in tbar:
        indices = torch.randperm(H*W*T)
        
        train_loss = 0
        nchunks = 0
        for b_idx in range(0, H*W*T, maxpoints):
            b_indices = indices[b_idx:min(H*W*T, b_idx+maxpoints)]
            b_coords = coords[b_indices, ...].cuda()
            b_mask = im_mask[b_indices, ...]
            b_indices = b_indices.cuda()
            pixelvalues = model(b_coords[None, ...]).squeeze(axis = 0)
            
            with torch.no_grad():
                im_estim[b_indices, :] = pixelvalues
        
            loss = criterion(pixelvalues * b_mask, imten[b_indices, :] * b_mask)
            
            optim.zero_grad()
            loss.backward()
            optim.step()
            
            lossval = loss.item()
            train_loss += lossval
            nchunks += 1

        # if occupancy:
        #     mse_array[idx] = volutils.get_IoU(im_estim, imten, mcubes_thres)
        # else:
        mse_array[idx] = train_loss/nchunks
        # print(mse_array[idx])
        time_array[idx] = time.time()
        scheduler.step()
        
        im_estim_vol = im_estim.reshape(H, W, T, L)
        
        if lossval < best_mse:
            best_mse = lossval
            best_img = copy.deepcopy(im_estim)

        # if sys.platform == 'win32':
        #     cv2.imshow('GT', im[..., idx%T])
        #     cv2.imshow('Estim', im_estim_vol[..., idx%T].detach().cpu().numpy())
        #     cv2.waitKey(1)
        
        # tbar.set_description('%.4e'%mse_array[idx])
        # tbar.refresh()
        # psnrval = -10*np.log10(mse_array[idx])
        # tbar.set_description('%.1f'%psnrval)
        # tbar.refresh()
        positive_psnrval = -10*torch.log10(torch.mean((imten - im_estim) ** 2) / torch.max(imten))
        tbar.set_description('%.1f'%positive_psnrval)
        tbar.refresh()
        
    total_time = time.time() - tic
    nparams = utils.count_parameters(model)
    
    best_img = best_img.reshape(H, W, T, L).detach().cpu().numpy()
    
    if posencode:
        nonlin = 'posenc'
        
    # Save data
    os.makedirs('results/%s'%expname, exist_ok=True)
    
    indices, = np.where(time_array > 0)
    time_array = time_array[indices]
    mse_array = mse_array[indices]

    
    spectral_img_gt= get_spectral_proj(im) / 2
    if L > 30:
        plt.imshow(spectral_img_gt[:,:,[26,16,6]])
    else:
        plt.imshow(spectral_img_gt[:,:,:])
    plt.savefig('results/%s/%s_gt.png'%(expname, nonlin))
    plt.close()
    spectral_img_estim= get_spectral_proj(best_img) / 2
    if L > 30:
        plt.imshow(spectral_img_estim[:,:,[26,16,6]])
    else:
        plt.imshow(spectral_img_estim[:,:,:])
    plt.savefig('results/%s/%s_estim.png'%(expname, nonlin))
    plt.close()
    print('img PSNR: ', 10*np.log10(np.max(spectral_img_gt) / np.mean((spectral_img_gt - spectral_img_estim) ** 2)))
    # ind1 = best_img.reshape(H*W,T,L)[:,,L]
    # ind2 = im.reshape(H*W,T,L)[:,(np.argmax((np.sum(im, axis = 3)).reshape(-1, T), axis = 1)),:]

    mdict = {'mse_array': mse_array,
             'best_img': best_img,
             'img': im,
             'time_array': time_array-time_array[0],
             'nparams': utils.count_parameters(model)}
    io.savemat('results/%s/%s.mat'%(expname, nonlin), mdict)
    
    # Generate a mesh with marching cubes if it is an occupancy volume
    # if occupancy:
    #     savename = 'results/%s/%s.dae'%(expname, nonlin)
    #     volutils.march_and_save(best_img, mcubes_thres, savename, True)
    
    print('Total time %.2f minutes'%(total_time/60))
    # if occupancy:
    #     print('IoU: ', volutils.get_IoU(best_img, im, mcubes_thres))
    # else:
    # print('PSNR: ', utils.psnr(im, best_img))


    print('PSNR: ', 10*np.log10(np.max(im) / np.mean((im - best_img) ** 2)))
    print('Total pararmeters: %.2f million'%(nparams/1e6))
    
    
