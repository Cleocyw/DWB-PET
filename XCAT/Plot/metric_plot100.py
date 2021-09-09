import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import numpy as np
import random
import torch
import glob
import torch.nn as nn
import torchvision
from scipy.io import loadmat
from mpl_toolkits.axes_grid1 import make_axes_locatable
import sys

#argv should be folder name, e.g. 'cat_mean_frame/'
if len(sys.argv)>1:
    folder_name=sys.argv[1]

path = folder_name + 'npy/'

cc_nx_array = np.zeros([800,14])
cc_nnx_array = np.zeros([800,14])
psnr_nx_array = np.zeros([800,14])
psnr_nnx_array = np.zeros([800,14])
for i in range(14):
    cc_nx_array[:,i]=np.load(path+'cc_nx-{}.npy'.format(i+1))
    cc_nnx_array[:,i]=np.load(path+'cc_nnx-{}.npy'.format(i+1))
    psnr_nx_array[:,i]=np.load(path+'psnr_nx-{}.npy'.format(i+1))
    psnr_nnx_array[:,i]=np.load(path+'psnr_nnx-{}.npy'.format(i+1))

fig1 = plt.figure()
plt.title('Correlation coefficient')
plt.xticks()
plt.xlabel('time frame')
plt.yticks()
plt.ylabel('iterations')
cc_img1 = plt.imshow(cc_nx_array[0:100,:],cmap='magma',aspect='auto')
plt.clim(0.5,cc_nx_array.max())
plt.colorbar(cc_img1)
plt.savefig(folder_name+'cc_nx.png')
plt.close(fig1)

loc_list = []
for i in range(14):
    loc = np.argmax(cc_nnx_array[0:100,i]) #find the number of iteration that gives the biggest cc_nnx value
    loc_list.append(loc)

print(loc_list)

fig2 = plt.figure()
plt.title('Correlation coefficient')
plt.xticks()
plt.xlabel('time frame')
plt.yticks()
plt.ylabel('iterations')
plt.plot(range(14),loc_list) #this plot the blue curve
cc_img2 = plt.imshow(cc_nnx_array[0:100,:],cmap='magma',aspect='auto')
plt.clim(0.5,cc_nnx_array.max())
plt.colorbar(cc_img2)
plt.savefig(folder_name+'cc_nnx.png')
plt.close(fig2)


fig3 = plt.figure()
plt.title('PSNR')
plt.xticks()
plt.xlabel('time frame')
plt.yticks()
plt.ylabel('iterations')
psnr_img1 = plt.imshow(psnr_nx_array[0:100,:],aspect='auto',cmap='magma')
plt.colorbar(psnr_img1)
plt.savefig(folder_name+'psnr_nx.png')
plt.close(fig3)

fig4 = plt.figure()
plt.title('PSNR')
plt.xticks()
plt.xlabel('time frame')
plt.yticks()
plt.ylabel('iterations')
psnr_img2 = plt.imshow(psnr_nnx_array[0:100,:],aspect='auto',cmap='magma')
plt.colorbar(psnr_img2)
plt.savefig(folder_name+'psnr_nnx.png')
plt.close(fig4)

