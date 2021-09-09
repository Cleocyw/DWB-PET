#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
#from torchvision.transforms import transforms
#from torch.utils.data import DataLoader
#from torch.optim import Adam
#from torch.autograd import Variable
#from torchvision import datasets
from scipy.io import loadmat
#import pathlib
import math
#import copy
#import torch.nn.functional as F
#from skimage.measure import compare_psnr
#from scipy.ndimage import gaussian_filter
from mpl_toolkits.axes_grid1 import make_axes_locatable
import sys



if len(sys.argv)>1:
    file_name = sys.argv[1]
    folder_name=sys.argv[2]
    location = sys.argv[3]
if location == 'tumor':
    x = 96
    y = 128
    z = 92
elif location == 'liver':
    x = 150
    y = 130
    z = 110
file_name = '../../SAN/inm/DWB-PET/raw_mat_files/'+ file_name
TAC_folder = folder_name+'TAC/'
if not os.path.exists(TAC_folder):
    os.makedirs(TAC_folder)
path = folder_name + 'npy/'

# get non-noisy TACs
def get_TACperfect(file_name):
    tac_perfect = []
    data = loadmat(file_name)
    whole_array= data['data']
    for i in range(14):
        pixel = whole_array[x,y,z,i]
        tac_perfect.append(pixel)
    return tac_perfect
        
tac_perfect = get_TACperfect(file_name)

# this gives TAC of noisy xcat
tac_measured = []
for i in range(14):
    measured = np.load(path+'real-{}.npy'.format(i+1))
    pixel_measured = measured[x,y]
    tac_measured.append(pixel_measured)



#this gives denoised DIP xcat TAC
tac_denoised=np.zeros([10,14])
for j in range(10):  #every 80 iterations draw a TAC. has 800 iteration in total. 
    for i in range(14):# time is 14
        denoised = np.load(path+'output:{}-{}.npy'.format(i+1,(j+1)*80))
        pixel_denoised = denoised[x,y] 
        tac_denoised[j][i]=pixel_denoised



#plot TAC
for i in range(10):
    fig = plt.figure()
    plt.plot(range(14), tac_perfect,label='non-nosiy xcat:TAC in'+location)
    plt.plot(range(14), tac_measured,label='noisy xcat:TAC in'+location)
    plt.plot(range(14), tac_denoised[i],label='DIP Denoised:TAC in'+location)
    plt.title('Iter {} :TAC in'+location.format(80*(i+1)))
    plt.legend()
    plt.xlabel('time')
    plt.ylabel('Acitivity [a.u.]')
    plt.savefig(TAC_folder+'TAC_{}-{}.png'.format(location,80*(i+1)))
    plt.close(fig)#not display the image

