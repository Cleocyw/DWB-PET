#!/usr/bin/env python
# coding: utf-8

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import numpy as np
import random
import torch
import glob
import torch.nn as nn
from scipy.io import loadmat
import math
from mpl_toolkits.axes_grid1 import make_axes_locatable
import sys



if len(sys.argv)>1:
    file_name = sys.argv[1]
    folder_name=sys.argv[2]


file_name = '../../SAN/inm/DWB-PET/raw_mat_files/'+ file_name
TAC_folder = folder_name+'TAC/'
if not os.path.exists(TAC_folder):
    os.makedirs(TAC_folder)
path = folder_name + 'npy/'
x = 128
y = 130
z = 270


# this gives TAC of noisy label
tac_measured = []
for i in range(14):
    measured = np.load(path+'label-{}.npy'.format(i+1))
    pixel_measured = measured[x,y]
    tac_measured.append(pixel_measured)


#this gives denoised DIP denoised TAC
tac_denoised = []
 
for i in range(14):# time is 14
    denoised = np.load(path+'output:{}-{}.npy'.format(i+1,450)) #450 is the stopping time
    pixel_denoised = denoised[x,y] 
    tac_denoised.append(pixel_denoised)


#plot TAC
fig = plt.figure()
plt.plot(range(14), tac_measured,label='noisy label:TAC in heart')
plt.plot(range(14), tac_denoised[i],label='DIP Denoised:TAC in heart')
plt.legend()
plt.xlabel('time')
plt.ylabel('Acitivity [a.u.]')
plt.savefig(TAC_folder+'TAC-{}.png'.format(450))
plt.close(fig)#not display the image

