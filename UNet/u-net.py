#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import torch
import glob
import torch.nn as nn
import torchvision
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.autograd import Variable
from torchvision import datasets

import pathlib
import matplotlib.pyplot as plt
import torch.nn.functional as F


# In[2]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# In[3]:


def activation_layer(activation:str):

    if activation == 'leaky':
        return nn.LeakyReLU(negative_slope=0.1)


# In[4]:


def normalization_layer(normalization: str,
                      num_channels: int, dim:int):
    if dim == 2:
        if normalization == 'BN':
            return nn.BatchNorm2d(num_channels)
    elif dim == 3:
        if normalization == 'BN':
            return nn.BatchNorm3d(num_channels)


# In[5]:


def pooling_layer(pooling:str, dim:int):
    if dim == 2:
        if pooling == "max":
            return nn.MaxPool2d(kernel_size=2,stride=2,padding=0)
        #if pooling == 'stride':
           # return nn.Conv2d()
    if dim == 3:
        if pooling == "max":
            return nn.MaxPool3d(kernel_size=2,stride=2,padding=0)


# In[6]:


def conv_layer(in_chs, out_chs, kernel_size, stride, padding, dim):
    if dim == 2:
        return nn.Conv2d(in_chs, out_chs, kernel_size, stride, padding)
    elif dim == 3:
        return nn.Conv3d(in_chs, out_chs, kernel_size, stride, padding)


# In[7]:


def up_sample_layer(up_sample,in_chs = None, out_chs = None, kernel_size = 2, stride = 2, dim = 3):
    if up_sample == 'transposed':
        if dim == 2:
            return nn.ConvTranspose2d(in_chs, out_chs, kernel_size,stride)
        elif dim == 3:
            return nn.ConvTranspose3d(in_chs, out_chs, kernel_size,stride)
    else:
        return nn.Upsample(scale_factor=2, mode=up_sample)


# In[8]:


def Cat(tensor1, tensor2):
    
    x = torch.cat((tensor1, tensor2), 1)

    return x


# In[9]:


def Add (tensor1, tensor2):
    
    x = torch.add(tensor1, tensor2)
    
    return x


# In[22]:


class DownBlock(nn.Module):
    """
    represent a block of left part of the U shape.
    it contains two convolution layers, 
    each followed by a batch normalization (BN) and a leaky rectified,
    and a downsampling layer followed by a BN and leakyRElu
    
    """

    def __init__(self,
                 in_ch,
                 out_ch,
                 pooling: str = "max",
                 stride_pooling = True,
                 kernel_size: int = 3,
                 stride:int = 1,
                 padding: int = 1,
                 activation: str = 'leaky',
                 normalization: str = 'BN',
                 dim: int = 2):
        super().__init__()

        self.in_ch = in_ch
        self.out_ch = out_ch
        self.pooling = pooling
        self.stride_pooling = stride_pooling
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.activation = activation
        self.normalization = normalization
        self.dim = dim
        
        self.activation_layer = activation_layer(self.activation)
        self.normalization_layer = normalization_layer(normalization=self.normalization, num_channels=self.out_ch,
                                           dim=self.dim)       
        self.pooling_layer = pooling_layer(pooling = self.pooling, dim=self.dim)
        self.stride_layer = conv_layer(self.out_ch, self.out_ch, kernel_size = self.kernel_size, stride = 2, padding = self.padding, 
                                          dim = self.dim)
        
        #self.tensor_to_cat = nn.ModuleList()
        self.conv_layer1 = conv_layer(self.in_ch, self.out_ch, kernel_size = self.kernel_size, stride = self.stride, padding = self.padding, 
                                          dim = self.dim)
        self.conv_layer2 = conv_layer(self.out_ch, self.out_ch, kernel_size = self.kernel_size, stride = self.stride, padding = self.padding, 
                                          dim = self.dim)




    def forward(self, x):
        x = self.conv_layer1(x)
        x = self.normalization_layer(x)
        x = self.activation_layer(x)
        x = self.conv_layer2(x)
        x = self.normalization_layer(x)
        x = self.activation_layer(x)
        connect_layer = x
        if self.stride_pooling:
            x = self.stride_layer(x)
            
        else:
            x =  self.pooling_layer(x)
        x = self.normalization_layer(x)
        x = self.activation_layer(x)
                                                       
        return x,connect_layer


# In[ ]:


def crop(down_level, up_level):
    """
    the input down_level is each level's last tensor on the left (down) side. e.g. [1,1,256,256,64]; up_level is the first
    tensor on the right (up) side.
    Center-crops the encoder_layer to the size of the decoder_layer,
    so can catanate encoder layer to decoder layer
    This is only necessary for input sizes != 2**n for 'same' padding and always required for 'valid' padding.
    """
    if down_level.shape[2:] != up_level.shape[2:]:
        down_shape = down_level.shape[2:]
        up_shape = up_level.shape[2:]
#down_shape should bigger than up_shape
        if down_level.dim() == 4:  # 2D
            down_level = encoder_layer[
                            :,
                            :,
                            ((down_shape[0] - up_shape[0]) // 2):((down_shape[0] + up_shape[0]) // 2),
                            ((down_shape[1] - up_shape[1]) // 2):((down_shape[1] + up_shape[1]) // 2)
                            ]
        elif down_level.dim() == 5:  # 3D
            down_level = down_level[
                            :,
                            :,
                            ((down_shape[0] - up_shape[0]) // 2):((down_shape[0] + up_shape[0]) // 2),
                            ((down_shape[1] - up_shape[1]) // 2):((down_shape[1] + up_shape[1]) // 2),
                            ((down_shape[2] - up_shape[2]) // 2):((down_shape[2] + up_shape[2]) // 2),
                            ]
    return down_level, up_level


# In[44]:


class UpBlock(nn.Module):
    """

    """

    def __init__(self,
                 in_ch,
                 out_ch,
                 concatenate:bool = False,
                 add : bool = False,
                 Crop:bool = False,
                 kernel_size = 3,
                 stride = 1,
                 padding = 1,
                 activation: str = 'leaky',
                 normalization: str = "BN",
                 dim: int = 3,
                 up_sample: str = 'nearest'
                 ):
        super().__init__()

        self.in_ch =in_ch
        self.out_ch = out_ch
        self.concatenate = concatenate
        self.add = add
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.activation = activation
        self.normalization = normalization
        self.dim = dim
        self.up_sample = up_sample
        self.Crop = Crop
        

    
        self.activation_layer = activation_layer(self.activation)
     
        self.up_sample_layer = up_sample_layer(up_sample = self.up_sample)
        
        self.conv_layer1 = conv_layer(self.in_ch, self.out_ch, kernel_size = self.kernel_size, stride = self.stride, padding = self.padding, 
                                          dim = self.dim)
        if self.add:
            self.conv_layer2 = conv_layer(self.out_ch, self.out_ch, kernel_size = self.kernel_size, stride = self.stride, padding = self.padding, 
                                          dim = self.dim)
        elif self.concatenate:
            self.conv_layer2 = conv_layer(self.in_ch, self.out_ch, kernel_size = self.kernel_size, stride = self.stride, padding = self.padding, 
                                          dim = self.dim)            
        self.norm_layer = normalization_layer(normalization=self.normalization, num_channels=self.out_ch,
                                           dim=self.dim)        
            
    def forward(self, x, connect_layer):
        """ 

        """
        #deconv + upsample
        x = self.conv_layer1(x)
        x = self.up_sample_layer(x)
        
        #merge
        if self.concatenate:
            x = Cat(connect_layer,x)
        elif self.add:
            x = Add(connect_layer,x)
        
        #conv+bn+lrelu
        x = self.conv_layer2(x)
        x = self.norm_layer(x)
        x = self.activation_layer(x)
        #conv+bn+lrelu
        x = self.conv_layer2(x)
        x = self.norm_layer(x)
        x = self.activation_layer(x)
        

        return x


# In[63]:


class UNet(nn.Module):
    def __init__(self,
                 chs = [1,16,32,64,128],
                 concatenate:bool = False,
                 add:bool = False,
                 Crop:bool=False,
                 pooling = "max",
                 stride_pooling = True,
                 kernel_size = 3,
                 stride = 1,
                 padding = 1,
                 activation: str = 'leaky',
                 normalization: str = "BN",
                 dim: int = 3,
                 up_sample: str = 'nearest'
                 ):
        super().__init__()

        self.chs = chs
        self.depth = len(chs)-2
        self.pooling = pooling
        self.stride_pooling = stride_pooling
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.activation = activation
        self.normalization = normalization
        self.dim = dim
        self.concatenate = concatenate
        self.add = add
        self.up_sample = up_sample
        self.Crop = Crop
        
        
        self.encoder = nn.ModuleList([])
        self.decoder = nn.ModuleList([])
        for i in range(self.depth):
            encoder_layer = DownBlock(
                 in_ch=self.chs[i],
                 out_ch=self.chs[i+1],
                 #concatenate = True,
                 stride_pooling = True,
                 pooling = self.pooling,
                 kernel_size = self.kernel_size,
                 stride = self.stride,
                 padding = self.padding,
                 activation = self.activation,
                 normalization = self.normalization,
                 dim= self.dim)
            
            self.encoder.append(encoder_layer)
            
            decoder_layer = UpBlock(
                 in_ch = self.chs[-1-i],
                 out_ch = self.chs[-2-i],
                 concatenate= self.concatenate,
                 add = self.add,
                 Crop=self.Crop,
                 kernel_size = self.kernel_size,
                 stride = self.stride,
                 padding = self.padding,
                 activation= self.activation,
                 normalization = self.normalization,
                 dim = self.dim,
                 up_sample= self.up_sample) 
            
            self.decoder.append(decoder_layer)

                       
        
    def forward(self,x):
        connect_list = []
        
        #encoder path
        for i in range(self.depth):
            block = self.encoder[i]
            x,connect_layer = block(x)
            connect_list.append(connect_layer)
            
        #bottom block
        
        act_layer = activation_layer(self.activation)
        conv_layer1 = conv_layer(self.chs[-2], self.chs[-1], kernel_size = self.kernel_size, stride = self.stride, padding = self.padding, 
                                          dim = self.dim)          
        conv_layer2 = conv_layer(self.chs[-1], self.chs[-1], kernel_size = self.kernel_size, stride = self.stride, padding = self.padding, 
                                          dim = self.dim)
        norm_layer = normalization_layer(normalization=self.normalization, num_channels=self.chs[-1],
                                           dim=self.dim)  
        x = conv_layer1(x)
        x = norm_layer(x)
        x = act_layer(x)
        x = conv_layer2(x)
        x = norm_layer(x)
        x = act_layer(x)
        
        #decoder path
        for i in range(self.depth):
            layer_to_connect = connect_list[-1-i]
            block = self.decoder[i]
            x = block(x,layer_to_connect)
            
        #last layer: 16 to 1
        conv_layer_final = conv_layer(self.chs[1], self.chs[0], kernel_size = self.kernel_size, stride = self.stride, padding = self.padding, 
                                          dim = self.dim)
        norm_layer_final = normalization_layer(normalization=self.normalization, num_channels=self.chs[0],
                                           dim=self.dim) 
        x = conv_layer_final(x)
        x = norm_layer_final(x)
        x = act_layer(x)
            
        return x
                
            



model = UNet(chs = [1,16,32,64,128],
                 concatenate= False,
                 add = True,
                 Crop=False,
                 pooling = "max",
                 stride_pooling = True,
                 kernel_size = 3,
                 stride = 1,
                 padding = 1,
                 activation= 'leaky',
                 normalization = "BN",
                 dim= 2,
                 up_sample = 'nearest').to(device)


# In[94]:


torch.save(model, "UNet.pth")

#model = torch.load("UNet.pth")



# start to try real image data
# take one slice of image from xcat.mat

# In[114]:


from scipy.io import loadmat
P = loadmat('xcat.mat')
p = P['data']
p1 = p[:,:,256:256+32,3]
P1= torch.from_numpy(p1)
P1 = P1.unsqueeze(0)
P1 = P1.unsqueeze(0)




# # add noise

# The function torch.randn produces a tensor with elements drawn from a Gaussian distribution of zero mean and unit variance. Multiply by sqrt(0.1) to have the desired variance.
# 
# x = torch.zeros(5, 10, 20, dtype=torch.float64)
# x = x + (0.1**0.5)*torch.randn(5, 10, 20)

# In[116]:


P1 = P1[:,:,:,0]
poisson_noise = torch.poisson(P1)*0.1




# In[118]:


image = P1+poisson_noise
label = P1


# In[117]:




optimizer = Adam(model.parameters(), lr=1e-3)
# mean-squared error loss
criterion = nn.MSELoss()
#criterion = nn.CrossEntropyLoss()
#criterion = nn.BCELoss()
if torch.cuda.is_available():
    model = model.cuda()
    criterion = criterion.cuda()


# In[122]:


def train(epoch):
    model.train()
    train_loss = 0.0
    
    x_train, y_train = Variable(image), Variable(label)

    if torch.cuda.is_available():
        x_train = x_train.cuda()           

        y_train = y_train.cuda()
            
    optimizer.zero_grad() 
    
    output = model(x_train) #give us prediction
        #print(outputs.shape)
        #outputs = torch.argmax(outputs, dim=1)

    loss = criterion(output,y_train)
    loss.backward()
    optimizer.step()
        
    #train_loss += loss.cpu().data*image.size(0)
    train_loss += loss.item()
        
    print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch, train_loss))

    
 

num_epochs = 20
train_loss = 0
for epoch in range(num_epochs):
    train(epoch)


# training error and test error plot
