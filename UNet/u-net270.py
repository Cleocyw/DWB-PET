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
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.autograd import Variable
from torchvision import datasets
from scipy.io import loadmat
import pathlib
import math

import torch.nn.functional as F
from skimage.measure import compare_psnr



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




def activation_layer(activation:str):

    if activation == 'leaky':
        return nn.LeakyReLU(negative_slope=0.1)
    
def normalization_layer(normalization: str,
                      num_channels: int, dim:int):
    if dim == 2:
        if normalization == 'BN':
            return nn.BatchNorm2d(num_channels)
    elif dim == 3:
        if normalization == 'BN':
            return nn.BatchNorm3d(num_channels)
        
def pooling_layer(pooling:str, dim:int):
    if dim == 2:
        if pooling == "max":
            return nn.MaxPool2d(kernel_size=2,stride=2,padding=0)
        elif pooling == 'avg':
            return nn.AvgPool2d(kernel_size=2,stride=2,padding=0)

    if dim == 3:
        if pooling == "max":
            return nn.MaxPool3d(kernel_size=2,stride=2,padding=0)
        elif pooling == 'avg':
            return nn.AvgPool3d(kernel_size=2,stride=2,padding=0)

def conv_layer(in_chs, out_chs, kernel_size, stride, padding, dim):
    if dim == 2:
        return nn.Conv2d(in_chs, out_chs, kernel_size, stride, padding)
    elif dim == 3:
        return nn.Conv3d(in_chs, out_chs, kernel_size, stride, padding)

def up_sample_layer(up_sample,in_chs = None, out_chs = None, kernel_size = 2, stride = 2, dim = 3):
    if up_sample == 'transposed':
        if dim == 2:
            return nn.ConvTranspose2d(in_chs, out_chs, kernel_size,stride)
        elif dim == 3:
            return nn.ConvTranspose3d(in_chs, out_chs, kernel_size,stride)
    else:
        return nn.Upsample(scale_factor=2, mode=up_sample) # mode can be 'nearest', 'bilinear' ,...
    
def Cat(tensor1, tensor2):    
    x = torch.cat((tensor1, tensor2), 1)
    return x

def Add (tensor1, tensor2):    
    x = torch.add(tensor1, tensor2)    
    return x


def autocrop(down_layer: torch.Tensor, up_layer: torch.Tensor):
    """
    Crop the tensors from  encoder and decoder pathways so that they
    can be merged when "skip to connect".
    This is only necessary for input sizes != 2**n for 'same' padding and always required for 'valid' padding.
    Input:
        down_layer: the last tensor(/layer) in each downBlock, which is just the connect_layer of the output of DownBlock.
        up_layer: the layer in UpBlock that need to be connected.
    Return:
        a tuple of two croped layers 
    """
    ndim = down_layer.dim()  # give the dim of the tensor , should be 5 in this work

    if down_layer.shape[2:] == up_layer.shape[2:]:  # the images are of same size so no need to crop anything
        return down_layer, up_layer

    # Step 1: Handle odd shape input layer from the encoder block.
    #  by cropping up_layer by 1 in the dim that is odd in down_layer because up_layer would be increased by value 1 when upsampling.
    # this is the situation in our work
    
    ds = down_layer.shape[2:] #e.g. (256,256,53)
    us = up_layer.shape[2:] #e.g. (256,256,54)
    croped_up = [u - ((u - d) % 2) for d, u in zip(ds, us)] #so it only crop 54 (from UpBlock layer) by one. (256,256,54) ->(256,256,53)

    if ndim == 4:
        up_layer = up_layer[:, :, :croped_up[0], :croped_up[1]]
    if ndim == 5:
        up_layer = up_layer[:, :, :croped_up[0], :croped_up[1], :croped_up[2]]

    # Step 2: when down_shape > up_shape,do center-crop to make down layer smaller.
    #this happens during downsampling when padding = 0, so the down layer will get smaller.
    ds = down_layer.shape[2:]
    us = up_layer.shape[2:]

    assert ds[0] >= us[0], f'{ds, us}'
    assert ds[1] >= us[1]
    if ndim == 4:
        down_layer = down_layer[
            :,
            :,
            (ds[0] - us[0]) // 2:(ds[0] + us[0]) // 2,
            (ds[1] - us[1]) // 2:(ds[1] + us[1]) // 2
        ]
    elif ndim == 5:
        assert ds[2] >= us[2]
        down_layer = down_layer[
            :,
            :,
            ((ds[0] - us[0]) // 2):((ds[0] + us[0]) // 2),
            ((ds[1] - us[1]) // 2):((ds[1] + us[1]) // 2),
            ((ds[2] - us[2]) // 2):((ds[2] + us[2]) // 2),
        ]
    return down_layer, up_layer



class DownBlock(nn.Module):
    """
    represent a block from the left part of  U shape.
    it contains two convolution layers, 
    each followed by a batch normalization (BN) and a leaky rectified,
    and a downsampling layer followed by a BN and leakyRElu
      
    
    """

    def __init__(self,
                 in_ch,
                 out_ch,
                 stride_pooling:bool,
                 pooling: str = "max",     
                 kernel_size: int = 3,
                 stride:int = 1,
                 padding: int = 1,
                 activation: str = 'leaky',
                 normalization: str = 'BN',
                 dim: int = 2
                 ):
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
    
    

class Latent(nn.Module):
    """
    Latent, also called bottleneck, represents the bottom middle part of the UNet.
    In this work, it contains a conv+BN+LeakyRelu + conv+BN+LeakyRelu.
    """
    def __init__(self,
                 in_ch,
                 out_ch,
                 kernel_size = 3,
                 stride = 1,
                 padding = 1,
                 activation: str = 'leaky',
                 normalization: str = "BN",
                 dim: int = 2

                 ):
        super().__init__()

        self.in_ch =in_ch
        self.out_ch = out_ch
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.activation = activation
        self.normalization = normalization
        self.dim = dim
        
        self.activation_layer = activation_layer(self.activation)
        self.norm_layer = normalization_layer(normalization=self.normalization, num_channels=self.out_ch,
                                           dim=self.dim) 
        self.conv_layer1 = conv_layer(self.in_ch, self.out_ch, kernel_size = self.kernel_size, stride = self.stride, padding = self.padding, 
                                          dim = self.dim)          
        self.conv_layer2 = conv_layer(self.out_ch, self.out_ch, kernel_size = self.kernel_size, stride = self.stride, padding = self.padding, 
                                          dim = self.dim)
    def forward(self,x):
        x = self.conv_layer1(x)
        x = self.norm_layer(x)
        x = self.activation_layer(x)
        x = self.conv_layer2(x)
        x = self.norm_layer(x)
        x = self.activation_layer(x)
        
        return x
        

        
class UpBlock(nn.Module):
    """
    it corresponds to "red arrow+blue arrow+ blue arrow", i.e.
    [decon_layer (half the number of channels)+ Upsampling (double image size)]+
    [conv+bn+leaky]+[con+bn+leaky]
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
                 dim: int = 2,
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
            self.conv_layer3 = conv_layer(self.out_ch, self.out_ch, kernel_size = self.kernel_size, stride = self.stride, padding = self.padding, 
                                          dim = self.dim)
        self.norm_layer = normalization_layer(normalization=self.normalization, num_channels=self.out_ch,
                                           dim=self.dim)        
            
    def forward(self, x, connect_layer):

        #deconv + upsample
        x = self.conv_layer1(x) #128 -> 64
        x = self.up_sample_layer(x) # 32*32 -> 64*64
        if self.Crop:
            connect_layer,x = autocrop(connect_layer,x)
        #merge
        if self.concatenate:
            x = Cat(connect_layer,x) #64 -> 128
            x = self.conv_layer2(x) #128->64
            x = self.norm_layer(x) 
            x = self.activation_layer(x)
            x = self.conv_layer3(x) #64 -> 64
            x = self.norm_layer(x)
            x = self.activation_layer(x)
            
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
    
        

class last_block(nn.Module):
    """
    it's the last block of layers after the UpBlock to make channel 16 into channel 1.
    it contains conv+bn+leakyRelu
    """
    def __init__(self,
                 in_ch,
                 out_ch,
                 kernel_size = 3,
                 stride = 1,
                 padding = 1,
                 activation: str = 'leaky',
                 normalization: str = "BN",
                 dim: int = 2

                 ):
        super().__init__()
        
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.activation = activation
        self.normalization = normalization
        self.dim = dim
        
        self.conv_layer_final = conv_layer(self.in_ch, self.out_ch, kernel_size = self.kernel_size, stride = self.stride, padding = self.padding, 
                                          dim = self.dim)
        self.norm_layer_final = normalization_layer(normalization=self.normalization, num_channels=self.out_ch,
                                           dim=self.dim) 
        
        
        
    def forward(self,x):

        x = self.conv_layer_final(x)
        x = self.norm_layer_final(x)
        #x = act_layer(x)
        #x = nn.Sigmoid()(x)
        #x = nn.Linear(256,256)(x)
        return x


# In[36]:


class UNet(nn.Module):
    """
    it combines DownBlock + Latent + UpBlock + the final conv_layer block.
    we want to follow the UNet from the paper, so  here depth is 3, which means
    the UNet will first run DownBlock for three times,
    then reach the bottom, and will run Latent,
    then will run UpBlock for three times,
    then we add the last_block to make channels from 16 -> 1

    """
    def __init__(self,
                 stride_pooling:bool,
                 chs = [1,16,32,64,128],
                 concatenate:bool = False,
                 add:bool = False,
                 Crop:bool=True,
                 pooling = "max",
                 
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
        self.latent = Latent(in_ch = self.chs[-2],
                             out_ch = self.chs[-1],
                             kernel_size = self.kernel_size,
                             stride = self.stride,
                             padding = self.padding,
                             activation = self.activation,
                             normalization = self.normalization,
                             dim = self.dim)
        self.last_block = last_block(in_ch = self.chs[1],
                             out_ch = self.chs[0],
                             kernel_size = self.kernel_size,
                             stride = self.stride,
                             padding = self.padding,
                             activation = self.activation,
                             normalization = self.normalization,
                             dim = self.dim)
        
        for i in range(self.depth):
            encoder_layer = DownBlock(
                 in_ch=self.chs[i],
                 out_ch=self.chs[i+1],
                 #concatenate = True,
                 stride_pooling = self.stride_pooling,
                 pooling = self.pooling,
                 kernel_size = self.kernel_size,
                 stride = self.stride,
                 padding = self.padding,
                 activation = self.activation,
                 normalization = self.normalization,
                 dim= self.dim)
            
            self.encoder.append(encoder_layer) #encoder is the modulelist, and it appends each downblocks
            
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
            
        self.set_weights()
        

    @staticmethod        
    def weight_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            #nn.init.xavier_normal(m.weight)
            nn.init.constant(m.bias, 0)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('Sigmoid') != -1:
            nn.init.xavier_normal(m.weight)
        #elif classname.find('Leaky') != -1:
            #nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        elif classname.find('Linear') != -1:
            nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            
   
    def set_weights(self):
        for i,m in enumerate(self.modules()):
            self.weight_init(m)
            
        
    def forward(self,x):
        connect_list = [] #it contains the layer from encoder path which need to skip to connect
        
        #encoder path
        for i in range(self.depth):
            block = self.encoder[i]
            x,connect_layer = block(x)
            connect_list.append(connect_layer)
            
        #bottom block: the middle and bottom part of UNet
        
        x = self.latent(x)
        
        #decoder path
        for i in range(self.depth):
            layer_to_connect = connect_list[-1-i]
            block = self.decoder[i]
            x = block(x,layer_to_connect)
            
        #last layer : 16 to 1
        x = self.last_block(x)
        
        
            
        return x
                
            


# # upload image and noise


def upload_image(file_name):
    """
    This function uploads and normalize the data by subtract mean and then devided by std
    input:
        file_name is a str, e.g. 'xcat.mat'
    returns:
        Pmean: the mean of temporal frames. Shape(1,1,256,256,400)
        blurry_P : each temporal frames has been added Gaussian noise with sigma = 5. Shape(1,1,256,256,400,14)
        Pgt :ground-truth / original image. Shape(1,1,256,256,400,14), which is not normalized.
    """
    P = loadmat(file_name)
    p = P['data']

    P1= torch.from_numpy(p)
    shape = P1.shape
    height = shape[0]
    width = shape[1]
    volume = shape[2]
    time = shape[3]
    
    noisyP = P1 # has 14 temporal frames
    Pgt = P1.unsqueeze(0)
    Pgt = Pgt.unsqueeze(0)# shape (1,1,256,256,400,14)
    
    #mean temporal frame
    P1 = torch.mean(P1,dim=3) 
    P = (P1-P1.mean())/(P1.std()) #normalization
    P = P.unsqueeze(0)
    Pmean = P.unsqueeze(0) #shape will be (1,1,256,256,400)

    #each temporal frame+gaussian noise
    g_noise = torch.normal(0,5,(height,width,volume,time)) #create a gaussian with mean 0, sigma 5
    noisyP=noisyP + g_noise  
    for i in range(time):
        blurry_P = (noisyP[:,:,:,i]-noisyP[:,:,:,i].mean())/(noisyP[:,:,:,i].std()) #normalize the blurry image
        noisyP[:,:,:,i] = blurry_P

    blurry_P = noisyP.unsqueeze(0)
    blurry_P = blurry_P.unsqueeze(0)# shape (1,1,256,256,400,14)
    
    
    return height,width,volume,time,Pmean,blurry_P,Pgt
    
    


# # save the output images
# 



def save_output_img(output_img,epoch,i,folder,train_loss,psnr,cc): # i is for telling which time (from 1 to 14); 
                                                                   #folder has three types because there are three kids of input_image
    fig = plt.figure()
    ax = fig.add_subplot(111)   
    plot = plt.imshow(output_img[0,0,:,:,36])
    #plt.clim(0,0.9*label_max)#have same color range
    plt.title('Time:{},Epoch: {}, Loss: {:.5f}, PSNR: {:.5f},CC:{:.5f}'.format(i+1,epoch+1, train_loss,psnr,cc))

    plt.savefig(folder+'output_image{}-{}.png'.format(i+1,epoch+1)) #e.g. output_image1-1000 means the 1000th image at time 1
    plt.close(fig)#not display the image


# # train function



def optim(optimizer:str = 'Adam'):
    if optimizer == 'Adam':
        return Adam(model.parameters(), lr=1e-3)
    elif optimizer == 'SGD':
        return torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    
def loss(criterion:str = 'MSE'):
    if criterion == 'MSE':
        return nn.MSELoss()
    
    
def train_setup(model,criterion_name,optimizer_name,input_image,label_image,epoch,i,folder):
    optimizer = optim(optimizer_name)
    criterion = loss(criterion_name)


    if torch.cuda.is_available():
        model = model.cuda()
        criterion = criterion.cuda()
        
    def train(epoch):
    #strat TRAIN mode
        model.train()
        train_loss = 0.0
        
        x_train, label= Variable(input_image), Variable(label_image)
    

        if torch.cuda.is_available():
        
            x_train = x_train.cuda()           
            label = label.cuda()
    
        optimizer.zero_grad()
        output = model(x_train)
        output_img = output.detach().cpu().numpy()
        
        loss = criterion(output,label)
    #compute gradient
        loss.backward()
    #update parameters
        torch.nn.utils.clip_grad_norm_(model.parameters(),1)
        optimizer.step()    
         
    
        train_loss = loss.item()
        
        #correlation coeeficient
        cm = np.corrcoef(output_img[0,0,:,:,:].flat,label_image[0,0,:,:,:].numpy().flat)#16 is because it's the middle slice from volume=53
        cc = cm[0,1]
        
        #psnr
        psnr = compare_psnr(label_image.numpy(), output_img,1)        
        
        loss_list.append(train_loss)
        cc_list.append(cc)
        psnr_list.append(psnr)
        

        
        print('Time:{} \tEpoch: {} \tTraining Loss: {:.6f} \tPSNR: {:.6f} \tCC:{:6f}'.format(i,epoch, train_loss,psnr,cc))
        
        
        if (epoch+1) %30 == 0:  #this only display the output of tevery 30 iteration
            
            save_output_img(output_img,epoch,i,folder,train_loss,psnr,cc)
            
    
    return train(epoch)
    
 


# #  training process



import sys

if len(sys.argv)>1:
    file_name=sys.argv[1]
folder = '../../SAN/inm/DWB-PET/raw_mat_files/'
file_name = folder+file_name

height,width,volume,time,Pmean,blurry_P,Pgt = upload_image(file_name)
Pmean53 = Pmean[:,:,:,:,250:303] #take a trunk of 53
blurry53 = blurry_P[:,:,:,:,250:303,:]
Pgt53 = Pgt[:,:,:,:,250:303,:]
uniform_noise = torch.randn(1,1,height,width,53)




criterion_name = 'MSE'
optimizer_name = 'Adam'

input_image2 = uniform_noise
input_image1 = Pmean53 #Pmean
input_image3 = blurry53


# # create a folder for all the output images

# In[27]:


folder2 = file_name[-7:-4]+'_noise/'
folder1 = file_name[-7:-4]+'_mean_frame/'
folder3 = file_name[-7:-4]+'_blurryImage/'


if not os.path.exists(folder1):
    os.makedirs(folder1)
if not os.path.exists(folder2):
    os.makedirs(folder2)
if not os.path.exists(folder3):
    os.makedirs(folder3)


# # iterate the training

# In[44]:


num_epochs = 1200


#when input is uniform noise

for i in range(time): #time = 14
    loss_list = []
    cc_list = []
    psnr_list = []
    model = UNet(stride_pooling = True,
                 chs = [1,16,32,64,128],
                 concatenate= False,
                 add = True,
                 Crop=True,
                 pooling = "max",                 
                 kernel_size = 3,
                 stride = 1,
                 padding = 1,
                 activation= 'leaky',
                 normalization = "BN",
                 dim= 3,
                 up_sample= 'nearest')

    label_image = Pgt53[:,:,:,:,:,i]
    
    fig_star =plt.figure()
    ax1 = fig_star.add_subplot(111)  
    plt.imshow(label_image[0,0,:,:,20],label='Ground truth image')
    plt.title('Ground truth image at time {}'.format(i+1))
    plt.savefig(folder1+'label-time:{}.png'.format(i+1))
    plt.close(fig_star)#not display the image
    
    for epoch in range(num_epochs):
        train_setup(model,criterion_name,optimizer_name,input_image1,label_image,epoch,i,folder1)
        
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)  
    plt.plot(loss_list,label='Training loss')
    plt.title('Time {}-Training Loss'.format(i+1))
    plt.savefig(folder1+'loss-{}.png'.format(i+1))
    plt.close(fig1)#not display the image
    
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)  
    plt.plot(psnr_list,label='PSNR')
    plt.title('Time {}-PSNR'.format(i+1))
    plt.savefig(folder1+'PSNR-{}.png'.format(i+1))
    plt.close(fig2)

    fig3 = plt.figure()
    ax3 = fig3.add_subplot(111)  
    plt.plot(cc_list,label='cc')
    plt.title('Time {}-Pearson correlation coefficient'.format(i+1))
    plt.savefig(folder1+'cc-{}.png'.format(i+1))
    plt.close(fig3)

#when input is the mean temporal frame

for i in range(time): #time = 14
    loss_list = []
    cc_list = []
    psnr_list = []
    model = UNet(stride_pooling = True,
                 chs = [1,16,32,64,128],
                 concatenate= False,
                 add = True,
                 Crop=True,
                 pooling = "max",                 
                 kernel_size = 3,
                 stride = 1,
                 padding = 1,
                 activation= 'leaky',
                 normalization = "BN",
                 dim= 3,
                 up_sample= 'nearest')

    label_image = Pgt53[:,:,:,:,:,i]
    
    fig_star =plt.figure()
    ax1 = fig_star.add_subplot(111)  
    plt.imshow(label_image[0,0,:,:,20],label='Ground truth image')
    plt.title('Ground truth image at time {}'.format(i+1))
    plt.savefig(folder2+'label-{}.png'.format(i+1))
    plt.close(fig_star)#not display the image
    
    for epoch in range(num_epochs):
        train_setup(model,criterion_name,optimizer_name,input_image2,label_image,epoch,i,folder2)
        
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)  
    plt.plot(loss_list,label='Training loss')
    plt.title('Time {}-Training Loss'.format(i+1))
    plt.savefig(folder2+'loss-{}.png'.format(i+1))
    plt.close(fig1)#not display the image
    
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)  
    plt.plot(psnr_list,label='PSNR')
    plt.title('Time {}-PSNR'.format(i+1))
    plt.savefig(folder2+'PSNR-{}.png'.format(i+1))
    plt.close(fig2)

    fig3 = plt.figure()
    ax3 = fig3.add_subplot(111)  
    plt.plot(cc_list,label='cc')
    plt.title('Time {}-Pearson correlation coefficient'.format(i+1))
    plt.savefig(folder3+'cc-{}.png'.format(i+1))
    plt.close(fig3)
    
    
#when input is the blurry image

for i in range(time): #time = 14
    loss_list = []
    cc_list = []
    psnr_list = []
    model = UNet(stride_pooling = True,
                 chs = [1,16,32,64,128],
                 concatenate= False,
                 add = True,
                 Crop=True,
                 pooling = "max",                 
                 kernel_size = 3,
                 stride = 1,
                 padding = 1,
                 activation= 'leaky',
                 normalization = "BN",
                 dim= 3,
                 up_sample= 'nearest')
    label_image = Pgt53[:,:,:,:,:,i]
    input_image = blurry53[:,:,:,:,:,i]
    
    fig_star =plt.figure()
    ax1 = fig_star.add_subplot(111)  
    plt.imshow(label_image[0,0,:,:,20],label='Ground truth image')
    plt.title('Ground truth image at time {}'.format(i+1))
    plt.savefig(folder3+'label-{}.png'.format(i+1))
    plt.close(fig_star)#not display the image
    
    for epoch in range(num_epochs):
        train_setup(model,criterion_name,optimizer_name,input_image,label_image,epoch,i,folder3)
        
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)  
    plt.plot(loss_list,label='Training loss')
    plt.title('Time {}-Training Loss'.format(i+1))
    plt.savefig(folder3+'loss-{}.png'.format(i+1))
    plt.close(fig1)#not display the image
    
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)  
    plt.plot(psnr_list,label='PSNR')
    plt.title('Time {}-PSNR'.format(i+1))
    plt.savefig(folder3+'PSNR-{}.png'.format(i+1))
    plt.close(fig2)

    fig3 = plt.figure()
    ax3 = fig3.add_subplot(111)  
    plt.plot(cc_list,label='cc')
    plt.title('Time {}-Pearson correlation coefficient'.format(i+1))
    plt.savefig(folder3+'cc-{}.png'.format(i+1))
    plt.close(fig3)
    

 