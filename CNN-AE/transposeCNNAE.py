#!/usr/bin/env python
# coding: utf-8

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

# %matplotlib inline


# In[2]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# In[3]:


from os import walk
for (dirpath, dirnames, filenames) in walk("./sars-covid-data"):
    print("Directory path: ", dirpath)
    print("Folder name: ", dirnames)


# In[4]:


# transform
transformer = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.Grayscale(),#make greyscale one channel
    transforms.ToTensor(), #0-1, numpy to tensor
    transforms.Normalize([0.5],
                        [0.5]) # 0-1 to -1,1 and normalized
])


# In[5]:


train_path ='./sars-covid-data'
dataset = datasets.ImageFolder(train_path, transform=transformer)


train, test = torch.utils.data.random_split(dataset, [2000,481])


train_loader = DataLoader(train,batch_size =32, shuffle = True)


# In[8]:


test_loader = DataLoader(test, batch_size =32, shuffle = True)


# from torchvision.utils import make_grid
# from torchvision.utils import save_image
# from IPython.display import Image

# save images not show images (torchvision)
# or matplot (instead of shoing images, just save it)

# def show_images(images, nmax=64):
#     fig, ax = plt.subplots(figsize=(8, 8))
#     ax.set_xticks([]); ax.set_yticks([])
#     ax.imshow(make_grid((images.detach()[:nmax]), nrow=8).permute(1, 2, 0))
# def show_batch(dl, nmax=64):
#     for images in dl:
#         show_images(images, nmax)
#         break

# In[28]:

# In[9]:


train_images, train_labels = next(iter(train_loader))
test_images, test_labels = next(iter(test_loader))


# def show_img(img):
#     img = img/2 + 0.5 #[-1,1] -> [0,1]
#     plt.imshow(img.permute(1,2,0))
#     # because the input of plt.imshow is (imagesize,imagesize,channels), the format of img is (channels,imagesize,imagesize)

# In[15]:


classes = ['covid','non-covid']

# for up and down, probably need to remove the linear layer (unet)
# 1. input = output
# 2. first want the middle part (very small)
# 3.from the small part see if covid or not

# class CNNnet(nn.Module):
#     
#     def __init__(self):
#         
#         super().__init__()
#         self.network = nn.Sequential(
# 
#             nn.Conv2d(1, 16, kernel_size = 3, stride=1, padding = 1),
#             nn.LeakyReLU(0.1),
#             nn.MaxPool2d(2,2),
#         
#             nn.Conv2d(16, 32, kernel_size = 3, stride = 1, padding = 1),
#             nn.LeakyReLU(0.1),           
#             nn.MaxPool2d(2,2), #(32*32*32)
# 
#             
#             nn.Flatten(),
#             nn.Linear(32768,128),#in features= , out features = 128
#             nn.ReLU(),
#             nn.Linear(128, 2)
#         )
#     
#     def forward(self, xb):
#         return self.network(xb)



class CnnAE(nn.Module):
    def __init__(self):
        super(CnnAE,self).__init__()
        
        #encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size = 3, stride=1, padding = 1),
        #size 128
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2,2),
            nn.Conv2d(16, 16, kernel_size = 3, stride = 1, padding = 1),
            #nn.Conv2d(16, 32, kernel_size = 3, stride = 1, padding = 1),
            
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2,2),
            
            nn.Conv2d(16, 32, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1)
            
            #size 32
        )

        
        #decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size = 3, stride=1, padding = 1),
            
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1),
            nn.ConvTranspose2d(in_channels=32,out_channels=16,
                           kernel_size=5,stride=2,output_padding=1,padding=2),
            
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.1),
            nn.Conv2d(16, 16, kernel_size = 3, stride=1, padding = 1),
        #size 128
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.1),
            #nn.ConvTranspose2d(in_channels=16, out_channels=16,
                           #kernel_size=5,stride=2,output_padding=1,padding=2),
            
            nn.ConvTranspose2d(in_channels=16, out_channels=1,
                           kernel_size=5,stride=2,output_padding=1,padding=2),
            
            nn.BatchNorm2d(1),
            nn.LeakyReLU(0.1),
            #nn.Conv2d(16, 16, kernel_size = 3, stride = 1, padding = 1),
           # nn.BatchNorm2d(16),
            #nn.LeakyReLU(0.1),
            #nn.Conv2d(16, 1, kernel_size = 3, stride = 1, padding = 1)
            
            
            #nn.Sigmoid()
        )
        
    def forward(self,x):
        x = self.encoder(x)
        x=self.decoder(x)
        return x



model = CnnAE().to(device)
# create an optimizer object
# Adam optimizer with learning rate 1e-3
optimizer = Adam(model.parameters(), lr=1e-3)
# mean-squared error loss
criterion = nn.MSELoss()
#criterion = nn.CrossEntropyLoss()
#criterion = nn.BCELoss()
if torch.cuda.is_available():
    model = model.cuda()
    criterion = criterion.cuda()

def train(epoch):
    model.train()
    train_accuracy = 0.0
    train_loss = 0.0
    
    total_step = len(train_loader)


    for i, (images,labels) in enumerate (train_loader):
        #images = torch.squeeze(images)
        if torch.cuda.is_available():
            #images = Variable(images.cuda())            
            images = images.to(device)
            #labels = Variable(labels.cuda())
            
        optimizer.zero_grad() 
    
        outputs = model(images) #give us prediction
        #print(outputs.shape)
        #outputs = torch.argmax(outputs, dim=1)
        loss = criterion(outputs,images)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.cpu().data*images.size(0)
        
    print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch, train_loss))


num_epochs = 10
train_loss = 0
for epoch in range(num_epochs):
    train(epoch)
    

output = model(train_images)



#Reconstructed Images

output = model(train_images)
output = output.detach().numpy()
print('Reconstructed Images')
fig, axes = plt.subplots(nrows=1, ncols=5, sharex=True, sharey=True, figsize=(12,4))
for idx in np.arange(5):
    ax = fig.add_subplot(1, 5, idx+1, xticks=[], yticks=[])
    plt.imshow(output[idx][0])
    ax.set_title(classes[train_labels[idx]])
# plt.show() 


# In[47]:


fig.savefig('transposeCnnAE.png')


# In[ ]:




