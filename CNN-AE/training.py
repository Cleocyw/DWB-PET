#!/usr/bin/env python
# coding: utf-8

# In[1]:


#load library
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

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# In[3]:


print(os.listdir("./sars-covid-data"))


# In[3]:


from os import walk
for (dirpath, dirnames, filenames) in walk("./sars-covid-data"):
    print("Directory path: ", dirpath)
    print("Folder name: ", dirnames)


# In[19]:


# transform
transformer = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.Grayscale(),#make greyscale one channel
    transforms.ToTensor(), #0-1, numpy to tensor
    transforms.Normalize([0.5],
                        [0.5]) # 0-1 to -1,1 and normalized
])


# In[20]:


#Dataloader
train_path ='./sars-covid-data'
dataset = datasets.ImageFolder(train_path, transform=transformer)
# either make here one channel or remove two channels later 


# In[21]:


dataset[0][0]


# In[22]:


dataset_subset = torch.utils.data.Subset(dataset, np.random.choice(len(dataset), 2, replace=False))


# In[ ]:


#train_loader = DataLoader(dataset, batch_size =128, shuffle = True)


# In[23]:


train_loader = DataLoader(dataset_subset, shuffle = True)


# In[24]:


images, labels = next(iter(train_loader))


# In[25]:


#print(labels)
#images.size()
dataset_subset[0][0]


# In[21]:


#autoencoder




# The torch.nn.Linear layer creates a linear function (θx + b), with its parameters initialized (by default). This means we will call an activation/non-linearity for such layers.
# 
# The in_features parameter dictates the feature size of the input tensor to a particular layer, e.g. in self.encoder_hidden_layer, it accepts an input tensor with the size of [N, input_shape] where N is the number of examples, and input_shape is the number of features in one example.
# 
# The out_features parameter dictates the feature size of the output tensor of a particular layer. Hence, in the self.decoder_output_layer, the feature size is kwargs["input_shape"], denoting that it reconstructs the original data input.
# 
# The forward() function defines the forward pass for a model. This is the function invoked when we pass input tensors to an instantiated object of a torch.nn.Module class.

# In[ ]:

# ## CNN

# In[26]:


class CNNnet(nn.Module):
    
    def __init__(self):
        
        super().__init__()
        self.network = nn.Sequential(

            nn.Conv2d(1, 16, kernel_size = 3, stride=1, padding = 1),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2,2),
        
            nn.Conv2d(16, 32, kernel_size = 3, stride = 1, padding = 1),
            nn.LeakyReLU(0.1),           
            nn.MaxPool2d(2,2), #(32*32*32)

            
            nn.Flatten(),
            nn.Linear(32768,128),#in features= , out features = 128
            nn.ReLU(),
            nn.Linear(128, 2)
        )
    
    def forward(self, xb):
        return self.network(xb)


# In[27]:


model = CNNnet().to(device)


# create an optimizer object
# Adam optimizer with learning rate 1e-3
optimizer = Adam(model.parameters(), lr=1e-3)

# mean-squared error loss
criterion = nn.MSELoss()


# In[34]:


total_step = len(train_loader)
loss_list = []
acc_list = []
#num_epochs = 
#for epoch in range(num_epochs):

for i, (images,labels) in enumerate (train_loader):
    outputs = model(images)
    loss = criterion(outputs,labels.float())
    loss_list.append(loss.item())
    
    #backword proprgation, adam optim
    #empty of grad 
    optimizer.zero_grad()
    loss.backward()
    #update grad
    optimizer.step()
    
    #accuracy
    total = labels.size(0)
    _,predicted = torch.max(outputs.data,1)
    correct = (predicted == labels).sum().item()
    acc_list.append(correct/total)


# In[36]:


print(acc_list)


# In[ ]:




