# basic libraries
import os
import sys
import time
import numpy as np
import pandas as pd

# graphics libraries
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import ipywidgets as widgets
import jupyterlab_widgets as lab
from IPython.display import display
import logging
logging.getLogger('matplotlib.font_manager').disabled = True
# use NMA plot style
#plt.style.use("https://raw.githubusercontent.com/NeuromatchAcademy/course-content/main/nma.mplstyle")
plt.style.use('seaborn-v0_8')
my_layout = widgets.Layout()
my_layout.width = '620px'

#machine Learning libraries
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import snntorch as snn
import snntorch.spikeplot as splt
from snntorch import utils
from snntorch import spikegen


#import from my scripts
main_dir = os.getcwd()
if main_dir not in sys.path:
    print('Adding the folder for the modules')
    sys.path.append(main_dir)
import importlib

importlib.reload(importlib.import_module('snn_experiments'))
importlib.reload(importlib.import_module('snn_plot_utils'))
importlib.reload(importlib.import_module('snn_models'))

from snn_experiments import *
from snn_plot_utils import *
from snn_models import *





# Create a costum class dataset with determinsitic spike rate encoding

class rate_encoded_mnist(torch.utils.data.Dataset):

    def __init__(self, batch_size, num_steps=100, gain=1, train=True, data_path=r'.\data', my_seed=42):

        self.batch_size = batch_size
        self.num_steps = num_steps
        self.gain = gain
        # define the transformation
        transform = transforms.Compose([
                    transforms.Resize((28,28)),
                    transforms.Grayscale(),
                    transforms.ToTensor(),
                    transforms.Normalize((0,), (1,)),
                    transforms.Lambda(lambda x: torch.squeeze(x)),
        ])
        my_generator = torch.Generator(device='cpu')
        my_generator.manual_seed(my_seed)
        self.my_generator = my_generator
        self.train = train
        self.dataset = datasets.MNIST(data_path, train=train, download=True, transform=transform)
        # divide the data from the targets
        self.data = self.dataset.data
        self.targets = self.dataset.targets

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        data_sample, target = self.dataset[idx]
        data_sample = self.rate_encode(data_sample, num_steps=self.num_steps, gain = self.gain)
        return data_sample, target
    
    def rate_encode(self, data_sample, num_steps, gain):
        data_sample = data_sample.view(-1)
        data_sample = data_sample*gain
        data_sample = torch.clamp(data_sample, 0, 1)
        data_sample = data_sample.repeat(num_steps,1)
        data_sample = torch.bernoulli(data_sample, generator=self.my_generator)
        return data_sample
    
    def visualize_rate_image(self, idx):
        data_sample, target = self.dataset[idx]
        data_sample = self.rate_encode(data_sample, num_steps=self.num_steps, gain = self.gain)
        data_sample = data_sample.view(self.num_steps, 28, 28)  # Reshape the data to match the desired shape
    
        # little animation to visualize an example of the spike data
        fig, ax = plt.subplots()
        # plot the animation
        anim = splt.animator(data_sample, fig, ax)
        print(f"The corresponding target is: {target}")

        # If you're feeling sentimental, you can save the animation: .gif, .mp4 etc.
        anim.save(f"Spike_rate_encoding_of_an_{target}.mp4")
        return
    
    def custom_collate_fn(self, batch):
        #C REATE THE BATCH ALONG THE RIGHT DIMENSION
        # batch is a list of (data, target) tuples
        # data: num_steps x 28 x 28 tensor
        # target: scalar tensor
        
        # Stack data tensors along a new dimension (axis=0)
        data_batch = torch.stack([item[0] for item in batch], dim=1)
        
        # Stack target tensors along a new dimension (axis=0)
        target_batch = torch.stack([torch.tensor(item[1]) for item in batch], dim=0)
        
        return data_batch, target_batch

    def get_loader(self):
        if self.train:
            # Split the train dataset into train and validation sets
            train_size = 50000
            val_size = 10000
            indices = torch.randperm(len(self.dataset), generator=self.my_generator).tolist()
            train_indices = indices[:train_size]
            val_indices = indices[train_size:train_size+val_size]
            train_sampler = torch.utils.data.SubsetRandomSampler(train_indices, generator=self.my_generator)
            val_sampler = torch.utils.data.SubsetRandomSampler(val_indices, generator=self.my_generator)
            train_loader = torch.utils.data.DataLoader(dataset=self, batch_size = self.batch_size, sampler=train_sampler, collate_fn=self.custom_collate_fn)
            val_loader = torch.utils.data.DataLoader(dataset=self, batch_size = self.batch_size, sampler=val_sampler, collate_fn=self.custom_collate_fn)
            return train_loader, val_loader
        elif not self.train:
            test_loader = torch.utils.data.DataLoader(dataset=self, batch_size = self.batch_size, shuffle=False, collate_fn=self.custom_collate_fn)
            return test_loader
        
    def get_subset(self, subset=100):
        mnist_subset = utils.data_subset(self, subset)
        subset_loader = DataLoader(mnist_subset, batch_size=self.batch_size, shuffle=True, collate_fn=self.custom_collate_fn)
        return subset_loader
        
        