# basic libraries
import os
import sys
import shutil
import time
import numpy as np
import pandas as pd

# graphics libraries
import matplotlib.pyplot as plt
from ipywidgets import interact, interactive, fixed, interact_manual
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

#import from my scripts
main_dir = os.getcwd()
if main_dir not in sys.path:
    print('Adding the folder for the modules')
    sys.path.append(main_dir)
import importlib

# machine learning libraries
import torch
import torch.nn as nn

importlib.reload(importlib.import_module('neurons'))
importlib.reload(importlib.import_module('learning_rules'))
importlib.reload(importlib.import_module('plot_utils'))
from neurons import *
from learning_rules import *
from plot_utils import *


####################################
#                                  #
#       DEFAULT PARAMETERS         #
#                                  #
####################################



####################################
#                                  #
#           SIMULATION             #
#                                  #
####################################



def generate_traces(pars, pre_spike_train_ex):
    """
    track of pre-synaptic spikes

    Args:
        pars               : parameter dictionary
        pre_spike_train_ex : binary spike train input of shape (num_steps, N_pre)

    Returns:
        traces             : array or torch tensor of shape (num_steps, N_pre)
    """

    # Get parameters
    tau_plus =  pars['tau_plus']
    dt = pars['dt']
    num_steps = pre_spike_train_ex.shape[0]

    # check i f we are working with numpy or torch
    if type(pre_spike_train_ex) == np.ndarray:
        traces = np.zeros_like(pre_spike_train_ex)
        decay_rate = 1-dt/tau_plus
    else:
        traces = torch.zeros_like(pre_spike_train_ex)    
        decay_rate = pars.get('beta_plus', 0.9)

    # loop for each time step
    for step in range(num_steps-1):

        # exponential decay
        traces[step+1,:] = decay_rate*traces[step, :] 
        # increment for the arriving spikes
        traces[step+1,:] += pre_spike_train_ex[step, :]


    return traces


##########################################
#                                        #
#    FUNCTIONS FOR GENERATING INPUTS     #
#                                        #
##########################################


def Poisson_generator(dt, rate, N_pre, num_steps, myseed=False, herz_rate = False, batch_size = 0):
    """Generates poisson trains

    Args:
    pars            : parameter dictionary
    rate            : float or array of shape (n) of spiking rate as prob pre bin or [Hz], constant in time for each spike train 
    n               : number of Poisson trains 
    myseed          : random seed. int or boolean

    Returns:
    pre_spike_train : spike train matrix, ith row represents whether
                        there is a spike in ith spike train over time
                        (1 if spike, 0 otherwise)
    """

    # set random seed
    if myseed:
        np.random.seed(seed=myseed)
    else:
        np.random.seed()

    # generate random variables uniformly distributed in [0,1]
    if batch_size:
        u_rand = np.random.rand(num_steps, batch_size, N_pre)
    else:
        u_rand = np.random.rand(num_steps, N_pre)

    # check if the rate is in Hz or in probability per bin
    if herz_rate:
        rate_per_bin = rate * (dt / 1000.)
    else:
        rate_per_bin = rate 
    
    # generate Poisson train
    poisson_train = 1. * (u_rand < rate_per_bin)

    return poisson_train



def random_shifted_trains(dt, 
                          num_steps, 
                          N_pre, 
                          N_pre_correlated = None, 
                          shift_values = [-5,5], 
                          rate = 0.05, 
                          my_seed  = 42):
    """
    ARGS:
    - pars: dictionary with the parameters
    - num_steps: number of time steps
    - N_pre: number of pre-synaptic neurons
    - N_pre_correlated: number of neurons with correlated spike times to an original spike train
    - shift_values: range of values for the random shifts of spike times
    - rate: rate of the original spike train and for the uncorrelated neurons
    - my_seed: random seed
    RETURNS:
    - pre_spk_train: input spike train of shape (num_steps, N_pre), with N_pre_correlated consectuive 
                     neurons with correlated spike times to an original spike train
    """
    # set random seed
    if my_seed:
        np.random.seed(seed=my_seed)

    if N_pre_correlated is None:
        N_pre_correlated = N_pre//2
    
    # check that the number of correlated neurons is less than the total number of neurons
    if N_pre_correlated > N_pre:
        print('The number of correlated neurons must be less than the total number of neurons')
        print('N_pre_correlated is set to N_pre//2')
        N_pre_correlated = N_pre//2
    # check if the number of correlated neurons is given, otherwise set it to half of the total number of neurons
    elif N_pre_correlated == None:
        N_pre_correlated = N_pre//2
    
    # generate the original spike train
    original_train = Poisson_generator(dt, rate = rate, N_pre = 1, num_steps = num_steps)

    # find the spike times of the original train
    spike_times = np.array(np.where(original_train==1))

    # generate the perturbed spike trains as a matrix of zeros
    perturbed_spike_trains = np.zeros((num_steps, N_pre_correlated))

    # perturb spike times
    for spk_t in spike_times[0]:
        random_shifts = np.random.randint(low = shift_values[0], high = shift_values[1], size=N_pre)
        within_the_range = lambda x: x > 0 and x < num_steps
        random_shifted_times = [spk_t + random_shifts[i] for i in range(N_pre_correlated) if within_the_range(spk_t + random_shifts[i])]
        for time, index in zip(random_shifted_times, range(N_pre_correlated)):
            perturbed_spike_trains[time, index] = 1

    pre_spk_train = Poisson_generator(dt, rate = rate, N_pre = N_pre-N_pre_correlated, num_steps = num_steps)
    pre_spk_train = np.concatenate(( perturbed_spike_trains, pre_spk_train), axis=1)
    return pre_spk_train, original_train



def random_offsets(dt, num_steps, N_pre, rate = 0.5, sections = 10, sort_shifts = True, length_ratio = 0.5):

    # check the length_ratio
    if length_ratio > 1:
        print('Length ratio must be smaller than 1')
        print('Half size is used')
        length_ratio = 0.5

    # generate the random 
    original_length = int(num_steps*length_ratio)
    I = Poisson_generator(dt, rate = rate, N_pre = N_pre, num_steps = original_length)

    # generate random offsets
    shift_values = np.random.randint(0,sections,N_pre)*(num_steps-original_length)//(sections-1)

    # sort the sections
    if sort_shifts:
        shift_values = np.sort(shift_values)

    I_original = np.concatenate([I, np.zeros((num_steps-original_length,N_pre))], axis=0)
    I_shifted = np.zeros((num_steps,N_pre))
    for i in range(N_pre):
        I_shifted[:,i] = np.roll(I_original[:,i], shift_values[i])

    first_section_index = np.where(shift_values == np.min(shift_values))[0]

    return I_shifted, I_original, first_section_index

