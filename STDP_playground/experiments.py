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



def default_pars(type_parameters = 'simple', **kwargs):
    '''
    Define a dictionary with the default parameters for the nuerons, the weight update rule and the simulation overall
    ARGS:
    - type_parameters (string): 'simple' or 'realistic' for the simplified or the more biologically plausible neuron model
    - kwargs: dictionary with the parameters to be updated or added
    RETURNS:
    - pars (dictionary): dictionary with the parameters
    '''
    if type_parameters not in ['simple', 'realistic']:
        print('The type of parameters must be simple or realistic')
        print('Using simple parameters')
        type_parameters = 'simple'

    s = type_parameters == 'simple'

    pars = {}

    # typical neuron parameters
    pars['threshold'] = 1 if s else -55.     # spike threshold [mV] !do not change
    pars['tau_m'] = 10.                      # membrane time constant [ms]
    pars['R'] = 1 if s else 0.1              # leak resistance [Ohm] with this resistance input current must be of the order of 100 mV !do not change
    pars['U_init'] = 0. if s else -65.       # initial potential [mV]  !do not change
    pars['U_reset'] = 0. if s else -75.      # reset potential [mV]    !do not change
    pars['U_resting'] = 0. if s else -75.    # leak reversal potential [mV]    !do not change  in Neuromatch course this is V_L
    pars['t_ref'] = 0. if s else 2.          # refractory time (ms)

    # neuron parameters for conductance based model
    pars['tau_syn_exc'] = 10. if s else 5.   # synaptic time constant [ms]
    pars['tau_syn_inh'] = 10. if s else 5.   # synaptic time constant [ms]
    pars['U_rev_exc'] = 1. if s else 0.      # excitatory reversal potential [mV]
    pars['U_rev_inh'] = -1. if s else -80.   # inhibitory reversal potential [mV]
    pars['max_g'] = 1. if s else 0.024       # maximum synaptic conductance [nS]

    # in the case of dynamic threshold
    pars['tau_thr'] = 20                     # threshold time constant [ms]
    pars['ratio_thr'] = 1.5 if s else 1.1    # relative increment in the threshold due to a spike

    # in case of noisy input
    pars['sigma'] = 1                        # standard deviation of the noise

    # random seed
    pars['my_seed'] = 42                     # seed for random number generation

    # time steps
    pars['dt'] = 1. if s else 0.1            # simulation time step [ms]  !do not change
                      

    # STDP parameters
    pars['A_plus'] = 0.02 if s else 0.008 * pars['max_g']    # magnitude of LTP
    pars['A_minus'] = 0.02 if s else 0.0088 * pars['max_g']  # magnitude of LTD 
    pars['tau_plus'] = 20                    # LTP time constant [ms]
    pars['tau_minus'] = pars['tau_plus']     # LTD time constant [ms]
    pars['tau_syn'] = 5.                     # synaptic time constant [ms] if synaptic decay is used
    pars['dynamic_weight_exponent'] = 0.01   # exponent for the dynamic weight constraint

    # weight parameters
    pars['w_max'] = 1. if s else 0.024       # maximum weight
    pars['w_min'] = 0.                       # minimum weight
    pars['w_init_value'] = 0.5 if s else 0.012 # initial value for the weights

    # neuron initialization parameters
    pars['refractory_time'] = False if s else True
    pars['dynamic_threshold'] = False
    pars['hard_reset'] = True

    # weights update rule parameters
    pars['constrain'] = 'Hard'               # weight constrain
    pars['short_memory_trace'] = False       # short memory trace for the traces

    # external parameters if any #
    for k in kwargs:
        pars[k] = kwargs[k]

    return pars



####################################
#                                  #
#           SIMULATION             #
#                                  #
####################################



def weight_initializer(pars, N_post, N_pre = None, I=None, type_init = 1, my_seed = 2024,  tensor = False):

    """
    ARGS:
    - I: input spike train with shape (num_steps, N_pre)
    - N_post: number of post-synaptic neurons
    - my_seed: seed for random number generation
    RETURNS:
    - W: initial weights with shape (N_pre, N_post)
    """

    if N_pre is None and I is not None:
        N_pre = np.shape(I)[1]
    elif N_pre is None and I is None:
        print('The number of pre-synaptic neurons must be given')
        return None
    elif N_pre is not None and I is not None:
        if N_pre != np.shape(I)[1]:
            print('The number of pre-synaptic neurons must be the same as the number of columns of the input spike train')
            return None
    



    # set random seed
    if my_seed:
        np.random.seed(seed=my_seed)

    if type_init == 1:
        # random uniform between w_min and w_max
        W = np.random.rand(N_post, N_pre) * (pars['w_max'] - pars['w_min']) + pars['w_min']
        

    elif type_init == 2:
        W = np.random.rand(N_post, N_pre) * 1/N_pre
        W = np.clip(W,  pars['w_min'], pars['w_max'])
        

    elif type_init == 3:
        # normalize the rows
        W = np.random.rand(N_post, N_pre) * (pars['w_max'] - pars['w_min']) + pars['w_min']
        W = W / W.sum(axis=1).reshape(-1,1)


    elif type_init == 4:
        # compute the mean fire rate for each pre-synaptic neuron
        mean_rate = np.mean(I, axis = 0)+1e-5
        
        # define the mean vector for the weights as inverse of the mean rate
        mean_vector = 1./mean_rate


        # initialize the weights as uniform random variables between 0 and 2 * mean_vector
        W = np.random.uniform(mean_vector, 1/N_pre , (N_post, N_pre))

        # clip _between 0 and 1
        W = np.clip(W, pars['w_min'], pars['w_max'])


    return W if not tensor else torch.from_numpy(W.T).float()




def simulation(
        pars,
        spk_input,
        W_init,                 
        neuron_type = LIFNeuron,  
        N_post = 1,                
        weight_rule = None, 
        my_seed = 2024,            
        ):
    """
    ARGS:
    - pars: dictionary with simulation parameters
    - spk_input: input spike train, a numpy vector of shape (time_steps, N_pre)
    - neuron_type: class for the type of neuron
    - N_post: number of post-synaptic neurons
    - weight_rule: class for the weight update
    - W_init: initial weights
    - my_seed: seed for random number generation
    - neuron_params: parameters for the neuron model
    - weight_update_params: parameters for the weight update
    RETURNS:
    - my_post_neuron: list of post-synaptic neurons
    - my_synapses: synapses object trained with the given rule   

    """
    
    num_steps = np.shape(spk_input)[0]
    N_pre = np.shape(spk_input)[1]

    # Initialize the post synaptic neurons
    my_post_neuron = [ neuron_type(pars) for i in range(N_post)]
    

    # Initialize the weights
    W = W_init


    # Initialize the synapses with the given update rule
    if weight_rule is not None:
        my_synapses = weight_rule(pars, N_pre, N_post, W_init = W )


    # start the simulation
    for t in range(num_steps):
        pre_spk = spk_input[t,:]
        I_inj = W @ pre_spk

        # colpo da maestro
        post_spk = [ my_post_neuron[i].forward(I_inj[i])[1] for i in range(N_post)]

        # update the weights
        if weight_rule is not None:
            my_synapses.update_weights([pre_spk, post_spk])
            W = my_synapses.W
        else:
            W = W
    if weight_rule is not None:
        return my_post_neuron, my_synapses
    else:
        return my_post_neuron




##########################################
#                                        #
#    FUNCTIONS FOR GENERATING INPUTS     #
#                                        #
##########################################


def repeat_ones(num_steps, N_pre, silent_time ):
    """
    ARGS:
    - num_steps: number of time steps
    - N_pre: number of pre-synaptic neurons
    - silent_time: time steps for the silent period
    RETURNS:
    - I: input spike train with all ones for the first silent_time and zeros for the remaining time
    """
    I = np.zeros((num_steps, N_pre))
    I[::silent_time, :] = 1
    return I



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



def half_growing_rate(dt, num_steps, N_pre, rate_ratio = 0.5):
    """
    ARGS:
    - pars: dictionary with the parameters
    - num_steps: number of time steps
    - N_pre: number of pre-synaptic neurons
    - rate_ratio: ratio between the first and the second half of the simulation
    RETURNS:
    - I: input spike train with rate growing linearly for 0 to 1 in the presynaptic neurons for the first half of the simulation and from 1 to rate_ratio in the second half
    """
    rate = np.arange(0,N_pre,1)/N_pre
    first_half = Poisson_generator(dt, rate = rate, N_pre = N_pre, num_steps = num_steps//2)
    second_half = Poisson_generator(dt, rate = rate * rate_ratio, N_pre = N_pre, num_steps = num_steps//2)
    I = np.concatenate([first_half, second_half ], axis=0) 
    return I



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



def signals_with_patterns(  
    dt, 
    N_pre ,
    num_steps,
    rate = 0.5,
    ratio_with_pattern = 0.5,
    repeats = 1,
    length_pattern = None,
    display_pattern = False,
    sort_neurons = True,
 ):
    # this is the wrong way to do it
    """
    ARGS:
    pars : (dict) parameters for the simulation
    N_pre : (int) number of pre-synaptic neurons
    num_steps : (int) number of time steps
    N_post : (int) number of post-synaptic neurons
    rate : (float) rate of the Poisson spike train (both for the pattern and the remaining signals)
    ratio_with_pattern : (float) ratio of the pre-synaptic neurons that will have the pattern
    repeats : (int) number of repeats for the pattern in each neuron
    length_pattern : (int) length of the pattern in time steps

    RETURNS:

    I_original : (np.array) original spike signals with shape (num_steps, N_pre)
    I_perturbed : (np.array) perturbed spike signals with shape (num_steps, N_pre) with the pattern inserted
    correlated_indexes : (np.array) indexes of the neurons presenting the pattern 
    perturbation_sites : (np.array) matrix with shape (num_steps, N_pre) with 1s where the pattern was inserted
    """
    
    # compute the number of correlated neurons
    num_correlated = int(N_pre*ratio_with_pattern)

    # se the length of the pattern
    if length_pattern is None:
        length_pattern = int(num_steps/(10*repeats))

    # generate a specific pattern
    pattern = Poisson_generator(dt, rate = rate, N_pre = 1, num_steps = length_pattern)

    # generate the original injected spike signals with shape (num_steps, N_pre)
    I_original = Poisson_generator(dt, rate = rate, N_pre = N_pre, num_steps = num_steps)

    # start times for the specific pattern in the simulation of the neurons with shape (repeats, N_pre)
    start_times = np.random.randint(0,num_steps-length_pattern, (repeats,N_pre))

    # chose the indexes for the correlated neurons 
    if sort_neurons:
        correlated_indexes = np.arange(num_correlated)
    else:
        correlated_indexes = np.random.choice(N_pre, num_correlated, replace = False)

    # make a copy of the original signal
    I_perturbed = I_original.copy()

    # initialize a matrix to keep track of the sites where the pattern is inserted
    perturbation_sites = np.zeros((num_steps, N_pre))

    # insert the pattern in the signal of the correlated neurons at the the given start times
    for neuron_index in correlated_indexes:
        for time_step in start_times[:, neuron_index]:
            I_perturbed[time_step:time_step+length_pattern, neuron_index] = pattern[:,0]
            perturbation_sites[time_step:time_step+length_pattern, neuron_index] = 1

    if display_pattern:
        display(pattern.T)

    return I_original,  I_perturbed, correlated_indexes, perturbation_sites