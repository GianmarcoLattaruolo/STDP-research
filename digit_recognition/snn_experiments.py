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




####################################
#                                  #
#       DEFAULT PARAMETERS         #
#                                  #
####################################


def mnist_pars(**kwargs):  #DA SISTEMARE
    '''
    Define a dictionary with the default parameters for the nuerons, the weight update rule and the simulation overall
    ARGS:
    - kwargs: dictionary with the parameters to be updated or added
    RETURNS:
    - pars (dictionary): dictionary with the parameters
    '''

    pars = {}
    
    # simulation parameters
    pars['dt'] = 1.                                   # (FIXED) simulation time step [ms]  
    pars['use_min_spk_number'] = False                # (FIXED) use the minimum number of spikes for a forward pass of a batch
    pars['min_spk_number'] = 5                        # (FIXED) minimum number of spikes for a forward pass of a batch
    pars['store_records'] = True                      # (FIXED) store the records of the simulation
    pars['store_subsampling_factor'] = 10             # (FIXED) subsampling factor in time steps for storing the records 
    pars['weight_initialization_type'] = 'clamp'      # type of weight initialization (still in development)
    pars['assignment_confidence'] = 0.01              # < of the difference between the max and the second max for the assignment of the label

    # typical neuron parameters
    pars['threshold'] = 1.0                           # (FIXED) spike threshold for the LIF neuron
    pars['alpha'] = 0.95                              # (FIXED) decaying factore for the neuron conductance
    pars['beta'] = 0.8                                # (FIXED) decaying factor for the neuron potential
    pars['reset_mechanism'] = 'subtract'              # (FIXED) reset mechanism for the neuron potential
    # parameters for dynamic threshold   
    pars['dynamic_threshold'] = False                 # (FIXED) use a dynamic threshold for the neuron
    pars['tau_theta'] = 20                            # (FIXED) time constant for the dynamic threshold
    pars['theta_add'] = 0.05                          # increment for the dynamic threshold in case of a spike
    # parameters for lateral inhibition
    pars['lateral_inhibition'] = False                # (FIXED) use lateral inhibition
    pars['lateral_inhibition_strength'] = 0.1         # strength of the lateral inhibition 
    # parameters for refractory period 
    pars['refractory_period'] = False                 # (FIXED) use refractory period
    pars['ref_time'] = 2.                             # refractory time in time steps

    # STDP parameters
    pars['STDP_type'] = None                          # type of STDP rule
    pars['beta_minus'] = 0.9                          # decay factor for the pre-synaptic trace
    pars['beta_plus'] = 0.9                           # decay factor for the post-synaptic trace
    pars['w_min'] = 0.0                               # (FIXED) minimum weight
    pars['w_max'] = 1.0                               # (FIXED) maximum weight
    pars['A_minus'] = 0.0002                          # magnitude of LTD if STDP_type is 'classic' or 'asymptotic
    pars['A_plus'] = 0.0001                           # magnitude of LTP if STDP_type is 'classic' or 'asymptotic
    pars['STDP_offset'] = 0.005                       # offset for the STDP rule if STDP_type is 'offset'
    pars['mu_exponent'] = 2                           # (FIXED) exponent for the dynamic weight constraint if STDP_type is 'offset'
    pars['learning_rate'] = 0.0001                    # learning rate for the weights if STDP_type is 'offset  

    # external parameters if any #
    for k in kwargs:
        pars[k] = kwargs[k]

    return pars
    

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



def weight_initializer(pars, N_post, N_pre = None, I=None, my_seed = 2024, type_init = 3, tensor = False):

    """
    ARGS:
    - I: input spike train with shape (num_steps, N_pre)
    - N_post: number of post-synaptic neurons
    - my_seed: seed for random number generation
    RETURNS:
    - W: initial weights with shape (N_pre, N_post)
    """
    if I is not None:
        type_init = 3

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
        # compute the mean fire rate for each pre-synaptic neuron
        mean_rate = np.mean(I, axis = 0)+1e-4
        
        # define the mean vector for the weights as inverse of the mean rate
        mean_vector = 1.5/mean_rate

        # divide by the pre-synaptic number of neurons
        mean_vector = mean_vector/N_pre

        print(f'mean vector max:{mean_vector.max()}')

        # initialize the weights as uniform random variables between 0 and 2 * mean_vector
        W = np.random.uniform(0, 2*mean_vector, (N_post, N_pre))

        # clip _between 0 and 1
        W = np.clip(W, 0, 1)

    elif type_init == 2:
        mean_rate = np.mean(I)+1e-4
        mean_value = 1.5/(mean_rate*N_pre)
        W = np.random.uniform(0, 2*mean_value, (N_post, N_pre))
        W = np.clip(W, 0, 1)

    elif type_init == 3:
        W = np.ones((N_post, N_pre)) * pars['w_init_value']

    return W if not tensor else torch.from_numpy(W.T).float()



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
    
    dt = pars['dt']
    num_steps = pre_spike_train_ex.shape[0]

    # check i f we are working with numpy or torch
    if type(pre_spike_train_ex) == np.ndarray:
        tau_plus =  pars['tau_plus']
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



class STDP_synapse:

    def __init__(self, pars, N_pre, N_post, W_init = None, **kwargs):
        
        
        # base attributes
        self.pars = pars
        self.dt = self.pars['dt']
        self.w_max = self.pars['w_max']
        self.w_min = self.pars['w_min']
        self.A_plus = self.pars['A_plus']
        self.A_minus = self.pars['A_minus']
        self.tau_plus = self.pars['tau_plus']
        self.tau_minus = self.pars['tau_minus']

        # user defined attributes
        self.N_pre = N_pre
        self.N_post = N_post
        constrain = self.pars.get('constrain', 'Hard')
        if constrain not in ['None', 'Hard', 'Dynamic']:
            print('constrain must be either None, Hard or Dynamic')
            print('constrain set to None')
            constrain = 'None'
            return
        self.constrain = constrain
        self.short_memory_trace = self.pars.get('short_memory_trace', False)

        # additional attributes
        for key, value in kwargs.items():
            setattr(self, key, value)        

        # Initialize weights and traces
        if W_init is not None:
            self.W = W_init
        else:
            # Set random seed
            np.random.seed(42)
            self.W = np.random.random(N_post, N_pre)
        self.traces = [np.zeros(N_pre), np.zeros(N_post)]

        # tracking attributes
        self.W_records = []
        self.W_records.append(self.W)
        self.pre_traces_records = []
        self.post_traces_records = []

    def update_weights(self, spikes):
        
        # unpack the variables
        pre_spk_vector , post_spk_vector = spikes
        pre_trace, post_trace = self.traces

        # update the traces:
        # expoenetial decay towards zero
        pre_trace = (1-self.dt/self.tau_plus)*pre_trace
        post_trace = (1-self.dt/self.tau_minus)*post_trace
        # contribution of spikes 
        # check what type pf traces we are using
        if self.short_memory_trace:
            # reset the traces to the rest value if the neuron spiked
            pre_trace += (self.pars.get('pre_trace_rest',1) - pre_trace ) * pre_spk_vector
            post_trace += (self.pars.get('post_trace_rest',1) - post_trace ) * post_spk_vector
        else:
            # add the spike contribution to the traces
            pre_trace += pre_spk_vector
            post_trace += post_spk_vector

        #store the results
        self.traces = [pre_trace, post_trace]


        # update the weights
        LTP = self.A_plus *  np.outer(post_spk_vector,pre_trace)
        LTD = self.A_minus *  np.outer(post_trace,pre_spk_vector)
        self.W = self.W + LTP - LTD 

        # weigths constrain
        if self.constrain == 'None':
            self.W = self.W
        elif self.constrain == 'Hard':
            self.W = np.clip(self.W, self.w_min, self.w_max)
        elif self.constrain == 'Dynamic':
            # dynamic weight constraint based on the current weight
            # there are some problems with the sign of the weights and the exponentiation
            self.A_plus = self.pars['A_plus'] * (np.absolute(self.w_max - self.W))**self.pars.get('dynamic_weight_exponent',1)*np.sign(self.w_max - self.W)
            self.A_minus = self.pars['A_minus'] * (np.absolute(self.W - self.w_min))**self.pars.get('dynamic_weight_exponent',1)*np.sign(self.W - self.w_min)
            
        #store the values
        self.pre_traces_records.append(pre_trace)
        self.post_traces_records.append(post_trace)
        self.W_records.append(self.W)

    def get_records(self):
        # should I return different records for different constraints?
        return {'W':np.array(self.W_records), 'pre_trace':np.array(self.pre_traces_records),'post_trace':np.array(self.post_traces_records)}
    
    def reset_records(self):
        self.W_records = []
        self.pre_traces_records = []
        self.post_traces_records = []

    def reset_weights(self):
        self.W = np.random.random(self.N_post, self.N_pre)
        self.traces = [np.zeros(self.N_pre), np.zeros(self.N_post)]
        self.W_records = []
        self.W_records.append(self.W)
        self.pre_traces_records = []
         
    def plot_synapses(self, post_index = 0, time_in_ms = False, subsampling = 1, trace_index_list = None):

        if time_in_ms:
            dt=self.pars['dt']
            label_x = 'Time (ms)'
        else:
            dt=1
            label_x = 'Time steps'

        # check if the index of the post-syn neuron is correct
        if post_index > self.N_post:
            print(f'Post-synaptic index must be less than {self.N_post}')
            return

        # retrive the records
        W = np.array(self.W_records)[1:,post_index,:]
        pre_trace = np.array(self.pre_traces_records)
        post_trace = np.array(self.post_traces_records)
        num_steps = pre_trace.shape[0]
        time_steps = np.arange(0, num_steps, 1)*dt

        # initialize the figure
        fig,ax = plt.subplots(4, figsize=(12, 14), gridspec_kw={'height_ratios': [1, 2, 2, 2]})#, sharex=True)

        # plot the weights as an image
        fig.colorbar(ax[0].imshow(W.T, cmap = 'viridis', aspect='auto'), ax=ax[0], orientation='vertical', fraction = 0.01, pad = 0.01)
        ax[0].set_xlabel(label_x)
        ax[0].grid(False)
        ax[0].set_ylabel('Synaptic weights')
        ax[0].set_title(f'Post-synaptic neuron: {post_index}')
        
        # plot the weights as a line plot
        ax[1].plot(time_steps[::subsampling], W[ ::subsampling,:], lw=1., alpha=0.7)
        # plot orizontal lines for the weight limits
        if self.constrain == 'Dynamic':
            ax[1].axhline(self.w_max, color='r', linestyle='--', lw=1)
            
        ax[1].set_xlabel(label_x)
        ax[1].set_ylabel('Weight')


        # check the index of the trace to higlight
        if trace_index_list is None:
            trace_index_list = [int(self.N_pre/4), int(self.N_pre/2), int(3*self.N_pre/4)]
        elif max(trace_index_list) > self.N_pre:
            print(f'Trace indexes must be less than {self.N_pre}')
            return
        n = len(trace_index_list)

        ax[2].plot(time_steps, pre_trace, lw=1., alpha=0.05)
        df = pd.DataFrame(pre_trace[:,trace_index_list])
        df.plot(ax=ax[2], color = ['r','g', 'b'], lw=1., alpha=1, legend=False)
        #ax[0].plot(time_steps, , lw=1., alpha=1)#, color = 'r')
        ax[2].set_title(f'Pre-synaptic traces - {n} higlighted')
        ax[2].set_xlabel(label_x)
        ax[2].set_ylabel('traces')

        ax[3].plot(time_steps, post_trace, lw=1., alpha=0.7)
        ax[3].set_xlabel(label_x)
        ax[3].set_ylabel('Post-synaptic traces')

        plt.tight_layout()
        plt.show()



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

