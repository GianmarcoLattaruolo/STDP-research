# basic libraries
import os
import sys
import shutil
import time
import numpy as np
import pandas as pd
from tqdm import tqdm

# graphics libraries
import matplotlib
import matplotlib.pyplot as plt
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets
import jupyterlab_widgets as lab
from IPython.display import display
import logging
logging.getLogger('matplotlib.font_manager').disabled = True
plt.style.use('seaborn-v0_8')
my_layout = widgets.Layout()
my_layout.width = '620px'

#machine Learning libraries
#machine Learning libraries
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import snntorch as snn
import snntorch.spikeplot as splt
from snntorch import utils
from snntorch import spikegen
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


#import from my scripts
main_dir = os.getcwd()
if main_dir not in sys.path:
    print('Adding the folder for the modules')
    sys.path.append(main_dir)
import importlib

importlib.reload(importlib.import_module('snn_experiments'))
importlib.reload(importlib.import_module('snn_plot_utils'))

from snn_experiments import *
from snn_plot_utils import *


class snn_mnist(nn.Module):


    # initialization methods
    def __init__(self, pars, input_size, n_neurons):
        super().__init__()

        requires_grad = pars.get('require_grad', False)

        # model parameters
        self.pars = pars
        self.alpha = pars.get('alpha', 0.9)
        self.beta = pars.get('beta', 0.8)
        self.n_neurons = n_neurons
        self.time_steps = np.arange(pars['num_steps'])*pars['dt']
        

        # STDP parameters
        self.beta_minus = pars['beta_minus']
        self.w_max = pars.get('w_max', 1.0)
        self.w_min = pars.get('w_min', 0.0)

        # SNN structure
        self.fc = nn.Linear(input_size, n_neurons, bias=False)
        self.fc.weight.data = torch.clamp(self.fc.weight.data, min=self.w_min, max=self.w_max)
        self.lif = snn.Synaptic(alpha = self.alpha,
                                beta = self.beta, 
                                threshold = pars['threshold'], 
                                reset_mechanism = pars['reset_mechanism'],
                                init_hidden = False,
                                )
        
        # tracking dictionary
        self.neuron_records = {'mem': [], 'spk': [], 'syn': [], 'threshold': []}
        self.synapse_records = {'pre_trace': [], 'post_trace': [], 'W': []}


    def get_records(self):

        # transform the list of tensors in a single numpy array for each record
        mem_record = torch.stack(self.neuron_records['mem']).detach().numpy()
        spk_record = torch.stack(self.neuron_records['spk']).detach().numpy()
        syn_record = torch.stack(self.neuron_records['syn']).detach().numpy()
        if self.pars['dynamic_threshold']:
            threshold_record = torch.stack(self.neuron_records['threshold']).detach().numpy()
        else:
            threshold_record = None
        pre_trace = self.synapse_records['pre_trace'].detach().numpy()
        post_trace = torch.stack(self.synapse_records['post_trace']).detach().numpy()
        W_record = torch.stack(self.synapse_records['W']).detach().numpy()

        # create the dictionary
        neurons_rec = {'mem': mem_record, 'spk': spk_record, 'syn': syn_record, 'threshold': threshold_record}
        synapse_rec = {'pre_trace': pre_trace, 'post_trace': post_trace, 'W': W_record}

        return neurons_rec, synapse_rec


    def reset_records(self):
        self.neuron_records = {'mem': [], 'spk': [], 'syn': [], 'threshold': []}
        self.synapse_records = {'pre_trace': [], 'post_trace': [], 'W': []}
        return
    

    # simulation with STDP methods
    def forward(self, image):
        with torch.no_grad():
            # init syn and mem
            syn, mem  = self.lif.init_synaptic()

            # retrive the batch size
            self.batch_size = image.shape[1]
            if self.pars['dynamic_threshold']:
                self.theta = torch.zeros((self.batch_size, self.n_neurons))
            if self.pars['refractory_period']:
                self.refractory_times = torch.zeros((self.batch_size, self.n_neurons))
                self.ref_time = self.pars['ref_time']

            # generate the pre synaptic traces
            pre_syn_traces = generate_traces(self.pars, image)    
            self.synapse_records['pre_trace'] = pre_syn_traces

            # initialize the post synaptic traces
            self.post_syn_traces = torch.zeros((self.batch_size, self.n_neurons))

            # reinitialize the refractory period
            self.refractory_times = torch.zeros((self.n_neurons))

            # iterate on the time steps
            for time_step in range(self.pars['num_steps']):

                if self.pars['refractory_period'] and time_step > 0:
                    mem_prev = mem
                    syn_prev = syn

                    # update the refractory period using the spikes of the previous time step
                    self.refractory_times = self.refractory_times - 1 + spk * self.ref_time

                # run the fc layer
                pre_train = self.fc(image[time_step])

                # run the lif neuron
                spk, syn, mem = self.lif(pre_train, syn, mem)

                # store the membrane potential and conductance
                self.neuron_records['mem'].append(mem)
                self.neuron_records['syn'].append(syn)            
                
                # check if some neurons are in refractory time 
                # for them we reset to the previous values the mem and syn attributes
                if self.pars['refractory_period'] and time_step > 0:

                    # retrive the index of the neurons in refractory period
                    refractory_neurons = (self.refractory_times > 0) + 0.0

                    # neurons not in refractory period preservs their spikes
                    spk = spk * (1 - refractory_neurons) 

                    # the mem and syn of the neurons in refractory period is reset to the previous value
                    mem = mem * (1 - refractory_neurons) + mem_prev * refractory_neurons
                    syn = syn * (1 - refractory_neurons) + syn_prev * refractory_neurons

                    # we update the records
                    self.neuron_records['mem'][-1] = mem
                    self.neuron_records['syn'][-1] = syn

                # store the spike
                self.neuron_records['spk'].append(spk)

                if self.pars['lateral_inhibition']:
                    syn = self.lateral_inhibition(spk, syn)

                if self.pars['dynamic_threshold']:
                    self.lif.threshold = self.dynamic_threshold(spk)

                # retrive the trace at the current time step to allow STDP to work
                self.pre_syn_traces = pre_syn_traces[time_step,:]

                # update post synaptic traces
                self.update_post_synaptic_traces(spk)

                # update the weights
                self.STDP_update(spk, image[time_step])

        return 



    def lateral_inhibition(self, spk, syn):
        # this function can handle batch_size > 1
        if torch.sum(spk) == 0:
            return syn
        else:
            first_neuron_index = np.argmax(spk.detach().numpy(), axis=1)
            # inhibit the conductance of all the others
            temp = torch.ones_like(syn)
            temp[np.arange(self.batch_size), first_neuron_index ] = 0
            syn = syn  - temp * self.pars['lateral_inhibition_strength']
        return syn
    

    def update_post_synaptic_traces(self, spk):
        # this function can handle batch_size > 1
        self.post_syn_traces = self.beta_minus*self.post_syn_traces + spk
        self.synapse_records['post_trace'].append(self.post_syn_traces)
        return


    @staticmethod
    def generate_traces(pars, pre_spike_train_ex):
        # this function can handle batch_size > 1
        """
        track of pre-synaptic spikes

        Args:
            pars               : parameter dictionary
            pre_spike_train_ex : binary spike train input of shape (num_steps, input_size)

        Returns:
            traces             : torch tensor of shape (num_steps, input_size)
        """

        # Get parameters
        num_steps = pre_spike_train_ex.shape[0]
        traces = torch.zeros_like(pre_spike_train_ex)    
        decay_rate = pars.get('beta_plus', 0.9)

        # loop for each time step
        for step in range(num_steps-1):

            # exponential decay
            traces[step+1] = decay_rate*traces[step] 
            # increment for the arriving spikes
            traces[step+1] += pre_spike_train_ex[step]


        return traces
    

    def STDP_update(self, spk, pre_train):
        # this function can handle batch_size > 1
        
        if self.pars['STDP_type'] == None:
            # no weights updates
            # used this for the test phase
            W = self.fc.weight.data
        
        elif self.pars['STDP_type'] == 'classic':
            
            # compute LTP and LTD
            A_plus, A_minus = self.pars['A_plus'], self.pars['A_minus']

            # reshape spike and traces to allow the outer product
            LTP = A_plus * torch.tensordot(spk.T, self.pre_syn_traces, dims=1) 
            LTD = A_minus * torch.tensordot(self.post_syn_traces.T, pre_train, dims=1)

            # update the weights
            W = self.fc.weight.data + LTP - LTD
            W = torch.clamp(W, min=self.w_min, max=self.w_max)
            
        
        elif self.pars['STDP_type'] == 'offset':
            # the STDP used by the authors of the paper
            if spk.sum() == 0:
                W = self.fc.weight.data
            else:
                thresholded_pre_traces = self.pre_syn_traces - self.pars['STDP_offset']
                dynamic_weights_constrain = (self.w_max - self.fc.weight.data)**self.pars['mu_exponent']
                dW = self.pars['learning_rate'] * torch.tensordot(thresholded_pre_traces, dynamic_weights_constrain, dims=1)
                W = self.fc.weight.data + dW
        else:
            raise NotImplementedError("Only classic STDP is implemented")
        
        # store the weights
        self.fc.weight.data = W
        self.synapse_records['W'].append(W)

        return


    def dynamic_threshold(self, spk):

        # addition to the threshold
        beta_theta = (1-self.pars['dt']/self.pars['tau_theta'])
        self.theta = beta_theta * self.theta + self.pars['theta_add'] * spk

        # decaying toward the resting threshold
        threshold = self.pars['threshold'] + self.theta

        # store the threshold
        self.neuron_records['threshold'].append(threshold)

        return threshold
    
    # test phase
    def forward_test(self, image):
        # retrive the batch size
        self.batch_size = image.shape[1]

        # reset only the neuron records 
        self.neuron_records = {'mem': [], 'spk': [], 'syn': [], 'threshold': []}

         # init syn and mem
        syn, mem  = self.lif.init_synaptic()

        # reinitialize the refractory period
        self.refractory_times = torch.zeros((self.batch_size, self.n_neurons))

        # iterate on the time steps
        for time_step in range(self.pars['num_steps']):

            # run the fc layer
            pre_train = self.fc(image[time_step])

            # run the lif neuron
            spk, syn, mem = self.lif(pre_train, syn, mem)

            # store the membrane potential and conductance
            self.neuron_records['mem'].append(mem)
            self.neuron_records['syn'].append(syn)

            # check if some neurons are in refractory time 
            # for them we reset to the previous values the mem and syn attributes
            # we adjust the records
            if self.pars['refractory_period']:
                # retrive the index of the neurons in refractory period
                refractory_neurons = torch.where(self.refractory_times>0)[0]
                # neurons not in refractory period preservs their spikes
                spk = spk * (1 - refractory_neurons) 
                # the mem and syn of the neurons in refractory period is reset to the previous value
                mem = mem * (1 - refractory_neurons) + self.neuron_records['mem'][-1] * refractory_neurons
                syn = syn * (1 - refractory_neurons) + self.neuron_records['syn'][-1] * refractory_neurons
                # we update the records
                self.neuron_records['mem'][-1] = mem
                self.neuron_records['syn'][-1] = syn

                # update the refractory period
                self.refractory_times = self.refractory_times - 1 + spk * self.ref_time
                
            # store the spike
            self.neuron_records['spk'].append(spk)

            if self.pars['lateral_inhibition']:
                syn = self.lateral_inhibition(spk, syn)

            if self.pars['dynamic_threshold']:
                self.lif.threshold = self.dynamic_threshold(spk)

        frequency = torch.stack(self.neuron_records['spk']).mean(axis = 0).detach().numpy()
        
        return frequency


    # plotting methods
    def plot_neuron_records(self, n_neurons_to_plot = 1):

        syn_conductance = torch.stack(self.neuron_records['syn']).detach().numpy()[:, :n_neurons_to_plot]
        mem_potential = torch.stack(self.neuron_records['mem']).detach().numpy()[:, :n_neurons_to_plot]
        spk_rec = torch.stack(self.neuron_records['spk'])

        fig, ax = plt.subplots(3, figsize=(12,10), sharex=True,  gridspec_kw={'height_ratios': [1, 1, 0.5]})
        lw = max(100/self.pars['num_steps'], 0.5)

        # plot the neuron conductance
        ax[0].plot(self.time_steps,syn_conductance, alpha = 0.8, lw = lw)
        ax[0].set_ylabel('Conductance')
        ax[0].set_title('Synaptic Conductance')
            
        # plot the membrane potential
        ax[1].plot(self.time_steps, mem_potential, alpha = 0.8, lw=lw)
        if self.neuron_records['threshold']:
            ax[1].plot(torch.stack(self.neuron_records['threshold']).detach().numpy()[:, :n_neurons_to_plot], '--', color = 'red')
        else:
            ax[1].hlines(self.pars['threshold'], 0, mem_potential.shape[0], 'k', 'dashed', color = 'red')
        ax[1].set_ylabel('Membrane Potential')
        ax[1].set_title('Membrane Potential')

        # plot the spikes
        height = ax[2].bbox.height
        ax[2].scatter(*torch.where(spk_rec), s=2*height/self.n_neurons, c="black", marker="|", lw = lw)
        ax[2].set_ylim([0-0.5, self.n_neurons-0.5])
        ax[2].set_title(f"Mean output firing rate: {spk_rec.mean():.2f} ")
        ax[2].set_ylabel("Neuron index")
        ax[2].set_xlabel("Time step")

        plt.show()
        return


    def plot_synapse_records(self, n_neurons_to_plot = 1, weight_index = 0):

        #THIS DOES NOT WORK PROPERLY
        
        pre_syn_traces = self.synapse_records['pre_trace'].detach().numpy()[:,:n_neurons_to_plot]
        post_syn_traces = torch.stack(self.synapse_records['post_trace']).detach().numpy()
        weight_history = torch.stack(self.synapse_records['W']).detach().numpy()[:,weight_index,:n_neurons_to_plot]

        fig, ax = plt.subplots(3, figsize=(12,10), sharex=True,  gridspec_kw={'height_ratios': [0.6, 0.6, 1]})
        lw = max(100/self.pars['num_steps'], 0.5)
        # pre synaptic traces plot
        ax[0].plot(self.time_steps, pre_syn_traces, alpha = 0.8, lw = lw)
        ax[0].set_ylabel('Pre-trace')
        ax[0].set_title('Pre-synaptic traces')

        # post synaptic traces plot
        ax[1].plot(self.time_steps, post_syn_traces, alpha = 0.8, lw = lw)
        ax[1].set_ylabel('Post-trace')
        ax[1].set_title('Post-synaptic traces')

        # weights plot
        ax[2].plot(weight_history, alpha = 0.5)
        ax[2].set_ylabel('Weight')
        ax[2].set_title(f'Synaptic weights history of neuron {weight_index}')

        plt.show()
        return
    



####################################
#                                  #
#      TRAIN AND EVALUATION        #
#                                  #
####################################
    
def train_model(model, data_train, data_val, num_epochs = 1, min_spk_number = 1, num_steps = 100, verbose = 0) :   
    
    records_full = {f'epochs_{i}': {f'image_{j}': [] for j in range(len(data_train))} for i in range(num_epochs)}

    with torch.no_grad():
        for epochs in range(num_epochs):
            # Iterate through minibatches
            start_time = time.time()
            with tqdm(total=len(data_train), unit='image', ncols=100) as pbar:

                for index, (data_it, targets_it) in enumerate(data_train):
                    flag = True
                    iter_flag = 0

                    # store the weights momentanearly
                    W = model.fc.weight.data.clone()

                    while flag:
                        
                        # reset the original weights
                        model.fc.weight.data = W

                        # generate spike data
                        data_it = data_it + torch.ones_like(data_it) * 0.01 * iter_flag

                        # generate spike data
                        spike_data = spikegen.rate(data_it, num_steps=num_steps, gain = 1)
                        
                        # forward pass
                        model.eval()
                        model.forward(spike_data)

                        # check that all the neurons have emitted at least min_spk_number spikes
                        flag = np.min(model.get_records()[0]['spk'].sum(axis = 0)) < min_spk_number
                        iter_flag += 1

                    # Update progress bar
                    pbar.update(1) 


                    records_full[f'epochs_{epochs}'][f'image_{index}'] = model.get_records()[1]['W']
                    model.reset_records()

                # assign the label to the neurons
                #temp_assignments = assign_neurons_to_classes(model, data_val, verbose = 0 )

                # compute the accuracy so far
                #accuracy = classify_test_set(model, data_val, temp_assignments, verbose = 0)

                #pbar.set_postfix(accuracy=f'{accuracy:.4f}', time=f'{time.time() - start_time:.2f}s')
                pbar.set_postfix( time=f'{time.time() - start_time:.2f}s')


        
    return model, records_full




def assign_neurons_to_classes(model, data_val, my_seed = None, verbose = 1):
    # set torch seed
    if my_seed is not None:
        torch.manual_seed(42)

    # store the rates to assign the neurons to classes
    num_classes = 10

    # retrive the number of postsynaptic neurons and steps from the model
    N_post = model.n_neurons
    num_steps = model.pars['num_steps']

    rates_dict = {f'neuron_{i}': {j: 0 for j in range(num_classes)} for i in range(N_post)}
    assignments = {f'neuron_{i}': 0 for i in range(N_post)}

    start_time = time.time()
    for index, (data_it, targets_it) in enumerate(data_val):

        if index % 100 == 0 and verbose > 0:
            print(f"Processing image {index} --- {(time.time() - start_time):.2f} seconds ---")
        
        # generate spike data
        spike_data = spikegen.rate(data_it, num_steps = num_steps, gain = model.pars['gain'])
        
        # forward pass
        model.eval()
        f = model.forward_test(spike_data)

        for neuron_index in range(N_post):
            rates_dict[f'neuron_{neuron_index}'][int(targets_it)] += f[neuron_index]


    # assign each neuron to a class
    for neuron_index in range(N_post):

        max_class = max(rates_dict[f'neuron_{neuron_index}'], key=rates_dict[f'neuron_{neuron_index}'].get)
        assignments[f'neuron_{neuron_index}'] = max_class

    return assignments



def classify_test_set(model, data_test, assignments, my_seed = None, verbose=1):
    # use the assignments to classify the test set
    if my_seed is not None:
        torch.manual_seed(42)

    accuracy_record = 0
    num_steps = model.pars['num_steps']

    start_time = time.time()
    for index, (data_it, targets_it) in enumerate(data_test):

        if index % 100 == 0 and verbose > 0:
            print(f"Processing image {index} --- {(time.time() - start_time):.2f} seconds ---")

        # generate spike data
        spike_data = spikegen.rate(data_it, num_steps=num_steps, gain = 1)
        
        # forward pass
        model.eval()
        f = model.forward_test(spike_data)

        # sum the rate of each neuron of a given class
        rates_dict = {j: 0 for j in range(10)}
        for neuron_index in range(model.n_neurons):
            rates_dict[assignments[f'neuron_{neuron_index}']] += f[neuron_index]

        # retrive the class with the maximum rate
        max_class = max(rates_dict, key=rates_dict.get)

        # compare the class of the max neuron with the target
        if max_class == targets_it:
            accuracy_record += 1

    # display the accuracy
    accuracy_record = accuracy_record / len(data_test)
    
    return accuracy_record









