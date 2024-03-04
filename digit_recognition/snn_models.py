# basic libraries
import os
import sys
import shutil
import time
import numpy as np
import pandas as pd

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
import torch
import torch.nn as nn
import snntorch as snn
import snntorch.spikeplot as splt
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
    def __init__(self, pars, input_size, n_neurons):
        super().__init__()

        requires_grad = pars.get('require_grad', False)

        # model parameters
        self.pars = pars
        self.alpha = pars.get('alpha', 0.9)
        self.beta = pars.get('beta', 0.8)
        self.n_neurons = n_neurons
        self.time_steps = np.arange(pars['num_steps'])*pars['dt']
        if self.pars['dynamic_threshold']:
            self.theta = torch.zeros((n_neurons), requires_grad=requires_grad)
        if self.pars['refractory_period']:
            self.refractory_times = torch.zeros((n_neurons), requires_grad=requires_grad)
            self.ref_time = pars['ref_time']

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


    def reset_redords(self):
        self.neuron_records = {'mem': [], 'spk': [], 'syn': [], 'threshold': []}
        self.synapse_records = {'pre_trace': [], 'post_trace': [], 'W': []}
        return
    

    def forward(self, image):
        # init syn and mem
        syn, mem  = self.lif.init_synaptic()

        # reshape the image
        image = image.view(self.pars['num_steps'], -1)

        # generate the pre synaptic traces
        pre_syn_traces = generate_traces(self.pars, image)
        self.synapse_records['pre_trace'] = pre_syn_traces

        # initialize the post synaptic traces
        self.post_syn_traces = torch.zeros((self.n_neurons))

        # reinitialize the refractory period
        self.refractory_times = torch.zeros((self.n_neurons))

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

            # retrive the trace at the current time step to allow STDP to work
            self.pre_syn_traces = pre_syn_traces[time_step,:]

            # update post synaptic traces
            self.update_post_synaptic_traces(spk)

            # update the weights
            self.STDP_update(spk, image[time_step])

        return 


    def forward_test(self, image):

        # reset only the neuron records 
        self.neuron_records = {'mem': [], 'spk': [], 'syn': [], 'threshold': []}

         # init syn and mem
        syn, mem  = self.lif.init_synaptic()

        # reshape the image
        image = image.view(self.pars['num_steps'], -1)

        # reinitialize the refractory period
        self.refractory_times = torch.zeros((self.n_neurons))

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



    def lateral_inhibition(self, spk, syn):
        first_neuron_index = torch.where(spk)[0][0]
        # inhibit the conductance of all the others
        temp = torch.ones_like(syn)
        temp[first_neuron_index] = 0
        syn = syn  - temp * self.pars['lateral_inhibition_strength']
        return
    

    def update_post_synaptic_traces(self, spk):
        self.post_syn_traces = self.beta_minus*self.post_syn_traces + spk
        self.synapse_records['post_trace'].append(self.post_syn_traces)
        return


    @staticmethod
    def generate_traces(pars, pre_spike_train_ex):
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
            traces[step+1,:] = decay_rate*traces[step, :] 
            # increment for the arriving spikes
            traces[step+1,:] += pre_spike_train_ex[step, :]


        return traces
    

    def STDP_update(self, spk, pre_train):
        
        if self.pars['STDP_type'] == None:
            # no weights updates
            # used this for the test phase
            pass
        
        elif self.pars['STDP_type'] == 'classic':
            
            # compute LTP and LTD
            A_plus, A_minus = self.pars['A_plus'], self.pars['A_minus']
            LTP = A_plus * torch.outer(spk, self.pre_syn_traces) 
            LTD = A_minus * torch.outer(self.post_syn_traces, pre_train)

            # update the weights
            W = self.fc.weight.data + LTP - LTD
            W = torch.clamp(W, min=self.w_min, max=self.w_max)
            self.fc.weight.data = W
            
        else:
            raise NotImplementedError("Only classic STDP is implemented")
        
        # store the weights
        self.synapse_records['W'].append(W)

        return


    def dynamic_threshold(self, spk):

        # addition to the threshold
        beta_theta = (1-dt/self.pars['tau_theta'])
        self.theta = beta_theta * self.theta + self.pars['theta_add'] * spk

        # decaying toward the resting threshold
        self.threshold = self.pars['threshold'] + self.theta

        # store the threshold
        self.neuron_records['threshold'].append(self.threshold)

        return
    

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
            ax[1].plot(torch.stack(self.neuron_records['threshold']).detach().numpy()[:, :n_neurons_to_plot], 'k--', color = 'red')
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