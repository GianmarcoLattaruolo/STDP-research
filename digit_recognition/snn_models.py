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
import torch
import torch.nn as nn
import snntorch as snn
import snntorch.spikeplot as splt
from snntorch import utils
from snntorch import spikegen
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# optimization libraries
import optuna
from optuna.study import MaxTrialsCallback
from optuna.trial import TrialState


#import from my scripts
main_dir = os.getcwd()
if main_dir not in sys.path:
    print('Adding the folder for the modules')
    sys.path.append(main_dir)
import importlib

importlib.reload(importlib.import_module('snn_experiments'))
importlib.reload(importlib.import_module('snn_plot_utils'))
importlib.reload(importlib.import_module('snn_datasets'))


global mnist_pars
global rate_encoded_mnist
from snn_experiments import *
from snn_plot_utils import *
from snn_datasets import *





##########################################
#                                        #
#   MAIN CLASS TO CREATE A SNN MODEL     #
#                                        #
##########################################



class snn_mnist(nn.Module):


    # initialization methods
    def __init__(self, pars, input_size, n_neurons):
        super().__init__()


        # model parameters
        self.pars = pars
        self.alpha = pars.get('alpha', 0.9)
        self.beta = pars.get('beta', 0.8)
        self.n_neurons = n_neurons
                       

        # STDP parameters
        self.beta_plus = pars['beta_plus']
        self.beta_minus = pars['beta_minus']
        self.w_max = pars.get('w_max', 1.0)
        self.w_min = pars.get('w_min', 0.0)

        # SNN structure
        self.input_size = input_size
        self.fc = nn.Linear(input_size, n_neurons, bias=False)
        self.fc.weight.data = self.weight_initializer()
        self.lif = snn.Synaptic(alpha = self.alpha,
                                beta = self.beta, 
                                threshold = pars['threshold'], 
                                reset_mechanism = pars['reset_mechanism'],
                                # reset_delay = False # thuis argument is not implemented in version 0.6.4 but might be useful
                                )
        
        # tracking dictionary
        if pars['store_records']:
            self.t = pars['store_subsampling_factor'] # this does not apply to the spikes
        # all the data will be stored in torch.float16 type to save memory
        self.neuron_records = {'mem': [], 'spk': [], 'syn': [], 'threshold': []}
        self.synapse_records = {'pre_trace': [], 'post_trace': [], 'W': []}
        self.forward_count = 0
        self.num_input_spikes = 0


    def weight_initializer(self): # sill in development
        
        if self.pars.get('weight_initialization_type', 'clamp') == 'clamp':
            return torch.clamp(self.fc.weight.data, min=self.w_min, max=self.w_max)
        elif self.pars['weight_initialization_type'] == 'shift':
            W = self.fc.weight.data
            W = W - W.min() + 0.001
            return W
        elif self.pars['weight_initialization_type'] == 'norm_column':
            W =  torch.rand_like(self.fc.weight.data)*(self.w_max - self.w_min) + self.w_min
            # normalize the weights rows to 1
            W = W / W.sum(axis=1).reshape(-1,1)
            return W
        else:
            raise NotImplementedError("Only clamp, shift and norm_column weight initialization are implemented")


    def get_records(self):
        """
        Returns the records of the simulation
        Args:
            None
        Returns:
            records: dictionary with the records of the simulation with inside
            mem: numpy array of shape (num_steps * forward_count * batch_size, n_neurons)
            spk: numpy array of shape (num_steps * forward_count * batch_size, n_neurons)
            syn: numpy array of shape (num_steps * forward_count * batch_size, n_neurons)
            threshold: numpy array of shape (num_steps * forward_count * batch_size, n_neurons)
            pre_trace: numpy array of shape (num_steps * forward_count * batch_size, input_size)
            post_trace: numpy array of shape (num_steps * forward_count * batch_size, n_neurons)
            W: numpy array of shape (num_steps * forward_count , n_neurons, input_size)
        """

        # transform the list of tensors in a single numpy array for each record
        mem_record = torch.stack(self.neuron_records['mem']).detach().numpy().reshape(-1, self.n_neurons)
        spk_record = torch.stack(self.neuron_records['spk']).reshape(-1, self.n_neurons)
        syn_record = torch.stack(self.neuron_records['syn']).detach().numpy().reshape(-1, self.n_neurons)
        if self.pars['dynamic_threshold']:
            threshold_record = torch.stack(self.neuron_records['threshold']).detach().numpy().reshape(-1, self.n_neurons)
        else:
            threshold_record = None
        pre_trace = torch.stack(self.synapse_records['pre_trace']).detach().numpy().reshape(-1, self.input_size)
        post_trace = torch.stack(self.synapse_records['post_trace']).detach().numpy().reshape(-1, self.n_neurons)
        W_record = torch.stack(self.synapse_records['W']).detach().numpy().reshape(-1, self.n_neurons, self.input_size)

        # create the dictionary
        records = {'mem': mem_record, 
                   'spk': spk_record, 
                   'syn': syn_record, 
                   'threshold': threshold_record,
                   'pre_trace': pre_trace, 
                   'post_trace': post_trace, 
                   'W': W_record}

        return records


    def reset_records(self):
        self.neuron_records = {'mem': [], 'spk': [], 'syn': [], 'threshold': []}
        self.synapse_records = {'pre_trace': [], 'post_trace': [], 'W': []}
        self.forward_count = 0

        return
    

    # simulation with STDP methods
    def forward(self, image):
                
        # with STDP we don't need to compute the gradients
        with torch.no_grad():

            # init syn and mem
            syn, mem  = self.lif.init_synaptic()

            # initialize auxiliary attributes and get the pre synaptic traces
            pre_syn_traces = self.initialize_auxiliary_attributes(image)

            # initialize the post synaptic traces
            post_syn_traces = torch.zeros((self.batch_size, self.n_neurons))

            # iterate on the time steps
            for time_step in range(image.shape[0]):

                # indicator of the time step to store the records
                store = self.pars['store_records'] and time_step % self.t == 0

                # store the previous values of mem and syn to allow the refractory period to work
                if self.pars['refractory_period'] and time_step > 0:
                    mem_prev = mem
                    syn_prev = syn
                    spk_prev = spk

                    # update the refractory period using the spikes of the previous time step
                    self.refractory_times = self.refractory_times + spk_prev * self.ref_time
                

                # run the fc layer
                pre_train = self.fc(image[time_step])

                # run the lif neuron
                spk, syn, mem = self.lif(pre_train, syn, mem)

                # store the membrane potential and conductance
                if store:
                    self.neuron_records['mem'].append(mem.to(dtype=torch.float16))
                    self.neuron_records['syn'].append(syn.to(dtype=torch.float16))            
                
                # check if some neurons are in refractory time 
                # for them we reset to the previous values the mem and syn attributes
                if self.pars['refractory_period'] and time_step > 0:
                     spk, syn, mem = self.compute_refractory_times(spk,spk_prev, mem_prev, syn_prev, mem, syn, store = store)

                # store the spike at each time step
                self.neuron_records['spk'].append(spk.to(dtype=torch.bool))

                # apply lateral inhibition
                if self.pars['lateral_inhibition']:
                    syn = self.lateral_inhibition(spk, syn, mem)

                # eventually update the threshold
                if self.pars['dynamic_threshold']:
                    self.lif.threshold = self.dynamic_threshold(spk, store = store)

                # update post synaptic traces
                post_syn_traces = self.update_post_synaptic_traces(spk, post_syn_traces, store = store)

                # update the weights
                self.STDP_update(spk, image[time_step], post_syn_traces, pre_syn_traces[time_step], store = store)

        return 


    def initialize_auxiliary_attributes(self, image):
        """
        With this method we exploit the kwnoledge of the batch size
        to initialize the attributes that are batch dependent.
        Morover we generate the pre synaptic traces
        Args:
            image: torch tensor of shape (num_steps, batch_size, input_size)
        Returns:
            pre_syn_traces: torch tensor of shape (num_steps, batch_size, n_neurons)
        """
        
        # retrive the batch size and number of time steps
        self.batch_size = image.shape[1]
        self.num_steps = image.shape[0]

        # intialize other attributes if needed:

        # tensor for the threshold dynamics of each neuron in each batch
        if self.pars['dynamic_threshold']:
            self.theta = torch.zeros((self.batch_size, self.n_neurons))

        # tensor for the refractory period of each neuron in each batch
        if self.pars['refractory_period']:
            self.refractory_times = torch.zeros((self.batch_size, self.n_neurons))
            self.ref_time = self.pars['ref_time']

        # generate the pre synaptic traces
        pre_syn_traces = self.generate_traces(image)

        if self.pars['store_records']:
            self.synapse_records['pre_trace'].append(pre_syn_traces[::self.t].to(dtype=torch.float16))
            self.forward_count += 1
            self.num_input_spikes += torch.sum(image)

        return pre_syn_traces


    def compute_refractory_times(self, spk, spk_prev, mem_prev, syn_prev, mem, syn, store=True):
        """
        With this method we compute the refractory period of the neurons
        We first identify the neurons that are in refractory period
        Then we reset the spikes of that neurons if any
        Finally we reset the mem and syn attributes of these neurons to the previous values
        and decrease the refractory period by 1
        Args:
            spk: torch tensor of shape (batch_size, n_neurons)
            mem_prev: torch tensor of shape (batch_size, n_neurons)
            syn_prev: torch tensor of shape (batch_size, n_neurons)
            mem: torch tensor of shape (batch_size, n_neurons)
            syn: torch tensor of shape (batch_size, n_neurons)
        Returns:
            spk: torch tensor of shape (batch_size, n_neurons)
            syn: torch tensor of shape (batch_size, n_neurons)
            mem: torch tensor of shape (batch_size, n_neurons)
        """
        # for the neurons that have just spiked we must accept the new resetted mem and syn values
        mem_prev = mem * spk_prev + mem_prev * (1 - spk_prev)
        syn_prev = syn * spk_prev + syn_prev * (1 - spk_prev)

        # retrive the index of the neurons in refractory period
        refractory_neurons = ((self.refractory_times > 0) + 0.0)  

        # neurons not in refractory period preservs their spikes
        spk = spk * (1 - refractory_neurons) 

        # the mem and syn of the neurons in refractory period is reset to the previous value
        mem = mem * (1 - refractory_neurons) + mem_prev * refractory_neurons
        syn = syn * (1 - refractory_neurons) + syn_prev * refractory_neurons

        # we update the records
        if store:   
            self.neuron_records['mem'][-1] = mem
            self.neuron_records['syn'][-1] = syn

        # decrease the refractory period
        self.refractory_times = self.refractory_times - 1 

        # return the corrected spk, mem and syn
        return spk, syn, mem


    def lateral_inhibition(self, spk, syn, mem):
        """
        With this method we implement the lateral inhibition
        If there have been no spikes we don't do anything
        if there have been spikes we inhibit the conductance of all the other neurons 
        but the first one we find in to have spiked following the order of the neurons
        Args:
            spk: torch tensor of shape (batch_size, n_neurons)
            syn: torch tensor of shape (batch_size, n_neurons)
            mem: torch tensor of shape (batch_size, n_neurons)
        Returns:
            syn: torch tensor of shape (batch_size, n_neurons)
        """
        # this function can handle batch_size > 1

        # check if there are no spikes
        if torch.sum(spk) == 0:
            # no changes to the sunaptic conductance
            return syn
        else:
            # find the first neuron that spiked by finding the highest membrane potential (that must be above 1)
            first_neuron_index = np.argmax(mem.detach().numpy(), axis=1)

            # initialize a tensor of shape (batch_size, n_neurons) with ones
            temp = torch.ones_like(syn)

            # for each bacth we set to 0 the indicator of the first neuron that spiked
            temp[np.arange(self.batch_size), first_neuron_index ] = 0

            # we subtract the same value from all the other neurons' conductance
            syn = syn  - temp * self.pars['lateral_inhibition_strength']

            # we eventually clamp to positive values
            syn = torch.clamp(syn, min=0)

            # return the corrected synaptic conductance
            return syn
    

    def dynamic_threshold(self, spk, store = True):
        """
        With this method we change the values of the threshold of the neurons
        by adding a dynamic variable theta:
        we first compute the decaying factor at each time steps of theta
        then we add a same value to the thetas of all the neurons that have spiked
        finally we update the threshold of the neurons and store the new values
        Args:
            spk: torch tensor of shape (batch_size, n_neurons)
        Returns:
            threshold: torch tensor of shape (batch_size, n_neurons)
        """
        # this function can handle batch_size > 1

        # decaying factor
        beta_theta = (1-self.pars['dt']/self.pars['tau_theta'])

        # update the theta tensor
        self.theta = beta_theta * self.theta + self.pars['theta_add'] * spk

        # update the threshold
        threshold = self.pars['threshold'] + self.theta

        # add a store check for the test phase
        if store:
            self.neuron_records['threshold'].append(threshold.to(dtype=torch.float16))

        return threshold
    

    def update_post_synaptic_traces(self, spk, previous_post_syn_traces, store=True):
        """
        with this method we compute the post synaptic traces at each time step
        unlike the pre synaptic traces for which we could compute them all at once
        The traces decay of a factor beta_minus and are incremented by the spikes.
        Args:
            spk:                      torch tensor of shape (batch_size, n_neurons)
            previous_post_syn_traces: torch tensor of shape (batch_size, n_neurons)
        Returns:
            new_post_syn_traces: torch tensor of shape (batch_size, n_neurons)
        """
        # this function can handle batch_size > 1

        # compute the new post synaptic traces
        new_post_syn_traces = self.beta_minus*previous_post_syn_traces + spk

        # store the new post synaptic traces
        if store:
            self.synapse_records['post_trace'].append(new_post_syn_traces.to(dtype=torch.float16))

        # return the new post synaptic traces
        return new_post_syn_traces


    def generate_traces(self, pre_spike_train_ex):
        """
        with this method we generate the pre synaptic traces starting from the 
        entire spike train of the input layer
        Args:
            pre_spike_train_ex : binary spike train input of shape (num_steps, input_size)
        Returns:
            traces             : torch tensor of shape (num_steps, batch_size, input_size)
        """
        # this function can handle batch_size > 1

        # Get parameters
        num_steps = pre_spike_train_ex.shape[0]
        traces = torch.zeros_like(pre_spike_train_ex)    
        decay_rate = self.beta_plus

        # loop for each time step
        for step in range(num_steps-1):

            # exponential decay
            traces[step+1] = decay_rate*traces[step] 
            # increment for the arriving spikes
            traces[step+1] += pre_spike_train_ex[step]


        return traces
    

    def STDP_update(self, spk, pre_train, post_syn_traces, pre_syn_traces, store=True):
        # this function can handle batch_size > 1
        
        if self.pars['STDP_type'] == None:
            # no weights updates
            # used this for the test phase
            W = self.fc.weight.data
        
        elif self.pars['STDP_type'] == 'classic':

            # check if there are no spikes
            if torch.sum(spk) == 0 and torch.sum(pre_train) == 0:
                W = self.fc.weight.data
            else:

                # compute LTP and LTD
                A_plus, A_minus = self.pars['A_plus'], self.pars['A_minus']

                # reshape spike and traces to allow the outer product
                LTP = torch.tensordot(spk.T, pre_syn_traces, dims=1) / self.batch_size
                LTD = torch.tensordot(post_syn_traces.T, pre_train, dims=1) / self.batch_size

                # update the weights
                W = self.fc.weight.data  + A_plus * LTP  - A_minus *  LTD
                W = torch.clamp(W, min=self.w_min, max=self.w_max)
            
        elif self.pars['STDP_type'] == 'offset':
            # the STDP used by the authors of the paper
            if spk.sum() == 0:
                W = self.fc.weight.data
            else:
                thresholded_pre_traces = pre_syn_traces - self.pars['STDP_offset'] 
                # the offset might be replaced by a running average of the pre synaptic traces
                dynamic_weights_constrain = (self.w_max - self.fc.weight.data)**self.pars['mu_exponent']
                dW = self.pars['learning_rate'] * thresholded_pre_traces.mean(axis=0) * dynamic_weights_constrain
                W = self.fc.weight.data + dW

        elif self.pars['STDP_type'] == 'classic_asymptotic':

            # check if there are no spikes
            if torch.sum(spk) == 0 and torch.sum(pre_train) == 0:
                W = self.fc.weight.data
            else:

                # compute LTP and LTD
                A_plus, A_minus = self.pars['A_plus'], self.pars['A_minus']

                # reshape spike and traces to allow the outer product
                LTP = torch.tensordot(spk.T, pre_syn_traces, dims=1) / self.batch_size
                LTD = torch.tensordot(post_syn_traces.T, pre_train, dims=1) / self.batch_size

                # update the weights
                W = self.fc.weight.data 
                W = W + A_plus * LTP * (self.w_max - W)**2 - A_minus *  LTD * (W - self.w_min)**2
                #W = torch.clamp(W, min=self.w_min, max=self.w_max)
        else:
            raise NotImplementedError("Only classic, offset and classic_asymptotic STDP are implemented")
        
        # store the weights
        self.fc.weight.data = W
        if store:
            self.synapse_records['W'].append(W.to(dtype=torch.float16))

        return


    # test phase
    def forward_test(self, image):
        """
        This forward method is meant to be used in the evaluation phase
        in which we don't need to update the weights and store all the history
        We only need to compute the firing rate of the neurons
        Args:
            image:  torch tensor of shape (num_steps, batch_size, input_size)
        Returns:
            rates:  the mean over time of the spikes elucitated
                    numpy array of shape (batch_size, n_neurons)
        """

        with torch.no_grad():

            # retrive the batch size
            self.batch_size = image.shape[1]

            # init syn and mem
            syn, mem  = self.lif.init_synaptic()

            # matrix of zeros for the rates
            num_spk = torch.zeros((self.batch_size, self.n_neurons))

            # reinitialize the refractory period
            if self.pars['refractory_period']:
                self.refractory_times = torch.zeros((self.batch_size, self.n_neurons))

            # iterate on the time steps
            for time_step in range(image.shape[0]):
                
                # store the previous values of mem and syn to allow the refractory period to work
                if self.pars['refractory_period'] and time_step > 0:
                        mem_prev = mem
                        syn_prev = syn

                        # update the refractory period using the spikes of the previous time step
                        self.refractory_times = self.refractory_times - 1 + spk * self.ref_time

                # run the fc layer
                pre_train = self.fc(image[time_step])

                # run the lif neuron
                spk, syn, mem = self.lif(pre_train, syn, mem)

                # add the new spikes
                num_spk += spk

                # check if some neurons are in refractory time 
                if self.pars['refractory_period']:
                    spk, syn, mem = self.compute_refractory_times(spk,mem_prev, syn_prev, mem, syn)

                # apply lateral inhibition
                if self.pars['lateral_inhibition']:
                    syn = self.lateral_inhibition(spk, syn, mem)

                # eventually update the threshold
                if self.pars['dynamic_threshold']:
                    self.lif.threshold = self.dynamic_threshold(spk, store = False)
        
        return num_spk.detach().numpy()


    # plot the simulation records, still in development
    def plot_simulation(self, neuron_index=0, n_highlight=10, plot_syn_conductance=False, plot_post_trace=False, plot_weights_distribution=False):
        # compute the total number of times steps experienced by the network in the training phase
        self.tot_sim_time = self.forward_count * self.num_steps * self.batch_size 

        # retrive the records
        records = self.get_records()

        num_subplots = 3  # Default subplots

        if plot_syn_conductance:
            num_subplots += 1
        if plot_post_trace:
            num_subplots += 1
        if plot_weights_distribution:
            num_subplots += 1

        fig, ax = plt.subplots(num_subplots, figsize=(12, 4 * num_subplots))
        lw = max(min(100 / self.tot_sim_time, 1), 0.5)

        # Plot the output spikes of the network
        tot_input_mean = self.num_input_spikes / (self.tot_sim_time * self.input_size)
        spk_rec = records['spk'].float()[-10000:,:]
        #reduced_spk = torch.stack([(self.get_records()['spk'][i:i + 100, :]+0.0).mean(axis=0)  for i in range(0, model.get_records()['spk'].shape[0], 100)]) + 0.0
        height = ax[0].bbox.height
        ax[0].scatter(*torch.where(spk_rec), s = height/self.n_neurons, c="black", marker="|", lw=0.5)  # scatter is incredible slow
        ax[0].set_ylim([0 - 0.5, self.n_neurons - 0.5])
        ax[0].set_title(f"Mean output rate: {spk_rec.mean():.5f} - Mean input rate: {tot_input_mean:.5f}, last 10000 time steps")
        ax[0].set_ylabel("Neuron index")

        # Plot the membrane potential of the chosen neuron
        mem_rec = records['mem'][:, neuron_index]
        threshold_rec = records['threshold']
        ax[1].plot(mem_rec, alpha=0.8, lw=lw)
        if threshold_rec is not None:
            ax[1].plot(threshold_rec[:, neuron_index], color='red', linestyle='--')
        else:
            ax[1].hlines(1, 1, self.tot_sim_time/self.t + 1, color='red', linestyle='--')
        ax[1].set_ylabel("Membrane Potential")
        ax[1].set_title("Membrane Potential and Output Spikes")

        # Plot the synaptic conductance of the chosen neuron if selected
        if plot_syn_conductance:
            syn_rec = records['syn'][:, neuron_index]
            ax[2].plot(syn_rec, alpha=0.8, lw=lw)
            ax[2].set_ylabel("Neuron conductance")
            ax[2].set_title(f"Neuron conductance, mean: {syn_rec.mean():.2f}")

        # Plot the post synaptic traces of the chosen neuron if selected
        if plot_post_trace:
            post_trace_rec = records['post_trace'][:, neuron_index]
            ax[2 + plot_syn_conductance].plot(post_trace_rec, alpha=0.8, lw=lw)
            ax[2 + plot_syn_conductance].set_ylabel("Post-synaptic trace")
            ax[2 + plot_syn_conductance].set_title(f"Post-synaptic trace, mean: {post_trace_rec.mean():.2f}")

        # Plot the weights evolution of the chosen neuron
        W_rec = records['W'][:, neuron_index, :]
        cor_weights = pd.DataFrame(W_rec[:, :n_highlight])
        uncor_weights = pd.DataFrame(W_rec[:, n_highlight:])
        cor_weights.plot(ax=ax[2 + plot_syn_conductance + plot_post_trace], legend=False, color='tab:red', alpha=0.8, lw=2 * lw)
        uncor_weights.plot(ax=ax[2 + plot_syn_conductance + plot_post_trace], legend=False, color='tab:blue', alpha=0.2, lw=lw)
        ax[2 + plot_syn_conductance + plot_post_trace].set_ylabel("Synaptic Weight")
        ax[2 + plot_syn_conductance + plot_post_trace].set_xlabel("Time step")

        if plot_weights_distribution:
            # Plot the final weights distribution of the chosen neuron
            time_step = -1
            w_min = np.min(W_rec[time_step, :]) - 0.0001
            w_max = np.max(W_rec[time_step, :]) + 0.0001
            width = (w_max - w_min) / 51
            bins = np.arange(w_min, w_max, width)
            ax[3 + plot_syn_conductance + plot_post_trace].hist(W_rec[time_step, :n_highlight], bins=bins, color='tab:red', alpha=0.5, label=f'first {n_highlight} synapses')
            ax[3 + plot_syn_conductance + plot_post_trace].hist(W_rec[time_step, n_highlight:], bins=bins, color='tab:blue', alpha=0.5, label=f'remaining {self.input_size - n_highlight} synapses')
            ax[3 + plot_syn_conductance + plot_post_trace].set_xlabel("Synaptic Weight")
            ax[3 + plot_syn_conductance + plot_post_trace].set_ylabel("Frequency")
            ax[3 + plot_syn_conductance + plot_post_trace].legend(loc='best')
            ax[3 + plot_syn_conductance + plot_post_trace].set_title("Synaptic Weight Distribution")

        plt.tight_layout()
        plt.show()
        return 


    def plot_simulation_interactive(self, manual_update = False):

        # compute the total number of times steps experienced by the network in the training phase
        self.tot_sim_time = self.forward_count * self.num_steps * self.batch_size 

        # retrive the records
        records = self.get_records()

        def main_plot( neuron_index = 0, n_highlight = 10):

            # plot the records
            fig, ax = plt.subplots(3, figsize=(12,15), gridspec_kw={'height_ratios': [1,1,1]})
            lw = max(min(100/self.tot_sim_time,1), 0.5)

            # plot the output spikes of the network
            tot_input_mean = self.num_input_spikes / (self.tot_sim_time * self.input_size)
            spk_rec = records['spk'].float()[-10000:,:]
            height = ax[0].bbox.height
            ax[0].scatter(*torch.where(spk_rec), s=2*height, c="black", marker="|", lw = lw )
            ax[0].set_ylim([0-0.5, self.n_neurons-0.5])
            ax[0].set_title(f"Mean output rate: {spk_rec.mean():.5f} - Mean inputrate: {tot_input_mean:.5f} last 10000 time steps")
            ax[0].set_ylabel("Neuron index")

            # plot the membrane potential of the choosen neuron
            mem_rec = records['mem'][:,neuron_index]
            threshold_rec = records['threshold']
            ax[1].plot(mem_rec, alpha = 0.8, lw = lw)
            if threshold_rec is not None:
                ax[1].plot(threshold_rec[:,neuron_index], color = 'red', linestyle = '--')
            else:
                ax[1].hlines(1, 1, self.tot_sim_time/self.t + 1, color = 'red', linestyle = '--')
            ax[1].set_ylabel("Membrane Potential")
            ax[1].set_title("Membrane Potential and Output Spikes")

            # plot the weights evolution of the choosen neuron
            W_rec = records['W'][:,neuron_index,:]
            cor_weights = pd.DataFrame(W_rec[:, :n_highlight])
            uncor_weights = pd.DataFrame(W_rec[:, n_highlight:])
            cor_weights.plot(ax = ax[2], legend = False, color = 'tab:red', alpha = 0.8, lw = 2*lw)
            uncor_weights.plot(ax = ax[2], legend = False, color = 'tab:blue', alpha = 0.2, lw = lw)
            ax[2].set_ylabel("Synaptic Weight")
            ax[2].set_xlabel("Time step")

            return
        
        interactive_plot = widgets.interactive(
        main_plot,
        {'manual': manual_update, 'manual_name': 'Update plot'},
        neuron_index = widgets.IntSlider(
            value=0, 
            min=0,
            max=self.n_neurons-1, 
            step=1, 
            description='Neuron index', 
            style = {'description_width': 'initial'},
            layout=widgets.Layout(width='600px'),
            continuous_update=False
            ),
        n_highlight = widgets.IntSlider(
            value=10,
            min=1,
            max=self.input_size-1,
            step=1,
            description='Number of highlighted synapses',
            style = {'description_width': 'initial'},
            layout=widgets.Layout(width='600px'),
            continuous_update=False
            )
        )

        display(interactive_plot)

        return



####################################
#                                  #
#      TRAIN AND EVALUATION        #
#                                  #
####################################
    


def train_model(model, train_loader, val_loader = None, num_epochs = 1, val_accuracy_record = True) :   
    
    if val_accuracy_record and val_loader is None:
        raise ValueError("If val_accuracy_record is set to True, val_loader must be provided")
    
    # retrive the minimum number of spikes to be emitted at each forward pass
    min_spk_number = model.pars['min_spk_number']
    record_spk = []

    # train the model
    with torch.no_grad():
        for epochs in range(num_epochs):
            # Iterate through minibatches
            start_time = time.time()
            with tqdm(total=len(train_loader), unit='batch', ncols=120+40*val_accuracy_record) as pbar:

                for data_it, _ in train_loader:

                    # forward pass
                    model.eval()
                    model.forward(data_it)
                    
                    # check that all the neurons have emitted at least min_spk_number spikes
                    spk_emitted = torch.stack(model.neuron_records['spk']) + 0. # shape (num_steps , batch_size, n_neurons)
                    flag = np.min(spk_emitted.numpy().sum(axis = 0)) < min_spk_number
                    if flag:
                        anspnpi = spk_emitted.numpy().sum(axis = 0).mean()
                        raise ValueError(f"One Neuron have not emitted at least {min_spk_number} spikes, anspnpi {anspnpi:.2e}")
                    else:
                        # reset just the spk records
                        model.neuron_records['spk'] = []
                        record_spk.append(spk_emitted)


                    # Update progress bar
                    pbar.update(1) 

                if val_accuracy_record:
                    # assign the label to the neurons
                    temp_assignments = assign_neurons_to_classes(model, val_loader, verbose = 0 )

                    # compute the accuracy so far (anspni = average number of spikes per neuron per image)
                    accuracy, anspnpi = classify_test_set(model, val_loader, temp_assignments, verbose = 0)

                    # update the progress bar
                    pbar.set_postfix(acc=f'{accuracy:.4f}', time=f'{time.time() - start_time:.2f}s', ANSPNPI = f'{anspnpi:.2e}')
                else:
                    pbar.set_postfix( time=f'{time.time() - start_time:.2f}s')
        
        model.neuron_records['spk'] = record_spk

    return model



def assign_neurons_to_classes(model, val_loader, verbose = 1):
    """
    Function to assign each post synaptic neuron to a class from 0 to 10
    according to the highest firing rate
    Args:
        model: snn_mnist object
        val_loader: torch DataLoader object
        verbose: int, level of verbosity
    Returns:
        assignments: numpy array of shape (n_neurons, ) with the assigned class
        df: pandas DataFrame with the mean firing rates of each neuron to all the classes
            normalized by the number of samples of that class and the index of the max value
    """
    # store the rates to assign the neurons to classes
    num_classes = 10
    n_neurons = model.n_neurons

    spk_matrix = np.zeros((num_classes, n_neurons))
    sample_per_class_count = np.zeros(num_classes)

    for data_it, targets_it in val_loader:

        # retrive the number of steps
        num_steps = data_it.shape[0]
        
        # forward pass
        model.eval()
        num_spk = model.forward_test(data_it)  # numpy array of shape (batch_size x n_neurons)

        for num_spk_index,target in enumerate(targets_it.tolist()):  

            # in the row corresponding to the target add the firing rate of the neurons in response to that target
            spk_matrix[target] += num_spk[num_spk_index] 

            # update the count of samples for that class
            sample_per_class_count[target] += 1


    # devide each row by the number of samples
    spk_matrix = spk_matrix.T / sample_per_class_count

    # compute the confidence of the assignment: 
    #1 : sort the response rate of the neurons to each class
    sort_rate = np.sort(spk_matrix , axis=1) 
    #2 : check if the difference between the highest response and the second highest is significative
    conf_status = np.diff(sort_rate, n=1)[:,-1] > model.pars.get('assignment_confidence',0)

    # assign the neurons to the classes
    assignments = spk_matrix.argmax(axis=1)
    # build the dataframe reporting if the neuron assignment is confident
    df_assignments = pd.DataFrame([assignments,conf_status], index = ['assignments', 'conf_status']).T

    if verbose > 0:
        df_rate = pd.DataFrame(spk_matrix/num_steps, index = [i for i in range(n_neurons)], columns = [i for i in range(num_classes)])
        df_rate = pd.concat([df_rate,pd.DataFrame(sample_per_class_count).T])
        df_rate.index = df_rate.index.tolist()[:-1] + ['sample_count']
        df_rate['max_index'] = df_rate.idxmax(axis=1)
        return df_assignments, df_rate
    else:
        return df_assignments



def classify_test_set(model, test_loader, df_assignments, verbose = 1):
    # use the assignments to classify the test set

    accuracy_record = 0
    tot_num_spk = 0

    start_time = time.time()
    for index, (data_it, targets_it) in enumerate(test_loader):

        if (index+1) % 100 == 0 and verbose > 0:
            print(f"Processing batch {index+1}/{len(test_loader)} --- {(time.time() - start_time):.2f} seconds ---")

        # forward pass
        model.eval()
        num_spk = model.forward_test(data_it)  # out shape : batch_size x n_neurons
        tot_num_spk += num_spk.sum()

        # # identify for each image in the batch the highest response neuron assigned with confidence
        # confidences = list(df_assignments['conf_status'].values)
        # f_conf = f[:,confidences]
        # max_index = np.argmax(f_conf,axis=1)
        # prediction2 = df_assignments['assignments'][confidences][max_index]


        # Identify for each slice of the batch the first class assigned with confidence to the highest responsive neuron
        #1 : sort the indexes of the rates array
        sorted_response = np.argsort(num_spk, axis=1)  # batch_size x n_neurons with indexes sorted accordingly to f 
        #2 : initialize a prediction array with values not present in the classes (20)
        prediction = np.ones(num_spk.shape[0])*20
        #3 : for each image in the batch find the first class assigned with confidence the the highest responsive neuron
        for i in range(num_spk.shape[0]):
            #4 : for each neuron in the sorted_response array check if the assignment has been done with confidence
            for go_to_previous_index in range(num_spk.shape[1]):
                #5 : identify the index of the neuron with the highest response
                max_neuron_index = sorted_response[i,-1-go_to_previous_index]
                #6 : check if the assignment has been done with confidence
                if df_assignments.iloc[max_neuron_index]['conf_status']:
                    prediction[i] = df_assignments['assignments'][max_neuron_index]
                    break #  exit from the inner for loop if the assignment has been done with confidence
                else:
                    continue # go to the next neuron if the assignment has not been done with confidence
        
        # # test if the two method are equals
        # if np.sum(prediction2 != prediction) > 0:
        #     print("The two methods to assign the neurons to the classes are not equals")

        # check that all the neurons have been assigned with confidence
        if np.sum(prediction == 20) > 0:
            #print(f"Not all the neurons have been classified with confidence > {model.pars.get('assignment_confidence',0)}")
            pass
                
        # update the running accuracy
        accuracy_record += np.sum(prediction == targets_it.numpy())
            
    # display the accuracy
    accuracy_record = accuracy_record / (len(test_loader)*test_loader.batch_size)
    averaged_num_spk_per_neuron_per_image = tot_num_spk / (model.n_neurons * len(test_loader)*test_loader.batch_size)
    
    return accuracy_record, averaged_num_spk_per_neuron_per_image


##########################################
#                                        #
#     HYPERPARAMETERS OPTIMIZATION       #
#                                        #
##########################################

# first search : time constants and learning rates

def define_model(trial):

    STDP_type = trial.suggest_categorical('STDP_type', ['classic', 'offset'])
    if STDP_type == 'offset':
        STDP_offset = trial.suggest_float("STDP_offset", 0.0001, 0.1, log = True)
        learning_rate = trial.suggest_float("learn_rate", 0.0001, 0.1, log = True)
        A_minus, A_plus = 0.0, 0.0
        beta_plus = 0.0 # we don't need post synaptic traces if we use the offset STDP
    else:
        STDP_offset = 0.0
        A_minus = trial.suggest_float("A_minus", 0.00005, 0.001, log = True )
        A_plus = trial.suggest_float("A_plus", 0.00005, 0.001, log = True )
        beta_plus = trial.suggest_float("beta_plus", 0.1, 1.0 )
        learning_rate = 0.0 # we don't need learning rate if we use the classic STDP

    pars = mnist_pars(
        weight_initialization_type = 'clamp',
        STDP_type = STDP_type,
        A_minus = A_minus,
        A_plus = A_plus,
        beta_minus = trial.suggest_float("beta_minus", 0.1, 1.0 ),
        beta_plus = beta_plus,
        STDP_offset = STDP_offset,
        learning_rate = learning_rate,
        alpha = trial.suggest_float("alpha", 0.1, 1.0 ),
        beta = trial.suggest_float("beta", 0.1, 1.0 ),
        dynamic_threshold = True,
        tau_theta = trial.suggest_float("tau_theta", 10, 1000, log = True),
        theta_add = trial.suggest_float("theta_add", 0.2, 2.0, step = 0.2),
        lateral_inhibition = True,
        lateral_inhibition_strength = trial.suggest_float("inhi_strength", 0.01, 10.0, log = True),
        store_records = False,
        assignment_confidence = trial.suggest_float('ass_conf', 0.0, 0.01, step = 0.001)    
    )

    # define the model
    input_size = 28*28
    n_neurons = 100
    model = snn_mnist(pars, input_size, n_neurons)
    trial.set_user_attr('weight_initialization_type','clamp')
    return model



def objective(trial):
    # in this objective there are 13 hyperparameters to optimize 

    # use cuda if available
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

    # define the model
    model = define_model(trial).to(device)

    # define the data loader
    batch_size = 100
    num_steps = 200
    gain = 1.
    min_rate = 0.005
    mnist_train = rate_encoded_mnist(batch_size, num_steps=num_steps, gain=gain, min_rate = min_rate, train=True, my_seed = 42)
    sub_train, sub_val, sub_test = mnist_train.get_subset_train_val_test(subset = 50)

    # train the model
    num_epochs = 3
    with torch.no_grad():
        for epochs in range(num_epochs):
            start_time = time.time()
            with tqdm(total=len(sub_train), unit='batch', ncols=120) as pbar:

                # Iterate through minibatches
                for data_it, _ in sub_train:

                    # forward pass
                    model.forward(data_it)

                    # Update progress bar
                    pbar.update(1)

                # assign the label to the neurons
                temp_assignments = assign_neurons_to_classes(model, sub_val, verbose = 0 )

                # compute the accuracy so far
                accuracy, mean_rate = classify_test_set(model, sub_test, temp_assignments, verbose = 0)

                # update the progress bar
                pbar.set_postfix(acc=f'{accuracy:.4f}', time=f'{time.time() - start_time:.2f}s', rate = f'{mean_rate:.2e}')

                # report the accuracy
                trial.report(accuracy, epochs)

                # handle pruning
                if trial.should_prune() or mean_rate < 0.00001:
                    raise optuna.exceptions.TrialPruned()
        
        # set a user attribute to keep tracks of other statistics of the model trained
        trial.set_user_attr('mean_rate',mean_rate)
        trial.set_user_attr('assign_mean_conf',np.mean(temp_assignments['conf_status']+0.0))


    return accuracy


# second search : number of neurons, batch size, num_steps, gain, min_rate

def objective_2(trial):
    # in this objective there 5 hyperparameters to optimize

    # use cuda if available
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

    # define the model
    pars = mnist_pars(
        weight_initialization_type = 'shift',
        STDP_type = 'classic',
        A_minus = 0.0007,
        A_plus = 0.0006,
        beta_minus = 0.85,
        beta_plus = 0.45,
        reset_mechanism = 'subtract',
        alpha = 0.95,
        beta = 0.8,
        dynamic_threshold = True,
        tau_theta = 12,
        theta_add = 2.0, # maybe greater
        refractory_period = False,
        lateral_inhibition = True,
        lateral_inhibition_strength = 1.3,
        store_records = False,
        assignment_confidence = 0.0,
    )

    # define the model
    input_size = 28*28
    n_neurons = trial.suggest_int("n_neurons", 128, 1024, step = 64)
    model = snn_mnist(pars, input_size, n_neurons).to(device)

    batch_size = trial.suggest_categorical("batch_size", [50, 100, 200, 400]) #  the val and test are just 400 samples
    num_steps = trial.suggest_int("num_steps", 200, 500, step = 50)
    gain = trial.suggest_float("gain", 0.9, 20.0, log=True)
    min_rate = trial.suggest_float("min_rate", 0.0, 0.5, step = 0.05)
    mnist_train = rate_encoded_mnist(batch_size, num_steps=num_steps, gain=gain, min_rate = min_rate, train=True, my_seed = 42)
    sub_train, sub_val, sub_test = mnist_train.get_subset_train_val_test(subset = 25)

    # train the model
    min_spk_number = model.pars['min_spk_number'] * batch_size
    num_epochs = 10
    with torch.no_grad():
        for epochs in range(num_epochs):
            start_time = time.time()
            with tqdm(total=len(sub_train), unit='batch', ncols=120) as pbar:

                # Iterate through minibatches
                for data_it, _ in sub_train:

                    # forward pass
                    model.forward(data_it)

                    # check the min spk number
                    flag = torch.stack(model.neuron_records['spk']).detach().numpy().sum() < min_spk_number
                    trial.set_user_attr('min_spk_n_reached', not flag)

                    # Update progress bar
                    pbar.update(1)

                # assign the label to the neurons
                temp_assignments = assign_neurons_to_classes(model, sub_val, verbose = 0 )

                # compute the accuracy so far
                accuracy, mean_rate = classify_test_set(model, sub_test, temp_assignments, verbose = 0)

                # update the progress bar
                pbar.set_postfix(acc=f'{accuracy:.4f}', time=f'{time.time() - start_time:.2f}s')

                # report the accuracy
                trial.report(accuracy, epochs)

                # handle pruning
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()
    
        #trial.set_user_attr('weight_initialization_type','clamp')
        trial.set_user_attr('mean_rate', mean_rate)
        #trial.set_user_attr('assign_mean_conf',np.mean(temp_assignments['conf_status']+0.0))
    return accuracy

    
# third search : STDP_classic_asymptotic

def objective_3(trial):
     # in this objective there 5 hyperparameters to optimize

    # use cuda if available
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

    # define the model
    pars = mnist_pars(
        weight_initialization_type = 'shift',
        STDP_type = 'classic',
        A_minus = trial.suggest_float("A_minus", 0.0001, 0.1, log = True ),
        w_min = 0.01,
        A_plus = trial.suggest_float("A_plus", 0.0001, 0.1, log = True ),
        beta_minus = 0.85,
        beta_plus = 0.45,
        reset_mechanism = 'subtract',
        alpha = 0.95,
        beta = 0.8,
        dynamic_threshold = True,
        tau_theta = 12,
        theta_add = trial.suggest_float("theta_add", 0.5, 5.0, step = 0.5),
        refractory_period = False,
        lateral_inhibition = True,
        lateral_inhibition_strength = trial.suggest_float("inhi_strength", 0.1, 10.0, log = True),
        store_records = False,
        assignment_confidence = 0.0,
    )

    # define the model
    input_size = 28*28
    n_neurons = trial.suggest_int("n_neurons", 128, 1024, step = 64)
    model = snn_mnist(pars, input_size, n_neurons).to(device)

    batch_size = 400 #  the val and test are just 400 samples
    num_steps = trial.suggest_int("num_steps", 200, 500, step = 50)
    gain = trial.suggest_float("gain", 0.9, 20.0, log=True)
    min_rate = trial.suggest_float("min_rate", 0.0, 0.5, step = 0.05)
    mnist_train = rate_encoded_mnist(batch_size, num_steps=num_steps, gain=gain, min_rate = min_rate, train=True, my_seed = 42)
    sub_train, sub_val, sub_test = mnist_train.get_subset_train_val_test(subset = 25)

    # train the model
    num_epochs = 5
    with torch.no_grad():
        for epochs in range(num_epochs):
            start_time = time.time()
            with tqdm(total=len(sub_train), unit='batch', ncols=120) as pbar:

                # Iterate through minibatches
                for data_it, _ in sub_train:

                    # forward pass
                    model.forward(data_it)

                    # Update progress bar
                    pbar.update(1)

                # assign the label to the neurons
                temp_assignments = assign_neurons_to_classes(model, sub_val, verbose = 0 )

                # compute the accuracy so far
                accuracy, mean_rate = classify_test_set(model, sub_test, temp_assignments, verbose = 0)

                # update the progress bar
                pbar.set_postfix(acc=f'{accuracy:.4f}', time=f'{time.time() - start_time:.2f}s')

                # report the accuracy
                trial.report(accuracy, epochs)

                # handle pruning
                if trial.should_prune() or model.fc.weight.data.mean() == 0.0:
                    raise optuna.exceptions.TrialPruned()
                
        trial.set_user_attr('mean weights', model.fc.weight.data.detach().numpy().mean(dtype=np.float64))
        #trial.set_user_attr('weight_initialization_type','clamp')
        trial.set_user_attr('mean_rate', mean_rate)
        #trial.set_user_attr('assign_mean_conf',np.mean(temp_assignments['conf_status']+0.0))

    return accuracy





