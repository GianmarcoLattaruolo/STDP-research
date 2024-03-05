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





####################################
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
        self.fc.weight.data = torch.clamp(self.fc.weight.data, min=self.w_min, max=self.w_max)
        self.lif = snn.Synaptic(alpha = self.alpha,
                                beta = self.beta, 
                                threshold = pars['threshold'], 
                                reset_mechanism = pars['reset_mechanism'],
                                # reset_delay = False # thuis argument is not implemented in version 0.6.4 but might be useful
                                )
        
        # tracking dictionary
        # all the data will be stored in torch.float16 type to save memory
        self.neuron_records = {'mem': [], 'spk': [], 'syn': [], 'threshold': []}
        self.synapse_records = {'pre_trace': [], 'post_trace': [], 'W': []}
        self.forward_count = 0


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
        spk_record = torch.stack(self.neuron_records['spk']).detach().numpy().reshape(-1, self.n_neurons)
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
        return
    

    # simulation with STDP methods
    def forward(self, image):
        self.forward_count += 1
        
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
                
                # store the previous values of mem and syn to allow the refractory period to work
                if self.pars['refractory_period'] and time_step > 0:
                    mem_prev = mem
                    syn_prev = syn

                    # update the refractory period using the spikes of the previous time step
                    self.refractory_times = self.refractory_times + spk * self.ref_time

                # run the fc layer
                pre_train = self.fc(image[time_step])

                # run the lif neuron
                spk, syn, mem = self.lif(pre_train, syn, mem)

                # store the membrane potential and conductance
                self.neuron_records['mem'].append(mem.to(dtype=torch.float16))
                self.neuron_records['syn'].append(syn.to(dtype=torch.float16))            
                
                # check if some neurons are in refractory time 
                # for them we reset to the previous values the mem and syn attributes
                if self.pars['refractory_period'] and time_step > 0:
                     spk, syn, mem = self.compute_refractory_times(spk,mem_prev, syn_prev, mem, syn)

                # store the spike
                self.neuron_records['spk'].append(spk.to(dtype=torch.bool))

                # apply lateral inhibition
                if self.pars['lateral_inhibition']:
                    syn = self.lateral_inhibition(spk, syn, mem)

                # eventually update the threshold
                if self.pars['dynamic_threshold']:
                    self.lif.threshold = self.dynamic_threshold(spk)

                # update post synaptic traces
                post_syn_traces = self.update_post_synaptic_traces(spk, post_syn_traces)

                # update the weights
                self.STDP_update(spk, image[time_step], post_syn_traces, pre_syn_traces[time_step])

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
        
        # retrive the batch size
        self.batch_size = image.shape[1]

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
        self.synapse_records['pre_trace'].append(pre_syn_traces.to(dtype=torch.float16))

        return pre_syn_traces


    def compute_refractory_times(self, spk, mem_prev, syn_prev, mem, syn):
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
    

    def update_post_synaptic_traces(self, spk, previous_post_syn_traces):
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
    

    def STDP_update(self, spk, pre_train, post_syn_traces, pre_syn_traces):
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
                LTP = A_plus * torch.tensordot(spk.T, pre_syn_traces, dims=1) 
                LTD = A_minus * torch.tensordot(post_syn_traces.T, pre_train, dims=1)

                # update the weights
                W = self.fc.weight.data + LTP - LTD
                W = torch.clamp(W, min=self.w_min, max=self.w_max)
            
        elif self.pars['STDP_type'] == 'offset':
            # the STDP used by the authors of the paper
            if spk.sum() == 0:
                W = self.fc.weight.data
            else:
                thresholded_pre_traces = pre_syn_traces - self.pars['STDP_offset']
                dynamic_weights_constrain = (self.w_max - self.fc.weight.data)**self.pars['mu_exponent']
                dW = self.pars['learning_rate'] * torch.tensordot(thresholded_pre_traces, dynamic_weights_constrain, dims=1)
                W = self.fc.weight.data + dW
        else:
            raise NotImplementedError("Only classic and paper5 STDP is implemented")
        
        # store the weights
        self.fc.weight.data = W
        self.synapse_records['W'].append(W.to(dtype=torch.float16))

        return


    # test phase
    def forward_test(self, image):
        """
        This forward method is meant to be used in the evaluation phase
        in which we don't need to update the weights and store all the history
        We only need to compute the firing rate of the neurons
        Args:
            image: torch tensor of shape (num_steps, batch_size, input_size)
        Returns:
            frequency: numpy array of shape (batch_size, n_neurons)
        """

        with torch.no_grad():

            # retrive the batch size
            self.batch_size = image.shape[1]

            # init syn and mem
            syn, mem  = self.lif.init_synaptic()

            # matrix of zeros for the rates
            rates = torch.zeros((self.batch_size, self.n_neurons))

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
                rates += spk

                # check if some neurons are in refractory time 
                if self.pars['refractory_period']:
                    spk, syn, mem = self.compute_refractory_times(spk,mem_prev, syn_prev, mem, syn)

                # apply lateral inhibition
                if self.pars['lateral_inhibition']:
                    syn = self.lateral_inhibition(spk, syn, mem)

                # eventually update the threshold
                if self.pars['dynamic_threshold']:
                    self.lif.threshold = self.dynamic_threshold(spk, store = False)

            # compute the firing rates from the number of spikes
            rates = rates / image.shape[0]
            rates = rates.detach().numpy()
        
        return rates


    # plotting methods
    def plot_neuron_records(self, n_neurons_to_plot = 1):

        syn_conductance = torch.stack(self.neuron_records['syn']).reshape(-1, self.n_neurons).detach().numpy()[:, :n_neurons_to_plot]
        mem_potential = torch.stack(self.neuron_records['mem']).reshape(-1, self.n_neurons).detach().numpy()[:, :n_neurons_to_plot]
        spk_rec = torch.stack(self.neuron_records['spk']).reshape(-1, self.n_neurons)

        fig, ax = plt.subplots(3, figsize=(12,10), sharex=True,  gridspec_kw={'height_ratios': [1, 1, 0.5]})
        lw = max(100/self.pars['num_steps'], 0.5)

        time_steps = np.arange(self.pars['num_steps'] * self.forward_count)*self.pars['dt']

        # plot the neuron conductance
        ax[0].plot(syn_conductance, alpha = 0.8, lw = lw)
        ax[0].set_ylabel('Conductance')
        ax[0].set_title(f'Synaptic Conductance during presentation of {self.batch_size} images')
            
        # plot the membrane potential
        ax[1].plot( mem_potential, alpha = 0.8, lw=lw)
        if self.pars['dynamic_threshold']:
            threshold_history = torch.stack(self.neuron_records['threshold']).reshape(-1, self.n_neurons).detach().numpy()
            ax[1].plot(threshold_history[:, :n_neurons_to_plot], '--', color = 'red')
        else:
            ax[1].hlines(self.pars['threshold'], 0, mem_potential.shape[0], 'k', 'dashed', color = 'red')
        ax[1].set_ylabel('Membrane Potential')
        ax[1].set_title(f'Membrane Potential during presentation of {self.batch_size} images for the first {n_neurons_to_plot} neurons')

        # plot the spikes
        height = ax[2].bbox.height
        ax[2].scatter(*torch.where(spk_rec), s=2*height/self.n_neurons, c="black", marker="|", lw = lw)
        ax[2].set_ylim([0-0.5, self.n_neurons-0.5])
        ax[2].set_title(f"Mean output firing rate: {spk_rec.mean():.5f} ")
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
    

def train_model(model, data_train, data_val, num_epochs = 1, min_spk_number = 1, num_steps = 100, my_seed = 42) :   
    
    #records_full = {f'epochs_{i}': {f'image_{j}': [] for j in range(len(data_train))} for i in range(num_epochs)}

    with torch.no_grad():
        for epochs in range(num_epochs):
            # Iterate through minibatches
            start_time = time.time()
            with tqdm(total=len(data_train), unit='batch', ncols=120) as pbar:

                for index, (data_it, targets_it) in enumerate(data_train):
                    flag = True
                    iter_flag = 0

                    # store the weights momentanearly
                    W = model.fc.weight.data.clone()

                    while flag and iter_flag < 10:
                        
                        # reset the original weights
                        model.fc.weight.data = W

                        # generate spike data
                        data_it = data_it + torch.ones_like(data_it) * 0.05 * iter_flag

                        # generate spike data 
                        # torch.manual_seed(my_seed)
                        spike_data = spikegen.rate(data_it, num_steps=num_steps, gain = model.pars['gain'])
                        
                        # forward pass
                        model.eval()
                        model.forward(spike_data)

                        # check that all the neurons have emitted at least min_spk_number spikes
                        flag = np.min(model.get_records()['spk'].sum(axis = 0)) < min_spk_number
                        iter_flag += 1
                        if not model.pars['use_min_spk_number']:
                            flag = False

                    # Update progress bar
                    pbar.update(1) 


                    #records_full[f'epochs_{epochs}'][f'image_{index}'] = model.get_records()['W'][::10]
                    #model.reset_records()

                # assign the label to the neurons
                temp_assignments = assign_neurons_to_classes(model, data_val, verbose = 0 )

                # compute the accuracy so far
                accuracy = classify_test_set(model, data_val, temp_assignments, verbose = 0)

                pbar.set_postfix(accuracy=f'{accuracy:.4f}', time=f'{time.time() - start_time:.2f}s')
                #pbar.set_postfix( time=f'{time.time() - start_time:.2f}s')

    return model



def assign_neurons_to_classes(model, data_val, my_seed = None, verbose = 1):
    # this function can be improved

    # set torch seed
    if my_seed is not None:
        torch.manual_seed(42)
    
    # store the rates to assign the neurons to classes
    num_classes = 10

    # retrive the number of postsynaptic neurons and steps from the model
    N_post = model.n_neurons
    num_steps = model.pars['test_num_steps']

    rates_dict = {f'neuron_{i}': {j: 0 for j in range(num_classes)} for i in range(N_post)}
    target_count_dict = {j: 0 for j in range(num_classes)}
    assignments = {f'neuron_{i}': 0 for i in range(N_post)}

    start_time = time.time()
    for index, (data_it, targets_it) in enumerate(data_val):

        if index % 100 == 0 and verbose > 0:
            print(f"Processing batch {index} --- {(time.time() - start_time):.2f} seconds ---")
        
        # generate spike data
        spike_data = spikegen.rate(data_it, num_steps = num_steps, gain = model.pars['gain'])
        
        # forward pass
        model.eval()
        f = model.forward_test(spike_data)

        for target in targets_it.tolist():
            target_count_dict[target] += 1

        for neuron_index in range(N_post):
            for f_index,target in enumerate(targets_it.tolist()):
                rates_dict[f'neuron_{neuron_index}'][target] += f[f_index, neuron_index]


    # assign each neuron to a class
    for neuron_index in range(N_post):
        # normalize the rates by the number of targets
        for target in range(num_classes):
            rates_dict[f'neuron_{neuron_index}'][target] = rates_dict[f'neuron_{neuron_index}'][target] / target_count_dict[target]

        max_class = max(rates_dict[f'neuron_{neuron_index}'], key=rates_dict[f'neuron_{neuron_index}'].get)
        assignments[f'neuron_{neuron_index}'] = max_class
    if verbose > 0:
        df = pd.DataFrame(rates_dict)
        df['num_samples'] = [target_count_dict.get(index) for index in df.index]
        df = df.T
        df['max index'] = df.idxmax(axis=1)
        return assignments, df
    else:
        return assignments



def classify_test_set(model, data_test, assignments, my_seed = None, verbose = 1):
    # use the assignments to classify the test set
    if my_seed is not None:
        torch.manual_seed(42)

    accuracy_record = 0
    num_steps = model.pars['num_steps']

    rates_dict = {j: 0 for j in range(10)}

    start_time = time.time()
    for index, (data_it, targets_it) in enumerate(data_test):

        if index % 100 == 0 and verbose > 0:
            print(f"Processing batch {index} --- {(time.time() - start_time):.2f} seconds ---")

        # generate spike data
        spike_data = spikegen.rate(data_it, num_steps=num_steps, gain = model.pars['gain'])
        
        # forward pass
        model.eval()
        f = model.forward_test(spike_data)

        # highest response neurons
        highest = f.argmax(axis=1)

        # convert the index of the neurons in the assigned classes
        prediction = [assignments[f'neuron_{i}'] for i in highest]

        # update the running accuracy
        accuracy_record += np.sum(np.array(prediction) == targets_it.numpy())
            
    # display the accuracy
    accuracy_record = accuracy_record / len(data_test.dataset)
    
    return accuracy_record









