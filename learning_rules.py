#!/usr/bin/env python3#!C:\Users\latta\miniconda3\envs\test\python3

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
from IPython.display import display
import logging
logging.getLogger('matplotlib.font_manager').disabled = True
# use NMA plot style
#plt.style.use("https://raw.githubusercontent.com/NeuromatchAcademy/course-content/main/nma.mplstyle")
plt.style.use('seaborn-v0_8')
my_layout = widgets.Layout()
my_layout.width = '620px'


class STDP_synapse:

    def __init__(self, pars, N_pre, N_post,
                 W_init = None,
                 hard_constrain = True, 
                 short_memory_trace = False,
                 seed=None):
        # Set random seed
        if seed is not None:
            np.random.seed(seed)
        
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
        self.hard_constrain = hard_constrain
        self.short_memory_trace = short_memory_trace

        # Initialize weights and traces
        if W_init is not None:
            self.W = W_init
        else:
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
        if self.hard_constrain == 'None':
            self.W = self.W
        elif self.hard_constrain == True:
            self.W = np.clip(self.W, self.w_min, self.w_max)
        else:
            # dynamic weight constraint based on the current weight
            self.A_plus = self.pars['A_plus'] * (self.w_max - self.W)**self.pars.get('dynamic_weight_exponent',1)
            self.A_minus = self.pars['A_minus'] * (self.W - self.w_min)**self.pars.get('dynamic_weight_exponent',1)
            
        #store the values
        self.pre_traces_records.append(pre_trace)
        self.post_traces_records.append(post_trace)
        self.W_records.append(self.W)

    def get_records(self):
        if self.hard_constrain:
            return {'W':np.array(self.W_records), 'pre_trace':np.array(self.pre_traces_records),'post_trace':np.array(self.post_traces_records)}
        else:
            # print("to be implemented")
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

        fig.colorbar(ax[0].imshow(W.T, cmap = 'viridis', aspect='auto'), ax=ax[0], orientation='vertical', fraction = 0.01, pad = 0.01)
        ax[0].set_xlabel(label_x)
        ax[0].grid(False)
        ax[0].set_ylabel('Synaptic weights')
        ax[0].set_title(f'Post-synaptic neuron: {post_index}')
        

        ax[1].plot(time_steps[::subsampling], W[ ::subsampling,:], lw=1., alpha=0.7)
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












if __name__ == "__main__":
    print('experiment is currentlyon the STDP-basic-experiments notebook')
