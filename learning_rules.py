#!/usr/bin/env python3#!C:\Users\latta\miniconda3\envs\test\python3

# basic libraries
import os
import sys
import shutil
import time
import numpy as np


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
        if self.hard_constrain == True:
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
            raise NotImplementedError
        
    def reset_records(self):
        self.W_records = []
        self.pre_traces_records = []
        self.post_traces_records = []


         












def STDP_rule(
        pars, # parameters dictionary
        W, # current weights
        traces, # STDP traces list [pre_synaptic, post_synaptic]  
        spikes, # Spikes [pre_synaptic, post_synaptic]
        **kwargs,
):
    """
    INPUTS:
    - pars    : parameters dictionary
    - W       : current weights of shape: (N_post, N_pre)
    - traces  : STDP traces list [pre_synaptic, post_synaptic] of shape (N_pre,), (N_pre,) respectively
    - spikes  : Spikes [pre_synaptic, post_synaptic] of shape (N_pre,), (N_post,) respectively

    RETURNS:
    - W       : updated weights    
    - traces  : updated STDP traces list [pre_synaptic, post_synaptic]

    """

    # retrive parameters
    dt = pars.get('dt', 0.1)
    A_plus = pars.get('A_plus', 0.01)
    A_minus = pars.get('A_minus', 0.01)
    tau_plus = pars.get('tau_plus', 20)
    tau_minus = pars.get('tau_minus', 20)
    Wmax = pars.get('Wmax', 1)
    Wmin = pars.get('Wmin', 0)

    if traces is None:
        traces = [np.zeros(np.shape(W)[1]), np.zeros(np.shape(W)[0])]
    # unpack the variables
    pre_syn_trace, post_syn_trace = traces
    pre_syn_spk, post_syn_spk = spikes

    # what the correct order of weight and traces update?

    # update the traces
    pre_syn_trace = pre_syn_trace*(tau_plus-1)/tau_plus + pre_syn_spk
    post_syn_trace = post_syn_trace*(tau_minus-1)/tau_minus + post_syn_spk

    # update the weights
    #print('depression',post_syn_trace)
    W =  W + A_plus*np.outer(post_syn_spk,pre_syn_trace) - A_minus*np.outer(post_syn_trace,pre_syn_spk)

    return W, [pre_syn_trace, post_syn_trace]