#!/usr/bin/env python3

# basic libraries
import os
import sys
import shutil
import time
import numpy as np

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
    pre_syn_trace = pre_syn_trace*(tau_plus-1)/tau_plus + A_plus*pre_syn_spk
    post_syn_trace = post_syn_trace*(tau_minus-1)/tau_minus + A_minus*post_syn_spk

    # update the weights
    #print('depression',post_syn_trace)
    W =  W + np.outer(post_syn_spk,pre_syn_trace) - np.outer(post_syn_trace,pre_syn_spk)

    return W, [pre_syn_trace, post_syn_trace]