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
from snn_experiments import *



##########################################
#                                        #
#      PLOT FUNCTIONS OF SECTION 1       #
#                                        #
##########################################



def plot_cur_mem_spk(cur_in, cur, mem, fmp, spk, thr_line=False, vline=False, title=False):
    num_steps = len(cur_in)
    # Generate Plots
    fig, ax = plt.subplots(4, figsize=(12,15), sharex=True,
                            gridspec_kw = {'height_ratios': [1, 1, 1, 0.4]})
    
    # Plot presynaptic spikes
    splt.raster(cur_in, ax = ax[0], s=400, c="tab:pink", marker="|")
    ax[0].set_ylabel("Neuron index")
    #ax[0].set_xlabel("Time step")
    if title:
        ax[0].set_title(title)

    # Plot input current
    ax[1].plot(cur, c="tab:orange")
    ax[1].set_xlim([0, num_steps])
    ax[1].set_ylabel("Input Current ($I_{in}$)")


    # Plot membrane potential
    ax[2].plot(mem)
    ax[2].plot(fmp, c="green", label="FMP")
    ax[2].set_ylabel("Membrane Potential ($U_{mem}$)")
    if thr_line:
        ax[2].axhline(y=thr_line, alpha=0.25, linestyle="dashed", c="red", linewidth=2, label = 'Thr')
    ax[2].legend(loc = 'best')
    plt.xlabel("Time step")

    # Plot output spike using spikeplot
    splt.raster(spk, ax[3], s=400, c="black", marker="|")
    if vline:
        ax[3].axvline(x=vline, ymin=0, ymax=6.75, alpha = 0.15, linestyle="dashed", c="black", linewidth=2, zorder=0, clip_on=False)
    plt.ylabel("Output spikes")
    plt.yticks([])

    plt.show()
    return



def plot_spk_cur_mem_spk(spk_in, syn_rec, mem_rec, mem_fmp_rec, spk_rec, title):

    # Generate Plots
    fig, ax = plt.subplots(4, figsize=(12,10), sharex=True,
                            gridspec_kw = {'height_ratios': [0.4, 1, 1, 0.4]})

    # Plot input current
    splt.raster(spk_in, ax[0], s=400, c="black", marker="|", lw=0.5)
    ax[0].set_ylabel("Input Spikes")
    ax[0].set_title("Synaptic Conductance-based Neuron Model With Input Spikes")
    ax[0].set_yticks([])

    # Plot membrane potential
    ax[1].plot(syn_rec.detach().numpy(), c="tab:orange")
    ax[1].set_ylabel("Synaptic Conductance ($I_{syn}$)")
    plt.xlabel("Time step")

    # Plot membrane potential
    ax[2].plot(mem_rec.detach().numpy())
    ax[2].plot(mem_fmp_rec.detach().numpy(), c="tab:green", label="FMP")
    ax[2].set_ylabel("Membrane Potential ($U_{mem}$)")
    ax[2].axhline(y=1, alpha=0.25, linestyle="dashed", c="red", linewidth=2)
    ax[2].legend(loc = 'best')
    plt.xlabel("Time step")

    # Plot output spike using spikeplot
    splt.raster(spk_rec, ax[3], s=400, c="black", marker="|")
    plt.ylabel("Output spikes")
    ax[3].set_yticks([])

    plt.show()
    return



##########################################
#                                        #
#      PLOT FUNCTIONS OF SECTION 2       #
#                                        #
##########################################



def plot_results_21(time_steps, cur_in, spk_record, weight_history):
    # plot the results
    fig, ax = plt.subplots(3, figsize=(12,10), sharex=True, gridspec_kw = {'height_ratios': [1, 0.4, 1]})

    # raster plot of the input spikes
    splt.raster(cur_in, ax = ax[0], s=400, c="tab:pink", marker="|")
    ax[0].set_ylabel("Neuron index")
    ax[0].set_title("STDP Synapse Model With Input Spikes")

    # raster plot of the output spikes
    splt.raster(spk_record, ax = ax[1], s=400, c="black", marker="|")
    ax[1].set_ylabel("Neuron index")

    # plot of the weights evolution
    ax[2].plot(time_steps, weight_history)
    ax[2].set_ylabel("Synaptic Weight")
    ax[2].set_xlabel("Time step")

    plt.show()
    return



def plot_raster_weight_distr(cur_in, spk_record, weight_history, N_pre_correlated):
    # plot the input and output raster plots, the weights evolution and the final weights distribution
    fig, ax = plt.subplots(4, figsize=(12,15), gridspec_kw = {'height_ratios': [1, 0.2, 1, 1]})

    # raster plot of the input spikes
    splt.raster(cur_in, ax = ax[0], s=400, c="black", marker="|", alpha = 0.3)
    ax[0].set_ylabel("Neuron index")
    ax[0].set_title("STDP Synapse Model With Input Spikes")

    # raster plot of the output spikes
    splt.raster(spk_record, ax = ax[1], s=400, c="black", marker="|", alpha = 0.8, lw = 0.3)
    ax[1].set_ylabel("Neuron index")
    ax[1].set_title(f'output spike rate {spk_record.mean()}')

    # plot of the weights evolution
    cor_weights = pd.DataFrame(weight_history[:, :N_pre_correlated])
    uncor_weights = pd.DataFrame(weight_history[:, N_pre_correlated:])
    cor_weights.plot(ax = ax[2], legend = False, color = 'tab:red', alpha = 0.8, lw = 1.)
    uncor_weights.plot(ax = ax[2], legend = False, color = 'tab:blue', alpha = 0.2, lw = 1.)
    ax[2].set_ylabel("Synaptic Weight")
    ax[2].set_xlabel("Time step")

    # plot the final weights distribution
    time_step = -1
    w_min = np.min(weight_history[time_step,:])-0.1
    w_max = np.max(weight_history[time_step,:])+0.1
    width = (w_max - w_min)/51
    bins = np.arange(w_min, w_max, width)
    ax[3].hist(weight_history[time_step,:N_pre_correlated], bins = bins, color = 'tab:red', alpha = 0.8, label = 'correlated')
    ax[3].hist(weight_history[time_step,N_pre_correlated:], bins = bins, color = 'tab:blue', alpha = 0.2, label = 'uncorrelated')
    ax[3].set_xlabel("Synaptic Weight")
    ax[3].set_ylabel("Frequency")
    ax[3].legend(loc = 'best')

    plt.show()
    return



##########################################
#                                        #
#      PLOT FUNCTIONS OF SECTION 3       #
#                                        #
##########################################


def plot_results_31(dt, cur_in, pre_trace, mem_rec, spk_rec, post_trace, weight_history, N_pre, N_post, num_steps):
    # plot the input spikes, membrane potentials, output spikes, pre and post synaptic traces and the weights evolution
    fig, ax = plt.subplots(6, figsize=(12,20), sharex=True, gridspec_kw = {'height_ratios': [1, 1, 1, 0.4, 1, 1]})
    time_steps = np.arange(num_steps) * dt

    #input spikes
    height = ax[0].bbox.height
    ax[0].scatter(*torch.where(cur_in), s=2*height/N_pre, c="black", marker="|", lw = max(100/num_steps, 0.5))
    ax[0].set_ylim([0-0.5, N_pre-0.5])
    ax[0].set_ylabel("Neuron index")
    ax[0].set_title("Input Spikes")

    # pre synaptic traces
    ax[1].plot(time_steps, pre_trace.detach().numpy(), alpha = 0.8, lw = max(100/num_steps, 0.5))
    ax[1].set_ylabel("Pre-synaptic trace")
    ax[1].set_title("Pre-synaptic trace")


    # membrane potential
    ax[2].plot(time_steps, mem_rec.detach().numpy(), alpha = 0.8, lw = max(100/num_steps, 0.5))
    ax[2].hlines(1, 0, num_steps, color = 'red', linestyle = '--')
    ax[2].set_ylabel("Membrane Potential")
    ax[2].set_title("Membrane Potential and Output Spikes")

    # output spikes
    height = ax[2].bbox.height
    ax[3].scatter(*torch.where(spk_rec), s=2*height/N_post, c="black", marker="|", lw = max(100/num_steps, 0.5))
    ax[3].set_ylim([0-0.5, N_post-0.5])
    ax[3].set_ylabel("Neuron index")
    ax[3].set_xlabel("Time step")


    # post synaptic traces
    ax[4].plot(time_steps, post_trace.detach().numpy(), alpha = 0.8, lw = max(100/num_steps, 0.5))
    ax[4].set_ylabel("Post-synaptic trace")
    ax[4].set_xlabel("Time step")

    # plot of the weights evolution
    ax[5].plot(time_steps, weight_history.detach().numpy()[:,0,:], alpha = 0.8, lw = max(100/num_steps, 0.5))
    ax[5].set_ylabel("Synaptic Weight")
    ax[5].set_xlabel("Time step")

    plt.show()
    return



def plot_results_32(dt, 
                    cur_in, 
                    pre_trace, 
                    cond_rec, 
                    mem_rec, 
                    spk_rec, 
                    post_trace, 
                    weight_history, 
                    N_pre, 
                    N_pre_correlated, 
                    N_post, 
                    num_steps,
                    threshold_rec = None):
    # plot the input spikes, membrane potentials, output spikes, pre and post synaptic traces and the weights evolution
    fig, ax = plt.subplots(8, figsize=(12,30), gridspec_kw = {'height_ratios': [1, 1, 1, 1, 0.4, 1, 1, 1]})
    time_steps = np.arange(num_steps) * dt
    lw = max(100/num_steps, 0.5)

    #input spikes
    height = ax[0].bbox.height
    ax[0].scatter(*torch.where(cur_in), s=2*height/N_pre, c="black", marker="|", lw = lw)
    ax[0].set_ylim([0-0.5, N_pre-0.5])
    ax[0].set_ylabel("Neuron index")
    ax[0].set_title("Input Spikes")

    # pre synaptic traces
    ax[1].plot(time_steps, pre_trace, alpha = 0.8, lw = lw)
    ax[1].set_ylabel("Pre-synaptic trace")
    ax[1].set_title("Pre-synaptic trace")

    # neuron conductance
    if type(cond_rec) == torch.Tensor:
        ax[2].plot(time_steps, cond_rec, alpha = 0.8, lw = lw)
    else:
        ax[2].plot(time_steps, cond_rec, alpha = 0.8, lw = lw)
    ax[2].set_ylabel("Neuron conductance")
    ax[2].set_title("Neuron conductance")

    # membrane potential
    ax[3].plot(time_steps + 1, mem_rec, alpha = 0.8, lw = lw)
    if threshold_rec is not None:
        ax[3].plot(threshold_rec[:,0], color = 'red', linestyle = '--')
    else:
        ax[3].hlines(1, 1, num_steps + 1, color = 'red', linestyle = '--')
    ax[3].set_ylabel("Membrane Potential")
    ax[3].set_title("Membrane Potential and Output Spikes")

    # output spikes
    height = ax[4].bbox.height
    ax[4].scatter(*torch.where(spk_rec), s=2*height/N_post, c="black", marker="|", lw = lw)
    ax[4].set_ylim([0-0.5, N_post-0.5])
    spk_rec = spk_rec.float()
    ax[4].set_title(f"Mean output firing rate: {spk_rec.mean():.2f} ")
    ax[4].set_ylabel("Neuron index")
    ax[4].set_xlabel("Time step")

    # post synaptic traces
    ax[5].plot(time_steps + 1, post_trace, alpha = 0.8, lw = lw)
    ax[5].set_ylabel("Post-synaptic trace")
    ax[5].set_xlabel("Time step")

    # plot of the weights evolution
    W = weight_history[:,0,:]
    cor_weights = pd.DataFrame(W[:, :N_pre_correlated])
    uncor_weights = pd.DataFrame(W[:, N_pre_correlated:])
    cor_weights.plot(ax = ax[6], legend = False, color = 'tab:red', alpha = 0.8, lw = lw)
    uncor_weights.plot(ax = ax[6], legend = False, color = 'tab:blue', alpha = 0.2, lw = lw)
    ax[6].set_ylabel("Synaptic Weight")
    ax[6].set_xlabel("Time step")

    # plot the final weights distribution
    time_step = -1
    w_min = np.min(W[time_step,:])-0.01
    w_max = np.max(W[time_step,:])+0.01
    width = (w_max - w_min)/51
    bins = np.arange(w_min, w_max, width)
    ax[7].hist(W[time_step,:N_pre_correlated], bins = bins, color = 'tab:red', alpha = 0.8, label = 'correlated')
    ax[7].hist(W[time_step,N_pre_correlated:], bins = bins, color = 'tab:blue', alpha = 0.2, label = 'uncorrelated')
    ax[7].set_xlabel("Synaptic Weight")
    ax[7].set_ylabel("Frequency")
    ax[7].legend(loc = 'best')

    plt.show()
    return



def plot_results_33(dt, num_steps, cur_in, pre_trace, cond_rec, N_pre, mem_rec, spk_rec, post_trace, weight_history, N_post, N_pre_correlated):
    # plot the input spikes, membrane potentials, output spikes, pre and post synaptic traces and the weights evolution
    fig, ax = plt.subplots(8, figsize=(12,30), gridspec_kw = {'height_ratios': [1, 1, 1, 1, 0.4, 1, 1, 1]})
    time_steps = np.arange(num_steps) * dt
    lw = max(100/num_steps, 0.5)

    #input spikes
    height = ax[0].bbox.height
    ax[0].scatter(*torch.where(cur_in), s=2*height/N_pre, c="black", marker="|", lw = lw)
    ax[0].set_ylim([0-0.5, N_pre-0.5])
    ax[0].set_ylabel("Neuron index")
    ax[0].set_title("Input Spikes")

    # pre synaptic traces
    ax[1].plot(time_steps, pre_trace.detach().numpy(), alpha = 0.8, lw = lw)
    ax[1].set_ylabel("Pre-synaptic trace")
    ax[1].set_title("Pre-synaptic trace")

    # neuron conductance
    ax[2].plot(time_steps, cond_rec.detach().numpy(), alpha = 0.8, lw = lw)
    ax[2].set_ylabel("Neuron conductance")
    ax[2].set_title(f"Neuron conductance, mean: {cond_rec.mean():.2f}")

    # membrane potential
    ax[3].plot(time_steps + 1, mem_rec.detach().numpy(), alpha = 0.8, lw = lw)
    ax[3].hlines(1, 1, num_steps + 1, color = 'red', linestyle = '--')
    ax[3].set_ylabel("Membrane Potential")
    ax[3].set_title("Membrane Potential and Output Spikes")

    # output spikes
    height = ax[4].bbox.height
    ax[4].scatter(*torch.where(spk_rec), s=2*height/N_post, c="black", marker="|", lw = lw)
    ax[4].set_ylim([0-0.5, N_post-0.5])
    ax[4].set_title(f"Mean output firing rate: {spk_rec.mean():.2f} ")
    ax[4].set_ylabel("Neuron index")
    ax[4].set_xlabel("Time step")

    # post synaptic traces
    ax[5].plot(time_steps + 1, post_trace.detach().numpy(), alpha = 0.8, lw = lw)
    ax[5].set_ylabel("Post-synaptic trace")


    # plot of the weights evolution
    post_index=1
    W = weight_history.detach().numpy()[:,post_index,:]
    cor_weights = pd.DataFrame(W[:, :N_pre_correlated])
    uncor_weights = pd.DataFrame(W[:, N_pre_correlated:])
    #mask = np.isin(np.arange(N_pre), first_section_index)
    #cor_weights =pd.DataFrame(W[:, mask])
    #uncor_weights = pd.DataFrame(W[:, ~mask])
    cor_weights.plot(ax = ax[6], legend = False, color = 'tab:red', alpha = 0.8, lw = lw)
    uncor_weights.plot(ax = ax[6], legend = False, color = 'tab:blue', alpha = 0.2, lw = lw)
    ax[6].set_title(f"Synaptic Weight of Neuron {post_index}")

    # plot the final weights distribution
    time_step = -1
    w_min = np.min(W[time_step,:])-0.01
    w_max = np.max(W[time_step,:])+0.01
    width = (w_max - w_min)/51
    bins = np.arange(w_min, w_max, width)
    ax[7].hist(W[time_step,:N_pre_correlated], bins = bins, color = 'tab:red', alpha = 0.8, label = 'correlated')
    ax[7].hist(W[time_step,N_pre_correlated:], bins = bins, color = 'tab:blue', alpha = 0.2, label = 'uncorrelated')
    ax[7].set_title(f"Final Synaptic Weight Distribution of Neuron {post_index}")
    ax[7].set_ylabel("Frequency")
    ax[7].legend(loc = 'best')

    #plt.tight_layout()

    plt.show()
    return



def all_post_weights_evolution(weight_history, N_pre_correlated, block_of_index):

    # retrive num_steps, N_post, N_pre from weight_history
    num_steps = weight_history.shape[0]
    N_pre = weight_history.shape[2]
    N_post = weight_history.shape[1]

    lw = max(100/num_steps, 0.5)
    fig, ax = plt.subplots(6, figsize=(12,5*N_post), gridspec_kw = {'height_ratios': [1, 0.5]* N_post})
    
    # plot of the weights evolution
    ax_number = 0
    for post_index in range(N_post):
        
        
        # Modify the weights based on post_index
        W = weight_history.detach().numpy()[:, post_index, :]
        
        # Plot synaptic weights
        mask = np.isin(np.arange(N_pre), block_of_index[post_index])
        cor_weights = pd.DataFrame(W[:, mask])
        uncor_weights = pd.DataFrame(W[:, ~mask])
        cor_weights.plot(ax=ax[ax_number], legend=False, color='tab:red', alpha=0.8, lw=lw)
        uncor_weights.plot(ax=ax[ax_number], legend=False, color='tab:blue', alpha=0.2, lw=lw)
        ax[ax_number].set_title(f"Synaptic Weight of Neuron {post_index}")

        # Plot final weights distribution
        time_step = -1
        w_min = np.min(W[time_step, :]) - 0.01
        w_max = np.max(W[time_step, :]) + 0.01
        width = (w_max - w_min) / 51
        bins = np.arange(w_min, w_max, width)
        ax[ax_number + 1].hist(W[time_step, :N_pre_correlated], bins=bins, color='tab:red', alpha=0.8, label='correlated')
        ax[ax_number + 1].hist(W[time_step, N_pre_correlated:], bins=bins, color='tab:blue', alpha=0.2, label='uncorrelated')
        ax[ax_number + 1].set_title(f"Final Synaptic Weight Distribution of Neuron {post_index}")
        ax[ax_number + 1].set_ylabel("Frequency")
        ax[ax_number + 1].legend(loc='best')

        ax_number += 2

    plt.tight_layout()
    plt.show()
    return

































