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
from IPython.display import clear_output
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

importlib.reload(importlib.import_module('neurons'))
importlib.reload(importlib.import_module('learning_rules'))
importlib.reload(importlib.import_module('plot_utils'))
importlib.reload(importlib.import_module('experiments'))

global simulation
global Poisson_generator
global half_growing_rate
global random_shifted_trains
global random_offsets
global weight_initializer

from experiments import *
from neurons import *
from learning_rules import *
from plot_utils import *









#########################################
#                                       #
#         STANDARD PLOT UTILS           #
#                                       # 
#########################################




def raster_plot(
        pars, 
        post_syn_spk, # output spikes as array of 0,1 of shape (time_steps, N_post) or as list of spike times
        pre_syn_spk = None, # input spikes as array of 0,1 of shape (time_steps, N_pre) or as list of spike times
        pre_syn_plot = True, # if True plot the input spikes
        title = 'Raster plot',
        time_in_ms = False,
        y_2_label = 'Output Spikes',
        perturbation_sites = None):
    
    """
    function to plot the raster plot of the input and output spikes
    INPUTS:
    - pars:             parameters of the simulation
    - pre_syn_spk:      input spikes as array of 0,1 of shape (num_steps, N_pre) or as list of spike times
    - post_syn_spk:     output spikes as array of 0,1 of shape (num_steps, N_post) or as list of spike times
    - pre_syn_plot:     if True plot the input spikes, otherwise it plots only the output spikes
    - title:            title of the plot
    - time_in_ms:       if True the x axis is in ms, otherwise in time steps
    - y_2_label:        label for the y axis of the output spikes
    """

    if time_in_ms:
        dt=pars['dt']
        label_x = 'Time (ms)'
    else:
        dt=1
        label_x = 'Time steps'

    # useful values
    
    if np.ndim(post_syn_spk) == 1:
        N_post = 1
        post_syn_spk = np.expand_dims(post_syn_spk, axis=1)
    else:
        N_post = np.shape(post_syn_spk)[1]

    # Generate Plots
    
    if pre_syn_plot:
        N_pre = np.shape(pre_syn_spk)[1]
        height_ratio = int(min(N_pre/ N_post,10))
        fig, ax = plt.subplots(2, figsize=(15,10), sharex=True, gridspec_kw = {'height_ratios': [height_ratio,1]})


        # convert spike record in spike times
        # note: even with 10000 presynaptic neurons and 10000 time steps this is not an heavy operation
        if perturbation_sites is None:
            pre_syn_spk_times = [np.array(np.where(pre_syn_spk[:,i]==1)[0])*dt for i in range(N_pre)]
            var = len(pre_syn_spk_times)
        else:
            pre_syn_spk_1 = pre_syn_spk * (1-perturbation_sites)
            pre_syn_spk_2 = pre_syn_spk * perturbation_sites
            pre_syn_spk_times_1 = [np.array(np.where(pre_syn_spk_1[:,i]==1)[0])*dt for i in range(N_pre)]
            pre_syn_spk_times_2 = [np.array(np.where(pre_syn_spk_2[:,i]==1)[0])*dt for i in range(N_pre)]
            var = len(pre_syn_spk_times_1)

            
        if perturbation_sites is None:
            ax[0].eventplot(pre_syn_spk_times, colors='black', lineoffsets=1,linewidth=1, linelengths=0.8, orientation='horizontal')
        else:
            ax[0].eventplot(pre_syn_spk_times_1, colors='black', lineoffsets=1,linewidth=1, linelengths=0.8, orientation='horizontal')
            ax[0].eventplot(pre_syn_spk_times_2, colors='red', lineoffsets=1,linewidth=1, linelengths=0.8, orientation='horizontal')

        # set y axis ticks corresponding to the neurons
        if N_pre > 10:
            ax[0].set_yticks(np.arange(0, var+1, round(N_pre/10)))
        ax[0].set_ylabel("Input Spikes")
        ax[0].set_title(title)


        # Plot output spikes
        post_syn_spk_times = [np.array(np.where(post_syn_spk[:,i]==1)[0])*dt for i in range(N_post)]

        ax[1].eventplot(post_syn_spk_times, colors='black', lineoffsets=1,linewidth=1, linelengths=0.8, orientation='horizontal')   
        ax[1].set_yticks(np.arange(0, N_post, max(round(N_post/10),5)))
        ax[1].set_ylabel(y_2_label)
        ax[1].set_xlabel(label_x)

        plt.show()
    else:

        fig, ax = plt.subplots(1, figsize=(15,5))
        # Plot output spikes
        if type(post_syn_spk) is list:
            post_syn_spk_times = post_syn_spk
        else:
            post_syn_spk_times = [np.array(np.where(post_syn_spk[:,i]==1)[0])*dt for i in range(N_post)]
        ax.eventplot(post_syn_spk_times, colors='black', lineoffsets=1,linewidth=1, linelengths=0.8, orientation='horizontal')   
        ax.set_yticks(np.arange(0, N_post, max(round(N_post/10),5)))
        ax.set_ylabel(y_2_label)
        ax.set_xlabel(label_x)
        ax.set_title(title)
        plt.show()














def weights_plot(pars, weights_history,time_step = None, time_in_ms = False, title = None, subsampling = 1):
    """
    Plot the weights changes during the simulation through a colored image, a graph and a histogram
    INPUT:
    - pars:                parameters of the simulation
    - weights_history:     synaptic weights over time for one post_synaptic neuron (num_steps, N_pre)
    - time_step:           time step in which we want to see the weights distribution
    - time_in_ms:          if True the x axis is in ms, otherwise in time steps
    - title:               title of the plot
    - subsampling:         subsampling factor in time for the weights plot (for long simulations)
    """

    # check if we want the time in ms
    if time_in_ms:
        dt=pars['dt']
        label_x = 'Time (ms)'
    else:
        dt=1
        label_x = 'Time steps'

    # useful values
    num_steps = weights_history.shape[0]
    time_steps = np.arange(0, num_steps, 1)*dt

    # set the default time step
    if time_step is None:
        time_step = num_steps-1
    elif time_step > num_steps:
        print(f'Time step must be less than {num_steps}')
        return
    
    # initialize the plot
    # fig,ax = plt.subplots(2, figsize=(12, 10), gridspec_kw={'height_ratios': [2, 1]})#, sharex=True)
    fig,ax = plt.subplots(1, figsize=(12, 6))#, sharex=True)


    # # plot the weights history as a colored image
    # weights_df = pd.DataFrame(weights_history)
    # weights_df.plot(ax = ax[0], legend = False, color = 'tab:blue', alpha = 0.2, lw = 1.)
    # ax[0].set_ylabel("Synaptic Weight")
    # ax[0].set_xlabel("Time step")
    # if title:
    #     ax[0].set_title(title)
    # else:
    #     ax[0].set_title('Synaptic weights over time')
      # plot the weights history as a colored image
    weights_df = pd.DataFrame(weights_history)
    weights_df.plot(ax = ax, legend = False, color = 'tab:blue', alpha = 0.2, lw = 0.5)
    ax.set_ylabel("Synaptic Weight")
    ax.set_xlabel("Time step")
    if title:
        ax.set_title(title)
    else:
        ax.set_title('Synaptic weights over time')
    
    # # plot the weights history as a graph
    # ax[1].plot(time_steps[::subsampling], weights_history[ ::subsampling,:], lw=1., alpha=0.7)
    # ax[1].axvline(time_step, 0., 1., color='red', ls='--')
    # ax[1].set_xlabel(label_x)
    # ax[1].set_ylabel('Weight')

    # plot the weights distribution at a given time step
    # w_min = np.min(weights_history[time_step,:])-0.1
    # w_max = np.max(weights_history[time_step,:])+0.1
    # width = (w_max - w_min)/51
    # bins = np.arange(w_min, w_max, width)
    # #g_dis, _ = np.histogram(weights_history[time_step,:], bins)
    # #ax[1].bar(bins[1:], g_dis, color='b', alpha=0.5, width=width)
    # ax[1].hist(weights_history[time_step,:], bins, color='b', alpha=0.5, facecolor = '#2ab0ff', edgecolor='#169acf', linewidth=0.5)
    # ax[1].set_xlabel('weights ditribution')
    # ax[1].set_ylabel('Number')
    # ax[1].set_title(f'Time step: {time_step}')
    plt.tight_layout()
    plt.show()
    















def plot_traces(pars, pre_trace_record, post_trace_record, 
                pre_trace_index_list = None, 
                post_trace_index_list = None,
                time_in_ms = False):
    """
    Plot the pre and post synaptic traces

    INPUT:
    - pars: parameter of the simulation
    - pre_trace_record: synaptic traces of the pre-synaptic neurons (num_steps, N_pre)
    - post_trace_record: synaptic traces of the post-synaptic neurons (num_steps, N_post)
    - pre_trace_index_list: list of indexes of the pre-synaptic traces to higlight
    - post_trace_index_list: list of indexes of the post-synaptic traces to higlight
    - time_in_ms: if True the x axis is in ms, otherwise in time steps
    """

    # check if we want the time in ms
    if time_in_ms:
        dt=pars['dt']
        label_x = 'Time (ms)'
    else:
        dt=1
        label_x = 'Time steps'

    N_pre = pre_trace_record.shape[1]
    # check the index of the pre traces to higlight
    if pre_trace_index_list is None:
        pre_trace_index_list = [int(N_pre/4), int(N_pre/2), int(3*N_pre/4)]
    elif max(pre_trace_index_list) > pre_trace_record.shape[1]:
        print(f'Trace indexes must be less than {pre_trace_record.shape[1]}')
        return
    n_pre = len(pre_trace_index_list)

    N_post = post_trace_record.shape[1]
    # check the index of the post traces to higlight
    if post_trace_index_list is None:
        post_trace_index_list = [int(N_post/4), int(N_post/2), int(3*N_post/4)]
    elif max(post_trace_index_list) > post_trace_record.shape[1]:
        print(f'Trace indexes must be less than {post_trace_record.shape[1]}')
        return
    n_post = len(post_trace_index_list)
    
    # useful values
    num_steps = pre_trace_record.shape[0]
    time_steps = np.arange(0, num_steps, 1)*dt

    fig,ax = plt.subplots(2, figsize=(12, 10), sharex=True)

    ax[0].plot(time_steps, pre_trace_record, lw=1., alpha=0.05)
    df = pd.DataFrame(pre_trace_record[:,pre_trace_index_list])
    color = ['r','g', 'b']*int(n_pre/3)+['r']*int(n_pre%3)
    df.plot(ax=ax[0], color = color, lw=1., alpha=1, legend=False)
    #ax[0].plot(time_steps, , lw=1., alpha=1)#, color = 'r')
    ax[0].set_title(f'Pre-synaptic traces - {n_pre} higlighted')
    ax[0].set_xlabel(label_x)
    ax[0].set_ylabel('Pre traces')

    if n_post < 3:
        ax[1].plot(time_steps, post_trace_record, lw=1., alpha=0.7)
        ax[1].set_title('Post-synaptic traces')
        ax[1].set_xlabel(label_x)
        ax[1].set_ylabel('Post traces')
    else:
        ax[1].plot(time_steps, post_trace_record, lw=1., alpha=0.05)
        df = pd.DataFrame(post_trace_record[:,post_trace_index_list])
        color = ['r','g', 'b']*int(n_post/3)+['r']*int(n_post%3)
        df.plot(ax=ax[1], color = color, lw=1., alpha=1, legend=False)
        ax[1].set_title(f'Post-synaptic traces - {n_post} higlighted')

    plt.tight_layout()
    plt.show()





#########################################
#                                       #
#        INTERACTIVE PLOT UTILS         #
#                                       # 
#########################################




def LIF_interactive_plot(pars_function, N_pre, num_steps,
                         
                         type_parameters = 'simple',
                         N_post = 1,
                         post_index = 0,
                         manual_update = True, 
                         time_in_ms = False,
                         my_seed = 2024,
                        ):
    """
    ARGS:


    RETURN:
    """

    # check if we want the time in ms
    temp_pars = pars_function()
    if time_in_ms:
        dt=temp_pars['dt']
        label_x = 'Time (ms)'
    else:
        dt=1
        label_x = 'Time steps'

    # check the post index and eventually reset
    if post_index >= N_post:
        print(f'Post index must be less than {N_post}')
        post_index = N_post-1


    # time steps for the x axis
    time_steps = np.arange(0, num_steps, 1)*dt

    # set the seed
    np.random.seed(my_seed)

    def main_plot(
        I_type,
        rate,
        tau_m = 20,
        refractory_time = True, 
        dynamic_threshold = True, 
        hard_reset = True,
        show_fmp = False,
        show_raster = False,
        tau_thr = 20,
        ratio_thr = 1.5,
        t_ref = 2,
    ):
        pars = pars_function(type_parameters = 'simple',
                             tau_m = tau_m,
                             tau_thr = tau_thr,
                             ratio_thr = ratio_thr,
                             t_ref = t_ref,
                             refractory_time = refractory_time,
                             dynamic_threshold = dynamic_threshold,
                             hard_reset = hard_reset)

        dt = pars['dt']

        if I_type == 'Poiss':
            I = Poisson_generator(dt, rate = rate, n = N_pre, num_steps = num_steps)
        elif I_type == 'Const':
            I = np.ones((num_steps, N_pre)) 
        elif I_type == 'Half':
            I = half_growing_rate(dt, num_steps, N_pre, rate_ratio = 0.5) 
        elif I_type == 'Shift':
            I,_ = random_shifted_trains(dt, num_steps, N_pre, rate = rate)
        elif I_type == 'Offsets':
            I,_,_= random_offsets(dt, num_steps, N_pre, rate = rate)
        else:
            print('I_type not recognized')
            print('I_type must be one of the following: Poiss, Const, Half, Shift, Offsets')
            print('Returning the Random input')
            I = Poisson_generator(dt, rate, n = N_pre, num_steps = num_steps)
        
        W_init = weight_initializer(pars, N_post, I=I)
        neurons = simulation(pars, I, W_init = W_init,
                              neuron_type = LIFNeuron, N_post = N_post)

        neurons[post_index].plot_records(show_fmp = show_fmp)

        if show_raster:
            get_post_spk_trains = lambda neurons : np.array([neurons[i].get_records()['spk'] for i in range(len(neurons))]).T
            raster_plot(pars, pre_syn_spk=I, post_syn_spk=get_post_spk_trains(neurons), title = 'Raster plot of the input and output spikes')

        return 
    
    # WIDGET CONSTRUCTION

    rate_widget = widgets.FloatSlider(
            value=0.5,
            min=0,
            max=1,
            step=0.01,
            description='Rate',
            layout=widgets.Layout(width='600px'),
            tooltip = 'Rate of the input',
            continuous_update=False
        )
    
    I_type_widget = widgets.ToggleButtons(
            options=['Poiss', 'Const', 'Half', 'Shift', 'Offsets'],
            value='Poiss',
            description='Input type:',
            layout=widgets.Layout(width='800px'),
            disabled=False,
            tooltips =['Random Poisson trains', 'Constant', 'half growing rate', 'random shifted trains', 'random offsets']
        )
    
    tau_m_widget = widgets.FloatSlider(
            value=20,
            min=1,
            max=1000,
            step=1,
            description='tau_m',
            layout=widgets.Layout(width='600px'),
            tooltip = 'Membrane time constant',
            continuous_update=False
        )    

    tau_thr_widget = widgets.FloatSlider(
            value=20,
            min=1,
            max=1000,
            step=1,
            description='tau_thr',
            layout=widgets.Layout(width='400px'),
            tooltip = 'Threshold time constant',
            continuous_update=False
        )
    
    ratio_thr_widget = widgets.FloatSlider(
            value=1.5,
            min=1,
            max=7,
            step=0.1,
            description='ratio_thr',
            layout=widgets.Layout(width='400px'),
            tooltip = 'Threshold ratio',
            continuous_update=False
        )
    
    t_ref_widget = widgets.IntSlider(
            value=2,
            min=1,
            max=50,
            step=1,
            description='t_ref',
            layout=widgets.Layout(width='400px'),
            tooltip = 'Refractory time',
            continuous_update=False
        )
    
    refractory_time_widget = widgets.Checkbox(
            value=False,
            description='Refractory time',
            disabled=False,
            indent=False,
            layout=widgets.Layout(width='200px'),
        )
    
    dynamic_threshold_widget = widgets.Checkbox(
            value=False,
            description='Dynamic threshold',
            disabled=False,
            indent=False,
            layout=widgets.Layout(width='200px'),
        )
    
    hard_reset_widget = widgets.Checkbox(
            value=True,
            description='Hard reset',
            disabled=False,
            indent=False,
            tooltip = 'if false the membrane resets by subtraction of the threshold value',
            layout=widgets.Layout(width='200px'),
        )
    
    show_fmp_widget = widgets.Checkbox(
            value=False,
            description='Show FMP',
            disabled=False,
            indent=False,
            tooltip = 'if True the Free Membrane Potential is shown',
            layout=widgets.Layout(width='200px'),
        )
    
    show_raster_widget = widgets.Checkbox(
            value=False,
            description='Show Raster',
            disabled=False,
            indent=False,
            tooltip = 'if True the Raster plot is shown',
            layout=widgets.Layout(width='200px'),
        )

    my_layout.width = '600px'

    interactive_plot = widgets.interactive(
        main_plot,
        {'manual': manual_update, 'manual_name': 'Update plot'},
        type_parameters = widgets.fixed(type_parameters),
        I_type = I_type_widget,
        rate = rate_widget,
        tau_m = tau_m_widget,
        tau_thr = tau_thr_widget,
        ratio_thr = ratio_thr_widget,
        t_ref = t_ref_widget,
        refractory_time = refractory_time_widget,
        dynamic_threshold = dynamic_threshold_widget,
        hard_reset = hard_reset_widget,
        show_fmp = show_fmp_widget,
        show_raster = show_raster_widget
    )

    pre = interactive_plot.children[:3]
    controls_neuron = widgets.HBox(interactive_plot.children[3:8])
    controls_1 = widgets.HBox(interactive_plot.children[8:10])
    output = interactive_plot.children[10:]


    final_widget = widgets.VBox([*pre, controls_neuron, controls_1, *output])

    return final_widget





def STDP_interactive_plot(pars_function, I, N_post = 10,
                          type_parameters = 'simple',
                          neuron_type = LIFNeuron,                
                          manual_update = True, 
                          time_in_ms = False,
                          highlight = [],
                          my_seed = 2024,
                          ):
    """
    Interactive plot to simulate the STDP learning rule given an input.

    ARGS:
    - pars_function: function that returns the parameters of the simulation (interactive changable):

        parameters of the neurons interactive changable:
        - dynamic_threshold: if True the threshold of the neurons is dynamic
        - hard_reset: if True the neurons have a hard reset
        - tau_m: membrane time constant
        parameters of the weight update rule  interactive changable:
        - A_plus and A_minus: parameters of the STDP rule
        - tau_plus and tau_minus: time constants of the STDP rule

    - I: input to the network (num_steps, N_pre) array of 0,1 fixed
    - N_post: number of post-synaptic neurons
    - manual_update: if True the plot is updated only when the button is pressed
    - time_in_ms: if True the x axis is in ms, otherwise in time steps
    - highlight: list of indexes of the pre-synaptic neurons to higlight in the plots
    - my_seed: seed for the random number generator

    RETURN:
    Interactive demo, Visualization of synaptic weights and neurons traces
    """
    num_steps = I.shape[0]
    N_pre = I.shape[1]
    # check if we want the time in ms
    temp_pars = pars_function()
    if time_in_ms:
        dt=temp_pars['dt']
        label_x = 'Time (ms)'
    else:
        dt=1
        label_x = 'Time steps'

    # time steps for the x axis
    time_steps = np.arange(0, num_steps, 1)*dt

    # set the seed
    np.random.seed(my_seed)

    def main_plot(
        type_parameters = 'simple',
        time_step = 1,
        post_index = 0,  
        dynamic_threshold = False,
        hard_reset = True,
        tau_m = 20,
        A_plus = 0.01,
        A_minus = 0.011,
        tau_plus = 20,
        tau_minus = 20,
    ):
        
        
        # inlcude all the plot from the post_synaptic neuron
        neuron_plot = 'Mem & Spk' # or 'Spikes'

        pars = pars_function(type_parameters = type_parameters,
                             A_plus = A_plus, A_minus = A_minus, 
                             tau_m = tau_m,
                             tau_plus = tau_plus, tau_minus = tau_minus,
                             dynamic_threshold = dynamic_threshold,
                             hard_reset = hard_reset,
                             )
        
        W_init = weight_initializer(pars, N_post, I=I, type_init = 3)

        neurons, syn = simulation(pars, I, neuron_type = neuron_type, weight_rule = STDP_synapse, N_post = N_post, W_init = W_init)
                
        selected_neuron = neurons[post_index]
        if neuron_plot == 'Spikes':
            post_spk_train = selected_neuron.get_records()['spk']
            n_figure = 3
            height_ratios = [4, 3, 1]
            fig_height = 10
        elif neuron_plot == 'Mem & Spk':
            post_spk_train = selected_neuron.get_records()['spk']
            mem = selected_neuron.get_records()['mem']
            if selected_neuron.dynamic_threshold:
                thr_records = selected_neuron.get_records()['thr']
            n_figure = 4
            height_ratios = [4, 3, 3, 1]
            fig_height = 12
        else:
            n_figure = 2
            height_ratios = [4, 3]
            fig_height = 8

        weights_history = syn.get_records()['W']
        
        s = 1 # subsampling seems not be usefull
        
        fig,ax = plt.subplots(n_figure, figsize=(10, fig_height), gridspec_kw={'height_ratios': height_ratios})#, sharex=True)

        # plot the weights
        x = time_steps[::s]
        y = weights_history[1 ::s,post_index,:]
        if len(highlight) > 0:
            alpha = 1/(N_pre-len(highlight)) + 0.05
            alpha1 = alpha/N_pre * len(highlight) + 0.1
            alpha2 = alpha/N_pre * (N_pre-len(highlight))
            mask = np.isin(np.arange(N_pre), highlight)
            df1 = pd.DataFrame(y[:,mask])
            df1.plot(ax=ax[0], color = 'r', lw=1., alpha=alpha1, legend=False)
            df2 = pd.DataFrame(y[:,~mask])  
            df2.plot(ax=ax[0], color = 'b', lw=1., alpha=alpha2, legend=False)
        else:
            ax[0].plot(x, y, lw=1.)
        ax[0].axvline(time_step, 0., 1., color='red', ls='--')
        if pars['constrain'] == 'Dynamic':
            ax[0].axhline(syn.w_max, 0., 1., color='green', ls='--')
            ax[0].axhline(syn.w_min, 0., 1., color='green', ls='--')
        ax[0].set_xlabel(label_x)
        ax[0].set_ylabel('Weight')


        # plot the weights distribution
        w_min = np.min(weights_history[time_step,:])-0.1
        w_max = np.max(weights_history[time_step,:])+0.1
        width = (w_max - w_min)/51
        bins = np.arange(w_min, w_max, width)
        if len(highlight) > 0:
            ax[1].hist(weights_history[time_step,post_index,mask], bins, color='r', alpha=0.8,  linewidth=0.5)
            ax[1].hist(weights_history[time_step,post_index,~mask], bins, color='b', alpha=0.4, linewidth=0.5)
        else:
            ax[1].hist(weights_history[time_step,post_index,:], bins, color='b', alpha=0.5, facecolor = '#2ab0ff', edgecolor='#169acf', linewidth=0.5)
        ax[1].set_xlabel(f'weights ditribution of neuron {post_index} at time step {time_step}')
        ax[1].set_ylabel('Number')
        #ax[1].set_title(f'Time step: {time_step}')

        if neuron_plot == 'Mem & Spk':
            # plot the membrane potential
            ax[2].plot(time_steps, mem)
            ax[2].set_ylabel("Membrane Potential ($U_{mem}$)")
            if selected_neuron.dynamic_threshold:
                ax[2].plot(time_steps, thr_records, c="red", linestyle="dashed", alpha=0.7, label="Threshold")
            else:
                ax[2].axhline(y=selected_neuron.threshold, alpha=0.5, linestyle="dashed", c="red", linewidth=2, label="Threshold")
            ax[2].legend( loc="best")
            plt.xlabel(label_x)

            # plot the spikes
            ax[3].eventplot(np.array(np.where(post_spk_train==1))*dt, color="black", linelengths=0.5, linewidths=1)
            ax[3].set_xlim(ax[0].get_xlim())
            ax[3].set_ylabel("Output spikes")
            ax[3].set_xlabel(label_x)
            plt.yticks([])
        elif neuron_plot == 'Spikes':
            # plot the spikes
            spk_times = np.array(np.where(post_spk_train==1))*dt
            ax[2].eventplot(spk_times, color="black", linelengths=0.5, linewidths=1)
            ax[2].set_xlim(ax[0].get_xlim())
            ax[2].set_ylabel("Output spikes")
            ax[2].set_xlabel(label_x)
            plt.yticks([])
        else:
            print(neuron_plot)

        plt.tight_layout()
        plt.show()
        return 


    # WIDGET CONSTRUCTION

    
    A_plus_widget = widgets.FloatLogSlider(
         value=0.008*0.024,
         base=2,
         min=-10, # max exponent of base
         max=10, # min exponent of base
         step=0.5, # exponent step
         description='A_plus',
         layout=widgets.Layout(width='400px'),
         tooltip = 'A_plus value for STDP',
         continuous_update=False
    )
    A_minus_widget = widgets.FloatLogSlider(
         value=0.0088*0.024,
         base=2,
         min=-10, # max exponent of base
         max=10, # min exponent of base
         step=0.5, # exponent step
         description='A_minus',
         layout=widgets.Layout(width='400px'),
         tooltip = 'A_minus value for STDP',
         continuous_update=False
    )
    tau_m_widget = widgets.FloatSlider(
         value=10,
         min=0.1,
         max=1000,
         step=1,
         description='tau_m',
         layout=widgets.Layout(width='600px'),
         tooltip = 'Membrane time constant',
    )
    tau_plus_widget = widgets.FloatSlider(
         value=20,
         min=1,
         max=1000,
         step=1,
         description='tau_plus',
         layout=widgets.Layout(width='400px'),
         tooltip = 'STDP potentiation time constant',
    )
    tau_minus_widget = widgets.FloatSlider(
         value=20,
         min=0.1,
         max=1000,
         step=1,
         description='tau_minus',
         layout=widgets.Layout(width='400px'),
         tooltip = 'STDP depression time constant',
    )


    my_layout.width = '600px'
    interactive_plot = widgets.interactive(
        main_plot,
        {'manual': manual_update, 'manual_name': 'Update plot'},
        type_parameters = widgets.fixed(type_parameters),
        time_step=widgets.IntSlider(
            min=0, 
            max=num_steps, 
            value = num_steps-1,
            step=num_steps//100,
            description = 'Time step',
            layout=my_layout,
            continuous_update=False
        ),
        post_index = widgets.IntSlider(
            min = 0,
            max = N_post-1,
            step = 1,
            layout=my_layout,
            continuous_update=False
        ),
        dynamic_threshold = widgets.Checkbox(
            value = False,
            description='Dynamic threshold for the neurons',
            disabled=False,
            indent=False
        ),
        hard_reset = widgets.Checkbox(
            value=False,
            description='Hard reset for the neurons',
            disabled=False,
            indent=False
        ),
        tau_m = tau_m_widget,
        A_plus = A_plus_widget,
        A_minus = A_minus_widget,
        tau_plus = tau_plus_widget,
        tau_minus = tau_minus_widget,
    )
    
    pre = interactive_plot.children[:2]
    controls_neuron = widgets.HBox(interactive_plot.children[2:4])
    controls_tau_m = interactive_plot.children[4]
    controls_A = widgets.HBox(interactive_plot.children[5:7])
    controls_tau_stdp = widgets.HBox(interactive_plot.children[7:9])
    output = interactive_plot.children[9:]

    final_widget = widgets.VBox([*pre, controls_neuron, controls_tau_m, controls_A , controls_tau_stdp, *output])
    #final_widget = interactive_plot
    #output = interactive_plot.children[-1]
    #output.layout.height = '350px'
    return final_widget

