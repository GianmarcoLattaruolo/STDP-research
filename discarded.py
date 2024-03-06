
# import base libraries
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import shutil

cwd = os.getcwd()

#torch
import torch
import torch.nn as nn
import torch.nn.functional as F

#snntorch
import snntorch as snn
from snntorch import utils
from snntorch import spikegen
from snntorch import spikeplot as splt
from snntorch import surrogate
from snntorch import backprop
from snntorch import functional as SF


def F(delta_t):
    # auxiliary function for each pair of spikes
    # I should improve this values
    tau_minus = 20
    tau_plus = 20
    A_minus = 0.0088
    A_plus = 0.008
    if delta_t <= 0:  # delta_t = t_pre-t_post
        return A_plus * np.exp(delta_t/tau_minus)
    else:
        return -A_minus * np.exp(-delta_t/tau_plus)

class two_neurons():
    """ 
    I want to simple simulate a STDP rule between two LIF neurons for certain time steps. 
    """
    def __init__(self, initial_weight = 0.5, beta = [0.8, 0.8] ):
        # we simply define the two beta values for the two neurons
        self.first_neuron = snn.Leaky(beta = beta[0])
        self.second_neuron = snn.Leaky(beta = beta[1])
        # we introduce the list to record the membrane potentials and spikes
        self.mem_rec1 = []
        self.spk_rec1 = []
        self.mem_rec2 = []
        self.spk_rec2 = []
        # intial values of membrane and spike
        self.initial_spk = torch.zeros(1)
        self.initial_mem = torch.zeros(1)
        # introduce the initial synapse weight
        self.synapse_weight = initial_weight    

    # we define the simulation function
    def forward(self, input):
        # input: a generic list of input current
        self.time_steps = len(input)
        mem1 = self.initial_mem
        mem2 = self.initial_mem
        # run a forwar pass through the neurons for the entire input
        for step in range(self.time_steps):

            # we first run the first neuron
            spk1, mem1 = self.first_neuron(input[step], mem1)

            # we multiply the spike with the weight and feed it to the second neuron
            synapse_pulse = spk1 * self.synapse_weight
            spk2 , mem2 = self.second_neuron(synapse_pulse, mem2)

            # append the values to the lists
            self.mem_rec1.append(mem1)
            self.spk_rec1.append(spk1)
            self.mem_rec2.append(mem2)
            self.spk_rec2.append(spk2)

        return 

    def visualize_forward(self, input):
        # convert spikes lists to tensors
        spk_rec1_t = torch.stack(self.spk_rec1).squeeze()
        spk_rec2_t = torch.stack(self.spk_rec2).squeeze()

        # visualize the forward pass in a simple plot
        fig, ax = plt.subplots(5, figsize=( 6, 8),
                                sharex=True,
                                gridspec_kw = {'height_ratios': [0.4, 1, 0.4, 1, 0.4]})
        
        # Plot input current
        ax[0].plot(input, c="tab:orange")
        #ax[0].set_ylim([0, ylim_max1])
        #ax[0].set_xlim([0, 200])
        ax[0].set_ylabel("Input Current ($I_{in}$)")
        ax[0].set_title("Simple forward pass of two neurons")

        # Plot the membrane potential of the first neuron
        ax[1].plot(self.mem_rec1)
        #ax[1].set_ylim([0, 10])
        ax[1].set_ylabel("Membrane Potential Of I neuron")
        thr_line = self.first_neuron.threshold  #.detach().numpy()
        ax[1].axhline(y=thr_line, alpha=0.25, linestyle="dashed", c="black", linewidth=2)
    

        # PLot the spike of the first neuron
        splt.raster(spk_rec1_t, ax[2], s=self.time_steps, c="black", marker="|")
        plt.ylabel("Spikes of I neuron")
        plt.yticks([])

        # Plot the membrane potential of the second neuron
        ax[3].plot(self.mem_rec2)
        #ax[3].set_ylim([0, 10])
        ax[3].set_ylabel("Membrane Potential Of II neuron")
        thr_line = self.second_neuron.threshold
        ax[3].axhline(y=thr_line, alpha=0.25, linestyle="dashed", c="black", linewidth=2)
        plt.xlabel("Time step")

        # PLot the spike of the second neuron
        splt.raster(spk_rec2_t, ax[4], s=self.time_steps, c="black", marker="|")
        plt.ylabel("Spikes of II neuron")
        plt.yticks([])

    

        return fig, ax
    
    def STDP_simulation(self, input, number_of_updates = 10):
        # we want to simulate the STDP rule between the two neurons
        # we implement a brute force method first

        weights = []
        for i in range(number_of_updates):
            # append the new weights
            weights.append(self.synapse_weight)

            # first reset the membrane potentials and spikes
            self.list_reset()
            
            # run a forward pass
            self.forward(input)

            # update the weight according to the time of the spikes
            spk_train1 = self.spk_rec1
            spk_train2 = self.spk_rec2
            update = self.STDP_rule(spk_train1, spk_train2)
            self.synapse_weight += update

        # append the last weight 
        weights.append(self.synapse_weight)

        return weights
    
    def list_reset(self):
        # we reset the lists
        self.mem_rec1, self.spk_rec1, self.mem_rec2, self.spk_rec2 = [],[],[],[]
        return

    def STDP_rule(self, spk_pre, spk_post):
        # we implement a simple STDP rule
        weight_update = 0
        for i in np.where(np.array(spk_pre).T[0]==1)[0]:
            for j in np.where(np.array(spk_post).T[0]==1)[0]:
                delta_t = i - j  # pre - post
                weight_update += F(i-j)

        return weight_update



class N_to_1_neurons():
    def __init__(self, initial_weights = 0.5, beta_values = [0.8, 0.8], neurons_in_first_layers = 10):
        self.N = neurons_in_first_layers
        self.fc1 = nn.Linear(self.N,1)
        self.lif1 = snn.Leaky(beta = beta_values[0])
        self.lif2 = snn.Leaky(beta = beta_values[1])

        # initialize hidden states
        self.mem1 = self.lif1.init_leaky()
        self.mem2 = self.lif2.init_leaky()

        # initialize synapse weights
        self.synapse_weight = initial_weights*torch.ones(self.N)

        # we introduce the list to record the membrane potentials and spikes
        self.mem_rec1 = []
        self.spk_rec1 = []
        self.mem_rec2 = []
        self.spk_rec2 = []

    def forward(self, input):
        self.time_steps = input.shape[0]
        mem1 = self.mem1
        mem2 = self.mem2
        for step in range(self.time_steps):

            # we first run the first neuron
            spk1, mem1 = self.lif1(input[step], mem1)

            # we multiply the spike with the weight and feed it to the second neuron
            synapse_pulse = torch.dot(spk1 ,self.synapse_weight)
            spk2 , mem2 = self.lif2(synapse_pulse, mem2)

            # append the values to the lists
            self.mem_rec1.append(mem1)
            self.spk_rec1.append(spk1)
            self.mem_rec2.append(mem2)
            self.spk_rec2.append(spk2)

    def visualize_forward(self, input):
        fig, ax = plt.subplots(4, figsize=( 6, 8),
                                sharex=True,
                                gridspec_kw = {'height_ratios': [1, 1, 1, 0.3 ]})
        
        # Plot input current
        ax[0].plot(input.detach().numpy())
        #ax[0].set_ylim([0, ylim_max1])
        #ax[0].set_xlim([0, 200])
        ax[0].set_ylabel("Input Current ($I_{in}$)")

        # Plot the first spikes train
        spk_rec1_t = torch.stack(self.spk_rec1)
        splt.raster(spk_rec1_t, ax[1], s=self.time_steps, c="black", marker="|")
        plt.ylabel("Spikes from first layer")
        plt.yticks([])

        # Plot the membrane potential of the output neuron
        ax[2].plot(torch.stack(self.mem_rec2).detach().numpy())
        ax[2].set_ylabel("Membrane Potential Of II neuron")
        thr_line = self.lif2.threshold
        ax[2].axhline(y=thr_line, alpha=0.25, linestyle="dashed", c="black", linewidth=2)
        plt.xlabel("Time step")

        # PLot the spike of the output neuron
        spk_rec2_t = torch.stack(self.spk_rec2)
        splt.raster(spk_rec2_t, ax[3], s=self.time_steps, c="black", marker="|")
        plt.ylabel("Spikes of II neuron")
        plt.yticks([])

        # show the figure
        plt.show()

        return fig, ax

    def STDP_simulation(self, input, number_of_updates = 10):
        # list to store all the subsequent weights updates
        weights = []
        for i in range(number_of_updates):
            # append the new weights
            weights.append(self.synapse_weight)

            # first reset the membrane potentials and spikes
            self.list_reset()
            
            # run a forward pass
            self.forward(input)

            # update the weight according to the time of the spikes
            spk_train1 = self.spk_rec1
            spk_train2 = self.spk_rec2
            update = self.STDP_rule(spk_train1, spk_train2)
            self.synapse_weight = weights[i] + update

        # append the last weight series
        weights.append(self.synapse_weight)


        return weights

    def list_reset(self):
        # we reset the lists
        self.mem_rec1, self.spk_rec1, self.mem_rec2, self.spk_rec2 = [],[],[],[]
        return

    def STDP_rule(self, spk_pre, spk_post):
        # we implement a simple STDP rule
        weight_update = torch.zeros(self.N)
        # convert the spike trains into tensor
        spk_pre = torch.stack(spk_pre)   # spk_pre is a tensor of shape (time_steps, N_neurons)
        spk_post = torch.stack(spk_post)

        # iterate over the spike times of the output neuron
        for j in np.where(spk_post.detach().numpy()==1)[0]:
            # iterate over the neuron in the first layer
            for n_neuron in range(self.N):
                # consider the spikes of the n_neuron
                spk_pre_n = spk_pre[:,n_neuron]
                # iterate over the spike times of the n_neuron
                for i in np.where(spk_pre_n.detach().numpy()==1)[0]:
                    delta_t = i - j  # pre - post
                    weight_update[n_neuron] += F(i-j)

        return weight_update






def test_two_neuron(input):
    #input = torch.zeros(40)
    #input[10:20] = 0.5
    #input[30:40] = 0.5
    #input = np.cos(np.linspace(0, 2*np.pi, 40))
    #input = spikegen.rate_conv(torch.rand((200, 784)))


    # we define the two neuron
    small_snn = two_neurons(initial_weight = 0.5, beta = [0.8, 0.8] )
    small_snn.forward(input)
    fig1, ax1 = small_snn.visualize_forward(input)
    # show the figure
    plt.show()
    weights = small_snn.STDP_simulation(input, number_of_updates = 10)
    small_snn.list_reset()
    small_snn.forward(input)
    fig2, ax2 = small_snn.visualize_forward(input)
    # show the new figure
    plt.show()



def test_N_to_1():
# first define the input current
    time_steps = 40 # twice the pattern
    N_neurons = 10
    n_updates = 100
    # half of the neurons have a random input
    input_random = np.random.randn(time_steps, N_neurons//2)/2
    # half of the neurons have a input with a pattern
    pattern = np.array([[1,2,4,8,16,32,64,128,0,128,0,0,0,0,0,0,0,0,0,0]])/15
    pattern = np.array([pattern, pattern]).reshape(1,-1)/5
    pattern = np.repeat(pattern, repeats=N_neurons//2, axis=0).T
    pattern = pattern + np.random.randn(time_steps, N_neurons//2)/10
    # concatenate the input
    input = np.concatenate([input_random, pattern], axis=1)
    input = torch.from_numpy(input).float()
    
    # create the simple snn N to 1
    my_snn = N_to_1_neurons(neurons_in_first_layers = N_neurons)
    # run a forward pass
    my_snn.forward(input)
    # visualize the results
    #_,_ = my_snn.visualize_forward(input)

    # run the STDP simulation
    weights = my_snn.STDP_simulation(input, number_of_updates = n_updates)
    # reset the list
    my_snn.list_reset()
    # run a forward pass
    my_snn.forward(input)
    # visualize the results
    #_,_ = my_snn.visualize_forward(input)


    # visualize the history of weights
    history_weights = torch.stack(weights).detach().numpy()
    plt.plot(history_weights)
    # show a little number for each line  on the graph
    for i in range(history_weights.shape[1]):
        plt.text(n_updates, history_weights[n_updates,i], str(i))
    plt.show()

    return my_snn, history_weights



# now I should optimize the STDP rule to scale my experiments


if __name__ =='__main__':

    my_snn, history_weights = test_N_to_1()

    print()

def base_simulation(
        pars,
        spk_input, # input spike train, a numpy vector of shape (time_steps, N_pre)
        neuron_type, # class for the neuron type
        weight_rule = None, # class for the weights update
        N_pre = 100, # number of pre-synaptic neurons
        W_init = None, # initial weights
        neuron_params = {}, # parameters for the neuron model
        weight_update_params = {}, # parameters for the weight update
):
    N_post = 1 # this is the base simulation, just one output neuron
    num_steps = np.shape(spk_input)[0]

    # Initialize the post-syn neuron
    my_neuron = neuron_type(pars, **neuron_params)
    

    # Check if the given inital weight are valid
    if W_init is None:
        W_init = np.random.rand(N_post, N_pre)
    else:
        assert np.shape(W_init)[1] == N_pre, 'W must have the same number of rows as the number of pre-synaptic neurons'

    # Initialize the synapses with the given update rule
    if weight_rule is not None:
        my_synapses = weight_rule(pars, N_pre, N_post, W_init = W_init,**weight_update_params)
    else:
        W = W_init 

    # start the simulation
    for t in range(num_steps):
        
        # get the spikes from the pre-synaptic neurons at time t
        pre_syn_spikes = spk_input[t,:] 
        # get the weights from the synapse
        if weight_rule is not None:
            W = my_synapses.W
        
        # compute the input current for the postsynaptic neuron
        #I = np.dot(pre_syn_spikes, W[:,0])
        I = (W @ pre_syn_spikes)[0]

        # run the neuron model
        _ , spk = my_neuron.forward(I)

        # update the weights
        if weight_rule is not None:
            spikes = [pre_syn_spikes, spk]
            my_synapses.update_weights(spikes)

    if weight_rule is not None:
        return my_neuron, my_synapses
    else:
        return my_neuron
    







    
def LIF(
    pars,                        # parameters dictionary
    I_inj,                       # input current [pA]
    mem = None,                  # membrane potential [mV]
    refractory_time = False,     
    dynamic_threshold = False,   
    hard_reset = True,           
    noisy_input = False,
    **kwargs,
):
    """
    INPUTS:
    - pars: parameter dictionary
    - I_inj: input current [pA]
    - mem: membrane potential [mV]
    - refractory_time: boolean, if True the neuron has a refractory period
    - dynamic_threshold: boolean, if True the threshold is dynamic
    - hard_reset: boolean, if True the reset is hard
    - noisy_input: boolean, if True the input is noisy
    
    REUTRNS:
    - membrane potential at the next time step
    - spikes produced by the neuron (0 or 1)
    """
    
    #retrive parameters
    tau_m = pars.get('tau_m', 10)
    R = pars.get('R', 0.1)
    U_resting = pars.get('U_resting', -75)
    threshold = pars.get('threshold', -55)
    dt = pars.get('dt', 0.1)
    if mem is None:
        mem = pars.get('U_init', -75)
    if noisy_input:
        sigma = pars.get('sigma', 1)
        I_inj += np.random.normal(0, sigma, 1)

    #mem = mem + (dt/tau_m)*(U_resting-mem + I_inj*R)
    spk = mem > threshold
    if spk:
        if hard_reset:
            mem = pars['U_reset']
        else:
            mem = mem - (threshold-U_resting)

    mem = (1-dt/tau_m)*mem + dt/tau_m * U_resting + dt/tau_m  * R * I_inj

    return mem, int(spk)



def Poisson_neuron(pars, # parameters dictionary
        I_inj, # input current [pA] 
        mem = None, # membrane potential [mV]
        refractory_time = False,
        dynamic_threshold = False,
):
    """ Simulate a Poisson neuron

    INPUT:
    - pars                : parameter dictionary
    - I_inj               : input current [pA]
    - mem                 : membrane potential [mV]
    - refractory_time     : boolean, if True the neuron has a refractory period 
    - dynamic_threshold   : boolean, if True the neuron has a dynamic threshold

    RETURNS:
    - mem                 : membrane potential for the next time step
    - spk                 : output spike  (0 if no spike, 1 if spike)

    """

    # retrive parameters
    tau_m = pars.get('tau_m', 10)
    R = pars.get('R', 10)   
    U_resting = pars.get('U_resting', -75)
    threshold = pars.get('threshold', -55)
    dt = pars.get('dt', 0.1)
    alpha = pars.get('alpha', 0.1)
    if mem is None:
        mem = pars.get('U_init', -75)
        
    mem = mem + (dt/tau_m)*(U_resting-mem + I_inj*R)     
    rate = alpha * (mem - threshold)
    spk = 1.*  np.random.random() < rate 
    if spk:
        mem = pars['U_resting']

    return mem, spk





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




    def simple_pars(**kwargs):
    '''
    Define the default parameters
    values come from COURSE 2 
    '''

    pars = {}

    # typical neuron parameters
    pars['threshold'] = 1.    # spike threshold [mV]
    pars['tau_m'] = 10.         # membrane time constant [ms]
    pars['R'] = 1            # leak resistance [Ohm] with this resistance input current must be of the order of 100 mV
    pars['U_init'] = 0.       # initial potential [mV]
    pars['U_reset'] = 0.      # reset potential [mV]
    pars['U_resting'] = 0.    # leak reversal potential [mV]
    pars['t_ref'] = 2.          # refractory time (ms)

    # in the case of dynamic threshold
    pars['tau_thr'] = 20         # threshold time constant [ms]
    pars['ratio_thr'] = 1.1     # relative increment in the threshold due to a spike

    # in the case of soft reset
    ## ?? some way to lower the membrane potential not to a constant value

    # random seed
    pars['my_seed'] = 42

    # time steps
    pars['dt'] = 1             # simulation time step [ms]

    # for Poisson models
    pars['alpha'] = 0.1          # scaling factor for the membrane to the rate

    # STDP parameters
    pars['A_plus'] = 0.008                   # magnitude of LTP
    pars['A_minus'] = pars['A_plus'] * 1.10  # magnitude of LTD 
    pars['tau_plus'] = 20                    # LTP time constant [ms]
    pars['tau_minus'] = pars['tau_plus']     # LTD time constant [ms]

    # weight parameters
    pars['w_max'] = 1.            # maximum weight
    pars['w_min'] = 0.            # minimum weight


    # external parameters if any #
    for k in kwargs:
        pars[k] = kwargs[k]

    return pars

s_pars = simple_pars()



    w_max_widget = widgets.FloatSlider(
         value=5,
         min=0,
         max=60,
         step=1,
         description='w_max',
         layout=widgets.Layout(width='400px'),
         tooltip = 'Maximum weight value  for STDP',
         continuous_update=False
    )
    w_min_widget = widgets.FloatSlider(
         value=0,
         min=-40,
         max=20,
         step=1,
         description='w_min',
         layout=widgets.Layout(width='400px'),
         tooltip = 'Minimum weight value  for STDP',
         continuous_update=False
    )


    
    R_widget = widgets.FloatLogSlider(
        value = 1,
        base = 2,
        min = -10,
        max = 6,
        step = 0.5,
        description = 'R',
        layout = widgets.Layout(width='400px'),
        tooltip = 'Resistence of the membrane',
        continuous_update=False
    )


constrain = widgets.ToggleButtons(
            options = ['None','Hard', 'Dynamic'],
            value = 'None',
            description = 'Constrain on the weights',
            disabled = False,
            button_style='', # 'success', 'info', 'warning', 'danger' or ''
            tooltips = ['No constrain', 'Hard constrain', 'Dynamic constrain'],
        )

        neuron_plot = widgets.ToggleButtons(
            options = ['None', 'Spikes', 'Mem & Spk'],
            value = 'None',
            description = 'Selected Neuron plots',
            disabled = False,
            button_style='', # 'success', 'info', 'warning', 'danger' or ''
            tooltips = ['No neuron plot', 'Spikes train in output plot', 'Membrane and spikes plots'],
        ),

        short_memory_trace = widgets.Checkbox(
            value=False,
            description='Short memory trace for STDP',
            disabled=False,
            indent=False
        ),


    dynamic_weight_exponent_widget = widgets.FloatLogSlider(
         value=0.01,
         base=2,
         min=-10, # max exponent of base
         max=2, # min exponent of base
         step=0.5, # exponent step
         description='Exponent',
         layout=widgets.Layout(width='400px'),
         tooltip = 'Exponent for the dynamic weight constrain',
         continuous_update=False
    )


N_pre = 100
num_steps = 1000
N_post = 100

I = np.concatenate([np.round(np.random.random((num_steps//3,N_pre))), np.round(np.random.random((num_steps//3,N_pre))-0.30),np.round(np.random.random((num_steps//3,N_pre)))], axis=0) 
pars = default_pars(R = 1, tau_thr = 40, ratio_thr = 1.5,
                    refractory_time = True, 
                    dynamic_threshold = True, 
                    hard_reset = False)

# first simulation
neurons = simulation(pars , N_post = N_post, spk_input= I , neuron_type=LIFNeuron)

post_spk = get_post_spk_trains(neurons)

raster_plot(pars, pre_syn_spk = I, post_syn_spk = post_spk, title = 'Raster plot of the input and output spikes')

# second simulation
neurons = simulation(pars , N_post = N_post, spk_input= post_spk, neuron_type=LIFNeuron)

post_spk_2 = get_post_spk_trains(neurons)

raster_plot(pars, pre_syn_spk = post_spk, post_syn_spk = post_spk_2, title = 'Raster plot of second output spikes', pre_syn_plot=False)

# third simulation
neurons = simulation(pars , N_post = N_post, spk_input= post_spk_2, neuron_type=LIFNeuron)

post_spk_3 = get_post_spk_trains(neurons)

raster_plot(pars, pre_syn_spk = post_spk_2, post_syn_spk = post_spk_3, title = 'Raster plot of third output spikes', pre_syn_plot=False)

# fourth simulation
neurons = simulation(pars , N_post = N_post, spk_input= post_spk_3, neuron_type=LIFNeuron)

post_spk_4 = get_post_spk_trains(neurons)

raster_plot(pars, pre_syn_spk = post_spk_3, post_syn_spk = post_spk_4, title = 'Raster plot of fourth output spikes', pre_syn_plot=False)



def syn_plot(pars, syn, 
             manual_update = True, 
             time_step = 1, 
             subsampling = False,   
             time_in_ms = False):
    """
    Plot the weights changes during the simulation as graph and as distribution at a given time step

    INPUT:
    - pars: parameter of the simulation
    - syn: synapse object containing the weights and the traces records
    - manual_update: if True the plot is updated only when the button is pressed
    - time_step: time step to plot the distribution
    - subsampling: subsampling of the weights plot in time
    - post_index: index of the post synaptic neuron
    - time_in_ms: if True the x axis is in ms, otherwise in time steps

    RETURN:
    Interactive demo, Visualization of synaptic weights as graph and distribution at a given time step
    """

    # useful values
    weights_history = syn.get_records()['W']
    N_post = weights_history.shape[1]
    num_steps = weights_history.shape[0]
    

    # check if we want the time in ms
    if time_in_ms:
        dt=pars['dt']
        label_x = 'Time (ms)'
    else:
        dt=1
        label_x = 'Time steps'

    # time steps for the x axis
    time_steps = np.arange(0, num_steps, 1)*dt

    # set the default time step
    if time_step is None:
        time_step = num_steps-10
    elif time_step > num_steps:
        print(f'Time step must be less than {num_steps}')
        return
    
    def main_plot(post_index, time_step = time_step, subsampling = False):
        # check i f subsampling is less than the number of time steps
        if subsampling:
            s = num_steps//10
        else:
            s = 1

        fig,ax = plt.subplots(2, figsize=(10, 8), gridspec_kw={'height_ratios': [1.5, 1]})#, sharex=True)

        # plot the weights
        x = time_steps[::s]
        y = weights_history[ ::s,post_index,:]
        ax[0].plot(x, y, lw=1., alpha=0.7)
        ax[0].axvline(time_step, 0., 1., color='red', ls='--')
        ax[0].set_xlabel(label_x)
        ax[0].set_ylabel('Weight')


        # plot the weights distribution
        w_min = np.min(weights_history[time_step,:])-0.1
        w_max = np.max(weights_history[time_step,:])+0.1
        width = (w_max - w_min)/51
        bins = np.arange(w_min, w_max, width)
        #g_dis, _ = np.histogram(weights_history[time_step,:], bins)
        #ax[1].bar(bins[1:], g_dis, color='b', alpha=0.5, width=width)
        ax[1].hist(weights_history[time_step,post_index,:], bins, color='b', alpha=0.5, facecolor = '#2ab0ff', edgecolor='#169acf', linewidth=0.5)
        ax[1].set_xlabel('weights ditribution')
        ax[1].set_ylabel('Number')
        #ax[1].set_title(f'Time step: {time_step}')
        plt.tight_layout()
        plt.show()
        return

    my_layout.width = '620px'


    interactive_plot = widgets.interactive(main_plot, 
                                            {'manual': manual_update, 'manual_name': 'Update plots'},
                                            post_index=widgets.IntSlider(
                                                min=0,
                                                max=N_post-1,
                                                step=1,
                                                layout=my_layout),
                                            time_step=widgets.IntSlider(
                                                value=time_step, 
                                                min=0, 
                                                max=num_steps, 
                                                step=num_steps//100,
                                                description='Time step',
                                                layout=my_layout),
                                            subsampling=subsampling)
    #output = interactive_plot.children[-1]
    #output.layout.height = '400px'    
    return interactive_plot
 







####################################
#                                  #
#       SNNTorch Discarded         #
#                                  #
####################################










# basic variables
N_pre = 100
N_pre_correlated = 10
N_post = 10
num_steps = 100
batch_size = 0 # for the moment we do not use batch size
rate = 0.1
# set torch seed
torch.manual_seed(42)

class syn_stdp_inh_hidden(nn.Module):

    def __init__(self, pars, N_pre, N_post = 1):
        super(syn_stdp_inh_hidden, self).__init__()

        # model parameters
        self.pars = pars
        self.alpha = pars.get('alpha', 0.9)
        self.beta = pars.get('beta', 0.8)
        self.w_max = pars.get('w_max', 1.0)
        self.w_min = pars.get('w_min', 0.0)

        # SNN structure
        self.fc = nn.Linear(N_pre, N_post, bias=False)
        # clamp the weight to positive values
        self.fc.weight.data = torch.clamp(self.fc.weight.data, min=self.w_min, max=self.w_max)
        # set the weight of the layer
        #W_init = weight_initializer(pars, N_pre, N_post, type_init = 3, tensor = True)
        #self.fc.weight = nn.Parameter(W_init)
        reset_mechanism = pars['reset_mechanism']
        self.lif = snn.Synaptic(alpha = self.alpha, 
                                beta = self.beta, 
                                threshold = pars['threshold'], 
                                reset_mechanism = reset_mechanism,
                                init_hidden = True
                                ) # inhibition is a feature that does not work properly yet
        self.lif2 = snn.Synaptic(alpha = self.alpha,
                                beta = self.beta,
                                threshold = pars['threshold'],
                                reset_mechanism = reset_mechanism,
                                init_hidden = False
                                )

    def forward(self, x, manual_inhibition = False):
        # initiliaze the membrane potential and the spike
        cond2, mem2 = self.lif2.init_synaptic()

        #tracking variables
        mem_rec = []
        spk_rec = []
        syn_rec = []
        mem2_rec = []
        # dovrei inizializzare tutto o come liste o come tensori...
        pre_syn_traces = generate_traces(self.pars, x)
        post_syn_traces = torch.zeros((x.shape[0]+1, N_post))
        weight_history = torch.zeros((x.shape[0]+1, N_post, N_pre))
        weight_history[0,:,:] = self.fc.weight.data

        for step in range(x.shape[0]):
            # run the fc layer
            cur_step = self.fc(x[step])
            # run thw lif neuron
            spk = self.lif(cur_step)
            spk, cond2, mem2 = self.lif2(spk, cond2, mem2) 
            # find the first neuron that spikes
            if spk.sum() > 0 and manual_inhibition:
                self.lif.syn = self.lateral_inhibition(spk, self.lif.syn)


            # store the membrane potential and the spike
            mem_rec.append(self.lif.mem)
            spk_rec.append(spk)
            syn_rec.append(self.lif.syn)
            mem2_rec.append(mem2)
            # updatae post synaptic traces
            beta_minus = self.pars.get('beta_minus',0.9)
            post_syn_traces[step+1, :] = beta_minus*post_syn_traces[step, :] + spk

            # update the weights
            weight_history[step,:,:] = self.fc.weight.data
            A_plus, A_minus = self.pars['A_plus'], self.pars['A_minus']
            LTP = A_plus * torch.outer(spk, pre_syn_traces[step,:]) 
            LTD = A_minus * torch.outer(post_syn_traces[step+1,:], x[step])
            self.fc.weight.data = self.fc.weight.data + LTP - LTD
            # hard constrain on the weights
            self.fc.weight.data = torch.clamp(self.fc.weight.data, min=self.w_min, max=self.w_max)


        post_syn_traces = post_syn_traces[:-1,:]
        weight_history = weight_history[:-1,:,:]

        self.records = {'mem': torch.stack(mem_rec), 
                        'spk': torch.stack(spk_rec), 
                        'syn':  torch.stack(syn_rec), 
                        'pre_trace': pre_syn_traces, 
                        'post_trace': post_syn_traces, 
                        'W': weight_history,
                        'mem2': torch.stack(mem2_rec)}

        return 
    
    def lateral_inhibition(self, spk, syn):
        first_neuron_index = torch.where(spk)[0][0]
        # inhibit the conductance of all the others
        temp = torch.ones_like(syn)
        temp[first_neuron_index] = 0
        syn = syn  - temp * 0.01 # reduce the conductance of all the neuron but the first to spike

        return syn

    



# parameters of the simulation
pars = default_pars(type_parameters='simple', 
                    w_init_value = 0.012,
                    alpha = 0.9,
                    beta = 0.8,
                    threshold = 1.0,
                    reset_mechanism = 'zero',
                    A_minus = 0.0088 * 0.024,
                    A_plus = 0.008 * 0.024,
                    beta_minus = 0.9,
                    beta_plus = 0.9)
dt = pars['dt']

# generate the input spikes
cur_in_numpy,_ = random_shifted_trains(dt, num_steps, N_pre, N_pre_correlated , rate = rate, shift_values=[-5,5]) 
#cur_in_numpy, _ , first_section_index = random_offsets(dt, num_steps, N_pre, rate = rate, sections = N_post, sort_shifts = True, length_ratio = 0.5)
cur_in = torch.from_numpy(cur_in_numpy)
cur_in = cur_in.to(dtype = torch.float32, device = device)

# intitilize the model
my_model = syn_stdp_inh_hidden(pars, N_pre, N_post)

# run the simulation
#my_model.train()
start_time = time.time()
my_model.forward(cur_in, manual_inhibition = False)
print(f"Forward pass time : --- {(time.time() - start_time):.2f} seconds ---")
mem_rec, spk_rec, syn_rec = my_model.records['mem'], my_model.records['spk'], my_model.records['syn'],
pre_trace, post_trace, weight_history =  my_model.records['pre_trace'], my_model.records['post_trace'], my_model.records['W']

plot_results_33(dt, num_steps, cur_in, pre_trace, syn_rec, N_pre, mem_rec, spk_rec, post_trace, weight_history, N_post, N_pre_correlated)



def mnist_pars(**kwargs):  #DA SISTEMARE
    

    '''
    Define a dictionary with the default parameters for the nuerons, the weight update rule and the simulation overall
    ARGS:
    - kwargs: dictionary with the parameters to be updated or added
    RETURNS:
    - pars (dictionary): dictionary with the parameters
    '''

    pars = {}
    # time steps
    pars['dt'] = 1.            # simulation time step [ms]  !do not change

    # typical neuron parameters
    pars['tau_m'] = 10.                      # membrane time constant [ms]
    pars['v_rest_e'] = -65.     # resting potential for excitatory neurons [mV]
    pars['v_rest_i'] = -60.     # resting potential for inhibitory neurons [mV]
    pars['v_reset_e'] = -65.   # reset potential for excitatory neurons [mV]
    pars['v_reset_i'] = -45.   # reset potential for inhibitory neurons [mV]
    pars['v_thresh_e'] = -52.  # threshold potential for excitatory neurons [mV]
    pars['v_thresh_i'] = -40.  # threshold potential for inhibitory neurons [mV]
    pars['refrac_e'] = 5.      # refractory time for excitatory neurons [ms]
    pars['refrac_i'] = 2.      # refractory time for inhibitory neurons [ms]


    # neuron parameters for conductance based model
    pars['tau_syn_exc'] = 10.   # synaptic time constant [ms]
    pars['tau_syn_inh'] = 10.     # synaptic time constant [ms]
    pars['U_rev_exc'] = 1.      # excitatory reversal potential [mV]
    pars['U_rev_inh'] = -1.   # inhibitory reversal potential [mV]
    pars['max_g'] = 1.     # maximum synaptic conductance [nS]

    # in the case of dynamic threshold
    pars['tau_theta'] = 1e7    # time constant for threshold added dynamic decay [ms]
    pars['theta_add'] = 0.05   # maximal increase in threshold voltage per spike [mV]                      

    # STDP parameters
    pars['STDP_offset'] = 0.4               # for the STDP type 1
    pars['A_plus'] =  0.008 * pars['max_g']    # magnitude of LTP
    pars['A_minus'] = 0.0088 * pars['max_g']  # magnitude of LTD 
    pars['nu'] = 0.0001
    pars['tau_plus'] = 20                    # LTP time constant [ms]
    pars['tau_minus'] = pars['tau_plus']     # LTD time constant [ms]
    pars['tau_syn'] = 5.                     # synaptic time constant [ms] if synaptic decay is used
    pars['dynamic_weight_exponent'] = 0.01   # exponent for the dynamic weight constraint

    # weight parameters
    pars['w_max'] = 1.       # maximum weight
    pars['w_min'] = 0.                       # minimum weight
    pars['w_init_value'] = 0.5  # initial value for the weights

    # neuron initialization parameters
    pars['refractory_time'] = False 
    pars['dynamic_threshold'] = False
    pars['hard_reset'] = True

    # weights update rule parameters
    pars['constrain'] = 'Hard'               # weight constrain
    pars['short_memory_trace'] = False       # short memory trace for the traces

    # dataset parameters
    pars['num_examples'] = 10000
    pars['N_pre'] = 784
    pars['N_post'] = 400  # in the original paper they had 400 excitatory neurons and 400 inhibitory neurons
    pars['num_steps_single'] = 350




from IPython.display import HTML
# little animation to visualize an example of the spike data
fig, ax = plt.subplots()

# extract just one sample
spike_data_sample = spike_data[:, 0, 0]
print(spike_data_sample.size())

# plot the animation
anim = splt.animator(spike_data_sample, fig, ax)
#%matplotlib widget
#%matplotlib inline

#HTML(anim.to_jshtml())
#HTML(anim.to_html5_video())
print(f"The corresponding target is: {targets_it[0]}")

# If you're feeling sentimental, you can save the animation: .gif, .mp4 etc.
anim.save("spike_mnist_test.mp4")










# basic variables
N_pre = 100
N_pre_correlated = 10
N_post = 10
num_steps = 100
batch_size = 0 # for the moment we do not use batch size
rate = 0.05
# set torch seed
torch.manual_seed(42)

class syn_stdp_inh(nn.Module):

    def __init__(self, pars, N_pre, N_post = 1):
        super(syn_stdp_inh, self).__init__()

        # model parameters
        self.pars = pars
        self.alpha = pars.get('alpha', 0.9)
        self.beta = pars.get('beta', 0.8)
        self.w_max = pars.get('w_max', 1.0)
        self.w_min = pars.get('w_min', 0.0)

        # SNN structure
        self.fc = nn.Linear(N_pre, N_post, bias=False)
        # clamp the weight to positive values
        self.fc.weight.data = torch.clamp(self.fc.weight.data, min=self.w_min, max=self.w_max)
        # set the weight of the layer
        #W_init = weight_initializer(pars, N_pre, N_post, type_init = 3, tensor = True)
        #self.fc.weight = nn.Parameter(W_init)
        reset_mechanism = 'zero' if pars['hard_reset'] else 'subtract'
        self.lif = snn.Synaptic(alpha = self.alpha, 
                                beta = self.beta, 
                                threshold = pars['threshold'], 
                                reset_mechanism = reset_mechanism,
                                ) # inhibition is a feature that does not work properly yet

    def forward(self, x, manual_inhibition = False):
        # initiliaze the membrane potential and the spike
        cond, mem = self.lif.init_synaptic()

        #tracking variables
        mem_rec = []
        spk_rec = []
        cond_rec = []
        # dovrei inizializzare tutto o come liste o come tensori...
        pre_syn_traces = generate_traces(self.pars, x)
        post_syn_traces = torch.zeros((x.shape[0]+1, N_post))
        weight_history = torch.zeros((x.shape[0]+1, N_post, N_pre))
        weight_history[0,:,:] = self.fc.weight.data

        for step in range(x.shape[0]):
            # run the fc layer
            cur_step = self.fc(x[step])
            # run thw lif neuron
            spk, cond,  mem = self.lif(cur_step, cond, mem)

            # find the first neuron that spikes
            if spk.sum() > 0 and manual_inhibition:
                cond = self.latera_inhibition(spk, cond)


            # store the membrane potential and the spike
            mem_rec.append(mem)
            spk_rec.append(spk)
            cond_rec.append(cond)

            # updatae post synaptic traces
            beta_minus = self.pars.get('beta_minus',0.9)
            post_syn_traces[step+1, :] = beta_minus*post_syn_traces[step, :] + spk

            # update the weights
            weight_history[step,:,:] = self.fc.weight.data
            A_plus, A_minus = self.pars['A_plus'], self.pars['A_minus']
            LTP = A_plus * torch.outer(spk, pre_syn_traces[step,:]) 
            LTD = A_minus * torch.outer(post_syn_traces[step+1,:], x[step])
            self.fc.weight.data = self.fc.weight.data + LTP - LTD
            # hard constrain on the weights
            self.fc.weight.data = torch.clamp(self.fc.weight.data, min=self.w_min, max=self.w_max)


        post_syn_traces = post_syn_traces[:-1,:]
        weight_history = weight_history[:-1,:,:]

        self.records = {'mem': torch.stack(mem_rec), 
                        'spk': torch.stack(spk_rec), 
                        'cond':  torch.stack(cond_rec), 
                        'pre_trace': pre_syn_traces, 
                        'post_trace': post_syn_traces, 
                        'W': weight_history}

        return 
    
    def latera_inhibition(self, spk, cond):
        first_neuron_index = torch.where(spk)[0][0]
        # inhibit the conductance of all the others
        temp = torch.ones_like(cond)
        temp[first_neuron_index] = 0
        cond = cond  - temp * 0.1 # reduce the conductance of all the neuron but the first to spike

        return cond

    



# parameters of the simulation
pars = default_pars(type_parameters='simple', 
                    w_init_value = 0.012,
                    alpha = 0.9,
                    beta = 0.8,
                    threshold = 1.0,
                    hard_reset = True,
                    A_minus = 0.0088 * 0.024,
                    A_plus = 0.008 * 0.024,
                    beta_minus = 0.9,
                    beta_plus = 0.9)
dt = pars['dt']

# generate the input spikes
cur_in_numpy,_ = random_shifted_trains(dt, num_steps, N_pre, N_pre_correlated , rate = rate, shift_values=[-5,5]) 
#cur_in_numpy, _ , first_section_index = random_offsets(dt, num_steps, N_pre, rate = rate, sections = N_post, sort_shifts = True, length_ratio = 0.5)
cur_in = torch.from_numpy(cur_in_numpy)
cur_in = cur_in.to(dtype = torch.float32, device = device)

# intitilize the model
my_model = syn_stdp_inh(pars, N_pre, N_post)

# run the simulation
#my_model.train()
start_time = time.time()
my_model.forward(cur_in, manual_inhibition = True)
print(f"Forward pass time : --- {(time.time() - start_time):.2f} seconds ---")
mem_rec, spk_rec, cond_rec, pre_trace, post_trace, weight_history = my_model.records['mem'], my_model.records['spk'], my_model.records['cond'], my_model.records['pre_trace'], my_model.records['post_trace'], my_model.records['W']

# plot results




#####################################################
#                                                   #
#  FROM THE ORIGINAL NOTEBOOK OF DIGIT RECOGNITION  #
#                                                   #
#####################################################


# 1. SNNTorch neurons
## 1.1 Leaky Integrate-and-Fire neuron
# basic variables
N_pre = 10
N_post = 1
num_steps = 100

#generate input spikes
cur_in = repeat_ones(num_steps, N_pre, silent_time = 7 )
#cur_in = np.ones((num_steps, N_pre))
# convert in torch tensor
cur_in = torch.from_numpy(cur_in)


# parameters of the simulation
pars = default_pars(type_parameters='simple', 
                    w_init_value = 0.11)
beta = np.exp(-pars['dt']/pars['tau_m'])
# initialize the weights
W = weight_initializer(pars, N_post, I = cur_in, type_init=3)
# convert in torch tensor
W = torch.from_numpy(W)

# initialize the neuron
reset_mechanism = 'zero' if pars['hard_reset'] else 'subtract' 
lif1 = snn.Leaky(beta=beta, threshold = pars['threshold'], 
                 learn_beta=False, 
                 learn_threshold=False, 
                 reset_mechanism=reset_mechanism,
                 #reset_delay = False
                 )
lif_fmp = snn.Leaky(beta=beta, threshold = 10000)

# run a simple simulation
mem = torch.zeros(1)
spk = torch.zeros(1)
mem_fmp = torch.zeros(1)
spk_fmp = torch.zeros(1)
mem_rec = []
spk_rec = []
mem_fmp_rec = []
spk_fmp_rec = []

# neuron simulation
for step in range(num_steps):
    step_cur_in = W @ cur_in[step]
    spk, mem = lif1.forward(step_cur_in, mem)
    _,mem_fmp = lif_fmp(step_cur_in, mem_fmp)
    mem_rec.append(mem)
    spk_rec.append(spk)
    mem_fmp_rec.append(mem_fmp)


# convert lists to tensors
mem_rec = torch.stack(mem_rec)
spk_rec = torch.stack(spk_rec)
mem_fmp_rec = torch.stack(mem_fmp_rec)


    
plot_cur_mem_spk(cur_in, (W @ cur_in.T)[0], mem_rec, mem_fmp_rec, spk_rec, thr_line=1, title="snn.Leaky Neuron Model")


## 1.2 Synaptic Conductance-based LIF 
#basic variables
N_pre = 10
N_post = 1
num_steps = 100

# parameters of the simulation
pars = default_pars(type_parameters='simple', 
                    w_init_value = 0.02)

# initialize the weights
W = weight_initializer(pars, N_post, I = cur_in, type_init=3)
W = torch.from_numpy(W)

#generate input spikes
rate=0.3
#cur_in = Poisson_generator(pars['dt'],rate, N_pre,num_steps)
#cur_in = np.ones((num_steps, N_pre))
cur_in = repeat_ones(num_steps, N_pre, silent_time = 8 )
cur_in = torch.from_numpy(cur_in)

# Temporal dynamics to compute on the basis of params
alpha = np.exp(-pars['dt']/pars['tau_syn_exc'])  # on SNN Torch tutorial this was 0.9 and I found 0.904
beta = np.exp(-pars['dt']/pars['tau_m'] * (1+pars['max_g']))  # on SNN Torch tutorial this was 0.8 and I found 0.8187

# Initialize 2nd-order LIF neuron
reset_mechanism = 'zero' if pars['hard_reset'] else 'subtract'
lif2 = snn.Synaptic(alpha=alpha, beta=beta,threshold = pars['threshold'],
                    learn_alpha=False, 
                    learn_beta=False, 
                    learn_threshold=False, 
                    reset_mechanism=reset_mechanism)
lif2_fmp = snn.Synaptic(alpha=alpha, beta=beta, threshold=100000 ) # second neuron with no threshold to visualize the FMP


# Initialize hidden states and output
syn, mem = lif1.init_synaptic() # random initialization of synapse and membrane
syn_fmp, mem_fmp = lif_fmp.init_synaptic() # random initialization of synapse and membrane
syn_rec = []
mem_rec = []
spk_rec = []
mem_fmp_rec = []
syn_fmp_rec = []

# Simulate neurons
for step in range(num_steps):
    step_cur_in = W @ cur_in[step]
    spk_out, syn, mem = lif2(step_cur_in, syn, mem)
    _, syn_fmp, mem_fmp = lif2_fmp(step_cur_in, syn_fmp, mem_fmp)
    spk_rec.append(spk_out)
    syn_rec.append(syn)
    mem_rec.append(mem)
    mem_fmp_rec.append(mem_fmp)


# convert lists to tensors
spk_rec = torch.stack(spk_rec)
syn_rec = torch.stack(syn_rec)
mem_rec = torch.stack(mem_rec)
mem_fmp_rec = torch.stack(mem_fmp_rec)



plot_spk_cur_mem_spk((W @ cur_in.T)[0], syn_rec, mem_rec, mem_fmp_rec, spk_rec, title="snn.Synaptic Neuron Model")
# 2. STDP preliminaries experiments
## 2.1 LIF
# basic variables
N_pre = 50
N_post = 10
num_steps = 100

# parameters of the simulation
pars = default_pars(type_parameters='simple', 
                    w_init_value = 0.1)
time_steps = np.arange(num_steps) * pars['dt']
rate = 0.05

#generate input spikes
#cur_in = repeat_ones(num_steps, N_pre, silent_time = 8 )
cur_in = Poisson_generator(pars['dt'], rate, N_pre, num_steps)
cur_in = torch.from_numpy(cur_in)

# initialize the weights
W = weight_initializer(pars, N_post, I = cur_in, type_init=3) * np.random.rand(N_post,N_pre) # noise added to appreciate different STDP dynamics
W = torch.from_numpy(W)

# initialize postsynaptic neurons
beta = np.exp(-pars['dt']/pars['tau_m'])
rm = 'zero' if pars['hard_reset'] else 'subtract' 
thr = pars['threshold']
post_neurons = [ snn.Leaky(beta = beta, threshold=thr, reset_mechanism=rm) for _ in range(N_post)]

# initialize the tracking variables
mem_record = np.zeros((num_steps+1, N_post))
spk_record = np.zeros((num_steps+1, N_post))
mem_record[0,:] = np.asarray([0 for i in range(N_post)])
spk_record[0,:] = np.asarray([0 for i in range(N_post)])

# intialize the synapses
my_synapses = STDP_synapse(pars, N_pre, N_post, W_init = W ) # this works even if now W is a tensor!

# run the simulation
for step in range(num_steps):
    # current injected at this time step
    pre_syn_spikes = cur_in[step]
    cur_in_step = W @ pre_syn_spikes

    #spike generated by the layer
    for i in range(N_post):
        spk_record[step+1,i],mem_record[step+1,i] = post_neurons[i](cur_in_step[i], mem_record[step,i])

    post_syn_spk = spk_record[step+1,:]
    
    # update the weights
    my_synapses.update_weights([pre_syn_spikes.detach().numpy(),post_syn_spk])
    W = my_synapses.W

# convert the results to torch tensors
mem_record = torch.from_numpy(mem_record[1:,:])
spk_record = torch.from_numpy(spk_record[1:,:])


plot_results_21(time_steps, cur_in, spk_record, my_synapses.get_records()['W'][1:,1,:])
## 2.2 Conductance-based LIF
# basic variables
N_pre = 10
N_post = 1
num_steps = 100

# parameters of the simulation
pars = default_pars(type_parameters='simple', 
                    w_init_value = 0.1)
time_steps = np.arange(num_steps) * pars['dt']
rate = 0.05

#generate input spikes
#cur_in = repeat_ones(num_steps, N_pre, silent_time = 8 )
cur_in = Poisson_generator(pars['dt'], rate, N_pre, num_steps)
cur_in = torch.from_numpy(cur_in)

# initialize the weights
W = weight_initializer(pars,  N_post, I=cur_in, type_init=3) #* np.random.rand(N_post,N_pre) # noise added to appreciate different STDP dynamics
W = torch.from_numpy(W)

# initialize postsynaptic neurons
beta = np.exp(-pars['dt']/pars['tau_m'] *(1+pars['max_g']))
alpha = np.exp(-pars['dt']/pars['tau_syn_exc'])
rm = 'zero' if pars['hard_reset'] else 'subtract' 
thr = pars['threshold']
post_neurons = [ snn.Synaptic(alpha = alpha, beta = beta, threshold=thr, reset_mechanism=rm) for _ in range(N_post)]

# initialize the tracking variables
mem_record = np.zeros((num_steps+1, N_post))
spk_record = np.zeros((num_steps+1, N_post))
cond_record = np.zeros((num_steps+1, N_post))
#mem_record[0,:] = np.asarray([0 for i in range(N_post)])
#spk_record[0,:] = np.asarray([0 for i in range(N_post)])
#cond_record[0,:] = np.asarray([0 for i in range(N_post)])

# intialize the synapses
my_synapses = STDP_synapse(pars, N_pre, N_post, W_init = W ) # this works even if now W is a tensor!

# run the simulation
for step in range(num_steps):
    # current injected at this time step
    pre_syn_spikes = cur_in[step]
    cur_in_step = W @ pre_syn_spikes

    #spike generated by the layer
    for i in range(N_post):
        spk, cond, mem = post_neurons[i](cur_in_step[i], cond_record[step, i], mem_record[step,i])
        spk_record[step+1,i] = spk
        mem_record[step+1,i] = mem
        cond_record[step+1,i] = cond

    post_syn_spk = spk_record[step+1,:]
    
    # update the weights
    my_synapses.update_weights([pre_syn_spikes.detach().numpy(),post_syn_spk])
    W = my_synapses.W

# convert the results to torch tensors
mem_record = torch.from_numpy(mem_record[1:,:])
spk_record = torch.from_numpy(spk_record[1:,:])

plot_results_21(time_steps, cur_in, spk_record, my_synapses.get_records()['W'][1:,0,:])



# 3. Simple SNNs built with SNNTorch
## 3.1 Build an SNN with 1 FC layer, Leaky Neuron, STDP
# basic variables
N_pre = 28*28
N_post = 1
batch_size = 0 # for the moment we do not use batch size
num_steps = 100
rate = 0.05

class snn_stdp0(nn.Module):

    def __init__(self, pars, N_pre, N_post = 1, neuron_type = 'Leaky'):
        super(snn_stdp0, self).__init__()
        self.pars = pars
        self.beta = pars.get('beta', 0.8)
        self.w_max = pars.get('w_max', 1.0)
        self.w_min = pars.get('w_min', 0.0)
        self.fc = nn.Linear(N_pre, N_post, bias=False)
        # clamp the weight to positive values
        self.fc.weight.data = torch.clamp(self.fc.weight.data, min=self.w_min, max=self.w_max)
        # set the weight of the layer
        #W_init = weight_initializer(pars, N_pre, N_post, type_init = 3, tensor = True)
        #self.fc.weight = nn.Parameter(W_init)
        reset_mechanism = 'zero' if pars['hard_reset'] else 'subtract'
        self.lif = snn.Leaky(beta = beta, threshold = pars['threshold'], reset_mechanism = reset_mechanism)

    def forward(self, x):
        # initiliaze the membrane potential and the spike
        mem = self.lif.init_leaky()

        #tracking variables
        mem_rec = []
        spk_rec = []

        pre_syn_traces = generate_traces(self.pars, x)
        post_syn_traces = torch.zeros((x.shape[0]+1, N_post))
        weight_history = torch.zeros((x.shape[0]+1, N_post, N_pre))
        weight_history[0,:,:] = self.fc.weight.data

        for step in range(x.shape[0]):
            # run the fc layer
            cur_step = self.fc(x[step])
            # run thw lif neuron
            spk, mem = self.lif(cur_step, mem)

            # store the membrane potential and the spike
            mem_rec.append(mem)
            spk_rec.append(spk)

            # updatae post synaptic traces
            beta_minus = self.pars.get('beta_minus',0.9)
            post_syn_traces[step+1, :] = beta_minus*post_syn_traces[step, :] + spk

            # update the weights
            weight_history[step,:,:] = self.fc.weight.data
            A_plus, A_minus = self.pars['A_plus'], self.pars['A_minus']
            LTP = A_plus * torch.outer(spk, pre_syn_traces[step,:]) 
            LTD = A_minus * torch.outer(post_syn_traces[step+1,:], x[step])
            self.fc.weight.data = self.fc.weight.data + LTP - LTD
            # hard constrain on the weights
            self.fc.weight.data = torch.clamp(self.fc.weight.data, min=self.w_min, max=self.w_max)


        post_syn_traces = post_syn_traces[:-1,:]
        weight_history = weight_history[:-1,:,:]



        return torch.stack(mem_rec), torch.stack(spk_rec), pre_syn_traces, post_syn_traces, weight_history
    



# parameters of the simulation
pars = default_pars(type_parameters='simple', 
                    w_init_value = 0.012,
                    alpha = 0.9,
                    beta = 0.8,
                    threshold = 1.0,
                    hard_reset = True,
                    A_minus = 0.0088 ,
                    A_plus = 0.008,
                    beta_minus = 0.9,
                    beta_plus = 0.9)

# generate the input spikes
cur_in = Poisson_generator(pars['dt'], rate, N_pre, num_steps, batch_size = batch_size)
cur_in = torch.from_numpy(cur_in)
cur_in = cur_in.to(dtype = torch.float32, device = device)

# intitilize the model
my_model = snn_stdp0(pars, N_pre, N_post, 'Synaptic')

# run the simulation
my_model.train()
mem_rec, spk_rec, pre_trace, post_trace, weight_history = my_model.forward(cur_in)

plot_results_31(pars['dt'], cur_in, pre_trace, mem_rec, spk_rec, post_trace, weight_history, N_pre, N_post, num_steps)







