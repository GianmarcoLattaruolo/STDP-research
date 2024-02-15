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