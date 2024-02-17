#!C:\Users\latta\miniconda3\envs\test\python3

# basic libraries
from cProfile import label
import os
import sys
import shutil
import time
import numpy as np

main_dir = os.getcwd()
if main_dir not in sys.path:
    print('Adding the folder for the modules')
    sys.path.append(main_dir)
import importlib
importlib.reload(importlib.import_module('learning_rules'))
from learning_rules import *

# graphics libraries
import matplotlib.pyplot as plt
import logging
logging.getLogger('matplotlib.font_manager').disabled = True
# use NMA plot style
#plt.style.use("https://raw.githubusercontent.com/NeuromatchAcademy/course-content/main/nma.mplstyle")
plt.style.use('seaborn-v0_8')




class LIFNeuron:

    def __init__(self, pars, 
                 refractory_time = False,  
                 dynamic_threshold = False,   
                 hard_reset = True,           
                 noisy_input = False,
                 **kwargs):
        """
        - pars: parameter dictionary
        - refractory_time: boolean, if True the neuron has a refractory period
        - dynamic_threshold: boolean, if True the threshold is dynamic
        - hard_reset: boolean, if True the reset is hard
        - noisy_input: boolean, if True the input is noisy
        """
        # base attributes
        self.pars = pars
        self.tau_m = self.pars['tau_m']
        self.R = self.pars['R']
        self.U_resting = self.pars['U_resting']
        self.threshold = self.pars['threshold']
        self.mem = self.pars['U_init']

        # user defined attributes
        self.hard_reset = hard_reset
        self.refractory_time = refractory_time
        self.dynamic_threshold = dynamic_threshold
        self.noisy_input = noisy_input
        
        # additional attributes
        for key, value in kwargs.items():
            setattr(self, key, value)
        if self.dynamic_threshold:
            self.tau_thr = self.pars['tau_thr']
        if self.refractory_time:
            self.t_ref = self.pars['t_ref']
            self.t_ref_left = 0

        # tracking attributes
        self.spk_record = []
        self.mem_record = []
        self.I_inj_record = []
        if self.dynamic_threshold:
            self.threshold_record = []

    def forward(self, I_inj):
        """
        REUTRNS:
        - membrane potential at the next time step
        - spikes produced by the neuron (0 or 1)
        """

        if self.noisy_input:
            # sum a gaussian variable to the injected current
            sigma = self.pars.get('sigma', 1)
            I_inj += np.random.normal(0, sigma, 1)

        self.I_inj_record.append(I_inj)

        if self.refractory_time:
            # check if we are in the refractory period of the neuron
            if self.t_ref_left > 0:
                # update the refractory period remaining
                self.t_ref_left = self.t_ref_left - 1
                # store the records
                self.mem_record.append(self.mem)
                self.spk_record.append(0)
                if self.dynamic_threshold:
                    self.threshold_record.append(self.threshold)
                # exit the forward
                return self.mem, 0
        
        # check if there's a spike
        spk = self.mem > self.threshold
        self.spk_record.append(int(spk))

        if spk:
            # reset the membrane potential
            if self.hard_reset:
                self.mem = self.pars['U_reset']
            else:
                self.mem = self.mem-(self.threshold-self.U_resting)

            # increase the threshold if wanted
            if self.dynamic_threshold:
                self.threshold = self.pars['ratio_thr'] * (self.threshold-self.U_resting) + self.U_resting

            # reset the refractory period count down
            if self.refractory_time:
                self.t_ref_left = self.t_ref

        # discretization time step [ms]
        dt = self.pars['dt']

        # update the membrane potential:
        # exponential decay
        self.mem  *= (1-dt/self.tau_m)
        # decaying towards the resting potential
        self.mem += dt/self.tau_m * self.U_resting 
        # add the contribution due to the incoming current
        self.mem += dt/self.tau_m * self.R * I_inj
        # store the membrane potential
        self.mem_record.append(self.mem)

        if self.dynamic_threshold:
            # update the membrane threshold
            # exponential decay
            self.threshold *= (1-dt/self.tau_thr)
            # decaying toward the resting threshold
            self.threshold += dt/self.tau_thr *self.pars['threshold'] 
            # store the threshold
            self.threshold_record.append(self.threshold)


        
        return self.mem, spk

    def get_records(self):
        if self.dynamic_threshold:
            return {'mem': np.array(self.mem_record), 'spk': np.array(self.spk_record), 'I_inj': np.array(self.I_inj_record), 'thr': np.array(self.threshold_record)}
        else:
            return {'mem': np.array(self.mem_record), 'spk': np.array(self.spk_record), 'I_inj': np.array(self.I_inj_record)}
        
    def reset_records(self):
        self.spk_record = []
        self.mem_record = []
        self.I_inj_record = []
        if self.dynamic_threshold:
            self.threshold_record = []

    def plot_records(self, title=False, time_in_ms = False):

        if time_in_ms:
            dt=self.pars['dt']
            label_x = 'Time (ms)'
        else:
            dt=1
            label_x = 'Time steps'

        records = self.get_records()
        cur = records['I_inj']
        mem = records['mem']
        spk = records['spk']
        
        number_of_steps = len(cur)
        T = number_of_steps*dt
        time = np.arange(0, number_of_steps*dt, dt)
        # Generate Plots
        fig, ax = plt.subplots(3, figsize = (15, 12), sharex=True,
                            gridspec_kw = {'height_ratios': [1, 1, 0.4]})

        # Plot input current
        ax[0].plot(time,cur, c="tab:orange")
        ax[0].set_ylabel("Input Current ($I_{in}$)")
        if title:
            ax[0].set_title(title)

        # Plot membrane potential
        ax[1].plot(time, mem)
        ax[1].set_ylabel("Membrane Potential ($U_{mem}$)")
        if self.dynamic_threshold:
            ax[1].plot(time, records['thr'], c="red", linestyle="dashed", alpha=0.5, label="Threshold")
        else:
            ax[1].axhline(y=self.threshold, alpha=0.25, linestyle="dashed", c="red", linewidth=2, label="Threshold")
        ax[1].legend( loc="best")
        plt.xlabel(label_x)

        # Plot output spike using spikeplot
        #ax[2].plot(time,spk, c="black", marker="|", linestyle='None', markersize=20, markeredgewidth=1.5)
        ax[2].eventplot(np.array(np.where(spk==1))*dt, color="black", linelengths=0.5, linewidths=1)
        plt.ylabel("Output spikes")
        plt.yticks([])

        plt.show()





class PoissonNeuron:

 

    def __init__(self, pars, 
                 refractory_time = False,  
                 dynamic_threshold = False,   
                 hard_reset = True,
                 **kwargs):
        """
        - pars: parameter dictionary
        - refractory_time: boolean, if True the neuron has a refractory period
        - dynamic_threshold: boolean, if True the threshold is dynamic
        """
        # base attributes
        self.pars = pars
        self.tau_m = self.pars['tau_m']
        self.R = self.pars['R']
        self.U_resting = self.pars['U_resting']
        self.threshold = self.pars['threshold']
        self.alpha = self.pars['alpha']
        self.mem = self.pars['U_init']

        # user defined attributes
        self.refractory_time = refractory_time
        self.dynamic_threshold = dynamic_threshold
        self.hard_reset = hard_reset
        

        # additional attributes
        for key, value in kwargs.items():
            setattr(self, key, value)   
        if self.refractory_time:
            self.t_ref = self.pars['t_ref']
            self.t_ref_left = 0
        if self.dynamic_threshold:
            self.tau_thr = self.pars['tau_thr']

        # tracking attributes
        self.spk_record = []
        self.mem_record = []
        self.rate_record = []
        self.I_inj_record = []
        if self.dynamic_threshold:
            self.threshold_record = []

    def forward(self, I_inj):
        # sigmoid function
        def sigmoid(x, alpha):
            return 1/(1+np.exp(-alpha*x))
        
        # record the input current
        self.I_inj_record.append(I_inj)

        if self.refractory_time:
            # check if we are in the refractory period of the neuron
            if self.t_ref_left > 0:
                # update the refractory period remaining
                self.t_ref_left -= 1
                # store the records
                self.mem_record.append(self.mem)
                self.rate_record.append(0)
                self.spk_record.append(0)
                if self.dynamic_threshold:
                    self.threshold_record.append(self.threshold)
                # exit the forward
                return self.mem, 0

        # compute the rate from the membrane potential
        rate = self.alpha * (self.mem - self.threshold) # I should improve this step according to the theory
        rate = rate * (rate>0)
        #rate = sigmoid(self.mem-self.threshold, self.alpha)
        self.rate_record.append(rate)
        # check if there's a spike
        spk = np.random.random() < rate
        # store the spike
        self.spk_record.append(int(spk))

        if spk:
            # reset the membrane potential
            if self.hard_reset:
                self.mem = self.pars['U_reset']
            else:
                self.mem = self.mem-(self.threshold-self.U_resting)

            # increase the threshold if wanted
            if self.dynamic_threshold:
                self.threshold = self.pars['ratio_thr'] * (self.threshold-self.U_resting) + self.U_resting

            # reset the refractory period count down
            if self.refractory_time:
                self.t_ref_left = self.t_ref

   


        # discretization time step [ms]
        dt = self.pars['dt']
        # update the membrane potential
        self.mem = self.mem + (dt/self.tau_m)*(self.U_resting-self.mem + I_inj*self.R)
        # store the membrane potential
        self.mem_record.append(self.mem)

        if self.dynamic_threshold:
            # update the membrane threshold
            # exponential decay
            self.threshold *= (1-dt/self.tau_thr)
            # decaying toward the resting threshold
            self.threshold += dt/self.tau_thr *self.pars['threshold'] 
            # store the threshold
            self.threshold_record.append(self.threshold)


        return self.mem, int(spk)

    def get_records(self):
        if self.dynamic_threshold:
            return {'mem': np.array(self.mem_record), 'spk': np.array(self.spk_record), 'rate': np.array(self.rate_record),'I_inj':np.array(self.I_inj_record),'thr': np.array(self.threshold_record)}
        else:
            return {'mem': np.array(self.mem_record), 'spk': np.array(self.spk_record), 'rate': np.array(self.rate_record), 'I_inj':np.array(self.I_inj_record)}
        
    def reset_records(self):
        self.spk_record = []
        self.mem_record = []
        self.I_inj_record = []
        if self.dynamic_threshold:
            self.threshold_record = []

    def plot_records(self, title=False, time_in_ms = False):
        
        if time_in_ms:
            dt=self.pars['dt']
            label_x = 'Time (ms)'
        else:
            dt=1
            label_x = 'Time steps'

        records = self.get_records()
        rate = records['rate']
        mem = records['mem']
        spk = records['spk']
        
        number_of_steps = len(rate)
        T = number_of_steps*dt
        time = np.arange(0, number_of_steps*dt, dt)
        # Generate Plots
        fig, ax = plt.subplots(4, figsize = (15, 17), sharex=True,
                            gridspec_kw = {'height_ratios': [1, 1, 1, 0.4]})

        # Plot input current
        ax[0].plot(time,records['I_inj'], c="tab:orange")
        ax[0].set_ylabel("Input Current ($I_{in}$)")
        if title:
            ax[0].set_title(title)

        #Plot the rate
        ax[1].plot(time,rate, c="tab:green")
        ax[1].set_ylabel("Varying rate $\lambda(t)$")


        # Plot membrane potential
        ax[2].plot(time, mem)
        ax[2].set_ylabel("Membrane Potential ($U_{mem}$)")
        if self.dynamic_threshold:
            ax[2].plot(time, records['thr'], c="red", linestyle="dashed", alpha=0.5, label="Threshold")
        else:
            ax[2].axhline(y=self.threshold, alpha=0.25, linestyle="dashed", c="red", linewidth=2, label="Threshold")
        ax[2].legend( loc="best")
        plt.xlabel(label_x)

        # Plot output spike using spikeplot
        ax[3].eventplot(np.array(np.where(spk==1))*dt, color="black", linelengths=0.5, linewidths=1)
        plt.ylabel("Output spikes")
        plt.yticks([])

        plt.show()






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
    

if __name__ == "__main__":
    print(help(LIF))



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
