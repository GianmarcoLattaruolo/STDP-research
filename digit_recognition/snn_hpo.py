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
importlib.reload(importlib.import_module('snn_models'))


global mnist_pars
global rate_encoded_mnist
from snn_experiments import *
from snn_plot_utils import *
from snn_datasets import *
from snn_models import *



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



# I need to build a more systematic research
# the structure will be the following:
def snn_hpo_function(
     sampler = optuna.samplers.TPESampler(seed=42),
     pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1, interval_steps=1,  n_min_trials=10),

):
    # 0. the following parameters are fixed
    fixed_params = {
        # simulation parameters
        'dt' : 1.,                                  # simulation time step [ms]  
        'min_spk_number' : 5,                       # minimum number of spikes for a forward pass of a batch
        'store_records' : False,                    # store the records of the simulation
        
        # typical neuron parameters
        'threshold' : 1.0,                          # spike threshold for the LIF neuron
        'alpha' : 0.95,                             # decaying factore for the neuron conductance
        'beta' : 0.8,                               # decaying factor for the neuron potential
        'reset_mechanism' : 'subtract',             # reset mechanism for the neuron potential
        # parameters for dynamic threshold   
        'dynamic_threshold' : True,                 # use a dynamic threshold for the neuron
        'tau_theta' : 20.,                          # time constant for the dynamic threshold                       
        # parameters for lateral inhibition
        'lateral_inhibition': True,                 # use lateral inhibition
        'lateral_inhibition_strength': 0.1,         # strength of the lateral inhibition 

        # STDP parameters
        'beta_minus' : 0.25,                        # decay factor for the pre-synaptic trace
        'beta_plus' : 0.25,                         # decay factor for the post-synaptic trace

    }

    # define a function for the STDP parameters depending on the type
    def STDP_params(STDP_type, trial):

        return stdp_params
    # 1. Execute a grid search for each STDP type

    STDP_types = ['classic', 'offset', 'asymptotic']
    for STDP in STDP_types:
        main_hpo_dir = main_dir+'\\'+STDP

        # 2. Fixed a type of STDP run small grid search to find good params that stabilize the post-synaptic rates

        study_name_rate = STDP + 'stabilize_rate'
        study_storage_rate = f'sqlite:///{main_hpo_dir}/{study_name_rate}.db'
        # initialize the study
        study_rate = optuna.create_study(
            direction="minimize", 
            study_name = study_name_rate,
            storage = study_storage_rate,
            load_if_exists = True,
            sampler = sampler,
            pruner = pruner
        )
        study_rate.set_user_attr(f"{STDP}", "rate stabilization")
        # define the objective:
        def objective_rate(trial):
            # define the dataset and get the loader
            batch_size = 200 # it should not influence the rate
            num_steps = 100 # fixed it does not influence the rate
            gain = trial.suggest_float("gain", 0.9, 20.0, log=True)
            min_rate = trial.suggest_float("min_rate", 0.0, 0.5, step = 0.1)
            subset_loader = rate_encoded_mnist(batch_size, num_steps=num_steps, gain=gain, min_rate = min_rate, train=True, my_seed = 42).get_subset(100)
            # define the model
            ...
            pars = mnist_pars(
                **STDP_params(STDP, trial)
                **fixed_params
            )
            n_neurons = ...
            snn_mnist(pars, n_neurons)
            # make just one epoch to observe the rate
            for data, _ in subset_loader:
                ...
                # logging to weight and biases for better visualization

            return np.abs(rate-rate_desired)
        
        # optimize the study
        study_rate.optimize(
            objective, 
            n_trials = 2000, 
            timeout = 36000,
            n_jobs = 4, 
            catch = (), 
            callbacks = [MaxTrialsCallback(2000, states=(TrialState.COMPLETE,))], 
            gc_after_trial = True, 
            show_progress_bar = False
        )

        # 3. With the range found in the previous step, run a larger grid search to find the best parameters for accuracy
        
        study_name = STDP + 'stabilize_rate'
        study_storage = f'sqlite:///{main_hpo_dir}/{study_name}.db'
        # initialize the study
        study = optuna.create_study(
            direction="maximize", 
            study_name = study_name,
            storage = study_storage,
            load_if_exists = True,
            sampler = sampler,
            pruner = pruner
        )
        study.set_user_attr(f"{STDP}", "accuracy maximization")
        # define the objective:
        def objective_accuracy(trial):
            # define the datasets and get the loaders
            batch_size = ...
            num_steps = ...
            gain = ...
            min_rate = ...
            mnist_train = rate_encoded_mnist(batch_size, num_steps=num_steps, gain=gain, min_rate = min_rate, train=True, my_seed = 42)
            sub_train, sub_val, sub_test = mnist_train.get_subset_train_val_test(subset = 50)
            # define the model
             
            pars = mnist_pars(
                **fixed_params
            )




            return accuracy

    # 4. compare the results between the different STDP types and choose the best
    # 5. run a final grid search to find the best parameters for the best STDP type, optimizing also the dataset parameters    

    return results    






























