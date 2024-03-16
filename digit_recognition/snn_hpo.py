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
import wandb


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



# set a user attribute to keep tracks of other statistics of the model trained
# trial.set_user_attr('test_list', test_list)
# trial.set_user_attr('mean_rate',mean_rate)
# trial.set_user_attr('assign_mean_conf',np.mean(temp_assignments['conf_status']+0.0))



# I need to build a more systematic research
# the structure will be the following:
class snn_hpo:
    
    def __init__(self, STDP_type, 
        n_trials = 1000,
        timeout = 3600,
        accuracy_epochs = 10,
        n_jobs = os.cpu_count(),
        main_dir = r'C:\Users\latta\GitHub\STDP-research\digit_recognition',
        seed = 42,
        pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1, interval_steps=1,  n_min_trials=20),

    ):
        
        # base parameters of the optimization
        self.STDP_type = STDP_type
        self.n_trials = n_trials
        self.timeout = timeout
        self.accuracy_epochs = accuracy_epochs
        self.n_jobs = n_jobs
        self.main_dir = main_dir
        self.sampler = optuna.samplers.TPESampler(seed=seed)
        self.pruner = pruner
        # use cuda if available
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
            
        # The following parameters are fixed
        self.fixed_params = {
            # simulation parameters
            'dt' : 1.,                                  # simulation time step [ms]  
            'min_spk_number' : 5,                       # minimum number of spikes for a forward pass of a batch per neuron per image
            'store_records' : False,                    # store the records of the simulation
            
            # typical neuron parameters
            'threshold' : 1.0,                          # spike threshold for the LIF neuron
            'alpha' : 0.95,                             # decaying factore for the neuron conductance
            'beta' : 0.8,                               # decaying factor for the neuron potential
            'reset_mechanism' : 'subtract',             # reset mechanism for the neuron potential
            # parameters for dynamic threshold   
            'dynamic_threshold' : True,                 # use a dynamic threshold for the neuron
            'tau_theta' : 1000.,                          # time constant for the dynamic threshold                       
            # parameters for lateral inhibition
            'lateral_inhibition': True,                 # use lateral inhibition
            'lateral_inhibition_strength': 0.1,         # strength of the lateral inhibition 
            # parameters for the refractory period
            'refractory_period' : True,                # use a refractory period

            # STDP parameters
            'beta_minus' : 0.25,                        # decay factor for the pre-synaptic trace
            'beta_plus' : 0.25,                         # decay factor for the post-synaptic trace
            'mu_exponent' : 2.,                         # exponent for the dynamic constrain on the weights

        }

        # tracking parameters for the optimization
        self.rate_tracking = {}

        return

    # define a function for the STDP parameters depending on the type
    def STDP_params(self, trial):
        # retrive the STDP type
        STDP_type = self.STDP_type

        if STDP_type == 'classic':
            A_minus = trial.suggest_float("A_minus", 0.00001, 0.001, log = True ) # from prelimanires grid search A_minus is much greater than A_plus
            A_plus = trial.suggest_float("A_plus", 0.000001, 0.0001, log = True )
            
            stdp_params = {'A_minus' : A_minus,'A_plus' : A_plus}

        elif STDP_type == 'offset':

            STDP_offset = trial.suggest_float("STDP_offset", 0.0001, 1.0, log = True)
            leaning_rate = trial.suggest_float("learn_rate", 0.00001, 0.1, log = True)

            stdp_params = {'STDP_offset' : STDP_offset,'leaning_rate' : leaning_rate}
        elif STDP_type == 'asymptotic':

            A_minus = trial.suggest_float("A_minus", 0.001, 10.0, log = True ) # from prelimanires grid search A_minus is much greater than A_plus
            A_plus = trial.suggest_float("A_plus", 0.0001, 0.1, log = True )

            stdp_params = {'A_minus' : A_minus,'A_plus' : A_plus}

        else:
            raise ValueError(f'Unknown STDP type: {STDP_type}')
        
        # common parameters
        stdp_params['STDP_type'] =  STDP_type
        stdp_params['theta_add'] = trial.suggest_float("theta_add", 0.0, 20.0, step = 0.5)
        stdp_params['ref_time'] = trial.suggest_int("ref_time", 0, 20, step = 2)

        return stdp_params

    # define a function to log a posteriori the trials to wandb
    def log_to_wandb(self, study_type = 'accuracy'): # still in development
        if study_type == 'rate':
            # check that study, study_name and pars are defined
            if not hasattr(self, 'rate_study'):
                raise ValueError('Log to wandb is possible only after the optimization of the study')
            # otherwise
            study, study_name, pars = self.study, self.study_name, self.pars
            # log the trials a posteriori
            for trial in study.trials:
                if trial.state != optuna.trial.TrialState.FAIL:
                    for key, value in trial.params.items():
                        pars[key] = value
                    config = pars
                    config['trial.number'] = trial.number
                    config['trial.state'] = trial.state
                    run = wandb.init(
                        project = 'digit-recognition',
                        entity = 'g-latta',
                        name = f'trial_{trial.number}',
                        config = config,
                        group = study_name,
                        reinit = True
                    )
                    # log to wandb
                    run.log(
                        data = {'accuracy': trial.value, 
                                'mean_rate': trial.user_attrs['min_spk_n_reached']},
                                )

                    # close the wandb run
                    run.finish()  
        elif study_type == 'accuracy':
            pass
        return 


    def my_callback(self):
        # still in development
        # I would like to print only when a new best trial is found
        def logging_callback(study, frozen_trial):
            previous_best_value = study.user_attrs.get("previous_best_value", None)
            if previous_best_value != study.best_value:
                study.set_user_attr("previous_best_value", study.best_value)
                print(
                    "Trial {} finished with best value: {} and parameters: {}. ".format(
                    frozen_trial.number,
                    frozen_trial.value,
                    frozen_trial.params,
                    )
                )
        return logging_callback
    
    # define a function to compute an error measure on the rate
    @staticmethod
    def f_error(anspnpi, asrpts): 
        """
        I would like to have:
        - the averge number of spikes per neuron per image around 5-40 (considering 100 time steps)
        - the average spike rate per neuron at each time step quite low (0.05-0.4 spikes) as implied by the first point
        - the variance of the average spike rate per neuron at each time step as low as possible
        """
        error = 0.0
        f_out_interval_error = lambda x: (100-25*x)*(x<=4)+(5*x-200)/3*(x>=40)
        spk_std = np.std(asrpts)
        spk_mean = np.mean(asrpts)
        np.random.seed(42)
        correction_factor = 100/np.std(np.random.random(100))
        error = f_out_interval_error(anspnpi) + correction_factor * spk_std + 100 * (spk_mean - 0.05) * (spk_mean > 0.05)
        if error<0:
            print(f'anspnpi = {anspnpi:.2e}, spk_std = {spk_std:.2e}, spk_mean = {spk_mean:.2e}, error = {error:.2e}')
            raise ValueError('Error is negative')
        # in the wrost cases the error should be around 200
        return error

    # define a function to calculate the best ranges and values from the rate stabilization study
    def best_range_calculation(self, trial): 

        # initialize two empty dictionaries
        best_ranges = {}
        best_values = {}

        # retrive all the trials from rate study
        sort_trials_df = self.study_rate.trials_dataframe().sort_values(by='value', ascending=True)

        # select only the complete trials
        sort_trials_df = sort_trials_df[sort_trials_df['state'] == 'COMPLETE']

        # select only the good trials, the first 200
        good_trials = sort_trials_df.head(200)
        
        # retrive the best trial to fix some values
        best_trial = self.study_rate.best_trial.params

        best_values['gain'] = best_trial['gain']
        best_values['min_rate'] = best_trial['min_rate']
        best_values['weight_initializer'] = best_trial['weight_initializer']
        # best_values['theta_add'] = best_trial['theta_add']
        # best_values['ref_time'] = best_trial['ref_time']

        # calculate the best ranges
        if self.STDP_type == 'classic':
            min_A_minus = good_trials['params_A_minus'].min()
            max_A_minus = good_trials['params_A_minus'].max()
            min_A_plus = good_trials['params_A_plus'].min()
            max_A_plus = good_trials['params_A_plus'].max()
            theta_add_max = good_trials['params_theta_add'].max()
            theta_add_min = good_trials['params_theta_add'].min()
            ref_time_max = good_trials['params_ref_time'].max()
            ref_time_min = good_trials['params_ref_time'].min()
            best_ranges['A_minus'] = trial.suggest_float('A_minus',min_A_minus, max_A_minus)
            best_ranges['A_plus'] = trial.suggest_float('A_plus',min_A_plus, max_A_plus)
            best_ranges['theta_add'] = trial.suggest_float('theta_add',theta_add_min, theta_add_max)
            best_ranges['ref_time'] = trial.suggest_int('ref_time',ref_time_min, ref_time_max)

        elif self.STDP_type == 'offset':
            min_learnin_rate = good_trials['params_learning_rate'].min()
            max_learnin_rate = good_trials['params_learning_rate'].max()
            min_STDP_offset = good_trials['params_STDP_offset'].min()
            max_STDP_offset = good_trials['params_STDP_offset'].max()
            best_ranges['learning_rate'] = trial.suggest_float('learning_rate',min_learnin_rate, max_learnin_rate)
            best_ranges['STDP_offset'] = trial.suggest_float('STDP_offset',min_STDP_offset, max_STDP_offset)
            
        elif self.STDP_type == "asymptotic":
            min_A_minus = good_trials['params_A_minus'].min()
            max_A_minus = good_trials['params_A_minus'].max()
            min_A_plus = good_trials['params_A_plus'].min()
            max_A_plus = good_trials['params_A_plus'].max()
            best_ranges['A_minus'] = trial.suggest_float('A_minus',min_A_minus, max_A_minus)
            best_ranges['A_plus'] = trial.suggest_float('A_plus',min_A_plus, max_A_plus)

        return best_ranges, best_values
    

    # define a function to initialize the study for the rate stabilization
    def initialize_rate_study(self):
        STDP = self.STDP_type
        main_hpo_dir = self.main_dir+'\\HPO'

        # 2. Fixed a type of STDP run small grid search to find good params that stabilize the post-synaptic rates

        study_name_rate = STDP + '_STDP_stab_rate'
        self.study_name = study_name_rate
        study_storage_rate = f'sqlite:///{main_hpo_dir}/{study_name_rate}.db'
        # initialize the study
        study_rate = optuna.create_study(
            direction="minimize", 
            study_name = study_name_rate,
            storage = study_storage_rate,
            load_if_exists = True,
            sampler = self.sampler,
            pruner = self.pruner
        )
        study_rate.set_user_attr(f"{STDP}", "rate stabilization")

        self.study_rate = study_rate
        return 

    # define the method to optimize the study on the rate stabilization:
    def optimize_rate(self):
        # check if self has study_rate attribute
        if not hasattr(self, 'study_rate'):
            raise ValueError('The study for the rate stabilization has not been initialized')
        # otherwise
        def objective(trial):

            # define the dataset and get the loader
            batch_size = 200 # it should not influence the rate
            num_steps = 100 # fixed it does not influence the rate
            min_rate = trial.suggest_float("min_rate", 0.0, 0.2, step=0.01)
            gain = trial.suggest_float("gain", 0.5, 20.0, step = 0.5)
            subset_loader = rate_encoded_mnist(batch_size, num_steps=num_steps, gain=gain, min_rate = min_rate, train=True, my_seed = 42).get_subset(30) # 2000 samples in the subset
            
            # define the model
            pars = mnist_pars(
                weight_initializer = trial.suggest_categorical('weight_initializer', ['clamp', 'shift', 'norm_row', 'random']),
                **self.STDP_params(trial),  # this two dictionaries must not share any key
                **self.fixed_params
            )
            n_neurons = 50 # fixed it does not influence the rate
            model = snn_mnist(pars, 28*28, n_neurons).to(self.device)

            # make just one epoch to observe the rate
            with tqdm(total=len(subset_loader), unit='batch', ncols=120) as pbar:

                # Iterate through minibatches  # with batch size 200 we have 3 batches
                for batch, (data_it, _) in enumerate(subset_loader):
                    # reset neuron records for spk
                    model.neuron_records['spk'] = []

                    # forward pass
                    model.forward(data_it)

                    # retrive the last rate statistics
                    anspnpi = model.anspnpi[-1]
                    asrpts = model.asrpts[-1]

                    # compute the stabizilation error
                    error = snn_hpo.f_error(anspnpi, asrpts)

                    # report
                    trial.report(error, batch)

                    # Update progress bar
                    pbar.update(1)
                    pbar.set_postfix(ANSPNPI = f'{anspnpi:.2e}')

                    # handle pruning
                    if trial.should_prune():
                        raise optuna.exceptions.TrialPruned()
            
            # set user attributes
            total_anspnpi = np.mean(model.anspnpi[-len(subset_loader):], dtype = np.float64)
            first_anspnpi = np.mean(model.anspnpi[-len(subset_loader)], dtype = np.float64)
            last_anspnpi = np.mean(model.anspnpi[-1], dtype = np.float64)
            mean_weights = model.fc.weight.data.detach().numpy().mean(dtype=np.float64)
            trial.set_user_attr('total_anspnpi', total_anspnpi)
            trial.set_user_attr('first_anspnpi', first_anspnpi)
            trial.set_user_attr('last_anspnpi', last_anspnpi)
            trial.set_user_attr('mean_weights', mean_weights)

            return error

        
        # optimize the study
        self.study_rate.optimize(
            objective, 
            n_trials = self.n_trials, 
            timeout = self.timeout,
            n_jobs = self.n_jobs, 
            catch = (), 
            callbacks = [MaxTrialsCallback(7000, states=(TrialState.COMPLETE,)), self.my_callback()], 
            gc_after_trial = True, 
            show_progress_bar = False
        )

        return
    
    # define a function to print the statistics of the study
    def print_statistics(self):
        flag = True
        if hasattr(self, 'study_rate'):
            print(f"Study for the rate stabilization statistics of {self.STDP_type} STDP: ")
            study = self.study_rate
            pruned_traials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
            complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

            print(f"")
            print(f"  Number of finished trials: {len(study.trials)}")
            print(f"  Number of pruned trials: {len(pruned_traials)}")
            print(f"  Number of complete trials: {len(complete_trials)}")

            print("Best trial:")
            best_trial = study.best_trial

            print(f"  Value: {best_trial.value}")

            print("  Params: ")
            for key, value in best_trial.params.items():
                print(f"    {key}: {value}")
            flag = False

            print(f'Best Ranges')
            sort_trials_df = study.trials_dataframe().sort_values(by='value', ascending=True)
            sort_trials_df = sort_trials_df[sort_trials_df['state'] == 'COMPLETE']
            good_trials = sort_trials_df.head(1000)
            if self.STDP_type == 'classic' or self.STDP_type == 'asymptotic':
                min_A_minus = good_trials['params_A_minus'].min()
                max_A_minus = good_trials['params_A_minus'].max()
                min_A_plus = good_trials['params_A_plus'].min()
                max_A_plus = good_trials['params_A_plus'].max()
                print(f'A_minus: {min_A_minus:.2e} - {max_A_minus:.2e}')
                print(f'A_plus: {min_A_plus:.2e} - {max_A_plus:.2e}')
                # plot the distribution of A_minus and A_plus
                plt.hist(good_trials['params_A_minus'], bins = 20)
                plt.title('A_minus distribution')
                plt.show()
                plt.hist(good_trials['params_A_plus'], bins = 20)
                plt.title('A_plus distribution')
                plt.show()
            elif self.STDP_type == 'offset':
                min_learnin_rate = good_trials['params_learn_rate'].min()
                max_learnin_rate = good_trials['params_learn_rate'].max()
                min_STDP_offset = good_trials['params_STDP_offset'].min()
                max_STDP_offset = good_trials['params_STDP_offset'].max()
                print(f'learning_rate: {min_learnin_rate:.2e} - {max_learnin_rate:.2e}')
                print(f'STDP_offset: {min_STDP_offset:.2e} - {max_STDP_offset:.2e}')
                # plot the histogram of param_learning_rate
                plt.hist(good_trials['params_learn_rate'], bins = 20)
                plt.title('Learning rate distribution')
                plt.show()


        if hasattr(self, 'study_accuracy'):
            print(f"Study for the accuracy optimization statistics: ")
            study = self.study_accuracy
            pruned_traials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
            complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

            print(f"")
            print(f"  Number of finished trials: {len(study.trials)}")
            print(f"  Number of pruned trials: {len(pruned_traials)}")
            print(f"  Number of complete trials: {len(complete_trials)}")

            print("Best trial:")
            trial = study.best_trial

            print(f"  Value: {trial.value}")

            print("  Params: ")
            for key, value in trial.params.items():
                print(f"    {key}: {value}")
            flag = False
        if flag:
            print('No study has been initialized')

        return
    
    # define a function to initialize the study for the accuracy optimization
    def initialize_accuracy_study(self):

        # 3. With the range found in the previous step, run a larger grid search to find the best parameters for accuracy
        STDP = self.STDP_type
        main_hpo_dir = self.main_dir+'\\HPO'

        study_name = STDP + '_STDP_optimize_acc'
        study_storage = f'sqlite:///{main_hpo_dir}/{study_name}.db'
        # initialize the study
        study = optuna.create_study(
            direction="maximize", 
            study_name = study_name,
            storage = study_storage,
            load_if_exists = True,
            sampler = self.sampler,
            pruner = self.pruner
        )
        study.set_user_attr(f"{STDP}", "accuracy maximization")

        self.study_accuracy = study
        return
    
    # define the method to optimize study on the accuracy
    def optimize_accuracy(self):
        # check if self has study_rate attribute
        if not hasattr(self, 'study_accuracy'):
            raise ValueError('The study for the accuracy optimization has not been initialized')
        # check if a best trial for the rate stabilization is available
        if not hasattr(self, 'study_rate'):
            raise ValueError('The study for the rate stabilization has not been initialized')
        if len(self.study_rate.trials) < 2:
            raise ValueError('The study for the rate stabilization has not been completed')
        
        # define the function to optimize the accuracy
        def objective_accuracy(trial):
            
            # calculate the best ranges and values from the rate stabilization study
            best_ranges, best_values = self.best_range_calculation(trial)

            # define the datasets and get the loaders
            batch_size = trial.suggest_categorical("batch_size", [50, 100, 200, 400])
            num_steps = trial.suggest_int("num_steps", 200, 500, step = 50)
            gain = best_values['gain']
            min_rate = best_values['min_rate']
            mnist_train = rate_encoded_mnist(batch_size, num_steps=num_steps, gain=gain, min_rate = min_rate, train=True, my_seed = 42)
            sub_train, sub_val, sub_test = mnist_train.get_subset_train_val_test(subset = 25)

            # define the model
            pars = mnist_pars(
                assignment_confidence = trial.suggest_float('ass_conf', 0.00001,0.01, log = True),
                **best_values,
                **best_ranges, 
                **self.fixed_params,
            )
            # define the model
            n_neurons = trial.suggest_int("n_neurons", 128, 1024, step = 64)
            model = snn_mnist(pars,28*28, n_neurons).to(self.device)

            # set user attributes
            trial.set_user_attr('weight_initialization_type',best_values['weight_initializer'])
            trial.set_user_attr('min_spk',[])

            # train the model
            num_epochs = 5
            for epochs in range(num_epochs):
                start_time = time.time()

                with tqdm(total=len(sub_train), unit='batch', ncols=120) as pbar:
                    # Iterate through minibatches
                    for data_it, _ in sub_train:
                        # reset the spk records before the forward pass
                        model.neuron_records['spk'] = []

                        # forward pass
                        model.forward(data_it)

                        # check the min spk number
                        min_spk = torch.stack(model.neuron_records['spk']).detach().numpy().sum(dtype = np.float64) 
                        trial.user_attrs['min_spk'].append(min_spk)

                        # Update progress bar
                        pbar.update(1)

                    # assign the label to the neurons
                    temp_assignments = assign_neurons_to_classes(model, sub_val, verbose = 0 )

                    # compute the accuracy so far
                    accuracy, anspnpi = classify_test_set(model, sub_test, temp_assignments, verbose = 0)

                    # update the progress bar
                    pbar.set_postfix(acc=f'{accuracy:.4f}', time=f'{time.time() - start_time:.2f}s', ANSPNPI = f'{anspnpi:.2e}')

                    # report the accuracy
                    trial.report(accuracy, epochs)

                    # handle pruning
                    if trial.should_prune():
                        raise optuna.exceptions.TrialPruned()
                
            # set user attributes
            total_anspnpi = np.mean(model.anspnpi[-len(sub_train):], dtype = np.float64)
            first_anspnpi = np.mean(model.anspnpi[-len(sub_train)], dtype = np.float64)
            last_anspnpi = np.mean(model.anspnpi[-1], dtype = np.float64)
            mean_weights = model.fc.weight.data.detach().numpy().mean(dtype=np.float64)
            trial.set_user_attr('total_anspnpi', total_anspnpi)
            trial.set_user_attr('first_anspnpi', first_anspnpi)
            trial.set_user_attr('last_anspnpi', last_anspnpi)
            trial.set_user_attr('mean_weights', mean_weights)
        
            return accuracy
        
        # optimize the study
        self.study_accuracy.optimize(
            objective_accuracy, 
            n_trials = self.n_trials, 
            timeout = self.timeout,
            n_jobs = self.n_jobs, 
            catch = (), 
            callbacks = [MaxTrialsCallback(7000, states=(TrialState.COMPLETE,))], 
            gc_after_trial = True, 
            show_progress_bar = False
        )

        return



############################################################################################################
    # 4. compare the results between the different STDP types and choose the best
    ...    
    # 5. run a final grid search to find the best parameters for the best STDP type, optimizing also the dataset parameters    
    ...
   




















