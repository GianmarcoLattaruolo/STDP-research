# STDP-research
---
## Summary


This repository contain the code for my Master Thesis with title "Spike-Time Dependent Plasticity Learning Techniques for Event-Based Signals".
My work focused on the Spike Timing Dependent Plasticity (STDP) learning rules which is a particular type of algorithm developed to be biologically plausible and mathematically tractable. Spiking Neural Netowrks (SNS) are the type machine learning architecture within we can test this algorithm. In this repository you can find a thourough analysis of both SNS and STDP, with implementation from scratch and subsequent exploitation of dedicated libraries such as SNNTorch. For mathematical proofs and techinical consideration you can ask me to provide you my thesis.

---
## Repository Structure

The repository structure is the following and it is essentially divided in the two different phases of my research:

- **STDP_playground**: This folder contains all the basic implementations and experiments with each object built from scratch. I did not use any kind of machine learning library in the code in this folder.
  - *STDP_basic_experiments.ipynb* : this is the main notebook from which all the code belonging to the other sripts can be used running the exeriments and visalizing the results.
  - *neurons.py*: this scipt contains the classes for the neurons utilized which are the Leaky-Integrate and Fire (LIF), the synaptic LIF and Poisson neuron models (everything built from scratch) 
  - *learning_rules.py*: this script defines the class for simulating synapses with STDP learning rule.
  - *experiments.py*:here some auxuliary functions are introduced in order to be able to run the experiments: a function for the default parameters, a function for running the simulation for several steps and many ones for generating specific spike trains as inputs.
  - *plot_utils.py*: this script contains all the utilities to plot the results such as the history of the synaptic weights during the simulation.


- **digit_recognition**: in this second phase of the study a more systematic approach towards classic machine learning has been choosen. The problem of making MNIST dataset suitable for SNNs has been addressed, then some effort has been putin order to utilize the PyTorch and SNNTorch libraries, integrating them with the STDP algorithm and its variants. Finally an attempt to classify the digit from the MNIST dataset has been initiated. Notice that this is a still a work in progress and the future results evetually founded will be part of my second Master Thesis.
  - *digit_recognition.ipynb*: as before this is the main notebook from which all the code belonging to the other sripts can be used running the exeriments and visalizing the results. first the experiments performed in the STDP_playground are replicated using the SNNTorch objects. Then the MNIST dataset is loaded and encoded to be used in the SNNs and several auxiliary functions for training, assignment and classification are presented. Subsequetly a thourough Hyperparameter optimization is performed, testing three different variants of the STDP learning rule. In the choice of the STDP variants and in the SNN structure I mainly followed the work by P. U. Diehl and M. Cook. [*“Unsupervised learning of digit recognition using spike-timing-dependent plasticity”*](https://github.com/peter-u-diehl/stdp-mnist). Since the Grid searches were computationally expensive, the results are not conclusive and the final training and the classification of the net is still a work in progress.
  - *snn_dataset.py*: this script contains the class for the pytorch dataloader containing the spike rate encoding of the MNIST dataset.
  - *snn_experiments.py*: this scripts contains many auxiliary functions to run the experiments as in the previous phase, but this time with the SNNTorch objects. 
  - *snn_hpo.py*: this script contains the class for the hyperparameter optimization of the SNNs performed thanks to the **Optuna** library.
  - *snn_models.py*: here the class for the SNN object has been defined, hereditating from the torch .nn.Module object. The STDP learning rule has been implemented overwriting the forward method and turning off the autograd engine.
  - *snn_plot_utils*: all plot functions divided by notebook sections.  
  - *HPO*: this folder contains the results of the hyperparameter optimization performed with Optuna. the files are meant to be open in the Optuna dashboard.
  - *animation*: this folder contains animation of some encoded digits from the MNIST dataset, with hyperparameters.
  - *data*: this folder contains the MNIST dataset downloaded from the torchvision library.
  - *result.txt*: here some results of the hyperparameter optimization are stored.

Finally there are other files that are common to both phases of the research, such as the files for the conda environment and the requirements, the gitignore file and the license.


---

 





