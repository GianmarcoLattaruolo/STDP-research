# STDP-research

## Summary


This repository contain the code for my Master Thesis with title "Spike-Time Dependent Plasticity Learning Techniques for Event-Based Signals".
My work focused on the Spike Timing Dependent Plasticity (STDP) learning rules which is a particular type of algorithm developed to be biologically plausible and mathematically tractable. Spiking Neural Netowrks (SNS) are the type machine learning architecture within we can test this algorithm. In this repository you can find a thourough analysis of both SNS and STDP, with implementation from scratch and subsequent exploitation of dedicated libraries such as SNNTorch. For mathematical proofs and techinical consideration you can ask me to provide you my thesis.

## Repository Structure

The repository structure is the following and it is essentially divided in the two different phases of my reasearch:

- **STDP_playground**: in this folder you can find all the implementation and the basic experiments with every object built from scratch. I didn'tuse any type of machine learning library in this folder's code.
  - STDP_basic_experiments.ipynb
  - neurons.py
  - learning_rules.py
  - experiments.py
  - plot_utils.py


- **digit_recognition**
  - HPO
  - animation
  - data
  - digit_recognition.ipynb
  - result.txt
  - snn_dataset.py
  - snn_experiments.py
  - snn_hpo.py
  - snn_models.py
  - snn_plot_utils

  
- .gitignore
- requirements.txt
- torch+.yml
- LICENSE
- README.md
 





