# SAASBO

This repository contain a Python package
for SAASBO, an algorithm for high-dimensional Bayesian optimization described in
[High-Dimensional Bayesian Optimization with Sparse Axis-Aligned Subspaces](https://arxiv.org/abs/2103.00349).

## Abstract

Bayesian optimization (BO) is a powerful paradigm for efficient optimization of black-box objective functions. High-dimensional BO presents a particular challenge, in part because the curse of dimensionality makes it difficult to define -- as well as do inference over -- a suitable class of surrogate models. We argue that Gaussian process surrogate models defined on sparse axis-aligned subspaces offer an attractive compromise between flexibility and parsimony. We demonstrate that our approach, which relies on Hamiltonian Monte Carlo for inference, can rapidly identify sparse subspaces relevant to modeling the unknown objective function, enabling sample-efficient high-dimensional BO. In an extensive suite of experiments comparing to existing methods for high-dimensional BO we demonstrate that our algorithm, Sparse Axis-Aligned Subspace BO (SAASBO), achieves excellent performance on several synthetic and real-world problems without the need to set problem-specific hyperparameters.

### Requirements
Python 3.7, NumPy, SciPy, JAX, NumPyro


### File structure

Besides the core functionality we include:
- a script (saasgp_demo.py) that demonstrates how to fit a GP equipped with a SAAS prior
- a script (saasbo_demo.py) that demonstrates how to run SAASBO on the Hartmann6 function embedded in D=50 dimensions 
- a notebook (Branin100.ipynb) that demonstrates how to run SAASBO on the Branin function embedded in D=100 dimensions
