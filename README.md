# ScsgAdaptivity

This repository contains all Python code to replicate all numerical results in our paper [On the Adaptivity of Stochastic Gradient-Based Optimization](https://arxiv.org/abs/1904.04480).

## Python scripts
The folder `python/` contains all python code:

- `methods.py` implements all methods considered in the paper, including SCSG, SVRG, SARAH, Katyusha(ns), SGD and GD;

- `objective.py` implements the loss function and the gradient function of multi-class logistic regression;

- `process_data.py` is the script to transform external data into our standard data format: "XXX_A.npy" for the design matrix, "XXX_y.npy" for the class labels and "XXX_params.p" for other information. Due to the storage constraints, the raw data is excluded from the repo;

- `expr.py` is the script to run experiments on one dataset and one stepsize. See `./expr.py -h` for the options;

- `post_process_expr.py` post-processes the experimental results from `expr.py` by choosing the best tuned stepsize and recording the corresponding results for each dataset. It also outputs the best tuned stepsize for each method into "XXX_besteta.p";

- `compute_optim.py` computes the optimum value f(x*) by running SCSG with 5000 effective passes of data with the best tuned stepsize from "XXX_besteta.p" and outputs the result into "XXX_optim.p";

- `expr_plot.py` generate all figures;

- `utils.py` implements a few helpers 

## Replicating the Experiments
The folder `jobs/` contains all files that facilitate job submission to the cluster. 

- `SCSG_expr_params.txt` contains all 126 combinations of datasets and stepsizes
 
- Running the following code in a SLURM system to submit jobs

```
mkdir results
mkdir log
sbatch SCSG_job.sh
```

Please contact lihualei@stanford.edu for further questions.
