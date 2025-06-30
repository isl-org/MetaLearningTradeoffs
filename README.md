# PROJECT NOT UNDER ACTIVE MANAGEMENT #  
This project will no longer be maintained by Intel.  
Intel has ceased development and contributions including, but not limited to, maintenance, bug fixes, new releases, or updates, to this project.  
Intel no longer accepts patches to this project.  
 If you have an ongoing need to use this project, are interested in independently developing it, or would like to maintain patches for the open source software community, please create your own fork of this project.  
  
# Modeling and Optimization Trade-off in Meta-learning

This repository contains the code used to obtain the experimental results in the paper [Modeling and Optimization Trade-off in Meta-learning](https://arxiv.org/abs/2010.12916), Gao and Sener (NeurIPS 2020).

It is based on the full_code branch of the [ProMP](https://github.com/jonasrothfuss/ProMP) repository.

The code is written in Python 3. The part corresponding to the linear regression experiment only requires [NumPy](https://numpy.org), while the part corresponding to the reinforcement learning experiments also requires [Tensorflow](https://www.tensorflow.org/) and the [Mujoco](http://www.mujoco.org/) physics engine.
Some of the reinforcement learning environments can be found in this repository, and the rest are from [MetaWorld](https://github.com/rlworkgroup/metaworld).

## Installation

Please follow the installation instructions provided by the [ProMP](https://github.com/jonasrothfuss/ProMP) repository and the [MetaWorld](https://github.com/rlworkgroup/metaworld) repository. 
For the latter, please use the api-rework branch for compatibility (this has already been added to requirements.txt).

## Running the experiments

### Linear regression

Execute
```
python3 linear_regression/run_experiment.py --p 1 --beta 2 --seed 1
```
The figures can then be found in the folder `p-1_beta-2_seed-1/figures`.

### Reinforcement learning

To create all the executable scripts that we need to run, execute
```
python3 experiments/benchmark/run.py
```
They will be found in the folder `scripts`.
The training scripts are of the form `algorithm_environment_mode_seed.sh`, and the testing scripts are of the form `test_algorithm_environment_mode_seed_checkpoint.sh`.
- `algorithm` is replaced by `ppo` (DRS+PPO), `promp` (ProMP), `trpo` (DRS+TRPO), `trpomaml` (TRPO-MAML).
- `environment` and `mode` are replaced by 
  - `walker` and `params-interpolate` (Walker2DRandParams) 
  - `walker` and `goal-interpolate` (Walker2DRandVel)
  - `cheetah` and `goal-interpolate` (HalfCheetahRandVel)
  - `hopper` and `params-interpolate` (HopperRandParams)
  - `metaworld` and `ml1-push` (ML1-Push)
  - `metaworld` and `ml1-reach` (ML1-Reach)
  - `metaworld` and `ml10` (ML10)
  - `metaworld` and `ml45` (ML45)
- `seed`, the random seed, is replaced by integers 1-5.
- `checkpoint`, the policies stored at various stages during training, is replaced by integers 0-20.

After all runs are finished, the figures can be created by executing
```
python3 experiments/benchmark/summary.py
```
They will be found in the folder `results`.

## Acknowledgements

We would like to thank Charles Packer for help during the creation of the code for the reinforcement learning experiments.

## Citation

To cite this repository in your research, please reference the following [paper](https://arxiv.org/abs/2010.12916):

> Katelyn Gao and Ozan Sener. Modeling and Optimization Trade-off in Meta-Learning. *arXiv preprint arXiv:2010.12916* (2020).

```TeX
@misc{GaoSener2020,
  Author = {Katelyn Gao and Ozan Sener},
  Title = {Modeling and Optimization Trade-off in Meta-Learning},
  Year = {2020},
  Eprint = {arXiv:2010.12916},
}
```
