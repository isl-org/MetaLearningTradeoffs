import os
import json
import tensorflow as tf
import numpy as np
import argparse
import joblib

from rand_param_envs.hopper_rand_params import HopperRandParamsEnv
from rand_param_envs.walker2d_rand_params import Walker2DRandParamsEnv
from maml_zoo.envs.mujoco_envs.half_cheetah_rand_vel import HalfCheetahRandVelEnv
from maml_zoo.envs.mujoco_envs.walker2d_rand_vel import Walker2DRandVelEnv

from metaworld.benchmarks import ML1
from metaworld.benchmarks import ML10
from maml_zoo.envs.mujoco_envs.metaworld_ml45 import ML45

from maml_zoo.utils.utils import set_seed, ClassEncoder
from maml_zoo.baselines.linear_baseline import LinearFeatureBaseline
from maml_zoo.envs.normalized_env import normalize
from maml_zoo.logger import logger
from gym.utils.seeding import create_seed

from maml_zoo.policies.gaussian_mlp_policy import GaussianMLPPolicy
from maml_zoo.samplers.maml_sampler import MAMLSampler
from maml_zoo.samplers.base import SampleProcessor
from maml_zoo.algos.vpg import VPG
from maml_zoo.tester import Tester

def run_experiment(hyperparams):

    EXP_NAME = hyperparams['algorithm']+'/'+hyperparams['mode']
    training_dir = os.getcwd()+'/data/'+EXP_NAME+'/'+str(hyperparams['environment'])+'/run_'+str(hyperparams['run'])
    exp_dir = training_dir+'/checkpoint_'+str(hyperparams['checkpoint'])
    logger.configure(dir=exp_dir, format_strs=['stdout', 'log', 'csv'], snapshot_mode='gap', snapshot_gap=50)
    hyperparam_dict = dict(hyperparams)
    if hyperparams['environment'] == 'metaworld':
        del hyperparam_dict['train_env']
        del hyperparam_dict['test_env']
    json.dump(hyperparam_dict, open(exp_dir+'/params.json', 'w'), indent=2, sort_keys=True, cls=ClassEncoder)

    # Set random seed
    set_seed(hyperparams['seed'])

    # Instantiate environment
    test_baseline = hyperparams['baseline']() # This is fine since there is no scaling by default
    if 'ml' in hyperparams['mode']:
        env = normalize(hyperparams['train_env'])
        test_env = normalize(hyperparams['test_env'])
    else:
        env = normalize(hyperparams['train_env']())
        test_env = normalize(hyperparams['test_env']())

    # Make sure the architecture  matches the training one so the params are loaded correctly
    test_policy = GaussianMLPPolicy(
            name="eval_policy",
            obs_dim=np.prod(env.observation_space.shape),
            action_dim=np.prod(env.action_space.shape),
            hidden_sizes=hyperparams['hidden_sizes'],
            learn_std=hyperparams['learn_std'],
            hidden_nonlinearity=hyperparams['hidden_nonlinearity'],
            output_nonlinearity=hyperparams['output_nonlinearity'],
        )

    test_sampler = MAMLSampler(
            env = test_env,
            policy = test_policy,
            rollouts_per_meta_task=hyperparams['rollouts_per_meta_task'],
            meta_batch_size=1,
            max_path_length=hyperparams['max_path_length'],
            parallel=False,
        )

    test_sample_processor = SampleProcessor(
            baseline=test_baseline,
            discount=hyperparams['discount'],
            gae_lambda=hyperparams['gae_lambda'],
            normalize_adv=hyperparams['normalize_adv'],
            positive_adv=hyperparams['positive_adv'],
        )

    test_algo = VPG(
            policy=test_policy,
            learning_rate=hyperparams['learning_rate'],
            inner_type=hyperparams['inner_type'],
        )

    checkpoint_path = training_dir+'/itr_'+str(hyperparams['checkpoint']*hyperparams['checkpoint_gap'])+'.pkl'
    tester = Tester(
            algo=test_algo,
            policy=test_policy,
            load_policy=checkpoint_path,
            output=exp_dir,
            env=test_env,
            sampler=test_sampler,
            sample_processor=test_sample_processor,
            n_evals=hyperparams['n_evals'],
            n_updates=hyperparams['n_updates'],
        )

    tester.test()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--algorithm', type=str)
    parser.add_argument('--env', type=str)
    parser.add_argument('--mode', type=str)
    parser.add_argument('--run', type=int)
    parser.add_argument('--seed', type=int)

    parser.add_argument('--learning-rate', type=float)
    parser.add_argument('--checkpoint', type=int)

    args = parser.parse_args()

    hyperparams = {
        'algorithm': args.algorithm,
        'environment': args.env,
        'mode': args.mode,
        'run': args.run,
        'seed': args.seed,
        'learning_rate': args.learning_rate,
        'checkpoint': args.checkpoint
    }

    if args.env == 'walker':

        if args.mode == 'params-interpolate':
            hyperparams['train_env'] = Walker2DRandParamsEnv
            hyperparams['test_env'] = Walker2DRandParamsEnv
        elif args.mode == 'goal-interpolate':
            hyperparams['train_env'] = Walker2DRandVelEnv
            hyperparams['test_env'] = Walker2DRandVelEnv
        else:
            raise NotImplementedError

        hyperparams['baseline'] = LinearFeatureBaseline
        hyperparams['rollouts_per_meta_task'] = 20
        hyperparams['max_path_length'] = 200

        hyperparams['discount'] = 0.99
        hyperparams['gae_lambda'] = 1
        hyperparams['normalize_adv'] = True
        hyperparams['positive_adv'] = False

        hyperparams['hidden_sizes'] = (64, 64)
        hyperparams['learn_std'] = True
        hyperparams['hidden_nonlinearity'] = tf.tanh
        hyperparams['output_nonlinearity'] = None

        hyperparams['learning_rate'] = args.learning_rate

        if args.algorithm == 'ppo':
            hyperparams['checkpoint_gap'] = 100
            hyperparams['inner_type'] = 'likelihood_ratio'
        elif args.algorithm == 'promp':
            hyperparams['checkpoint_gap'] = 50
            hyperparams['inner_type'] = 'likelihood_ratio'
        elif args.algorithm == 'trpo':
            hyperparams['checkpoint_gap'] = 100
            hyperparams['inner_type'] = 'log_likelihood'
        elif args.algorithm == 'trpomaml':
            hyperparams['checkpoint_gap'] = 50
            hyperparams['inner_type'] = 'log_likelihood'
        else:
            raise NotImplementedError

        hyperparams['n_evals'] = 1000
        hyperparams['n_updates'] = 5

    elif args.env == 'hopper':

        if args.mode == 'params-interpolate':
            hyperparams['train_env'] = HopperRandParamsEnv
            hyperparams['test_env'] = HopperRandParamsEnv
        else:
            raise NotImplementedError

        hyperparams['baseline'] = LinearFeatureBaseline
        hyperparams['rollouts_per_meta_task'] = 20
        hyperparams['max_path_length'] = 200

        hyperparams['discount'] = 0.99
        hyperparams['gae_lambda'] = 1
        hyperparams['normalize_adv'] = True
        hyperparams['positive_adv'] = False

        hyperparams['hidden_sizes'] = (64, 64)
        hyperparams['learn_std'] = True
        hyperparams['hidden_nonlinearity'] = tf.tanh
        hyperparams['output_nonlinearity'] = None

        hyperparams['learning_rate'] = args.learning_rate

        if args.algorithm == 'ppo':
            hyperparams['checkpoint_gap'] = 100
            hyperparams['inner_type'] = 'likelihood_ratio'
        elif args.algorithm == 'promp':
            hyperparams['checkpoint_gap'] = 50
            hyperparams['inner_type'] = 'likelihood_ratio'
        elif args.algorithm == 'trpo':
            hyperparams['checkpoint_gap'] = 100
            hyperparams['inner_type'] = 'log_likelihood'
        elif args.algorithm == 'trpomaml':
            hyperparams['checkpoint_gap'] = 50
            hyperparams['inner_type'] = 'log_likelihood'
        else:
            raise NotImplementedError

        hyperparams['n_evals'] = 1000
        hyperparams['n_updates'] = 5

    elif args.env == 'cheetah':

        if args.mode == 'goal-interpolate':
            hyperparams['train_env'] = HalfCheetahRandVelEnv
            hyperparams['test_env'] = HalfCheetahRandVelEnv
        else:
            raise NotImplementedError

        hyperparams['baseline'] = LinearFeatureBaseline
        hyperparams['rollouts_per_meta_task'] = 20
        hyperparams['max_path_length'] = 200

        hyperparams['discount'] = 0.99
        hyperparams['gae_lambda'] = 1
        hyperparams['normalize_adv'] = True
        hyperparams['positive_adv'] = False

        hyperparams['hidden_sizes'] = (64, 64)
        hyperparams['learn_std'] = True
        hyperparams['hidden_nonlinearity'] = tf.tanh
        hyperparams['output_nonlinearity'] = None

        hyperparams['learning_rate'] = args.learning_rate

        if args.algorithm == 'ppo':
            hyperparams['checkpoint_gap'] = 100
            hyperparams['inner_type'] = 'likelihood_ratio'
        elif args.algorithm == 'promp':
            hyperparams['checkpoint_gap'] = 50
            hyperparams['inner_type'] = 'likelihood_ratio'
        elif args.algorithm == 'trpo':
            hyperparams['checkpoint_gap'] = 100
            hyperparams['inner_type'] = 'log_likelihood'
        elif args.algorithm == 'trpomaml':
            hyperparams['checkpoint_gap'] = 50
            hyperparams['inner_type'] = 'log_likelihood'
        else:
            raise NotImplementedError

        hyperparams['n_evals'] = 1000
        hyperparams['n_updates'] = 5

    elif args.env == 'metaworld':

        if args.mode == 'ml1-push':
            hyperparams['train_env'] = ML1.get_train_tasks('push-v1')
            hyperparams['test_env'] = ML1.get_test_tasks('push-v1')
        elif args.mode == 'ml1-reach':
            hyperparams['train_env'] = ML1.get_train_tasks('reach-v1')
            hyperparams['test_env'] = ML1.get_test_tasks('reach-v1')
        elif args.mode == 'ml10':
            hyperparams['train_env'] = ML10.get_train_tasks()
            hyperparams['test_env'] = ML10.get_test_tasks()
        elif args.mode == 'ml45':
            hyperparams['train_env'] = ML45.get_train_tasks()
            hyperparams['test_env'] = ML45.get_test_tasks()
        else:
            raise NotImplementedError

        hyperparams['baseline'] = LinearFeatureBaseline
        hyperparams['rollouts_per_meta_task'] = 10
        hyperparams['max_path_length'] = 150

        hyperparams['discount'] = 0.99
        hyperparams['gae_lambda'] = 1
        hyperparams['normalize_adv'] = True
        hyperparams['positive_adv'] = False

        hyperparams['hidden_sizes'] = (100, 100)
        hyperparams['learn_std'] = True
        hyperparams['hidden_nonlinearity'] = tf.tanh
        hyperparams['output_nonlinearity'] = None

        hyperparams['learning_rate'] = args.learning_rate

        if args.algorithm == 'ppo':
            hyperparams['checkpoint_gap'] = 1000
            hyperparams['inner_type'] = 'likelihood_ratio'
        elif args.algorithm == 'promp':
            hyperparams['checkpoint_gap'] = 500
            hyperparams['inner_type'] = 'likelihood_ratio'
        elif args.algorithm == 'trpo':
            hyperparams['checkpoint_gap'] = 1000
            hyperparams['inner_type'] = 'log_likelihood'
        elif args.algorithm == 'trpomaml':
            hyperparams['checkpoint_gap'] = 500
            hyperparams['inner_type'] = 'log_likelihood'
        else:
            raise NotImplementedError

        hyperparams['n_evals'] = 1000
        hyperparams['n_updates'] = 5

    else:
        raise NotImplementedError

    run_experiment(hyperparams)
