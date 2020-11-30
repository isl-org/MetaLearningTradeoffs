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
from maml_zoo.samplers.maml_sample_processor import MAMLSampleProcessor
from maml_zoo.algos.ppo import PPO
from maml_zoo.drs_trainer import Trainer

def run_experiment(hyperparams):

    EXP_NAME = 'ppo'
    EXP_NAME += '/'+hyperparams['mode']
    exp_dir = os.getcwd()+'/data/'+EXP_NAME+'/'+str(hyperparams['environment'])+'/run_'+str(hyperparams['run'])
    logger.configure(dir=exp_dir, format_strs=['stdout', 'log', 'csv'], snapshot_mode='gap', snapshot_gap=hyperparams['checkpoint_gap'])
    hyperparam_dict = dict(hyperparams)
    if hyperparams['environment'] == 'metaworld':
        # Metaworld classes cannot be saved to a json
        del hyperparam_dict['train_env']
    json.dump(hyperparam_dict, open(exp_dir+'/params.json', 'w'), indent=2, sort_keys=True, cls=ClassEncoder)

    # Set random seed
    set_seed(hyperparams['seed'])

    # Instantiate environment
    baseline = hyperparams['baseline']()
    if 'ml' in hyperparams['mode']:
        env = normalize(hyperparams['train_env'])
    else:
        env = normalize(hyperparams['train_env']())

    # Pick up training where it left off if necessary
    try:
        with open(exp_dir+'/itr.txt', 'r') as itr_file:
            for line in itr_file:
                start_itr = int(line)+1
    except:
        start_itr = 0

    # Instantiate learner classes
    with tf.Session() as sess:
        if start_itr > 0:
            load_path = exp_dir+'/itr_'+str(start_itr-1)+'.pkl'
            policy = joblib.load(load_path)['policy']
        else:
            policy = GaussianMLPPolicy(
                    name="train-policy",
                    obs_dim=np.prod(env.observation_space.shape),
                    action_dim=np.prod(env.action_space.shape),
                    hidden_sizes=hyperparams['hidden_sizes'],
                    learn_std=hyperparams['learn_std'],
                    hidden_nonlinearity=hyperparams['hidden_nonlinearity'],
                    output_nonlinearity=hyperparams['output_nonlinearity'],
            )

        sampler = MAMLSampler(
            env = env,
            policy = policy,
            rollouts_per_meta_task=hyperparams['rollouts_per_meta_task'],
            meta_batch_size=hyperparams['meta_batch_size'],
            max_path_length=hyperparams['max_path_length'],
            parallel=hyperparams['parallel'],
        )

        sample_processor = MAMLSampleProcessor(
            baseline=baseline,
            discount=hyperparams['discount'],
            gae_lambda=hyperparams['gae_lambda'],
            normalize_adv=hyperparams['normalize_adv'],
            positive_adv=hyperparams['positive_adv'],
        )

        algo = PPO(
            policy=policy,
            learning_rate=hyperparams['learning_rate'],
            clip_eps=hyperparams['clip_eps'],
            max_epochs=hyperparams['num_ppo_steps'],
        )

        trainer = Trainer(
            algo=algo,
            policy=policy,
            env=env,
            sampler=sampler,
            sample_processor=sample_processor,
            start_itr=start_itr,
            n_itr=hyperparams['n_itr'],
            sess=sess,
        )

        trainer.train()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--env', type=str)
    parser.add_argument('--mode', type=str)
    parser.add_argument('--run', type=int)

    parser.add_argument('--learning-rate', type=float)
    parser.add_argument('--seed', type=int)

    args = parser.parse_args()

    hyperparams = {
        'environment': args.env,
        'mode': args.mode,
        'run': args.run,
        'seed': args.seed
    }

    if args.env == 'walker':

        if args.mode == 'params-interpolate':
            hyperparams['train_env'] = Walker2DRandParamsEnv
        elif args.mode == 'goal-interpolate':
            hyperparams['train_env'] = Walker2DRandVelEnv
        else:
            raise NotImplementedError

        hyperparams['baseline'] = LinearFeatureBaseline
        hyperparams['rollouts_per_meta_task'] = 20
        hyperparams['max_path_length'] = 200
        hyperparams['parallel'] = True

        hyperparams['discount'] = 0.99
        hyperparams['gae_lambda'] = 1
        hyperparams['normalize_adv'] = True
        hyperparams['positive_adv'] = False

        hyperparams['hidden_sizes'] = (64, 64)
        hyperparams['learn_std'] = True
        hyperparams['hidden_nonlinearity'] = tf.tanh
        hyperparams['output_nonlinearity'] = None

        hyperparams['learning_rate'] = args.learning_rate
        if 'goal' in args.mode:
            hyperparams['num_ppo_steps'] = 5
        else:
            hyperparams['num_ppo_steps'] = 3
        hyperparams['clip_eps'] = 0.3

        hyperparams['n_itr'] = 2001
        hyperparams['checkpoint_gap'] = 100
        hyperparams['meta_batch_size'] = 40

    elif args.env == 'hopper':

        if args.mode == 'params-interpolate':
            hyperparams['train_env'] = HopperRandParamsEnv
        else:
            raise NotImplementedError

        hyperparams['baseline'] = LinearFeatureBaseline
        hyperparams['rollouts_per_meta_task'] = 20
        hyperparams['max_path_length'] = 200
        hyperparams['parallel'] = True

        hyperparams['discount'] = 0.99
        hyperparams['gae_lambda'] = 1
        hyperparams['normalize_adv'] = True
        hyperparams['positive_adv'] = False

        hyperparams['hidden_sizes'] = (64, 64)
        hyperparams['learn_std'] = True
        hyperparams['hidden_nonlinearity'] = tf.tanh
        hyperparams['output_nonlinearity'] = None

        hyperparams['learning_rate'] = args.learning_rate
        hyperparams['num_ppo_steps'] = 3
        hyperparams['clip_eps'] = 0.3

        hyperparams['n_itr'] = 2001
        hyperparams['checkpoint_gap'] = 100
        hyperparams['meta_batch_size'] = 40

    elif args.env == 'cheetah':

        if args.mode == 'goal-interpolate':
            hyperparams['train_env'] = HalfCheetahRandVelEnv
        else:
            raise NotImplementedError

        hyperparams['baseline'] = LinearFeatureBaseline
        hyperparams['rollouts_per_meta_task'] = 20
        hyperparams['max_path_length'] = 200
        hyperparams['parallel'] = True

        hyperparams['discount'] = 0.99
        hyperparams['gae_lambda'] = 1
        hyperparams['normalize_adv'] = True
        hyperparams['positive_adv'] = False

        hyperparams['hidden_sizes'] = (64, 64)
        hyperparams['learn_std'] = True
        hyperparams['hidden_nonlinearity'] = tf.tanh
        hyperparams['output_nonlinearity'] = None

        hyperparams['learning_rate'] = args.learning_rate
        hyperparams['num_ppo_steps'] = 5
        hyperparams['clip_eps'] = 0.3

        hyperparams['n_itr'] = 2001
        hyperparams['checkpoint_gap'] = 100
        hyperparams['meta_batch_size'] = 40

    elif args.env == 'metaworld':

        if args.mode == 'ml1-push':
            hyperparams['train_env'] = ML1.get_train_tasks('push-v1')
        elif args.mode == 'ml1-reach':
            hyperparams['train_env'] = ML1.get_train_tasks('reach-v1')
        elif args.mode == 'ml10':
            hyperparams['train_env'] = ML10.get_train_tasks()
        elif args.mode == 'ml45':
            hyperparams['train_env'] = ML45.get_train_tasks()
        else:
            raise NotImplementedError

        hyperparams['baseline'] = LinearFeatureBaseline
        hyperparams['rollouts_per_meta_task'] = 10
        hyperparams['max_path_length'] = 150
        hyperparams['parallel'] = True

        hyperparams['discount'] = 0.99
        hyperparams['gae_lambda'] = 1
        hyperparams['normalize_adv'] = True
        hyperparams['positive_adv'] = False

        hyperparams['hidden_sizes'] = (100, 100)
        hyperparams['learn_std'] = True
        hyperparams['hidden_nonlinearity'] = tf.tanh
        hyperparams['output_nonlinearity'] = None

        hyperparams['learning_rate'] = args.learning_rate
        hyperparams['num_ppo_steps'] = 5
        hyperparams['clip_eps'] = 0.3

        hyperparams['n_itr'] = 20001
        hyperparams['checkpoint_gap'] = 1000
        hyperparams['meta_batch_size'] = 20

    else:
        raise NotImplementedError

    run_experiment(hyperparams)

