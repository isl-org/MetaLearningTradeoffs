import tensorflow as tf
import numpy as np
import time
from maml_zoo.logger import logger
import joblib
import os
import json


class NumpyEncoder(json.JSONEncoder):
    """Ensures json.dumps doesn't crash on numpy types
    See: https://stackoverflow.com/questions/27050108/convert-numpy-type-to-python/27050186#27050186
    """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NumpyEncoder, self).default(obj)

class Tester(object):
    """
    Performs evaluation of MAML
    Args:
        algo (Algo) :
        env (Env) :
        sampler (Sampler) :
        sample_processor (SampleProcessor) :
        baseline (Baseline) :
        policy (Policy) :
        load_policy (file path):
        n_evals (int) : Number of trials (tasks) evaluated
        n_updates (int) : Number of gradient update steps per trial (task)
        sess (tf.Session) : current tf session (if we loaded policy, for example)
    """
    def __init__(
            self,
            algo,
            env,
            sampler,
            sample_processor,
            policy,
            load_policy,
            output,
            n_evals,
            n_updates,
            sess = None,
        ):

        self.algo = algo
        self.env = env
        self.sampler = sampler
        self.sample_processor = sample_processor
        self.baseline = sample_processor.baseline
        self.policy = policy
        self.load_policy = load_policy
        self.output_dir = output
        self.n_evals = n_evals
        self.n_updates = n_updates
        if sess is None:
            sess = tf.Session()
        self.sess = sess

    def test(self):
        """
        Tests meta-policy on env using algo for finetuning
        Pseudocode:
            for eval in n_evals:
                sampler.update_tasks()
                for update in n_updates:
                    sampler.sample()
                    algo.optimize_policy()
                sampler.sample() and record final reward, success rate, etc.
        """
        with self.sess.as_default() as sess:

            # initialize uninitialized vars  (only initialize vars that were not loaded)
            uninit_vars = [var for var in tf.global_variables() if not sess.run(tf.is_variable_initialized(var))]
            sess.run(tf.variables_initializer(uninit_vars))

            # load in policy params
            load_path = self.load_policy
            loaded_policy = joblib.load(load_path)['policy'] # Policy instance
            loaded_params_dict = loaded_policy.get_param_values()

            start_time = time.time()
            for trial in range(self.n_evals):

                self.policy.set_params(loaded_params_dict) # reset params at each trial to trained one
                trial_start_time = time.time()

                logger.log("\n ---------------- Trial %d ----------------" % (trial+1))

                logger.log("Sampling task/goal for this trial...")
                self.sampler.update_tasks()

                list_sampling_time, list_step_time, list_proc_samples_time = [], [], []
                all_total_rewards = np.zeros(self.sampler.rollouts_per_meta_task*(self.n_updates+1))
                avg_rollout_reward = np.zeros(self.n_updates+1) # all_total_rewards averaged over each update

                for update in range(self.n_updates+1):
                    logger.log('** Update ' + str(update+1) + ' **')

                    """ -------------------- Sampling --------------------------"""
                    
                    logger.log("Obtaining samples...")
                    time_env_sampling_start = time.time()
                    paths = self.sampler.obtain_samples(log=True, log_prefix='Update_%d-' % (update+1))
                    for l in range(self.sampler.rollouts_per_meta_task):
                        rollout = paths[0][l] # dictionary mapping obs, actions, etc. to trajectory
                        rollout_rewards = rollout['rewards']
                        all_total_rewards[(l+self.sampler.rollouts_per_meta_task*update)] = np.sum(rollout_rewards)
                        avg_rollout_reward[update] += (np.sum(rollout_rewards)-avg_rollout_reward[update])/(l+1) 
                        # incremental average computation
                    list_sampling_time.append(time.time() - time_env_sampling_start)

                    """ ----------------- Processing Samples ---------------------"""

                    logger.log("Processing samples...")
                    time_proc_samples_start = time.time()
                    samples_data = self.sample_processor.process_samples(paths[0], log='all', log_prefix='Update_%d-' % (update+1))
                    list_proc_samples_time.append(time.time() - time_proc_samples_start)
                    self.log_diagnostics(sum(list(paths.values()), []), prefix='Update_%d-' % (update+1))

                    """ ------------------- Policy Update --------------------"""

                    time_step_start = time.time()
                    if update < self.n_updates:
                        logger.log("Computing policy updates...")
                        self.algo.optimize_policy(samples_data)
                    list_step_time.append(time.time() - time_step_start)

                    """ ------------------- Logging Stuff --------------------------"""

                    logger.logkv('Trial', trial+1)
                    logger.logkv('Update', update+1)
                    logger.logkv('Time', time.time() - start_time)
                    logger.logkv('TrialTime', time.time() - trial_start_time)
                    logger.logkv('Time-Sampling', np.sum(list_sampling_time))
                    logger.logkv('Time-SampleProc', np.sum(list_proc_samples_time))
                    logger.logkv('Time-Updating', np.sum(list_step_time))

                    logger.dumpkvs()

                # Write to results file
                with open(os.path.join(self.output_dir, 'evaluation.json'), 'a') as results_file:
                    results_file.write(json.dumps({
                        'model': self.load_policy,
                        'avg_update_reward': avg_rollout_reward,
                        'all_rewards': all_total_rewards,
                    }, cls=NumpyEncoder))
                    results_file.write('\n')

        logger.log("Testing finished")
        self.sess.close()

    def log_diagnostics(self, paths, prefix):
        
        self.env.log_diagnostics(paths, prefix)
        self.policy.log_diagnostics(paths, prefix)
        self.baseline.log_diagnostics(paths, prefix)
