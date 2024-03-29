import tensorflow as tf
import numpy as np
import time
from maml_zoo.logger import logger


class Trainer(object):
    """
    Performs steps for DRS

    Args:
        algo (Algo) :
        env (Env) :
        sampler (Sampler) :
        sample_processor (SampleProcessor) :
        baseline (Baseline) :
        policy (Policy) :
        n_itr (int) : Number of iterations to train for
        start_itr (int) : Number of iterations policy has already trained for, if reloading
        sess (tf.Session) : current tf session (if we loaded policy, for example)
    """
    def __init__(
            self,
            algo,
            env,
            sampler,
            sample_processor,
            policy,
            n_itr,
            start_itr=0,
            task=None,
            sess=None,
            ):
        self.algo = algo
        self.env = env
        self.sampler = sampler
        self.sample_processor = sample_processor
        self.baseline = sample_processor.baseline
        self.policy = policy
        self.n_itr = n_itr
        self.start_itr = start_itr
        self.task = task
        if sess is None:
            sess = tf.Session()
        self.sess = sess

    def train(self):
        """
        Trains policy on env using algo

        Pseudocode:
            for itr in n_itr:
                for step in num_inner_grad_steps:
                    sampler.sample()
                    algo.compute_updated_dists()
                algo.optimize_policy()
                sampler.update_goals()
        """
        with self.sess.as_default() as sess:

            # initialize uninitialized vars  (only initialize vars that were not loaded)
            uninit_vars = [var for var in tf.global_variables() if not sess.run(tf.is_variable_initialized(var))]
            sess.run(tf.variables_initializer(uninit_vars))

            start_time = time.time()
            for itr in range(self.start_itr, self.n_itr):
                self.task = self.env.sample_tasks(self.sampler.meta_batch_size)
                self.sampler.set_tasks(self.task)
                itr_start_time = time.time()
                logger.log("\n ---------------- Iteration %d ----------------" % itr)
                logger.log("Sampling set of tasks/goals for this meta-batch...")

                """ -------------------- Sampling --------------------------"""
                
                logger.log("Obtaining samples...")
                time_env_sampling_start = time.time()
                paths = self.sampler.obtain_samples(log=True, log_prefix='train-')
                sampling_time = time.time() - time_env_sampling_start

                """ ----------------- Processing Samples ---------------------"""

                logger.log("Processing samples...")
                time_proc_samples_start = time.time()
                samples_data = self.sample_processor.process_samples(paths, log='all', log_prefix='train-')
                proc_samples_time = time.time() - time_proc_samples_start

                # Combine paths from different tasks into one path for PPO/TRPO optimization
                samples_data_flattened = {}
                for i in range(len(samples_data)): # for each task
                    for key in samples_data[i].keys():
                        if i == 0:
                            samples_data_flattened[key] = samples_data[i][key]
                        else:
                            if key == 'env_infos':
                                for env_key in samples_data[i][key].keys():
                                    samples_data_flattened[key][env_key] = np.concatenate((samples_data_flattened[key][env_key], samples_data[i][key][env_key]), axis=0)
                            elif key == 'agent_infos':
                                for agent_key in samples_data[i][key].keys():
                                    samples_data_flattened[key][agent_key] = np.concatenate((samples_data_flattened[key][agent_key], samples_data[i][key][agent_key]), axis=0)
                            else:
                                samples_data_flattened[key] = np.concatenate((samples_data_flattened[key], samples_data[i][key]), axis=0)
                self.log_diagnostics(sum(paths.values(), []), prefix='train-')

                """ ------------------ Policy Update ---------------------"""

                logger.log("Optimizing policy...")
                # This needs to take all samples_data so that it can construct graph for meta-optimization.
                time_optimization_step_start = time.time()
                self.algo.optimize_policy(samples_data_flattened)

                """ ------------------- Logging Stuff --------------------------"""

                logger.logkv('Itr', itr)
                logger.logkv('n_timesteps', self.sampler.total_timesteps_sampled)

                logger.logkv('Time-Optimization', time.time() - time_optimization_step_start)
                logger.logkv('Time-SampleProc', np.sum(proc_samples_time))
                logger.logkv('Time-Sampling', sampling_time)

                logger.logkv('Time', time.time() - start_time)
                logger.logkv('ItrTime', time.time() - itr_start_time)

                logger.log("Saving snapshot...")
                params = self.get_itr_snapshot(itr)
                logger.save_itr_params(itr, params)
                logger.log("Saved")

                logger.dumpkvs()
                if itr == 0:
                    sess.graph.finalize()

        logger.log("Training finished")
        self.sess.close()

    def get_itr_snapshot(self, itr):
        """
        Gets the current policy and env for storage
        """
        return dict(itr=itr, policy=self.policy, env=self.env, baseline=self.baseline)

    def log_diagnostics(self, paths, prefix):
        
        self.env.log_diagnostics(paths, prefix)
        self.policy.log_diagnostics(paths, prefix)
        self.baseline.log_diagnostics(paths, prefix)

