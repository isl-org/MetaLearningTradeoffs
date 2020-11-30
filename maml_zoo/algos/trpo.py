from maml_zoo.logger import logger
from maml_zoo.algos.base import Algo
from maml_zoo.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer

import tensorflow as tf
from collections import OrderedDict

class TRPO(Algo):

    '''
    Algorithm for TRPO

    Args:
        policy (Policy): policy object
        name (str): tf variable scope
        step_size (int): trust region size for the policy optimization

    '''

    def __init__(
            self,
            *args,
            name='trpo',
            step_size=0.01,
            **kwargs
            ):
        
        super(TRPO, self).__init__(*args, **kwargs)

        self.step_size = step_size
        self.name = name
        self._optimization_keys = ['observations', 'actions', 'advantages', 'agent_infos']

        # Assume that we are not using a recurrent policy
        self.optimizer = ConjugateGradientOptimizer()

        self.build_graph()

    def build_graph(self):

        """
        Creates the computation graph
        """

        """ Create Variables """

        """ ----- Build graph for the meta-update ----- """
        self.meta_op_phs_dict = OrderedDict()
        obs_ph, action_ph, adv_ph, dist_info_old_ph, all_phs_dict = self._make_input_placeholders('train')
        self.meta_op_phs_dict.update(all_phs_dict)

        # dist_info_vars_for_next_step
        distribution_info_vars = self.policy.distribution_info_sym(obs_ph)
        hidden_ph, next_hidden_var = None, None

        """ Outer variables """
        # meta-objective
        likelihood_ratio = self.policy.distribution.likelihood_ratio_sym(action_ph, dist_info_old_ph, distribution_info_vars)
        mean_kl = tf.reduce_mean(self.policy.distribution.kl_sym(dist_info_old_ph, distribution_info_vars))
        surr_obj = - tf.reduce_mean(likelihood_ratio * adv_ph)

        self.optimizer.build_graph(
            loss=surr_obj,
            target=self.policy,
            input_ph_dict=self.meta_op_phs_dict,
            leq_constraint=(mean_kl, self.step_size),
        )

    def optimize_policy(self, samples_data, log=True):

        """
        Performs policy optimization
        """

        input_dict = self._extract_input_dict(samples_data, self._optimization_keys, prefix='train')
        logger.log("Computing KL before")
        mean_kl_before = self.optimizer.constraint_val(input_dict)
        logger.log("Computing loss before")
        loss_before = self.optimizer.loss(input_dict)

        logger.log("Optimizing")
        self.optimizer.optimize(input_dict)

        logger.log("Computing loss after")
        loss_after = self.optimizer.loss(input_dict)
        logger.log("Computing KL after")
        mean_kl = self.optimizer.constraint_val(input_dict)

        if log:
            logger.logkv('MeanKLBefore', mean_kl_before)
            logger.logkv('MeanKL', mean_kl)
            logger.logkv('LossBefore', loss_before)
            logger.logkv('LossAfter', loss_after)
            logger.logkv('dLoss', loss_before-loss_after)

