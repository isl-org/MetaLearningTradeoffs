from metaworld.benchmarks.base import Benchmark
from metaworld.core.serializable import Serializable
from metaworld.envs.mujoco.multitask_env import MultiClassMultiTaskEnv
from metaworld.envs.mujoco.env_dict import HARD_MODE_ARGS_KWARGS, HARD_MODE_CLS_DICT

class ML45(MultiClassMultiTaskEnv, Benchmark, Serializable):

    '''
    Modified from https://github.com/rlworkgroup/metaworld/blob/api-rework/metaworld/benchmarks/ml45.py
    '''

    def __init__(self, env_type='train', sample_all=False):
        assert env_type == 'train' or env_type == 'test'
        Serializable.quick_init(self, locals())

        cls_dict = HARD_MODE_CLS_DICT[env_type]
        args_kwargs = HARD_MODE_ARGS_KWARGS[env_type]

        super().__init__(
            task_env_cls_dict=cls_dict,
            task_args_kwargs=args_kwargs,
            sample_goals=True,
            obs_type='plain',
            sample_all=sample_all)

        if env_type == 'test':
            self._max_plain_dim = 9 # same as training

# NOTE: Use the api-rework branch of the metaworld repo. The current master branch has been reorganized
# and is not guaranteed to work with our code.
