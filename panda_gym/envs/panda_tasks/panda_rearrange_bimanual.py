import numpy as np

from panda_gym.envs.core import RobotTaskEnv, BimanualTaskEnv
from panda_gym.envs.robots.panda import Panda
from panda_gym.envs.robots.panda_bound import PandaBound
from panda_gym.envs.tasks.rearrange_bimanual import RearrangeBimanual
from panda_gym.pybullet import PyBullet


class PandaRearrangeBimanualEnv(BimanualTaskEnv):
    """Rearrange task wih Bimanual robot.

    Args:
        render (bool, optional): Activate rendering. Defaults to False.
        num_blocks (int): >=1
        control_type (str, optional): "ee" to control end-effector position or "joints" to control joint values.
            Defaults to "ee".
    """

    def __init__(self, render=False, store_trajectory=False, store_video=False, seed=0, task_kwargs={}) -> None:
        sim = PyBullet(render=render, timestep=1.0/240, n_substeps=20)
        ''' choose robot type '''
        robot0 = PandaBound(sim, index=0, block_gripper=False, base_position=np.array([-0.6, -0.4, 0.0]),
                            control_type='ee', base_orientation=[0, 0, np.sqrt(2)/2, np.sqrt(2)/2])
        robot1 = PandaBound(sim, index=1, block_gripper=False, base_position=np.array([0.6, 0.4, 0.0]),
                            control_type='ee', base_orientation=[0, 0, -np.sqrt(2)/2, np.sqrt(2)/2])
        task = RearrangeBimanual(
            sim, robot0.get_ee_position, robot1.get_ee_position, seed=seed, **task_kwargs)
        super().__init__(robot0, robot1, task, store_trajectory=store_trajectory,
                         store_video=store_video, good_init_pos_rate=0, seed=seed)