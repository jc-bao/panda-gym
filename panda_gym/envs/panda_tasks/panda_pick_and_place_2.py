import numpy as np

from panda_gym.envs.core import BimanualTaskEnv
from panda_gym.envs.robots.panda import Panda
from panda_gym.envs.tasks.pick_and_place_2 import PickAndPlace2
from panda_gym.pybullet import PyBullet


class PandaPickAndPlace2Env(BimanualTaskEnv):
    """Pick and Place task wih Panda robot.

    Args:
        render (bool, optional): Activate rendering. Defaults to False.
        reward_type (str, optional): "sparse" or "dense". Defaults to "sparse".
        control_type (str, optional): "ee" to control end-effector position or "joints" to control joint values.
            Defaults to "ee".
    """

    def __init__(self, render: bool = False, reward_type: str = "sparse", control_type: str = "ee") -> None:
        sim = PyBullet(render=render)
        robot0 = Panda(sim, block_gripper=False, base_position=np.array([-0.6, 0.0, 0.0]), control_type=control_type, index = 0)
        robot1 = Panda(sim, block_gripper=False, base_position=np.array([0.5, 0.0, 0.0]), control_type=control_type, index = 1)
        task = PickAndPlace2(sim, reward_type=reward_type)
        super().__init__(robot0 = robot0, robot1 = robot1, task = task)