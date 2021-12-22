import numpy as np

from panda_gym.envs.core import BimanualTaskEnv
from panda_gym.envs.robots.panda import Panda
from panda_gym.envs.tasks.reach_bimanual import ReachBimanual
from panda_gym.pybullet import PyBullet


class PandaReachBimanualEnv(BimanualTaskEnv):
    """Stack task wih Panda robot.

    Args:
        render (bool, optional): Activate rendering. Defaults to False.
        num_blocks (int): >=1
        control_type (str, optional): "ee" to control end-effector position or "joints" to control joint values.
            Defaults to "ee".
    """

    def __init__(self, render: bool = False, control_type: str = "ee", has_object = False) -> None:
        sim = PyBullet(render=render)
        robot0 = Panda(sim, index=0,block_gripper=False, base_position=np.array([-0.775, 0.0, 0.0]), control_type=control_type, base_orientation = [0,0,0,1])
        robot1 = Panda(sim, index=1, block_gripper=False, base_position=np.array([0.775, 0.0, 0.0]),control_type=control_type, base_orientation = [0,0,1,0])
        task = ReachBimanual(sim, robot0.get_ee_position, robot1.get_ee_position, has_object = has_object)
        super().__init__(robot0, robot1, task)
