import numpy as np

from panda_gym.envs.core import RobotTaskEnv, BimanualTaskEnv
from panda_gym.envs.robots.panda import Panda
from panda_gym.envs.tasks.tower_bimanual import TowerBimanual
from panda_gym.pybullet import PyBullet


class PandaTowerBimanualEnv(BimanualTaskEnv):
    """Stack task wih Panda robot.

    Args:
        render (bool, optional): Activate rendering. Defaults to False.
        num_blocks (int): >=1
        control_type (str, optional): "ee" to control end-effector position or "joints" to control joint values.
            Defaults to "ee".
    """

    def __init__(self, render: bool = False, num_blocks: int = 1, control_type: str = "ee", reward_type = 'sparse') -> None:
        sim = PyBullet(render=render)
        robot0 = Panda(sim, index=0,block_gripper=False, base_position=np.array([-0.775, 0.0, 0.0]), control_type=control_type, base_orientation = [0,0,0,1])
        robot1 = Panda(sim, index=1, block_gripper=False, base_position=np.array([0.775, 0.0, 0.0]),control_type=control_type, base_orientation = [0,0,1,0])
        task = TowerBimanual(sim, num_blocks = num_blocks, get_ee_position0=robot0.get_ee_position, get_ee_position1=robot1.get_ee_position)
        super().__init__(robot0, robot1, task)
