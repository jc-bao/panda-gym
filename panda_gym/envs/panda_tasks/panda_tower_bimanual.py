import numpy as np

from panda_gym.envs.core import RobotTaskEnv, BimanualTaskEnv
from panda_gym.envs.robots.panda_bound import PandaBound
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

    def __init__(self, render: bool = False, num_blocks: int = 1, control_type: str = "ee", curriculum_type = None) -> None:
        sim = PyBullet(render=render)
        robot0 = PandaBound(sim, index=0,block_gripper=False, base_position=np.array([-0.775, 0.0, 0.0]), control_type=control_type, base_orientation = [0,0,0,1])
        robot1 = PandaBound(sim, index=1, block_gripper=False, base_position=np.array([0.775, 0.0, 0.0]),control_type=control_type, base_orientation = [0,0,1,0])
        if curriculum_type == 'gravity':
            has_gravaty_rate = 0
            other_side_rate = 0.5
        elif curriculum_type == 'other_side':
            has_gravaty_rate = 1
            other_side_rate = 0
        else:
            has_gravaty_rate = 1
            other_side_rate = 0.5
        task = TowerBimanual(sim, num_blocks = num_blocks, curriculum_type = curriculum_type, other_side_rate = other_side_rate, has_gravaty_rate = has_gravaty_rate)
        super().__init__(robot0, robot1, task)
