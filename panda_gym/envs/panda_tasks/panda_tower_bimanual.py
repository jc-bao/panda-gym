import numpy as np

from panda_gym.envs.core import RobotTaskEnv, BimanualTaskEnv
from panda_gym.envs.robots.panda import Panda
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

    def __init__(self, render: bool = False, num_blocks: int = 1, control_type: str = "ee", curriculum_type = None, \
        use_bound = False, use_musk = False, shared_op_space = False, gap_distance = 0.23, max_delay_steps = 0, \
            target_shape = 'any', reach_once = False, single_side = False, block_length = 5, os_rate = None, \
                max_num_need_handover = 10, max_move_per_step = 0.05, noise_obs = False, store_trajectory = False, \
                    parallel_robot = False, exchange_only = False, reward_type = 'normal', subgoal_generation = False) -> None:
        if gap_distance == None:
            gap_distance = block_length*0.04+0.03
        sim = PyBullet(render=render, timestep=1.0/240, n_substeps=20)
        if use_bound:
            robot0 = PandaBound(sim, index=0,block_gripper=False, base_position=np.array([-0.775, 0.0, 0.0]), control_type=control_type, base_orientation = [0,0,0,1])
            robot1 = PandaBound(sim, index=1, block_gripper=False, base_position=np.array([0.775, 0.0, 0.0]),control_type=control_type, base_orientation = [0,0,1,0]) 
        elif shared_op_space:
            base_x = 0.72 if gap_distance==0 else 0.5
            robot0 = Panda(sim, index=0,block_gripper=False, base_position=np.array([-base_x, 0.0, 0.0]), control_type=control_type, base_orientation = [0,0,0,1])
            robot1 = Panda(sim, index=1, block_gripper=False, base_position=np.array([base_x, 0.0, 0.0]),control_type=control_type, base_orientation = [0,0,1,0])
            robot0.neutral_joint_values = np.array([-8.62979537e-04, 6.67109107e-02, 8.93407819e-04, -2.71219648e+00, \
                -1.67254799e-04, 2.77888080e+00, 7.85577202e-01, 0, 0])
            robot1.neutral_joint_values = np.array([-8.62979537e-04, 6.67109107e-02, 8.93407819e-04, -2.71219648e+00, \
                -1.67254799e-04, 2.77888080e+00, 7.85577202e-01, 0, 0])
        elif parallel_robot:
            robot0 = Panda(sim, index=0,block_gripper=False, base_position=np.array([-0.7, -0.4, 0.0]), control_type=control_type, base_orientation = [0,0,np.sqrt(2)/2,np.sqrt(2)/2], max_move_per_step=max_move_per_step, noise_obs = noise_obs)
            robot1 = Panda(sim, index=1, block_gripper=False, base_position=np.array([0.7, 0.4, 0.0]),control_type=control_type, base_orientation = [0,0,-np.sqrt(2)/2,np.sqrt(2)/2], max_move_per_step=max_move_per_step, noise_obs = noise_obs)
            robot0.neutral_joint_values = np.array([-0.12593504068329087, 0.2317273297268855, -0.39855150509205445, -2.4891976287831454, 0.2079942120401763, 2.694932460185828, 1.6530547720778208])
            robot1.neutral_joint_values = np.array([-0.12593504068329087, 0.2317273297268855, -0.39855150509205445, -2.4891976287831454, 0.2079942120401763, 2.694932460185828, 1.6530547720778208])
        else:
            robot0 = Panda(sim, index=0,block_gripper=False, base_position=np.array([-0.775, 0.0, 0.0]), control_type=control_type, base_orientation = [0,0,0,1], max_move_per_step=max_move_per_step, noise_obs = noise_obs)
            robot1 = Panda(sim, index=1, block_gripper=False, base_position=np.array([0.775, 0.0, 0.0]),control_type=control_type, base_orientation = [0,0,1,0], max_move_per_step=max_move_per_step, noise_obs = noise_obs)
        # robot0.neutral_joint_values = np.array([0.01, 0.54, 0.003, -2.12, -0.003, 2.67, 0.80, 0.00, 0.00])
        # robot1.neutral_joint_values = np.array([0.01, 0.54, 0.003, -2.12, -0.003, 2.67, 0.80, 0.00, 0.00])
        if curriculum_type == 'gravity':
            has_gravaty_rate = 0
            other_side_rate = 0.5
            obj_not_in_hand_rate = 1
            goal_xyz_range=[0.4, 0.3, 0.2]
            obj_xyz_range=[0.3, 0.3, 0]
            goal_not_in_obj_rate = 1
        elif curriculum_type == 'hand_range_num_mix':
            has_gravaty_rate = 1
            other_side_rate = 0.75
            obj_not_in_hand_rate = 0.5
            goal_xyz_range=[0.3, 0.3, 0.2]
            obj_xyz_range=[0.2, 0.3, 0]
            # goal_xyz_range=[0.9, 0.3, 0.2]
            # obj_xyz_range=[0.7, 0.3, 0]
            goal_not_in_obj_rate = 1
        elif curriculum_type == 'other_side' or curriculum_type == 'mix' or curriculum_type == 'os_num_mix':
            has_gravaty_rate = 1
            other_side_rate = 0
            obj_not_in_hand_rate = 0.8
            goal_xyz_range=[0.4, 0.3, 0.2]
            obj_xyz_range=[0.3, 0.3, 0]
            goal_not_in_obj_rate = 1
        elif curriculum_type == 'in_hand' or curriculum_type == 'hand_num_mix':
            has_gravaty_rate = 1
            other_side_rate = 0.6
            obj_not_in_hand_rate = 0.4
            goal_xyz_range=[0.4, 0.3, 0.2]
            obj_xyz_range=[0.3, 0.3, 0]
            goal_not_in_obj_rate = 1
        elif curriculum_type == 'goal_z':
            has_gravaty_rate = 1
            other_side_rate = 0.5
            obj_not_in_hand_rate = 1
            goal_xyz_range=[0.4, 0.3, 0]
            obj_xyz_range=[0.3, 0.3, 0]
            goal_not_in_obj_rate = 1
        elif curriculum_type == 'goal_in_obj':
            has_gravaty_rate = 1
            other_side_rate = 0.5
            obj_not_in_hand_rate = 1
            goal_xyz_range=[0.4, 0.3, 0.2]
            obj_xyz_range=[0.3, 0.3, 0]
            goal_not_in_obj_rate = 0
        else:
            has_gravaty_rate = 1
            other_side_rate = 0.6
            obj_not_in_hand_rate = 0.8
            goal_xyz_range=[0.3, 0.4, 0] if shared_op_space else [0.4, 0.3, 0.2]
            obj_xyz_range= [0.3, 0.4, 0] if shared_op_space else [0.3, 0.3, 0]
            goal_not_in_obj_rate = 1
        if not os_rate == None: 
            other_side_rate = os_rate
        if gap_distance == 0:
            goal_xyz_range = [0.4, 0.3, 0]
            obj_xyz_range = goal_xyz_range
        task = TowerBimanual(sim, robot0.get_ee_position, robot1.get_ee_position, num_blocks = num_blocks, \
            curriculum_type = curriculum_type, other_side_rate = other_side_rate, has_gravaty_rate = has_gravaty_rate, \
                use_musk = use_musk, obj_not_in_hand_rate = obj_not_in_hand_rate, goal_xyz_range=goal_xyz_range, \
                    obj_xyz_range = obj_xyz_range, goal_not_in_obj_rate = goal_not_in_obj_rate, \
                        shared_op_space = shared_op_space, gap_distance = gap_distance, target_shape = target_shape, \
                            reach_once = reach_once, single_side = single_side, block_length=block_length, \
                                max_num_need_handover=max_num_need_handover, max_move_per_step = max_move_per_step, \
                                    noise_obs=noise_obs, exchange_only = exchange_only, parallel_robot = parallel_robot, \
                                        reward_type=reward_type, subgoal_generation=subgoal_generation)
        super().__init__(robot0, robot1, task, max_delay_steps = max_delay_steps, store_trajectory = store_trajectory)