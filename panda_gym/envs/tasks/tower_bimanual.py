from typing import Any, Dict, Tuple, Union

import numpy as np
import panda_gym

from panda_gym.envs.core import Task
from panda_gym.utils import distance

'''
Note: this environment add gripper pos to achieved goal. But as the goal and achieved goal will keep the same, it will
not matter
'''

class TowerBimanual(Task):
    def __init__(
        self,
        sim,
        get_ee_position0,
        get_ee_position1,
        distance_threshold=0.05,
        goal_xyz_range=[0.4, 0.3, 0.2],
        obj_xyz_range=[0.3, 0.3, 0],
        num_blocks = 1,
        target_shape = 'any',
        curriculum_type = None,
        other_side_rate = 0.5,
        has_gravaty_rate = 1,
        use_musk = False,
        obj_not_in_hand_rate = 1, 
        goal_not_in_obj_rate = 1, 
        shared_op_space = False, 
        gap_distance = 0.15, 
    ) -> None:
        super().__init__(sim)
        self.load_tabel = True
        self.gap_distance = gap_distance
        self.shared_op_space = shared_op_space
        self.distance_threshold = distance_threshold
        self.get_ee_position0 = get_ee_position0
        self.get_ee_position1 = get_ee_position1
        self.object_size = 0.04
        self.use_musk = use_musk
        self.other_side_rate = other_side_rate
        self.has_gravaty_rate = has_gravaty_rate
        self.curriculum_type = curriculum_type # gravity or other_side
        self.obj_not_in_hand_rate = obj_not_in_hand_rate
        self.goal_not_in_obj_rate = goal_not_in_obj_rate
        self.max_num_blocks = 6
        self.num_blocks = num_blocks
        self._max_episode_steps = 70*self.num_blocks
        self.target_shape = target_shape
        self.goal_xyz_range = goal_xyz_range
        self.num_not_musk = 1
        self.goal_range_low = np.array([0, -goal_xyz_range[1]/2, self.object_size/2])
        self.goal_range_high = np.array(goal_xyz_range) + self.goal_range_low
        self.obj_range_low = np.array([0.1*(not self.shared_op_space), -obj_xyz_range[1] / 2, self.object_size/2])
        self.obj_range_high = np.array(obj_xyz_range) + self.obj_range_low
        with self.sim.no_rendering():
            self._create_scene()
            self.sim.place_visualizer(target_position=np.zeros(3), distance=0.9, yaw=45, pitch=-30)
        self.ignore_goal_size = 0
        self.goal_size = 3

    def _create_scene(self) -> None:
        self.sim.create_plane(z_offset=-0.4)
        table_x = 0.3 if self.shared_op_space else 0.5 + self.gap_distance/2
        self.sim.create_table(length=1., width=0.7, height=0.4, x_offset=(-table_x))
        self.sim.create_table(length=1., width=0.7, height=0.4, x_offset=(table_x))
        # obj_range_size_half = (self.obj_range_high - self.obj_range_low)/ 2
        # obj_range_pos_0 = (self.obj_range_high + self.obj_range_low)/ 2
        # obj_range_pos_1 = (self.obj_range_high + self.obj_range_low)/ 2
        # obj_range_pos_1[0] = -obj_range_pos_1[0]
        # self.sim.create_box(
        #     body_name="debug_obj_0",
        #     half_extents=obj_range_size_half,
        #     mass=0.0,
        #     ghost=True,
        #     position=obj_range_pos_0,
        #     rgba_color=np.array([0, 0, 1, 0.1]),
        # )
        # self.sim.create_box(
        #     body_name="debug_obj_1",
        #     half_extents=obj_range_size_half,
        #     mass=0.0,
        #     ghost=True,
        #     position=obj_range_pos_1,
        #     rgba_color=np.array([0, 0, 1, 0.1]),
        # )
        # goal_range_size_half = (self.goal_range_high - self.goal_range_low)/ 2
        # goal_range_pos_0 = (self.goal_range_high + self.goal_range_low)/ 2
        # goal_range_pos_1 = (self.goal_range_high + self.goal_range_low)/ 2
        # goal_range_pos_1[0] = -goal_range_pos_1[0]
        # self.sim.create_box(
        #     body_name="debug_goal_0",
        #     half_extents=goal_range_size_half,
        #     mass=0.0,
        #     ghost=True,
        #     position=goal_range_pos_0,
        #     rgba_color=np.array([0, 1, 0, 0.05]),
        # )
        # self.sim.create_box(
        #     body_name="debug_goal_1",
        #     half_extents=goal_range_size_half,
        #     mass=0.0,
        #     ghost=True,
        #     position=goal_range_pos_1,
        #     rgba_color=np.array([0, 1, 0, 0.05]),
        # )
        self.use_small_obj = (self.gap_distance==0 or self.shared_op_space)
        for i in range(self.max_num_blocks):
            color = np.random.rand(3)
            self.sim.create_box(
                body_name="object"+str(i),
                half_extents=np.array([1 if self.use_small_obj else 3,1,1]) * self.object_size / 2,
                mass=0.5,
                position=np.array([1, 0.1*i - 0.3, self.object_size / 2]),
                rgba_color=np.append(color, 1),
            )
            self.sim.create_sphere(
                body_name="target"+str(i),
                radius=self.object_size / 1.9,
                mass=0.0,
                ghost=True,
                position=np.array([1, 0.1*i-0.3, 0.05]),
                rgba_color=np.append(color, 0.5),
            )

    def get_obs(self) -> np.ndarray:
        # position, rotation of the object
        obs = []
        for i in range(self.num_blocks):
            # [NOTE]trick: reset object orientation to aviod z rotation
            pos = self.sim.get_base_position("object"+str(i))
            ori = np.array([0, self.sim.get_base_rotation("object"+str(i))[1], 0])
            self.sim.set_base_pose("object"+str(i), pos, ori)
            obs.append(pos)
            obs.append(ori)
            obs.append(self.sim.get_base_velocity("object"+str(i)))
            obs.append(self.sim.get_base_angular_velocity("object"+str(i)))
        observation = np.array(obs).flatten()
        return observation

    def get_achieved_goal(self) -> np.ndarray:
        ag = []
        for i in range(self.num_blocks):
            ag.append(np.array(self.sim.get_base_position("object"+str(i))))
        ag = np.array(ag)
        if self.use_musk:
            ag[self.musk_index] = np.zeros(3)
        achieved_goal = (ag).flatten()
        return achieved_goal

    def reset(self) -> None:
        obj_pos = self._sample_objects()
        self.goal = self._sample_goal(obj_pos)
        for i in range(self.num_blocks):
            self.sim.set_base_pose("target"+str(i), self.goal[i*3:(i+1)*3], np.array([0.0, 0.0, 0.0, 1.0]))
            self.sim.set_base_pose("object"+str(i), obj_pos[i*3:(i+1)*3], np.array([0.0, 0.0, 0.0, 1.0]))
        # set gravity
        if self.np_random.uniform() < self.has_gravaty_rate:
            self.sim.physics_client.setGravity(0, 0, -9.81)
        else:
            self.sim.physics_client.setGravity(0, 0, 0)
        assert len(obj_pos)%self.num_blocks==0, 'task observation shape linear to num blocks'
        obs = self.get_obs()
        self.obj_obs_size = int(len(obs)/self.num_blocks)

    def _sample_goal(self, obj_pos) -> np.ndarray:
        goals = []
        if self.target_shape == 'tower':
            base_pos = self.np_random.uniform(self.goal_range_low, self.goal_range_high)
            base_pos[0] = np.random.choice([-1,1])* base_pos[0]
            for i in range(self.num_blocks):
                goals.append(np.array([0, 0, self.object_size*i])+base_pos)  # z offset for the cube center
            goals = np.array(goals).flatten()
            # goals = np.append(goals, [0]*6) # Note: this one is used to calculate the gripper distance
        elif self.target_shape == 'any':
            need_handover = False
            positive_side_goal_idx = []
            negative_side_goal_idx = []
            for i in range(self.num_blocks):
                obj_side = (float(obj_pos[i*3]>0)*2-1)
                if_same_side = (float(self.np_random.uniform()>self.other_side_rate)*2-1)
                goal_side = obj_side * if_same_side
                need_handover = need_handover or (if_same_side < 0)
                while True:
                    # sample goal
                    goal = self.np_random.uniform(self.obj_range_low, self.obj_range_high)
                    goal[0] = goal_side*goal[0]
                    if len(goals) == 0:
                        if goal_side > 0:
                            positive_side_goal_idx.append(i)
                        else:
                            negative_side_goal_idx.append(i)
                        goals.append(goal)
                        break
                    # if goal is satisfied, append
                    elif (np.linalg.norm(goal - obj_pos[i*3:i*3+3])) > self.distance_threshold*1.2:
                        x_size = self.object_size*1.5 if self.use_small_obj else self.object_size*4.5
                        if  min(abs(goals - goal)[..., 0]) > x_size or \
                            min(abs(goals - goal)[..., 1]) > self.object_size*1.5:
                            if goal_side > 0:
                                positive_side_goal_idx.append(i)
                            else:
                                negative_side_goal_idx.append(i)
                            goals.append(goal)
                            break
            if not need_handover: # make object in the air to learn pnp
                if len(positive_side_goal_idx) > 0:
                    idx = np.random.choice(positive_side_goal_idx)
                    goals[idx] = np.random.uniform(self.goal_range_low, self.goal_range_high)
                if len(negative_side_goal_idx) > 0: 
                    idx = np.random.choice(negative_side_goal_idx)
                    new_goal = np.random.uniform(self.goal_range_low, self.goal_range_high)
                    new_goal[0] = -new_goal[0]
                    goals[idx] = new_goal
            if self.curriculum_type == 'goal_in_obj':
                # goal in object rate, curriculum trick
                new_idx = np.arange(self.num_blocks)
                np.random.shuffle(new_idx)
                relabel_num = 0
                for j in new_idx[1:]:
                    if self.np_random.uniform() > self.goal_not_in_obj_rate: # get goal to obj
                        goals[j] = obj_pos[j*3:j*3+3]
                        relabel_num += 1
                if relabel_num == (self.num_blocks - 1):
                    new_goal = np.random.uniform(self.goal_range_low, self.goal_range_high)
                    new_goal[0] = np.random.choice([-1,1])*new_goal[0]
                    goals[new_idx[0]] = new_goal
        goals = np.array(goals)
        if self.use_musk:
            num_musk = self.num_blocks - self.num_not_musk
            self.musk_index = self.np_random.choice(np.arange(self.num_blocks), num_musk, replace=False)
            goals[self.musk_index] = np.zeros(3)
        goals = (goals).flatten()
        return goals

    def _sample_objects(self) -> np.ndarray:
        obj_pos = []
        # old version: sample obj according to goal
        # same_side_rate = 1 - self.other_side_rate
        # for i in range(self.num_blocks):
        #     # get target object side
        #     goal_side = (float(self.goal[i*3]>0)*2-1)
        #     if_same_side = (float(self.np_random.uniform()<same_side_rate)*2-1)
        #     self.if_same_side.append(if_same_side)
        #     obj_side = goal_side * if_same_side
        #     while True:
        #         pos = self.np_random.uniform(self.obj_range_low, self.obj_range_high)
        #         pos[0] = obj_side * pos[0]
        #         if (np.linalg.norm(pos - self.goal[i*3:i*3+3])) > self.distance_threshold*1.2:
        #             if len(obj_pos) == 0:
        #                 break
        #             elif min(np.linalg.norm(obj_pos - pos, axis = 1)) > self.object_size*2:
        #                 break
        #     obj_pos.append(pos)
        for i in range(self.num_blocks):
            # get target object side
            while True:
                pos = self.np_random.uniform(self.obj_range_low, self.obj_range_high)
                pos[0] = (self.np_random.choice([-1,1])) * pos[0]
                if len(obj_pos) == 0:
                    break
                elif min(np.linalg.norm(obj_pos - pos, axis = -1)) > self.object_size*4:
                    break
            obj_pos.append(pos)
        choosed_block_id = np.random.choice(np.arange(self.num_blocks))
        if self.np_random.uniform()>self.obj_not_in_hand_rate:
            if self.np_random.uniform()>0.5: 
                obj_pos[choosed_block_id] = self.get_ee_position0()
            else:
                obj_pos[choosed_block_id] = self.get_ee_position1()
        return np.array(obj_pos).flatten()

    def is_success(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> Union[np.ndarray, float]:
        # must be vectorized !!
        dists = [
            np.linalg.norm(achieved_goal[..., i * 3:(i + 1) * 3] - desired_goal[..., i * 3:(i + 1) * 3], axis=-1)
            for i in range(self.num_blocks)
        ]
        return float(np.all([d < self.distance_threshold for d in dists]))

    def compute_reward(self, achieved_goal, desired_goal, info: Dict[str, Any]) -> Union[np.ndarray, float]:
        delta = (achieved_goal - desired_goal).reshape(-1, self.num_blocks ,3)
        dist_block2goal = np.linalg.norm(delta, axis=-1)
        rew = -np.sum(dist_block2goal>self.distance_threshold, axis=-1, dtype = float)
        if self.shared_op_space or self.gap_distance == 0:
            ee_dis = info['ee_pos'][0] - info['ee_pos'][1]
            d = np.sqrt(0.5*(np.square(ee_dis[0]/0.12) + np.square(ee_dis[1]/0.24)))
            dis_rew = max(1.5-d, 0)
            rew -= dis_rew
        if len(rew) == 1 :
            return rew[0]
        else: # to process multi dimension input
            return rew

    def change(self, config = None):
        if self.curriculum_type == 'gravity':
            self.has_gravaty_rate = config
        elif self.curriculum_type == 'other_side':
            self.other_side_rate = config
        elif self.curriculum_type == 'in_hand':
            self.obj_not_in_hand_rate = config
        elif self.curriculum_type == 'goal_in_obj':
            self.goal_not_in_obj_rate = config
        elif self.curriculum_type == 'goal_z':
            self.goal_xyz_range[-1] = config*0.2
            self.goal_range_high = np.array(self.goal_xyz_range) + self.goal_range_low
        elif self.curriculum_type == 'num_blocks':
            self.num_blocks = int(config)
            self._max_episode_steps = 70 * self.num_blocks
        elif self.curriculum_type == 'musk':
            if self.num_not_musk < self.num_blocks:
                self.num_not_musk = int(config*self.num_blocks)+1
        elif self.curriculum_type == 'mix': # learn 1pnp -> expand num
            self.num_blocks = min(int(config), 6)
            self._max_episode_steps = 80 * self.num_blocks
        elif self.curriculum_type == 'mix_2': # learn rearrange fisrt -> multi
            # expand number of block first
            if self.num_blocks < 4:
                self.num_blocks = int(config)
                self._max_episode_steps = 60 * self.num_blocks
            else: # expand otherside rate
                self.other_side_rate = 0.72 # 50%need two handover
                self._max_episode_steps = 80 * self.num_blocks
        