from typing import Any, Dict, Tuple, Union

import numpy as np

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
        goal_xy_range=0.2,
        obj_xy_range=0.2,
        num_blocks = 1,
        target_shape = 'any', 
        goal_center = 0.2,
        same_side_rate = 1
    ) -> None:
        super().__init__(sim)
        self.distance_threshold = distance_threshold
        self.object_size = 0.04
        self.same_side_rate = same_side_rate
        self.goal_center = goal_center
        self.num_blocks = num_blocks
        self.target_shape = target_shape
        self.get_ee_position0 = get_ee_position0
        self.get_ee_position1 = get_ee_position1
        self.goal_range_low = np.array([-goal_xy_range / 2, -goal_xy_range / 1.8, 0])
        self.goal_range_high = np.array([goal_xy_range / 2, goal_xy_range / 1.8, 0])
        self.obj_range_low = np.array([-obj_xy_range / 2, -obj_xy_range / 2, 0])
        self.obj_range_high = np.array([obj_xy_range / 2, obj_xy_range / 2, 0])
        with self.sim.no_rendering():
            self._create_scene()
            self.sim.place_visualizer(target_position=np.zeros(3), distance=0.9, yaw=45, pitch=-30)

    def _create_scene(self) -> None:
        self.sim.create_plane(z_offset=-0.4)
        self.sim.create_table(length=1., width=0.7, height=0.4, x_offset=-0.575)
        self.sim.create_table(length=1., width=0.7, height=0.4, x_offset=0.575)
        for i in range(self.num_blocks):
            color = np.random.rand(3)
            self.sim.create_box(
                body_name="object"+str(i),
                half_extents=np.array([3,1,1]) * self.object_size / 2,
                mass=2.0,
                position=np.array([0.5*i, 0.0, self.object_size / 2]),
                rgba_color=np.append(color, 1),
            )
            self.sim.create_box(
                body_name="target"+str(i),
                half_extents=np.ones(3) * self.object_size / 1.9,
                mass=0.0,
                ghost=True,
                position=np.array([0.5*i, 0.0, 0.05]),
                rgba_color=np.append(color, 0.5),
            )

    def get_obs(self) -> np.ndarray:
        # position, rotation of the object
        obj_pos = []
        for i in range(self.num_blocks):
            # [NOTE]trick: reset object orientation to aviod z rotation
            pos = self.sim.get_base_position("object"+str(i))
            ori = [0, self.sim.get_base_rotation("object"+str(i))[1], 0]
            self.sim.set_base_pose("object"+str(i), pos, ori)
            obj_pos.append(pos)
            obj_pos.append(pos - self.get_ee_position0())
            obj_pos.append(pos - self.get_ee_position1())
            obj_pos.append(ori)
            obj_pos.append(self.sim.get_base_velocity("object"+str(i)))
            obj_pos.append(self.sim.get_base_angular_velocity("object"+str(i)))
        observation = np.array(obj_pos).flatten()
        return observation

    def get_achieved_goal(self) -> np.ndarray:
        ag = []
        for i in range(self.num_blocks):
            ag.append(np.array(self.sim.get_base_position("object"+str(i))))
        ag.append(self.get_ee_position0())
        ag.append(self.get_ee_position1())
        achieved_goal = np.array(ag).flatten()
        return achieved_goal

    def reset(self) -> None:
        self.goal = self._sample_goal()
        obj_pos = self._sample_objects()
        for i in range(self.num_blocks):
            self.sim.set_base_pose("target"+str(i), self.goal[i*3:(i+1)*3], np.array([0.0, 0.0, 0.0, 1.0]))
            self.sim.set_base_pose("object"+str(i), obj_pos[i*3:(i+1)*3], np.array([0.0, 0.0, 0.0, 1.0]))

    def _sample_goal(self) -> np.ndarray:
        goals = []
        if self.target_shape == 'tower':
            noise = self.np_random.uniform(self.goal_range_low, self.goal_range_high)
            for i in range(self.num_blocks):
                goals.append(np.array([self.goal_center*np.random.choice([-1,1]), 0.0, self.object_size / 2* (2*i+1)])+noise)  # z offset for the cube center
            goals = np.array(goals).flatten()
            goals = np.append(goals, [0]*6) # Note: this one is used to calculate the gripper distance
        elif self.target_shape == 'any':
            goals = [self.np_random.uniform(self.goal_range_low, self.goal_range_high)+[self.goal_center*np.random.choice([-1,1]), 0.0, self.object_size / 2]]
            for i in range(1, self.num_blocks):
                x_pos = self.goal_center * np.random.choice([-1,1])
                goal =  self.np_random.uniform(self.goal_range_low, self.goal_range_high)+[x_pos, 0.0, self.object_size / 2]
                while min(np.linalg.norm(goals - goal, axis = 1)) < self.object_size*2:
                    goal =  self.np_random.uniform(self.obj_range_low, self.obj_range_high)+[x_pos, 0.0, self.object_size / 2]
            goals.append(goal)
            goals = np.array(goals).flatten()
            goals = np.append(goals, [0]*6) # Note: this one is used to calculate the gripper distance
        return np.array(goals)

    def _sample_objects(self) -> np.ndarray:
        x_pos = self.goal_center * (float(self.goal[0]>0)*2-1) * (float(np.random.random_sample()<self.same_side_rate)*2-1)
        obj_pos = [self.np_random.uniform(self.obj_range_low, self.obj_range_high)+[x_pos, 0.0, self.object_size / 2]]
        while min(np.linalg.norm(obj_pos - self.goal.reshape(-1,3), axis = 1)) < self.distance_threshold*1.2:
            obj_pos = [self.np_random.uniform(self.obj_range_low, self.obj_range_high)+[x_pos, 0.0, self.object_size / 2]]
        for i in range(1, self.num_blocks):
            x_pos = self.goal_center * (float(self.goal[3*i]>0)*2-1) * (float(np.random.random_sample()<self.same_side_rate)*2-1)
            pos =  self.np_random.uniform(self.obj_range_low, self.obj_range_high)+[x_pos, 0.0, self.object_size / 2]
            while min(np.linalg.norm(obj_pos - pos, axis = 1)) < self.object_size*2 or min(np.linalg.norm(obj_pos - self.goal.reshape(-1,3), axis = 1)) < self.distance_threshold*1.2:
                pos =  self.np_random.uniform(self.obj_range_low, self.obj_range_high)+[x_pos, 0.0, self.object_size / 2]
            obj_pos.append(pos)
        return np.array(obj_pos).flatten()

    def is_success(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> Union[np.ndarray, float]:
        # must be vectorized !!
        subgoal_distances = self.subgoal_distances(achieved_goal, desired_goal)
        return float(np.all([d < self.distance_threshold for d in subgoal_distances]))

    def compute_reward(self, achieved_goal, desired_goal, info: Dict[str, Any]) -> Union[np.ndarray, float]:
        subgoal_distances = self.subgoal_distances(achieved_goal, desired_goal)
        # Part1: number of stacked blocks
        rew = np.sum([(d < self.distance_threshold).astype(np.float32) for d in subgoal_distances])
        # Part2: if all block are set, encourage arm to move away [Note: use musk to boost relabel]
        rew += self.gripper_pos_far_from_goals(achieved_goal, desired_goal) * (np.abs(rew - self.num_blocks)<0.01).astype(float)
        return rew

    def subgoal_distances(self, goal_a, goal_b):
        return [
            np.linalg.norm(goal_a[..., i * 3:(i + 1) * 3] - goal_b[..., i * 3:(i + 1) * 3], axis=-1) for i in
            range(self.num_blocks)
        ]
    
    def gripper_pos_far_from_goals(self, achieved_goal, goal):
        gripper_pos = achieved_goal[...,-3:]
        block_goals = goal[...,:-3]
        distances = [
            np.linalg.norm(gripper_pos - block_goals[...,i*3:(i+1)*3], axis=-1) for i in range(self.num_blocks)
        ]
        return np.all([d > self.distance_threshold * 2 for d in distances], axis=0).astype(float)

    def change(self, config = None):
        if config != None:
            self.same_side_rate = config['same_side_rate']