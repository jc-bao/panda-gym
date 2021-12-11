from typing import Any, Dict, Tuple, Union

import numpy as np

from panda_gym.envs.core import Task
from panda_gym.utils import distance


class Tower(Task):
    def __init__(
        self,
        sim,
        distance_threshold=0.1,
        goal_xy_range=0.3,
        obj_xy_range=0.3,
        num_obj = 1,
    ) -> None:
        super().__init__(sim)
        self.distance_threshold = distance_threshold
        self.object_size = 0.04
        self.num_obj = num_obj
        self.goal_range_low = np.array([-goal_xy_range / 2, -goal_xy_range / 2, 0])
        self.goal_range_high = np.array([goal_xy_range / 2, goal_xy_range / 2, 0])
        self.obj_range_low = np.array([-obj_xy_range / 2, -obj_xy_range / 2, 0])
        self.obj_range_high = np.array([obj_xy_range / 2, obj_xy_range / 2, 0])
        with self.sim.no_rendering():
            self._create_scene()
            self.sim.place_visualizer(target_position=np.zeros(3), distance=0.9, yaw=45, pitch=-30)

    def _create_scene(self) -> None:
        self.sim.create_plane(z_offset=-0.4)
        self.sim.create_table(length=1.1, width=0.7, height=0.4, x_offset=-0.3)
        for i in range(self.num_obj):
            color = np.random.rand(3)
            self.sim.create_box(
                body_name="object"+str(i),
                half_extents=np.ones(3) * self.object_size / 2,
                mass=2.0,
                position=np.array([0.5*i, 0.0, self.object_size / 2]),
                rgba_color=np.append(color, 1),
            )
            self.sim.create_box(
                body_name="target"+str(i),
                half_extents=np.ones(3) * self.object_size / 2,
                mass=0.0,
                ghost=True,
                position=np.array([0.5*i, 0.0, 0.05]),
                rgba_color=np.append(color, 0.5),
            )

    def get_obs(self) -> np.ndarray:
        # position, rotation of the object
        obj_pos = []
        for i in range(self.num_obj):
            obj_pos.append(np.array(self.sim.get_base_position("object"+str(i))))
            obj_pos.append(np.array(self.sim.get_base_rotation("object"+str(i))))
            obj_pos.append(np.array(self.sim.get_base_velocity("object"+str(i))))
            obj_pos.append(np.array(self.sim.get_base_angular_velocity("object"+str(i))))
        observation = np.array(obj_pos).flatten()
        return observation

    def get_achieved_goal(self) -> np.ndarray:
        ag = []
        for i in range(self.num_obj):
            ag.append(np.array(self.sim.get_base_position("object"+str(i))))
        achieved_goal = np.array(ag).flatten()
        return achieved_goal

    def reset(self) -> None:
        self.goal = self._sample_goal()
        obj_pos = self._sample_objects()
        for i in range(self.num_obj):
            self.sim.set_base_pose("target"+str(i), self.goal[i*3:(i+1)*3], np.array([0.0, 0.0, 0.0, 1.0]))
            self.sim.set_base_pose("object"+str(i), obj_pos[i*3:(i+1)*3], np.array([0.0, 0.0, 0.0, 1.0]))

    def _sample_goal(self) -> np.ndarray:
        goals = []
        noise = self.np_random.uniform(self.goal_range_low, self.goal_range_high)
        for i in range(self.num_obj):
            goals.append(np.array([0.0, 0.0, self.object_size / 2* (2*i+1)])+noise)  # z offset for the cube center
        return np.array(goals).flatten()

    def _sample_objects(self) -> np.ndarray:
        obj_pos = [self.np_random.uniform(self.obj_range_low, self.obj_range_high)+[0.0, 0.0, self.object_size / 2]]
        for _ in range(self.num_obj-1):
            pos = self.np_random.uniform(self.obj_range_low, self.obj_range_high) + [0.0, 0.0, self.object_size / 2]
            while min(np.linalg.norm(obj_pos - pos, axis = 1)) < self.object_size*2:
                pos =  + self.np_random.uniform(self.obj_range_low, self.obj_range_high)+[0.0, 0.0, self.object_size / 2]
            obj_pos.append(pos)
        return np.array(obj_pos).flatten()

    def is_success(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> Union[np.ndarray, float]:
        # must be vectorized !!
        rew = self.compute_reward(achieved_goal, desired_goal, None)
        return float(abs(rew-self.num_obj)<0.001)

    def compute_reward(self, achieved_goal, desired_goal, info: Dict[str, Any]) -> Union[np.ndarray, float]:
        rew = 0
        for i in range(self.num_obj):
            d = distance(achieved_goal[i*3:i*3+3], desired_goal[i*3:i*3+3])
            rew += float(d < self.distance_threshold)
        return rew
