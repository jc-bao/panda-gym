from typing import Any, Dict, Union, Tuple

import numpy as np

from panda_gym.envs.core import Task
from panda_gym.pybullet import PyBullet
from panda_gym.utils import distance


class PickAndPlace2(Task):
    def __init__(
        self,
        sim: PyBullet,
        reward_type: str = "sparse",
        distance_threshold: float = 0.05,
        goal_xy_range: float = 0.3,
        goal_z_range: float = 0.2,
        obj_xy_range: float = 0.3,
    ) -> None:
        super().__init__(sim)
        self.reward_type = reward_type
        self.distance_threshold = distance_threshold
        self.object_size = 0.04
        self.goal_range_low = np.array([-goal_xy_range / 2, -goal_xy_range / 2, 0])
        self.goal_range_high = np.array([goal_xy_range / 2, goal_xy_range / 2, goal_z_range])
        self.obj_range_low = np.array([-obj_xy_range / 2, -obj_xy_range / 2, 0])
        self.obj_range_high = np.array([obj_xy_range / 2, obj_xy_range / 2, 0])
        with self.sim.no_rendering():
            self._create_scene()
            self.sim.place_visualizer(target_position=[0.8,0,0], distance=0.9, yaw=45, pitch=-30)

    def _create_scene(self) -> None:
        """Create the scene."""
        self.sim.create_plane(z_offset=-0.4)
        self.sim.create_table(length=1.1, width=0.7, height=0.4, x_offset=-0.3)
        self.sim.create_table(length=1.1, width=0.7, height=0.4, x_offset=0.8)
        self.sim.create_box(
            body_name="object1",
            half_extents=np.ones(3) * self.object_size / 2,
            mass=1.0,
            position=np.array([0.0, 0.0, self.object_size / 2]),
            rgba_color=np.array([0.1, 0.9, 0.1, 1.0]),
        )
        self.sim.create_box(
            body_name="target1",
            half_extents=np.ones(3) * self.object_size / 2,
            mass=0.0,
            ghost=True,
            position=np.array([0.0, 0.0, 0.05]),
            rgba_color=np.array([0.1, 0.9, 0.1, 0.3]),
        )
        self.sim.create_box(
            body_name="object2",
            half_extents=np.ones(3) * self.object_size / 2,
            mass=1.0,
            position=np.array([0.0, 0.0, self.object_size / 2]),
            rgba_color=np.array([0.1, 0.1, 0.9, 1.0]),
        )
        self.sim.create_box(
            body_name="target2",
            half_extents=np.ones(3) * self.object_size / 2,
            mass=0.0,
            ghost=True,
            position=np.array([0.0, 0.0, 0.05]),
            rgba_color=np.array([0.1, 0.1, 0.9, 0.3]),
        )

    def get_obs(self) -> np.ndarray:
        # position, rotation of the object
        object1_position = np.array(self.sim.get_base_position("object1"))
        object1_rotation = np.array(self.sim.get_base_rotation("object1"))
        object1_velocity = np.array(self.sim.get_base_velocity("object1"))
        object1_angular_velocity = np.array(self.sim.get_base_angular_velocity("object1"))
        object2_position = np.array(self.sim.get_base_position("object2")) - [1.1,0,0] # trick: make center the same
        object2_rotation = np.array(self.sim.get_base_rotation("object2"))
        object2_velocity = np.array(self.sim.get_base_velocity("object2"))
        object2_angular_velocity = np.array(self.sim.get_base_angular_velocity("object2"))
        observation = np.concatenate(
            [
                object1_position,
                object1_rotation,
                object1_velocity,
                object1_angular_velocity,
                object2_position,
                object2_rotation,
                object2_velocity,
                object2_angular_velocity,
            ]
        )
        return observation

    def get_achieved_goal(self) -> np.ndarray:
        object1_position = self.sim.get_base_position("object1")
        object2_position = self.sim.get_base_position("object2")  - [1.1,0,0] # trick: make center the same
        achieved_goal = np.concatenate((object1_position, object2_position))
        return achieved_goal

    def get_goal(self) -> np.ndarray:
        goal = self.goal.copy()
        goal[3]-=1.1 # trick: make center the same
        return goal

    def reset(self) -> None:
        self.goal = self._sample_goal()
        object1_position, object2_position = self._sample_objects()
        self.sim.set_base_pose("target1", self.goal[:3], np.array([0.0, 0.0, 0.0, 1.0]))
        self.sim.set_base_pose("target2", self.goal[3:], np.array([0.0, 0.0, 0.0, 1.0]))
        self.sim.set_base_pose("object1", object1_position, np.array([0.0, 0.0, 0.0, 1.0]))
        self.sim.set_base_pose("object2", object2_position, np.array([0.0, 0.0, 0.0, 1.0]))


    def _sample_goal(self) -> np.ndarray:
        goal1 = np.array([0.0, 0.0, self.object_size / 2])  # z offset for the cube center
        goal2 = np.array([1.1, 0.0, self.object_size / 2])  # z offset for the cube center
        noise1 = self.np_random.uniform(self.goal_range_low, self.goal_range_high)
        noise2 = self.np_random.uniform(self.goal_range_low, self.goal_range_high)
        goal1 += noise1
        goal2 += noise2
        return np.concatenate((goal1, goal2))


    def _sample_objects(self) -> Tuple[np.ndarray, np.ndarray]:
        # while True:  # make sure that cubes are distant enough
        object1_position = np.array([0.0, 0.0, self.object_size / 2])
        object2_position = np.array([1.1, 0.0, self.object_size / 2])
        noise1 = self.np_random.uniform(self.obj_range_low, self.obj_range_high)
        noise2 = self.np_random.uniform(self.obj_range_low, self.obj_range_high)
        object1_position += noise1
        object2_position += noise2
        return object1_position, object2_position

    def is_success(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> Union[np.ndarray, float]:
        delta = (achieved_goal - desired_goal).reshape(-1 ,3)
        dist_block2goal = np.linalg.norm(delta, axis=-1)
        return float(np.all(dist_block2goal < self.distance_threshold))

    def compute_reward(self, achieved_goal, desired_goal, info: Dict[str, Any]) -> Union[np.ndarray, float]:
        delta = (achieved_goal - desired_goal).reshape(-1, 2 ,3)
        dist_block2goal = np.linalg.norm(delta, axis=-1)
        rew = -np.sum(dist_block2goal>self.distance_threshold, axis=-1, dtype = float)
        if len(rew) == 1 :
            return rew[0]
        else: # to process multi dimension input
            return rew
