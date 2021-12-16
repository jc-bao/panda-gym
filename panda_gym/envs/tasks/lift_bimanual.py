from typing import Any, Dict, Tuple, Union

import numpy as np

from panda_gym.envs.core import Task
from panda_gym.utils import distance

'''
Note: this environment add gripper pos to achieved goal. But as the goal and achieved goal will keep the same, it will
not matter
'''

class LiftBimanual(Task):
    def __init__(
        self,
        sim,
        distance_threshold=0.05,
        goal_xy_range=0.2,
        obj_xy_range=0.2,
        heavy_object_rate = 0.0,
    ) -> None:
        super().__init__(sim)
        self.distance_threshold = distance_threshold
        self.heavy_object_rate = heavy_object_rate
        self.object_size = 0.04
        self.goal_range_low = np.array([-goal_xy_range / 2, -goal_xy_range / 1.8, 0])
        self.goal_range_high = np.array([goal_xy_range / 2, goal_xy_range / 1.8, 0.2])
        self.obj_range_low = np.array([-obj_xy_range / 2, -obj_xy_range / 2, 0])
        self.obj_range_high = np.array([obj_xy_range / 2, obj_xy_range / 2, 0])
        with self.sim.no_rendering():
            self._create_scene()
            self.sim.place_visualizer(target_position=np.zeros(3), distance=0.9, yaw=45, pitch=-30)
        self.num_step = 0

    def _create_scene(self) -> None:
        self.sim.create_plane(z_offset=-0.4)
        self.sim.create_table(length=1., width=0.7, height=0.4, x_offset=-0.575)
        self.sim.create_table(length=1., width=0.7, height=0.4, x_offset=0.575)
        color = np.random.rand(3)
        self.sim.create_box(
            body_name="object",
            half_extents=np.array([12,1,1]) * self.object_size / 2,
            mass=0.10,
            position=np.array([0.5, 0.0, self.object_size / 2]),
            rgba_color=np.append(color, 1),
        )
        self.sim.create_box(
            body_name="target",
            half_extents=np.array([12,1,1]) * self.object_size / 2,
            mass=0.0,
            ghost=True,
            position=np.array([0.5, 0.0, 0.05]),
            rgba_color=np.append(color, 0.5),
        )

    def get_obs(self) -> np.ndarray:
        # position, rotation of the object
        object_position = self.sim.get_base_position("object")
        object_rotation = np.array([0, self.sim.get_base_rotation("object")[1], 0])
        self.sim.set_base_pose("object", object_position, object_rotation) # trick: keep not rotate
        object_velocity = self.sim.get_base_velocity("object")
        object_angular_velocity = self.sim.get_base_angular_velocity("object")
        observation = np.concatenate([object_position, object_rotation, object_velocity, object_angular_velocity])
        self.num_step += 1
        return observation

    def get_achieved_goal(self) -> np.ndarray:
        object_position = np.array(self.sim.get_base_position("object"))
        return object_position

    def reset(self) -> None:
        self.goal = self._sample_goal()
        object_position = self._sample_object()
        self.sim.set_base_pose("target", self.goal, np.array([0.0, 0.0, 0.0, 1.0]))
        self.sim.set_base_pose("object", object_position, np.array([0.0, 0.0, 0.0, 1.0]))
        self.num_step = 0
        # change weight to change task chellenage
        if np.random.rand() < self.heavy_object_rate:
            self.sim.physics_client.changeDynamics(self.sim._bodies_idx["object"], -1, mass = 2.00)
        else:
            self.sim.physics_client.changeDynamics(self.sim._bodies_idx["object"], -1, mass = 0.10)


    def _sample_goal(self) -> np.ndarray:
        """Sample a goal."""
        goal = np.array([0.0, 0.0, self.object_size / 2])  # z offset for the cube center
        noise = self.np_random.uniform(self.goal_range_low, self.goal_range_high)
        goal += noise
        return goal

    def _sample_object(self) -> np.ndarray:
        """Randomize start position of object."""
        object_position = np.array([0.0, 0.0, self.object_size / 2])
        noise = self.np_random.uniform(self.obj_range_low, self.obj_range_high)
        object_position += noise
        return object_position

    def is_success(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> Union[np.ndarray, float]:
        d = distance(achieved_goal, desired_goal)
        return np.array(d < self.distance_threshold, dtype=np.float64)

    def compute_reward(self, achieved_goal, desired_goal, info: Dict[str, Any]) -> Union[np.ndarray, float]:
        d = distance(achieved_goal, desired_goal)
        return -np.array(d > self.distance_threshold, dtype=np.float64)

    def change(self, config = None):
        if config != None:
            self.heavy_object_rate = config
