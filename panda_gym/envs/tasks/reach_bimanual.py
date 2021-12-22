from typing import Any, Dict, Union

import numpy as np
from numpy.core.defchararray import center

from panda_gym.envs.core import Task
from panda_gym.utils import distance


class ReachBimanual(Task):
    def __init__(
        self,
        sim,
        get_ee_position0,
        get_ee_position1,
        reward_type="sparse",
        distance_threshold=0.05,
        goal_range=0.3,
        has_object = False,
    ) -> None:
        super().__init__(sim)
        self.has_object = has_object
        self.object_size = 0.04
        self.reward_type = reward_type
        self.distance_threshold = distance_threshold
        self.get_ee_position0 = get_ee_position0
        self.get_ee_position1 = get_ee_position1
        self.goal_range_low = np.array([goal_range / 4, goal_range / 4, -goal_range/2.5])
        self.goal_range_high = np.array([goal_range, goal_range, goal_range/2.5])
        obj_xyz_range=[0.3, 0.3, 0]
        self.obj_range_low = np.array([0.1, -obj_xyz_range[1] / 2, self.object_size/2])
        self.obj_range_high = np.array(obj_xyz_range) + self.obj_range_low
        with self.sim.no_rendering():
            self._create_scene()
            self.sim.place_visualizer(target_position=np.zeros(3), distance=0.9, yaw=45, pitch=-30)

    def _create_scene(self) -> None:
        self.sim.create_plane(z_offset=-0.4)
        self.sim.create_table(length=1., width=0.7, height=0.4, x_offset=-0.575)
        self.sim.create_table(length=1., width=0.7, height=0.4, x_offset=0.575)
        self.sim.create_sphere(
            body_name="target0",
            radius=0.02,
            mass=0.0,
            ghost=True,
            position=np.zeros(3),
            rgba_color=np.array([0.1, 0.9, 0.1, 0.3]),
        )
        self.sim.create_sphere(
            body_name="target1",
            radius=0.02,
            mass=0.0,
            ghost=True,
            position=np.zeros(3),
            rgba_color=np.array([0.9, 0.1, 0.1, 0.3]),
        )
        if self.has_object:
            self.sim.create_box(
                body_name="object0",
                half_extents=np.ones(3) * self.object_size / 2,
                mass=2.0,
                position=np.array([0.0, 0.0, self.object_size / 2]),
                rgba_color=np.array([0.1, 0.9, 0.1, 1.0]),
            )
            self.sim.create_box(
                body_name="object1",
                half_extents=np.ones(3) * self.object_size / 2,
                mass=2.0,
                position=np.array([0.0, 0.0, self.object_size / 2]),
                rgba_color=np.array([0.9, 0.1, 0.1, 1.0]),
            )

    def get_obs(self) -> np.ndarray:
        if self.has_object:
            # position, rotation of the object
            object1_position = np.array(self.sim.get_base_position("object1"))
            object1_rotation = np.array(self.sim.get_base_rotation("object1"))
            object1_velocity = np.array(self.sim.get_base_velocity("object1"))
            object1_angular_velocity = np.array(self.sim.get_base_angular_velocity("object1"))
            object0_position = np.array(self.sim.get_base_position("object0"))
            object0_rotation = np.array(self.sim.get_base_rotation("object0"))
            object0_velocity = np.array(self.sim.get_base_velocity("object0"))
            object0_angular_velocity = np.array(self.sim.get_base_angular_velocity("object0"))
            observation = np.concatenate(
                [
                    object0_position,
                    object0_rotation,
                    object0_velocity,
                    object0_angular_velocity,
                    object1_position,
                    object1_rotation,
                    object1_velocity,
                    object1_angular_velocity,
                ]
            )
            return observation
        else:
            return np.array([])  # no tasak-specific observation

    def get_achieved_goal(self) -> np.ndarray:
        if self.has_object:
            object0_position = self.sim.get_base_position("object0")
            object1_position = self.sim.get_base_position("object1")
            obj_center = (object1_position + object0_position)/2
            self.sim.set_base_pose("target0", -self.goal/2 + obj_center, np.array([0.0, 0.0, 0.0, 1.0]))
            self.sim.set_base_pose("target1", self.goal/2 + obj_center, np.array([0.0, 0.0, 0.0, 1.0]))
            ag = (object1_position - object0_position)
        else:
            ee_position0 = np.array(self.get_ee_position0())
            ee_position1 = np.array(self.get_ee_position1())
            ee_center = (ee_position0 + ee_position1)/2
            self.sim.set_base_pose("target0", -self.goal/2 + ee_center, np.array([0.0, 0.0, 0.0, 1.0]))
            self.sim.set_base_pose("target1", self.goal/2 + ee_center, np.array([0.0, 0.0, 0.0, 1.0]))
            ag = (ee_position1 - ee_position0)
        return ag

    def reset(self) -> None:
        self.goal = self._sample_goal()
        object0_position, object1_position = self._sample_objects()
        self.sim.set_base_pose("object0", object0_position, np.array([0.0, 0.0, 0.0, 1.0]))
        self.sim.set_base_pose("object1", object1_position, np.array([0.0, 0.0, 0.0, 1.0]))

    def _sample_goal(self) -> np.ndarray:
        """Randomize goal."""
        goal = self.np_random.uniform(self.goal_range_low, self.goal_range_high)
        return goal

    def _sample_objects(self):
        # while True:  # make sure that cubes are distant enough
        object0_position = self.np_random.uniform(self.obj_range_low, self.obj_range_high)
        object0_position[0] = - object0_position[0]
        object1_position = self.np_random.uniform(self.obj_range_low, self.obj_range_high)
        return object0_position, object1_position

    def is_success(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> Union[np.ndarray, float]:
        d = distance(achieved_goal, desired_goal)
        return np.array(d < self.distance_threshold, dtype=np.float64)

    def compute_reward(self, achieved_goal, desired_goal, info: Dict[str, Any]) -> Union[np.ndarray, float]:
        d = distance(achieved_goal, desired_goal)
        if self.reward_type == "sparse":
            return -np.array(d > self.distance_threshold, dtype=np.float64)
        else:
            return -d

    def change(self, config = None):
        print('change called!')

