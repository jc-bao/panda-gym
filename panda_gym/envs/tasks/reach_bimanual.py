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
    ) -> None:
        super().__init__(sim)
        self.reward_type = reward_type
        self.distance_threshold = distance_threshold
        self.get_ee_position0 = get_ee_position0
        self.get_ee_position1 = get_ee_position1
        self.goal_range_low = np.array([goal_range / 4, goal_range / 4, -goal_range/2.5])
        self.goal_range_high = np.array([goal_range, goal_range, goal_range/2.5])
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

    def get_obs(self) -> np.ndarray:
        return np.array([])  # no tasak-specific observation

    def get_achieved_goal(self) -> np.ndarray:
        ee_position0 = np.array(self.get_ee_position0())
        ee_position1 = np.array(self.get_ee_position1())
        ee_center = (ee_position0 + ee_position1)/2
        self.sim.set_base_pose("target0", -self.goal/2 + ee_center, np.array([0.0, 0.0, 0.0, 1.0]))
        self.sim.set_base_pose("target1", self.goal/2 + ee_center, np.array([0.0, 0.0, 0.0, 1.0]))
        return (ee_position1 - ee_position0)

    def reset(self) -> None:
        self.goal = self._sample_goal()
        ee_position0 = np.array(self.get_ee_position0())
        ee_position1 = np.array(self.get_ee_position1())
        ee_center = (ee_position0 + ee_position1)/2
        self.sim.set_base_pose("target0", -self.goal/2 + ee_center, np.array([0.0, 0.0, 0.0, 1.0]))
        self.sim.set_base_pose("target1", self.goal/2 + ee_center, np.array([0.0, 0.0, 0.0, 1.0]))

    def _sample_goal(self) -> np.ndarray:
        """Randomize goal."""
        goal = self.np_random.uniform(self.goal_range_low, self.goal_range_high)
        return goal

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

