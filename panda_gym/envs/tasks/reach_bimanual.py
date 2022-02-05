from typing import Any, Dict, Union

import numpy as np
from numpy.core.defchararray import center

import panda_gym
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
        goal_range=0.35,
        has_object = False,
        absolute_pos = False,
        obj_not_in_hand_rate = 1, 
    ) -> None:
        super().__init__(sim)
        self.has_object = has_object
        self.absolute_pos = absolute_pos
        self.object_size = 0.04
        self.reward_type = reward_type
        self.distance_threshold = distance_threshold
        self.obj_not_in_hand_rate = obj_not_in_hand_rate
        self.get_ee_position0 = get_ee_position0
        self.get_ee_position1 = get_ee_position1
        self.goal_range_low = np.array([goal_range / 4, goal_range / 4, -goal_range/1.5])
        self.goal_range_high = np.array([goal_range, goal_range, goal_range/1.5])
        obj_xyz_range=[0.3, 0.3, 0]
        self.obj_range_low = np.array([0.1, -obj_xyz_range[1] / 2, self.object_size/2])
        self.obj_range_high = np.array(obj_xyz_range) + self.obj_range_low
        with self.sim.no_rendering():
            self._create_scene()
            self.sim.place_visualizer(target_position=np.zeros(3), distance=0.9, yaw=45, pitch=-30)
        self._max_episode_steps = 50

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
        self.sim.create_sphere(
            body_name="target2",
            radius=0.03,
            mass=0.0,
            ghost=True,
            position=np.zeros(3),
            rgba_color=np.array([0.1, 0.1, 0.9, 0.5]),
        )
        if self.has_object:
            self.sim.create_box(
                body_name="object0",
                half_extents=np.ones(3) * self.object_size / 2,
                mass=0.5,
                position=np.array([0.0, 0.0, self.object_size / 2]),
                rgba_color=np.array([0.1, 0.9, 0.1, 1.0]),
            )
            self.sim.create_box(
                body_name="object1",
                half_extents=np.ones(3) * self.object_size / 2,
                mass=0.5,
                position=np.array([0.0, 0.0, self.object_size / 2]),
                rgba_color=np.array([0.9, 0.1, 0.1, 1.0]),
            )
            # LOAD SHAPE TAMP
            # self.sim.physics_client.setAdditionalSearchPath(panda_gym.assets.get_data_path())
            # self.sim.loadURDF(
            #     body_name='object01',
            #     fileName='plate.urdf',
            #     basePosition=[-0.2,0,0.02],
            #     baseOrientation = [0,0,0,1],
            #     useFixedBase=False,
            # )
            # self.sim.loadURDF(
            #     body_name='object11',
            #     fileName='cup.urdf',
            #     basePosition=[0.2,0,0.02],
            #     baseOrientation = [0,0,0,1],
            #     useFixedBase=False,
            # )

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
            self.assemble_done = self.assemble_done or \
                (np.linalg.norm(object1_position-object0_position-self.goal[3:]) < self.distance_threshold)
            if self.assemble_done:
                object1_position = object0_position + self.goal[3:]
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
            if self.assemble_done:
                object1_position = object0_position + self.goal[3:]
            else:
                object1_position = self.sim.get_base_position("object1")
            obj_center = (object1_position + object0_position)/2
            if self.absolute_pos:
                self.sim.set_base_pose("target0", -self.goal[3:]/2 + obj_center, np.array([0.0, 0.0, 0.0, 1.0]))
                self.sim.set_base_pose("target1", self.goal[3:]/2 + obj_center, np.array([0.0, 0.0, 0.0, 1.0]))
                ag = np.append(object0_position, object1_position-object0_position)
            else:
                self.sim.set_base_pose("target0", -self.goal/2 + obj_center, np.array([0.0, 0.0, 0.0, 1.0]))
                self.sim.set_base_pose("target1", self.goal/2 + obj_center, np.array([0.0, 0.0, 0.0, 1.0]))
                ag = (object1_position - object0_position)
            # CHANGE TAMP
            # for i in range(2):
            #     pos = self.sim.get_base_position("object"+str(i))
            #     ori = np.array([0, self.sim.get_base_rotation("object"+str(i))[1], 0])
            #     self.sim.set_base_pose("object"+str(i), pos, ori)
            #     self.sim.set_base_pose("object"+str(i)+"1",pos,np.array([0.0, 0.0, 0.0, 1.0]))
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
        if self.has_object:
            self.sim.set_base_pose("object0", object0_position, np.array([0.0, 0.0, 0.0, 1.0]))
            self.sim.set_base_pose("object1", object1_position, np.array([0.0, 0.0, 0.0, 1.0]))
        if self.absolute_pos:
            self.sim.set_base_pose("target2", self.goal[:3], np.array([0.0, 0.0, 0.0, 1.0]))
        self.assemble_done = False
        

    def _sample_goal(self) -> np.ndarray:
        """Randomize goal."""
        if self.absolute_pos:
            goal_abs = self.np_random.uniform(self.obj_range_low, self.obj_range_high)
            goal_abs[0] = -goal_abs[0]
            goal_abs[2] += self.np_random.uniform(0, 0.27)
            goal_relative = self.np_random.uniform(self.goal_range_low, self.goal_range_high)
            # goal_relative = [0.12, 0, 0.08] # TAMP
            # goal_abs = [-0.4, 0.18, 0.18] # TAMP
            goal = np.append(goal_abs, goal_relative)
        else:
            goal = self.np_random.uniform(self.goal_range_low, self.goal_range_high)
        return goal

    def _sample_objects(self):
        # while True:  # make sure that cubes are distant enough
        if self.np_random.uniform()<self.obj_not_in_hand_rate:
            object0_position = self.np_random.uniform(self.obj_range_low, self.obj_range_high)
            object0_position[0] =- object0_position[0]
            object1_position = self.np_random.uniform(self.obj_range_low, self.obj_range_high)
        else:
            object0_position = np.array(self.get_ee_position0())
            object1_position = np.array(self.get_ee_position1())
        return object0_position, object1_position

    def is_success(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> Union[np.ndarray, float]:
        if self.absolute_pos:
            return np.array(\
                distance(achieved_goal[..., :3], desired_goal[..., :3]) < self.distance_threshold and \
                    distance(achieved_goal[..., 3:], desired_goal[..., 3:]) < self.distance_threshold
                    , dtype=np.float64)
        else:
            d = distance(achieved_goal, desired_goal)
            return np.array(d < self.distance_threshold, dtype=np.float64)

    def compute_reward(self, achieved_goal, desired_goal, info: Dict[str, Any]) -> Union[np.ndarray, float]:
        if self.absolute_pos:
            # if info['assemble_done']:
            d = distance(achieved_goal[..., :3], desired_goal[..., :3])
            rew = -np.array(d > self.distance_threshold, dtype=np.float64).flatten()
            # else:
            d = distance(achieved_goal[...,3:], desired_goal[..., 3:])
            rew -= np.array(d > self.distance_threshold, dtype=np.float64).flatten()
        else:
            d = distance(achieved_goal, desired_goal)
            rew = -np.array(d > self.distance_threshold, dtype=np.float64).flatten()
        if len(rew) == 1 :
            return rew[0]
        else: # to process multi dimension input
            return rew

    def change(self, config = None):
        self.obj_not_in_hand_rate = config