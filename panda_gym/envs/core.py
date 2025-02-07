from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple, Union

import gym
import gym.spaces
import gym.utils.seeding
import numpy as np
import torch
from sqlalchemy import case
import pickle

from panda_gym.pybullet import PyBullet
import panda_gym
import imageio
from pathlib import Path
class PyBulletRobot(ABC):
    """Base class for robot env.

    Args:
        sim (PyBullet): Simulation instance.
        body_name (str): The name of the robot within the simulation.
        file_name (str): Path of the urdf file.
        base_position (np.ndarray): Position of the base of the robot as (x, y, z).
    """

    def __init__(
        self,
        sim: PyBullet,
        body_name: str,
        file_name: str,
        base_position: np.ndarray,
        action_space: gym.spaces.Space,
        joint_indices: np.ndarray,
        joint_forces: np.ndarray,
        base_orientation: np.ndarray=[0,0,0,1],
    ) -> None:
        self.sim = sim
        self.body_name = body_name
        with self.sim.no_rendering():
            self._load_robot(file_name, base_position, base_orientation)
            self.setup()
        self.action_space = action_space
        self.joint_indices = joint_indices
        self.joint_forces = joint_forces

    def _load_robot(self, file_name: str, base_position: np.ndarray, base_orientation: np.ndarray) -> None:
        """Load the robot.

        Args:
            file_name (str): The URDF file name of the robot.
            base_position (np.ndarray): The position of the robot, as (x, y, z).
        """
        self.sim.loadURDF(
            body_name=self.body_name,
            fileName=file_name,
            basePosition=base_position,
            baseOrientation = base_orientation,
            useFixedBase=True,
        )

    def setup(self) -> None:
        """Called after robot loading."""
        pass

    @abstractmethod
    def set_action(self, action: np.ndarray) -> None:
        """Set the action. Must be called just before sim.step().

        Args:
            action (np.ndarray): The action.
        """

    @abstractmethod
    def get_obs(self) -> np.ndarray:
        """Return the observation associated to the robot.

        Returns:
            np.ndarray: The observation.
        """

    @abstractmethod
    def reset(self) -> np.ndarray:
        """Reset the robot and return the observation.

        Returns:
            np.ndarray: The observation.
        """

    def get_link_position(self, link: int) -> np.ndarray:
        """Returns the position of a link as (x, y, z)

        Args:
            link (int): The link index.

        Returns:
            np.ndarray: Position as (x, y, z)
        """
        return self.sim.get_link_position(self.body_name, link)

    def get_link_velocity(self, link: int) -> np.ndarray:
        """Returns the velocity of a link as (vx, vy, vz)

        Args:
            link (int): The link index.

        Returns:
            np.ndarray: Velocity as (vx, vy, vz)
        """
        return self.sim.get_link_velocity(self.body_name, link)

    def get_joint_angle(self, joint: int) -> float:
        """Returns the angle of a joint

        Args:
            joint (int): The joint index.

        Returns:
            float: Joint angle
        """
        return self.sim.get_joint_angle(self.body_name, joint)

    def get_joint_velocity(self, joint: int) -> float:
        """Returns the velocity of a joint as (wx, wy, wz)

        Args:
            joint (int): The joint index.

        Returns:
            np.ndarray: Joint velocity as (wx, wy, wz)
        """
        return self.sim.get_joint_velocity(self.body_name, joint)

    def control_joints(self, target_angles: np.ndarray) -> None:
        """Control the joints of the robot.

        Args:
            target_angles (np.ndarray): The target angles. The length of the array must equal to the number of joints.
        """
        self.sim.control_joints(
            body=self.body_name,
            joints=self.joint_indices,
            target_angles=target_angles,
            forces=self.joint_forces,
        )

    def set_joint_angles(self, angles: np.ndarray) -> None:
        """Set the joint position of a body. Can induce collisions.

        Args:
            angles (list): Joint angles.
        """
        self.sim.set_joint_angles(self.body_name, joints=self.joint_indices, angles=angles)

    def inverse_kinematics(self, link: int, position: np.ndarray, orientation: np.ndarray) -> np.ndarray:
        """Compute the inverse kinematics and return the new joint values.

        Args:
            link (int): The link.
            position (x, y, z): Desired position of the link.
            orientation (x, y, z, w): Desired orientation of the link.

        Returns:
            List of joint values.
        """
        inverse_kinematics = self.sim.inverse_kinematics(self.body_name, link=link, position=position, orientation=orientation)
        return inverse_kinematics


class Task(ABC):
    """Base class for tasks.
    Args:
        sim (PyBullet): Simulation instance.
    """

    def __init__(self, sim: PyBullet) -> None:
        self.sim = sim
        self.goal = None

    @abstractmethod
    def reset(self) -> None:
        """Reset the task: sample a new goal"""

    @abstractmethod
    def get_obs(self) -> np.ndarray:
        """Return the observation associated to the task."""

    @abstractmethod
    def get_achieved_goal(self) -> np.ndarray:
        """Return the achieved goal."""

    def get_goal(self) -> np.ndarray:
        """Return the current goal."""
        if self.goal is None:
            raise RuntimeError("No goal yet, call reset() first")
        else:
            return self.goal.copy()

    def seed(self, seed: Optional[int]) -> int:
        """Sets the random seed.

        Args:
            seed (Optional[int]): The desired seed. Leave None to generate one.

        Returns:
            int: The seed.
        """
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return seed

    @abstractmethod
    def is_success(
        self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info: Dict[str, Any] = {}
    ) -> Union[np.ndarray, float]:
        """Returns whether the achieved goal match the desired goal."""

    @abstractmethod
    def compute_reward(
        self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info: Dict[str, Any] = {}
    ) -> Union[np.ndarray, float]:
        """Compute reward associated to the achieved and the desired goal."""


class RobotTaskEnv(gym.GoalEnv):
    """Robotic task goal env, as the junction of a task and a robot.

    Args:
        robot (PyBulletRobot): The robot.
        task (Task): The task.
    """

    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(self, robot: PyBulletRobot, task: Task) -> None:
        assert robot.sim == task.sim, "The robot and the task must belong to the same simulation."
        self.sim = robot.sim
        self.robot = robot
        self.task = task
        try: 
            self.num_blocks = task.num_blocks
        except:
            print('num_blocks dose not exist')
        self.seed()  # required for init; can be changed later
        obs = self.reset()
        observation_shape = obs["observation"].shape
        achieved_goal_shape = obs["achieved_goal"].shape
        desired_goal_shape = obs["achieved_goal"].shape
        self.observation_space = gym.spaces.Dict(
            dict(
                observation=gym.spaces.Box(-10.0, 10.0, shape=observation_shape, dtype=np.float32),
                desired_goal=gym.spaces.Box(-10.0, 10.0, shape=achieved_goal_shape, dtype=np.float32),
                achieved_goal=gym.spaces.Box(-10.0, 10.0, shape=desired_goal_shape, dtype=np.float32),
            )
        )
        self.action_space = self.robot.action_space
        self.compute_reward = self.task.compute_reward
        self.robot_obs_size = len(self.robot.get_obs())
        self._max_episode_steps = self.task._max_episode_steps

    def _get_obs(self) -> Dict[str, np.ndarray]:
        robot_obs = self.robot.get_obs()  # robot state
        task_obs = self.task.get_obs()  # object position, velococity, etc...
        observation = np.concatenate([robot_obs, task_obs])
        achieved_goal = self.task.get_achieved_goal()
        return {
            "observation": observation,
            "achieved_goal": achieved_goal,
            "desired_goal": self.task.get_goal(),
        }

    def reset(self) -> Dict[str, np.ndarray]:
        with self.sim.no_rendering():
            self.robot.reset()
            self.task.reset()
        return self._get_obs()

    def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, Dict[str, Any]]:
        self.robot.set_action(action)
        self.sim.step()
        obs = self._get_obs()
        done = False
        info = {
            "is_success": self.task.is_success(obs["achieved_goal"], self.task.get_goal()), 
            "ee_pos": self.robot.get_ee_position(),
            }
        if hasattr(self.task, 'unstable_obj_idx'):
            info['unstable_obj_idx']=self.task.unstable_obj_idx
        if hasattr(self.task, 'unstable_state'):
            info['unstable_state']=self.task.unstable_state
        reward = self.task.compute_reward(obs["achieved_goal"], self.task.get_goal(), info)
        assert isinstance(reward, float)  # needed for pytype cheking
        return obs, reward, done, info

    def seed(self, seed: Optional[int] = None) -> int:
        """Setup the seed."""
        return self.task.seed(seed)

    def close(self) -> None:
        self.sim.close()

    def render(
        self,
        mode: str,
        width: int = 720,
        height: int = 480,
        target_position: np.ndarray = np.zeros(3),
        distance: float = 1.4,
        yaw: float = 45,
        pitch: float = -30,
        roll: float = 0,
    ) -> Optional[np.ndarray]:
        """Render.

        If mode is "human", make the rendering real-time. All other arguments are
        unused. If mode is "rgb_array", return an RGB array of the scene.

        Args:
            mode (str): "human" of "rgb_array". If "human", this method waits for the time necessary to have
                a realistic temporal rendering and all other args are ignored. Else, return an RGB array.
            width (int, optional): Image width. Defaults to 720.
            height (int, optional): Image height. Defaults to 480.
            target_position (np.ndarray, optional): Camera targetting this postion, as (x, y, z).
                Defaults to [0., 0., 0.].
            distance (float, optional): Distance of the camera. Defaults to 1.4.
            yaw (float, optional): Yaw of the camera. Defaults to 45.
            pitch (float, optional): Pitch of the camera. Defaults to -30.
            roll (int, optional): Rool of the camera. Defaults to 0.

        Returns:
            RGB np.ndarray or None: An RGB array if mode is 'rgb_array', else None.
        """
        return self.sim.render(
            mode,
            width=width,
            height=height,
            target_position=target_position,
            distance=distance,
            yaw=yaw,
            pitch=pitch,
            roll=roll,
        )

    def change(self,config = None):
        if config != None:
            self.task.change(config)
            self._max_episode_steps = self.task._max_episode_steps

class BimanualTaskEnv(gym.GoalEnv):
    """Bimanual task goal env, as the junction of a task and a robot.

    Args:
        robot (PyBulletRobot): The robot.
        task (Task): The task.
    """

    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(self, robot0: PyBulletRobot, robot1: PyBulletRobot, task: Task, \
        store_trajectory = False, store_video = False, good_init_pos_rate = 0, \
            seed = 123, ignore_obj_rate = 0, fix_horizon = True) -> None:
        """PandaBimanual Env

        Args:
            robot0 (PyBulletRobot): _description_
            robot1 (PyBulletRobot): _description_
            task (Task): _description_
            store_video (bool, optional): _description_. Defaults to False.
            good_init_pos_rate (int, optional): _description_. Defaults to 0.
            ignore_obj_rate (int, optional): if ignore one object. Defaults to 0.
        """
        assert robot0.sim == task.sim, "The robot and the task must belong to the same simulation."
        assert robot0.sim == robot1.sim, "The robot must belong to the same simulation."
        self.good_init_pos_rate = good_init_pos_rate
        self.ignore_obj_rate = ignore_obj_rate
        self.sim = robot0.sim
        self.robot0 = robot0
        self.robot1 = robot1
        self.task = task
        self.fix_horizon = fix_horizon
        try: 
            self.num_blocks = task.num_blocks
        except:
            print('num_blocks dose not exist')
        self.store_trajectory = store_trajectory
        if store_trajectory:
            self.num_trajectory = 0
            self.trajectory = {
                'is_success': False, 
                'time': [], 
                'panda0_ee': [], 
                'panda1_ee': [], 
                'panda0_finger': [], 
                'panda1_finger': [], 
                'panda0_joints': [],
                'panda1_joints': []
            }
        self.store_video = store_video
        if store_video:
            self.video = []
        self.seed(seed)  # required for init; can be changed later
        obs = self.reset()
        observation_shape = obs["observation"].shape
        achieved_goal_shape = obs["achieved_goal"].shape
        desired_goal_shape = obs["achieved_goal"].shape
        self.observation_space = gym.spaces.Dict(
            dict(
                observation=gym.spaces.Box(-10.0, 10.0, shape=observation_shape, dtype=np.float32),
                desired_goal=gym.spaces.Box(-10.0, 10.0, shape=achieved_goal_shape, dtype=np.float32),
                achieved_goal=gym.spaces.Box(-10.0, 10.0, shape=desired_goal_shape, dtype=np.float32),
                gripper_arr=gym.spaces.Box(
                    -np.inf, np.inf, shape=obs["gripper_arr"].shape, dtype="float32"
                ),
                object_arr=gym.spaces.Box(
                    -np.inf, np.inf, shape=obs["object_arr"].shape, dtype="float32"
                ),
                desired_goal_arr=gym.spaces.Box(
                    -np.inf, np.inf, shape=obs["desired_goal_arr"].shape, dtype="float32"
                ),
                achieved_goal_arr=gym.spaces.Box(
                    -np.inf, np.inf, shape=obs["achieved_goal_arr"].shape, dtype="float32"
                ),
            )
        )
        self.robot_obs_size_0 = len(self.robot0.get_obs())
        self.robot_obs_size_1 = len(self.robot1.get_obs())
        self.robot_obs_size = self.robot_obs_size_0 + self.robot_obs_size_1
        self.single_task_obs_size = int(len(self.task.get_obs())/self.task.num_blocks)
        self.single_goal_size = int(len(self.task.goal)/self.task.num_blocks)
        self.robot0_action_shape = self.robot0.action_space.shape[0]
        self.robot1_action_shape = self.robot1.action_space.shape[0]
        action_shape = self.robot0_action_shape + self.robot1_action_shape
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(action_shape,), dtype=np.float32)
        self.compute_reward = self.task.compute_reward
        self._max_episode_steps = self.task._max_episode_steps
        if self.sim.blender_record:
            self.sim.recorder.register_object(0, panda_gym.assets.get_data_path()+'/franka_panda/panda.urdf')
            self.sim.recorder.register_object(1, panda_gym.assets.get_data_path()+'/franka_panda/panda.urdf')
            self.sim.recorder.register_object(10, panda_gym.assets.get_data_path()+'/plate.urdf')
            self.sim.recorder.register_object(11, panda_gym.assets.get_data_path()+'/cup.urdf')

    def _get_obs(self) -> Dict[str, np.ndarray]:
        robot0_obs = self.robot0.get_obs()  # robot state
        robot1_obs = self.robot1.get_obs()  # robot state
        task_obs = self.task.get_obs()  # object position, velococity, etc...
        observation = np.concatenate([robot0_obs, robot1_obs, task_obs])
        achieved_goal = self.task.get_achieved_goal()
        return {
            # tmp
            "gripper_arr": np.concatenate([robot0_obs, robot1_obs]),
            "object_arr": task_obs.reshape(self.num_blocks, -1),
            "achieved_goal_arr": achieved_goal.reshape(self.num_blocks, -1),
            "desired_goal_arr": self.task.get_goal().reshape(self.num_blocks, -1),
            # real
            "observation": observation,
            "achieved_goal": achieved_goal,
            "desired_goal": self.task.get_goal(),
        }

    def reset(self, attr_dict = {}, panda0_init = None, panda1_init = None) -> Dict[str, np.ndarray]:
        with self.sim.no_rendering():
            self.robot0.reset()
            self.robot1.reset()
            self.task.reset(attr_dict)
        self.num_steps = 0
        if self.store_trajectory:
            self._store_trajectory()
        return self._get_obs()

    def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, Dict[str, Any]]:
        act0 = action[:self.robot0_action_shape]
        act1 = action[self.robot0_action_shape:]
        self.robot0.set_action(act0)
        self.robot1.set_action(act1)
        self.sim.step()
        obs = self._get_obs()
        done = False
        mask = np.zeros(self.num_blocks)
        mask[np.random.randint(self.num_blocks)] = np.random.uniform() < self.ignore_obj_rate
        # Note: make the info ndarray to store them in the buffer
        if_drop = (self.task.obj_pos.reshape(-1,3)[:,2] < 0).any()
        info = {
            "is_success": self.task.is_success(obs["achieved_goal"], self.task.get_goal()), 
            "dropout": np.max(self.task.init_obj_pos - self.task.obj_pos) < self.task.distance_threshold or if_drop,
            "ee_pos": np.array([self.robot0.get_ee_position(), self.robot1.get_ee_position()]),
            "gripper_pos": np.array([self.robot0.get_fingers_width(), self.robot0.get_fingers_width()]),
            "mask": mask # number of object to ignore. 1 is ignore
            }
        reward = self.task.compute_reward(obs["achieved_goal"], self.task.get_goal(), info)
        assert isinstance(reward, float)  # needed for pytype cheking
        self.num_steps += 1
        if self.store_trajectory:
            self.trajectory['is_success'] = info['is_success']
            self.trajectory['time'].append(self.num_steps/12)
            self.trajectory['panda0_ee'].append(self.robot0.get_ee_position())
            self.trajectory['panda1_ee'].append(self.robot1.get_ee_position())
            self.trajectory['panda0_finger'].append(self.robot0.get_fingers_width())
            self.trajectory['panda1_finger'].append(self.robot1.get_fingers_width())
            self.trajectory['panda0_joints'].append(np.array([self.robot0.get_joint_angle(joint=i) for i in range(7)]))
            self.trajectory['panda1_joints'].append(np.array([self.robot1.get_joint_angle(joint=i) for i in range(7)]))
        if self.store_video:
            self.video.append(self.render(mode='rgb_array'))
        if self.fix_horizon:
            done = self.num_steps >= self._max_episode_steps 
        else:
            done = self.num_steps >= self._max_episode_steps or bool(info['is_success'])
        return obs, reward, done, info

    def seed(self, seed: Optional[int] = None) -> int:
        """Setup the seed."""
        return self.task.seed(seed)

    def close(self) -> None:
        self.sim.close()

    def render(
        self,
        mode: str,
        width: int = 720,
        height: int = 480,
        target_position: np.ndarray = np.zeros(3),
        distance: float = 1.4,
        yaw: float = 45,
        pitch: float = -30,
        roll: float = 0,
    ) -> Optional[np.ndarray]:
        """Render.

        If mode is "human", make the rendering real-time. All other arguments are
        unused. If mode is "rgb_array", return an RGB array of the scene.

        Args:
            mode (str): "human" of "rgb_array". If "human", this method waits for the time necessary to have
                a realistic temporal rendering and all other args are ignored. Else, return an RGB array.
            width (int, optional): Image width. Defaults to 720.
            height (int, optional): Image height. Defaults to 480.
            target_position (np.ndarray, optional): Camera targetting this postion, as (x, y, z).
                Defaults to [0., 0., 0.].
            distance (float, optional): Distance of the camera. Defaults to 1.4.
            yaw (float, optional): Yaw of the camera. Defaults to 45.
            pitch (float, optional): Pitch of the camera. Defaults to -30.
            roll (int, optional): Rool of the camera. Defaults to 0.

        Returns:
            RGB np.ndarray or None: An RGB array if mode is 'rgb_array', else None.
        """
        return self.sim.render(
            mode,
            width=width,
            height=height,
            target_position=target_position,
            distance=distance,
            yaw=yaw,
            pitch=pitch,
            roll=roll,
        )

    def change(self,config = None):
        if not config is None:
            self.task.change(config)
            self._max_episode_steps = self.task._max_episode_steps

    def obs_parser(self, x, contain_goal = True, mode = 'mirror'):
        task_obs_size = self.single_task_obs_size * self.task.num_blocks
        goal_size = self.single_goal_size * self.task.num_blocks
        # robot0
        robot0_obs = x[..., :self.robot_obs_size_0]
        # robot0_pos = robot0_obs[..., :3]
        # robot0_vel = robot0_obs[..., 3:6]
        # robot0_finger = robot0_obs[..., 6]
        # robot1
        robot1_obs = x[..., self.robot_obs_size_0:self.robot_obs_size]
        # robot1_pos = robot1_obs[..., :3]
        # robot1_vel = robot1_obs[..., 3:6]
        # robot1_finger = robot1_obs[..., 6]
        # task
        task_obs = x[..., self.robot_obs_size:self.robot_obs_size+task_obs_size]
        # obj_pos, obj_ori, obj_vel, obj_angvel = [], [], [], []
        # for i in range(self.task.num_blocks):
        #     obj_pos.append(task_obs[..., 12*i:12*i+3])
        #     obj_ori.append(task_obs[..., 12*i+3:12*i+6])
        #     obj_vel.append(task_obs[..., 12*i+6:12*i+9])
        #     obj_angvel.append(task_obs[..., 12*i+9:12*i+12])
        if contain_goal:
            goal = x[..., self.robot_obs_size+task_obs_size:self.robot_obs_size+task_obs_size+goal_size]
        return {
            'mirror': np.concatenate(
                (
                    robot1_obs*([-1,-1,1]*2+[1]),
                    robot0_obs*([-1,-1,1]*2+[1]),
                    task_obs*([-1,-1,1,1,-1,1]*2*self.task.num_blocks),
                    goal*([-1,-1,1]*self.task.num_blocks),
                ),
                axis=-1),
        }[mode]

    def get_spaces(self):
        return self.observation_space, self.action_space

    def _set_init_state(self):
        good_start_pos = np.random.uniform() < self.good_init_pos_rate
        if good_start_pos:
            if np.random.uniform() > 0.5:
                panda0_init = np.random.uniform([-0.2, 0.2, 0.05], [0.0, -0.2, 0.2])
                if np.random.uniform() < self.good_init_pos_rate:
                    panda1_init = panda0_init + np.array([0.16, 0, 0])
            else:
                panda1_init = np.random.uniform([0.0, 0.2, 0.05], [0.2, -0.2, 0.2])
                if np.random.uniform() < self.good_init_pos_rate:
                    panda0_init = panda1_init - np.array([0.16, 0, 0])
        self.robot0.reset(init_pos = panda0_init)
        self.robot1.reset(init_pos = panda1_init)
        if good_start_pos:
            block_id = np.random.choice(np.arange(self.num_blocks))
            if np.random.uniform() > 0.5:
                obj_pos_dict={block_id: self.robot0.get_ee_position()+np.array([0.08,0,0])}
            else:
                obj_pos_dict={block_id: self.robot1.get_ee_position()-np.array([0.08,0,0])}

    def _store_trajectory(self):
        if self.trajectory['is_success']:
            self.num_trajectory += 1
            with open(f"/Users/reedpan/Downloads/tmp/{self.num_trajectory}.pkl", 'wb') as f:
                pickle.dump(self.trajectory, f, protocol=2)
            print(f"trajectory{self.num_trajectory} saved!")
            if self.store_video:
                if len(self.video)>0:
                    path = f'/Users/reedpan/Downloads/tmp/pic{self.num_trajectory}'
                    Path(path).mkdir(parents=True, exist_ok=True)
                    imageio.mimwrite(f'{path}/video.mp4', self.video , fps = 30)
                    # for j, image in enumerate(self.video):
                    #     imageio.imwrite(f'{path}/{self.num_trajectory}_{j}.png', image)
        self.trajectory = {
            'is_success': False, 
            'obj_init_pos': self.task.get_achieved_goal(),
            'goal': self.task.get_goal(),
            'time': [0],
            'panda0_ee': [self.robot0.get_ee_position()], 
            'panda1_ee': [self.robot1.get_ee_position()], 
            'panda0_finger': [self.robot0.get_fingers_width()], 
            'panda1_finger': [self.robot1.get_fingers_width()], 
            'panda0_joints': [np.array([self.robot0.get_joint_angle(joint=i) for i in range(7)])], 
            'panda1_joints': [np.array([self.robot1.get_joint_angle(joint=i) for i in range(7)])], 
        }
        self.video = []