import gym
import numpy as np
from panda_gym.pybullet import PyBullet
from panda_gym.envs.robots.panda_2 import Panda


class PandaBimanualHandoverEnv(gym.GoalEnv):
    def __init__(self, render: bool = False, num_blocks: int = 1, block_size=(0.2, 0.04, 0.04),
                 os_rate=0.5, obj_in_hand_rate=0.0, reward_type="sparse", initial_gap_distance=0.0):
        self.sim = PyBullet(render, 20, timestep=1.0 / 240)
        self.base_position0 = np.array([-0.6, -0.4, -0.1])
        self.base_position1 = np.array([0.6, 0.4, -0.1])
        self.gap_distance = 0.5
        self.current_gap_distance = initial_gap_distance
        self.robot0 = Panda(
            self.sim, base_position=self.base_position0, base_orientation=np.array([0, 0, np.sqrt(2)/2, np.sqrt(2)/2]), index=0,
            # eef_orientation=np.array([np.sqrt(2) / 2, np.sqrt(2) / 2, 0., 0.])
            eef_orientation=np.array([1.0, 0., 0., 0.])
        )
        self.robot1 = Panda(
            self.sim, base_position=self.base_position1, base_orientation=np.array([0, 0, -np.sqrt(2), np.sqrt(2)]), index=1,
            eef_orientation=np.array([0.0, 1.0, 0., 0.]))
        self.block_size = np.array(block_size)
        self.obj_range_low = np.array(
            [self.base_position0[0] - 0.2, self.base_position0[1] + 0.2, self.block_size[-1] / 2])
        self.obj_range_high = np.array(
            [self.base_position1[0] + 0.2, self.base_position1[1] - 0.2, self.block_size[-1] / 2])
        self.num_blocks = num_blocks
        self.distance_threshold = 0.05
        self.os_rate = os_rate
        self.obj_in_hand_rate = obj_in_hand_rate
        self.reward_type = reward_type
        self.seed()
        self._create_scene()
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
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(8,))

    def _create_scene(self):
        self.sim.create_plane(z_offset=-0.4)
        table_x = self.base_position1[0]
        self.sim.create_table(length=2 * table_x - self.gap_distance, width=0.7, height=0.4, x_offset=-table_x, index=0)
        self.sim.create_table(length=2 * table_x - self.gap_distance, width=0.7, height=0.4, x_offset=table_x, index=1)
        self.table0_pos = self.sim.get_base_position("table0")
        self.table1_pos = self.sim.get_base_position("table1")
        for i in range(self.num_blocks):
            color = np.random.rand(3)
            self.sim.create_box(
                body_name="object" + str(i),
                half_extents=self.block_size / 2,
                mass=0.2,
                position=np.array([2, 0.1 * i - 0.3, self.block_size[-1] / 2]),
                rgba_color=np.append(color, 1),
            )
            self.sim.create_sphere(
                body_name="target" + str(i),
                radius=self.block_size[-1] / 2,
                mass=0.0,
                ghost=True,
                position=np.array([2, 0.1 * i - 0.3, 0.05]),
                rgba_color=np.append(color, 0.5),
            )

    def reset(self):
        with self.sim.no_rendering():
            self.robot0.reset(np.random.uniform(
                low=[self.base_position0[0] - 0.2, self.base_position0[1] + 0.2, self.block_size[-1] / 2],
                high=[self.base_position0[0] + 0.4, self.base_position0[1] + 0.4, 0.1]
            ))
            self.robot1.reset(np.random.uniform(
                low=[self.base_position1[0] - 0.4, self.base_position1[1] - 0.4, self.block_size[-1] / 2],
                high=[self.base_position1[0] + 0.2, self.base_position1[1] - 0.2, 0.1]
            ))
            table0_pos = self.table0_pos.copy()
            table0_pos[0] += (self.gap_distance - self.current_gap_distance) / 2
            table1_pos = self.table1_pos.copy()
            table1_pos[0] -= (self.gap_distance - self.current_gap_distance) / 2
            self.sim.set_base_pose("table0", table0_pos, np.array([0, 0, 0, 1]))
            self.sim.set_base_pose("table1", table1_pos, np.array([0, 0, 0, 1]))
            self._task_reset()
            self._reset_callback()
        self.last_n_inplace = None
        return self._get_obs()

    def _sample_goal(self, obj_pos):
        goals = []
        for i in range(self.num_blocks):
            goal_pos = np.random.uniform(self.obj_range_low, self.obj_range_high)
            if np.random.uniform() < self.os_rate:
                if obj_pos[i][0] * goal_pos[0] > 0:
                    goal_pos[0] *= -1
            else:
                if obj_pos[i][0] * goal_pos[0] < 0:
                    goal_pos[0] *= -1
            goals.append(goal_pos)
        return np.concatenate(goals)

    def _task_reset(self):
        obj_pos = np.random.uniform([self.gap_distance / 2, self.obj_range_low[1], self.obj_range_low[2]],
                                    self.obj_range_high, size=(self.num_blocks, 3))
        obj_pos[:, 0] *= np.random.choice([1, -1], size=(self.num_blocks,))
        self.goal = self._sample_goal(obj_pos)
        for i in range(self.num_blocks):
            self.sim.set_base_pose("target" + str(i), self.goal[i * 3:(i + 1) * 3], np.array([0.0, 0.0, 0.0, 1.0]))
            self.sim.set_base_pose("object" + str(i), obj_pos[i], np.array([0.0, 0.0, 0.0, 1.0]))

    def _reset_callback(self):
        if np.random.uniform() < self.obj_in_hand_rate:
            obj_idx = np.random.randint(0, self.num_blocks)
            pos = self.sim.get_base_position("object" + str(obj_idx))
            if pos[0] < -self.gap_distance / 2 - 0.1:
                robot = self.robot0
            elif pos[0] > self.gap_distance / 2 + 0.1:
                robot = self.robot1
            else:
                if np.random.uniform() < 0.5:
                    robot = self.robot0
                else:
                    robot = self.robot1
            pos[0] += np.random.uniform(-self.block_size[0] / 2, self.block_size[0] / 2)
            target_joints = robot.inverse_kinematics(robot.ee_link, pos, robot.eef_orientation)
            target_joints[-2:] = 0.04
            robot.set_joint_angles(target_joints)

    def _get_obs(self):
        robot0_obs = self.robot0.get_obs()
        robot1_obs = self.robot1.get_obs()
        # object obs
        objects_obs = []
        for i in range(self.num_blocks):
            pos = self.sim.get_base_position("object" + str(i))
            rel_pos0 = pos - self.robot0.get_ee_position()
            rel_pos1 = pos - self.robot1.get_ee_position()
            ori = self.sim.get_base_rotation("object" + str(i), type="euler")
            vel = self.sim.get_base_velocity("object"+str(i))
            vela = self.sim.get_base_angular_velocity("object" + str(i))
            objects_obs.append(np.concatenate([pos, ori, rel_pos0, rel_pos1, vel, vela]))
        achieved_goal = np.concatenate([item[:3] for item in objects_obs])
        objects_obs = np.concatenate(objects_obs)
        return {"observation": np.concatenate([robot0_obs, robot1_obs, objects_obs]),
                "achieved_goal": achieved_goal,
                "desired_goal": self.goal.copy()}

    def get_obs(self):
        return self._get_obs()

    def step(self, action):
        self.robot0.set_action(action[:4])
        self.robot1.set_action(action[4:])
        self.sim.step()
        obs = self._get_obs()
        info = {"is_success": self.is_success(obs["achieved_goal"], obs["desired_goal"])}
        reward = self.compute_reward(obs["achieved_goal"], obs["desired_goal"], info)
        done = False
        return obs, reward, done, info

    def is_success(self, achieved_goal, desired_goal):
        success_per_object = np.stack([np.linalg.norm(
            achieved_goal[..., 3 * i: 3 * (i + 1)] - desired_goal[..., 3 * i: 3 * (i + 1)],
            axis=-1) < self.distance_threshold for i in range(self.num_blocks)], axis=-1)
        success = np.all(success_per_object, axis=-1)
        return success

    def compute_reward(self, achieved_goal, desired_goal, info):
        distance_per_object = np.stack([np.linalg.norm(
            achieved_goal[..., 3 * i: 3 * (i + 1)] - desired_goal[..., 3 * i: 3 * (i + 1)], axis=-1
        ) for i in range(self.num_blocks)], axis=-1)
        if self.reward_type == "sparse":
            success_per_object = distance_per_object < self.distance_threshold
            # reward = np.sum(success_per_object, axis=-1) / self.num_blocks  # ratio of in-place objects
            n_inplace = np.sum(success_per_object, axis=-1)
            reward = n_inplace / self.num_blocks if self.last_n_inplace is None else (n_inplace - self.last_n_inplace) / self.num_blocks
            self.last_n_inplace = n_inplace
            info.update({"n_inplace": n_inplace})
        elif self.reward_type == "dense":
            reward = -np.sum(distance_per_object, axis=-1)
        else:
            raise NotImplementedError
        return reward

    def render(self, mode="human"):
        return self.sim.render(
            mode,
            width=480,
            height=480,
            target_position=np.zeros(3),
            distance=1.5,
            yaw=60,
            pitch=-30,
            roll=0,
        )

    def change_gap_distance(self, distance):
        self.current_gap_distance = max(min(distance, self.gap_distance), 0)
