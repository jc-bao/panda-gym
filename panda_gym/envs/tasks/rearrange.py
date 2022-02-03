from typing import Any, Dict, Tuple, Union

import numpy as np

from panda_gym.envs.core import Task

class Rearrange(Task):
    def __init__(
        self,
        sim,
        reward_type="sparse",
        distance_threshold=0.05,
        goal_xy_range=0.3,
        obj_xy_range=0.3,
        num_blocks = 1,
        unstable_mode = False
    ) -> None:
        super().__init__(sim)
        self.seed(0)
        self.unstable_mode = unstable_mode
        self.unstable_state = False
        self.num_blocks = num_blocks
        self._max_episode_steps = 50*self.num_blocks
        self.reward_type = reward_type
        self.distance_threshold = distance_threshold
        self.object_size = 0.04
        self.goal_range_low = np.array([-goal_xy_range / 2, -goal_xy_range / 2, 0])
        self.goal_range_high = np.array([goal_xy_range / 2, goal_xy_range / 2, 0])
        self.obj_range_low = np.array([-obj_xy_range / 2, -obj_xy_range / 2, 0])
        self.obj_range_high = np.array([obj_xy_range / 2, obj_xy_range / 2, 0])
        with self.sim.no_rendering():
            self._create_scene()
            self.sim.place_visualizer(target_position=np.zeros(3), distance=0.9, yaw=45, pitch=-30)
        self.reset()
        self.obj_obs_size = int(len(self.get_obs())/self.num_blocks)

    def _create_scene(self) -> None:
        self.sim.create_plane(z_offset=-0.4)
        self.sim.create_table(length=1.1, width=0.7, height=0.4, x_offset=-0.3)
        for i in range(6):
            color = np.random.rand(3)
            self.sim.create_box(
                body_name="object"+str(i),
                half_extents=np.array([1,1,1]) * self.object_size / 2,
                mass=0.3,
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
            obs.append(self.sim.get_base_position("object"+str(i)))
            obs.append(self.sim.get_base_rotation("object"+str(i)))
            obs.append(self.sim.get_base_velocity("object"+str(i)))
            obs.append(self.sim.get_base_angular_velocity("object"+str(i)))
        observation = np.array(obs).flatten()
        if self.unstable_mode:
            observation = np.append(observation, np.array([self.unstable_mode, self.unstable_obj_idx/6], dtype=float))
            self.unstable_state = self.unstable_state or self.np_random.uniform()<0.05
        return observation

    def get_achieved_goal(self) -> np.ndarray:
        ag = []
        for i in range(self.num_blocks):
            ag.append(np.array(self.sim.get_base_position("object"+str(i))))
        ag = np.array(ag)
        achieved_goal = (ag).flatten()
        return achieved_goal

    def reset(self) -> None:
        obj_pos = self._sample_objects()
        self.goal = self._sample_goal()
        for i in range(self.num_blocks):
            self.sim.set_base_pose("target"+str(i), self.goal[i*3:(i+1)*3], np.array([0.0, 0.0, 0.0, 1.0]))
            self.sim.set_base_pose("object"+str(i), obj_pos[i*3:(i+1)*3], np.array([0.0, 0.0, 0.0, 1.0]))
        assert len(obj_pos)%self.num_blocks==0, 'task observation shape linear to num blocks'
        self.unstable_obj_idx = self.np_random.randint(low = 0, high = self.num_blocks)
        self.unstable_state = False # if block one object reward

    def _sample_goal(self) -> np.ndarray:
        goals = []
        for i in range(self.num_blocks):
            while True:
                goal = self.np_random.uniform(self.goal_range_low, self.goal_range_high)+[0,0,self.object_size/2]
                if len(goals) == 0:
                    break
                elif min(np.linalg.norm(goals - goal, axis = -1)) > self.object_size*1.5:
                    break
            goals.append(goal)
        return np.array(goals).flatten()

    def _sample_objects(self) -> Tuple[np.ndarray, np.ndarray]:
        obj_pos = []
        for i in range(self.num_blocks):
            # get target object side
            while True:
                pos = self.np_random.uniform(self.obj_range_low, self.obj_range_high) + self.object_size/2
                if len(obj_pos) == 0:
                    break
                elif min(np.linalg.norm(obj_pos - pos, axis = -1)) > self.object_size*1.5:
                    break
            obj_pos.append(pos)
        return np.array(obj_pos).flatten()

    def is_success(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> Union[np.ndarray, float]:
        dists = [
            np.linalg.norm(achieved_goal[..., i * 3:(i + 1) * 3] - desired_goal[..., i * 3:(i + 1) * 3], axis=-1)
            for i in range(self.num_blocks)
        ]
        return float(np.all([d < self.distance_threshold for d in dists]))


    def compute_reward(self, achieved_goal, desired_goal, info: Dict[str, Any]) -> Union[np.ndarray, float]:
        delta = (achieved_goal - desired_goal).reshape(-1, self.num_blocks ,3)
        dist_block2goal = np.linalg.norm(delta, axis=-1)
        rew = -np.sum(dist_block2goal>self.distance_threshold, axis=-1, dtype = float)
        if self.unstable_mode and info['unstable_state']:
            rew -= (dist_block2goal[..., info['unstable_obj_idx']]<self.distance_threshold)
        if len(rew) == 1 :
            return rew[0]
        else: # to process multi dimension input
            return rew

    def change(self, config):
        self.num_blocks = int(config)
        self._max_episode_steps = 50 * self.num_blocks
