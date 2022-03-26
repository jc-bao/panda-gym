from typing import Any, Dict, Tuple, Union
import numpy as np
import panda_gym
import gym
from scipy.stats import binom

from panda_gym.envs.core import Task
from panda_gym.utils import distance

class RearrangeBimanual(Task):
    def __init__(
        self,
        sim,
        get_ee_position0,
        get_ee_position1,
        seed = 0, 
        obj_xyz_range=[0.3, 0.4, 0],
        goal_z=0,
        num_blocks = 1, # number of blocks
        os_rate = 0.6, # init goal in different table
        os_num_dist = 'binominal', # other side number distribution 'uniform', 'binominal'
        obj_in_hand_rate = 0.2, # init obj in hand
        gap_distance = None, # if None, auto set
        debug_mode = False, # if show debug info
        base_ep_len = None, # total episode length=base_ep_len*num_blocks
    ) -> None:
        super().__init__(sim)
        self.seed(seed)
        # fixed parameters
        self.block_size = np.array([0.04*5, 0.04, 0.04])
        self.distance_threshold = 0.05
        # args parameters
        if gap_distance is None:
            self.gap_distance = self.block_size[0]*1.2
        else:
            self.gap_distance = gap_distance
        self.num_blocks = num_blocks
        self.os_rate = os_rate
        self.os_num_dist = {
            'uniform': np.ones(self.num_blocks+1)/(self.num_blocks+1), 
            'binominal': [binom.pmf(r, self.num_blocks, self.os_rate) \
                for r in np.arange(self.num_blocks+1)]
        }[os_num_dist]
        self.obj_in_hand_rate = obj_in_hand_rate
        if base_ep_len is None:
            self.base_ep_len = int(50/0.3*obj_xyz_range[0])
        else:
            self.base_ep_len = base_ep_len
        self._max_episode_steps = self.base_ep_len * self.num_blocks
        self.goal_space, self.obj_space = self._get_goal_obj_space(goal_z, obj_xyz_range)
        # sim parameters
        self.debug_mode = debug_mode
        self.get_ee_position0 = get_ee_position0
        self.get_ee_position1 = get_ee_position1
        with self.sim.no_rendering():
            self._create_scene()
            self.sim.place_visualizer(target_position=np.zeros(3), distance=1.2, yaw=45, pitch=-30)
        # gym env parameters
        self.obj_obs_size = int(len(self.reset())/self.num_blocks)

    def _create_scene(self) -> None:
        self.sim.create_plane(z_offset=-0.4)
        table_x = 0.5 + self.gap_distance/2
        self.sim.create_table(length=1., width=1, height=0.4, x_offset=(-table_x), index=0)
        self.sim.create_table(length=1., width=1, height=0.4, x_offset=(table_x), index=1)
        for i in range(self.num_blocks):
            color = self.np_random.rand(3)
            self.sim.create_box(
                body_name="object"+str(i),
                half_extents=self.block_size/2,
                mass=0.2,
                position=np.array([2, 0.1*i - 0.3, self.block_size[2]/2]),
                rgba_color=np.append(color, 1),
            )
            self.sim.create_sphere(
                body_name="target"+str(i),
                radius=self.block_size[2]/1.8,
                mass=0.0,
                ghost=True,
                position=np.array([2, 0.1*i-0.3, 0.05]),
                rgba_color=np.append(color, 0.5),
            )
        if self.debug_mode:
            self._show_goal_obj_space()

    def get_obs(self) -> np.ndarray:
        # position, rotation of the object
        obs = []
        for i in range(self.num_blocks):
            # [NOTE]trick: reset object orientation to aviod z rotation
            pos = self.sim.get_base_position("object"+str(i))
            ori = np.array([0, self.sim.get_base_rotation("object"+str(i))[1], 0])
            self.sim.set_base_pose("object"+str(i), pos, ori)
            obs.append(pos)
            obs.append(ori)
            obs.append(self.sim.get_base_velocity("object"+str(i)))
            obs.append(self.sim.get_base_angular_velocity("object"+str(i)))
        observation = np.array(obs).flatten()
        return observation

    def get_achieved_goal(self) -> np.ndarray:
        ag = []
        for i in range(self.num_blocks):
            pos = self.sim.get_base_position("object"+str(i))
            ag.append(pos)
        return np.array(ag).flatten()

    def reset(self, attr_dict = {}) -> None:
        '''manualy set attribute to certain value'''
        for k,v in attr_dict.items():
            if hasattr(self, k):
                setattr(self,k,v)
            else:
                print(f'[DEBUG] task has no attribute {k}')
        # get pos
        if not 'obj_pos' in attr_dict.keys():
            self.obj_pos = self._sample_objects()
        if not 'goal' in attr_dict.keys():
            # Note: call goal after call obj_pos. goal sampler need obj pos
            if 'num_need_handover' in attr_dict.keys():
                num_need_handover = attr_dict['num_need_handover']
            else:
                num_need_handover = self.np_random.choice(a=self.num_blocks+1, size=1,p=self.os_num_dist)[0]
            self.goal = self._sample_goal(num_need_handover)
        for i in range(self.num_blocks):
            self.sim.set_base_pose("target"+str(i), self.goal[i*3:(i+1)*3], np.array([0.0, 0.0, 0.0, 1.0]))
            self.sim.set_base_pose("object"+str(i), self.obj_pos[i*3:(i+1)*3], np.array([0.0, 0.0, 0.0, 1.0]))
        assert len(self.obj_pos)%self.num_blocks==0, 'task observation shape linear to num blocks'
        return self.get_obs()

    def _sample_goal(self, num_need_handover) -> np.ndarray:
        goals = []
        handover_idx = self.np_random.choice(np.arange(self.num_blocks), size=num_need_handover, replace=False)
        for i in range(self.num_blocks):
            obj_side = (float(self.obj_pos[i*3]>0)*2-1)
            if_same_side = -1 if i in handover_idx else 1
            goal_side = obj_side * if_same_side
            for _ in range(10): # max trail time: 10
                if self.num_blocks == 1 or num_need_handover == 0:
                    goal = self.goal_space.sample()
                else:
                    goal = self.obj_space.sample()
                goal[0] = goal_side*goal[0]
                if len(goals) == 0 or self._check_distance(goal, goals):
                    break
            goals.append(goal)
        goals = np.array(goals).flatten()
        return goals

    def _sample_objects(self) -> np.ndarray:
        obj_pos = []
        # random sample
        for _ in range(self.num_blocks):
            while True:
                pos = self.obj_space.sample()
                pos[0] = self.np_random.choice([-1,1]) * pos[0]
                if len(obj_pos) == 0 or self._check_distance(pos, obj_pos):
                    break
            obj_pos.append(pos)
        # set in hand
        if self.np_random.uniform()<self.obj_in_hand_rate:
            choosed_block_id = self.np_random.randint(self.num_blocks)
            if obj_pos[choosed_block_id][0] > 0:
                ee_pos = self.get_ee_position1()
            else:
                ee_pos = self.get_ee_position0()
            block_x_shift = self.np_random.uniform(low = -self.block_size[0]/3, high = self.block_size[0]/3)
            obj_pos[choosed_block_id] = ee_pos+np.array([block_x_shift,0,0])
        return np.array(obj_pos).flatten()

    def is_success(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> Union[np.ndarray, float]:
        # must be vectorized !!
        dists = [
            np.linalg.norm(achieved_goal[..., i * 3:(i + 1) * 3] - desired_goal[..., i * 3:(i + 1) * 3], axis=-1)
            for i in range(self.num_blocks)
        ]
        success = [d < self.distance_threshold for d in dists]
        return float(np.all(success))

    def compute_reward(self, achieved_goal, desired_goal, info: Dict[str, Any]) -> Union[np.ndarray, float]:
        delta = (achieved_goal - desired_goal).reshape(-1, self.num_blocks ,3)
        dist_block2goal = np.linalg.norm(delta, axis=-1)
        rew = -np.mean(dist_block2goal>self.distance_threshold, axis=-1, dtype = float)
        if len(rew) == 1 :
            return rew[0]
        else: # to process multi dimension input
            return rew

    def _get_goal_obj_space(self, goal_z, obj_xyz_range):
        obj_range_low = np.array([self.gap_distance/2+self.block_size[0]/2, -obj_xyz_range[1] / 2, self.block_size[2]/2])
        obj_range_high = np.array(obj_xyz_range) + obj_range_low
        obj_space = gym.spaces.Box(low = obj_range_low, high = obj_range_high)
        goal_range_low = obj_range_low
        goal_range_low[0] = 0 # extend goal space to gap
        goal_range_high = obj_range_high
        goal_range_high[2] += goal_z
        goal_space = gym.spaces.Box(low = goal_range_low, high = goal_range_high)
        return goal_space, obj_space

    def _show_goal_obj_space(self):
        obj_range_size_half = (self.obj_space.high - self.obj_space.low)/ 2
        obj_range_pos_0 = (self.obj_space.high + self.obj_space.low)/ 2
        obj_range_pos_1 = obj_range_pos_0.copy()
        obj_range_pos_1[0] = -obj_range_pos_1[0]
        self.sim.create_box(
            body_name="debug_obj_0",
            half_extents=obj_range_size_half,
            mass=0.0,
            ghost=True,
            position=obj_range_pos_0,
            rgba_color=np.array([0, 0, 1, 0.1]),
        )
        self.sim.create_box(
            body_name="debug_obj_1",
            half_extents=obj_range_size_half,
            mass=0.0,
            ghost=True,
            position=obj_range_pos_1,
            rgba_color=np.array([0, 0, 1, 0.1]),
        )
        goal_range_size_half = (self.goal_space.high - self.goal_space.low)/ 2
        goal_range_pos_0 = (self.goal_space.high + self.goal_space.low)/ 2
        goal_range_pos_1 = goal_range_pos_0.copy()
        goal_range_pos_1[0] = -goal_range_pos_1[0]
        self.sim.create_box(
            body_name="debug_goal_0",
            half_extents=goal_range_size_half,
            mass=0.0,
            ghost=True,
            position=goal_range_pos_0,
            rgba_color=np.array([0, 1, 0, 0.05]),
        )
        self.sim.create_box(
            body_name="debug_goal_1",
            half_extents=goal_range_size_half,
            mass=0.0,
            ghost=True,
            position=goal_range_pos_1,
            rgba_color=np.array([0, 1, 0, 0.05]),
        )
    
    def _check_distance(self, my_pos, others_pos):
        delta_pos = abs(others_pos - my_pos)
        return min(delta_pos[..., 0]) > self.block_size[0] * 1.2 or min(delta_pos[..., 1]) > self.block_size[1] * 1.2
    
    def change(self, config = None):
        pass