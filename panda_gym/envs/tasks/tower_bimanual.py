import enum
from typing import Any, Dict, Tuple, Union

import numpy as np
import panda_gym

from panda_gym.envs.core import Task
from panda_gym.utils import distance

'''
Note: this environment add gripper pos to achieved goal. But as the goal and achieved goal will keep the same, it will
not matter
'''


class TowerBimanual(Task):
    def __init__(
        self,
        sim,
        get_ee_position0,
        get_ee_position1,
        distance_threshold=0.05,
        goal_xyz_range=[0.4, 0.3, 0.2],
        obj_xyz_range=[0.3, 0.3, 0],
        num_blocks=1,
        target_shape='any',
        curriculum_type=None,
        other_side_rate=0.5,
        has_gravaty_rate=1,
        use_musk=False,
        obj_not_in_hand_rate=1,
        goal_not_in_obj_rate=1,
        shared_op_space=False,
        gap_distance=0.23,
        reach_once=False,
        single_side=False,
        block_length=3,
        max_num_need_handover=10,
        max_move_per_step=0.05,
        noise_obs=False,
        exchange_only=False,
        parallel_robot=False,
        reward_type='normal',
        subgoal_rate=0,
        debug_mode=False,
        use_task_distribution=True,
        base_ep_len=50,
    ) -> None:
        self.use_task_distribution = use_task_distribution
        self.task_distribution = np.ones(num_blocks+1)/(num_blocks+1)
        self.debug_mode = debug_mode
        self.subgoal_generation = (subgoal_rate > 0)
        self.subgoal_rate = subgoal_rate
        self.reward_type = reward_type
        self.noise_obs = noise_obs
        self.exchange_only = exchange_only
        self.parallel_robot = parallel_robot  # to use longer
        super().__init__(sim)
        self.max_move_per_step = max_move_per_step
        self.max_num_need_handover = max_num_need_handover
        self.block_length = block_length
        self.reach_once = reach_once  # if fix obj once reach
        self.single_side = single_side  # only generate obj/goal on the single side
        self.load_tabel = True
        self.gap_distance = gap_distance
        self.shared_op_space = shared_op_space
        self.distance_threshold = distance_threshold
        self.get_ee_position0 = get_ee_position0
        self.get_ee_position1 = get_ee_position1
        self.object_size = 0.04
        self.use_musk = use_musk
        self.other_side_rate = other_side_rate
        self.has_gravaty_rate = has_gravaty_rate
        self.curriculum_type = curriculum_type  # gravity or other_side
        self.obj_not_in_hand_rate = obj_not_in_hand_rate
        self.goal_not_in_obj_rate = goal_not_in_obj_rate
        self.max_num_blocks = 6
        self.num_blocks = num_blocks
        self.base_ep_len = base_ep_len
        self._max_episode_steps = self.base_ep_len * \
            self.num_blocks * int(0.05/self.max_move_per_step)
        self.target_shape = target_shape
        self.goal_xyz_range = goal_xyz_range
        self.obj_xyz_range = obj_xyz_range
        self._update_obj_goal_range()
        self.num_not_musk = 1
        with self.sim.no_rendering():
            self._create_scene()
            self.sim.place_visualizer(target_position=np.zeros(
                3), distance=0.9, yaw=45, pitch=-30)
        self.ignore_goal_size = 0
        self.goal_size = 3
        self.seed(1)
        self.reset()

    def _create_scene(self) -> None:
        self.sim.create_plane(z_offset=-0.4)
        table_x = 0.3 if self.shared_op_space else 0.5 + self.gap_distance/2
        self.sim.create_table(length=1., width=1.5,
                              height=0.4, x_offset=(-table_x), index=0)
        self.sim.create_table(length=1., width=1.5,
                              height=0.4, x_offset=(table_x), index=1)
        if self.parallel_robot:
            self.sim.create_box(
                body_name="panda_base_0",
                half_extents=np.array([0.2, 0.2, 0.2]),
                mass=0.0,
                ghost=True,
                position=np.array([-0.8, -0.4, -0.2]),
                rgba_color=np.array([1, 1, 1, 1]),
            )
            self.sim.create_box(
                body_name="panda_base_1",
                half_extents=np.array([0.2, 0.2, 0.2]),
                mass=0.0,
                ghost=True,
                position=np.array([0.8, 0.4, -0.2]),
                rgba_color=np.array([1, 1, 1, 1]),
            )
        if self.debug_mode:
            obj_range_size_half = (
                self.obj_range_high - self.obj_range_low) / 2
            obj_range_pos_0 = (self.obj_range_high + self.obj_range_low) / 2
            obj_range_pos_1 = (self.obj_range_high + self.obj_range_low) / 2
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
            goal_range_size_half = (
                self.goal_range_high - self.goal_range_low) / 2
            goal_range_pos_0 = (self.goal_range_high + self.goal_range_low) / 2
            goal_range_pos_1 = (self.goal_range_high + self.goal_range_low) / 2
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
        self.use_small_obj = (self.gap_distance == 0 or self.shared_op_space)
        for i in range(self.max_num_blocks):
            color = np.random.rand(3)
            self.sim.create_box(
                body_name="object"+str(i),
                half_extents=np.array(
                    [1 if self.use_small_obj else self.block_length, 1, 1]) * self.object_size / 2,
                mass=0.2,
                position=np.array([2, 0.1*i - 0.3, self.object_size / 2]),
                rgba_color=np.append(color, 1),
            )
            self.sim.create_sphere(
                body_name="target"+str(i),
                radius=self.object_size / 1.9,
                mass=0.0,
                ghost=True,
                position=np.array([2, 0.1*i-0.3, 0.05]),
                rgba_color=np.append(color, 0.5),
            )

    def get_obs(self) -> np.ndarray:
        # position, rotation of the object
        obs = []
        poses = []
        for i in range(self.num_blocks):
            # [NOTE]trick: reset object orientation to aviod z rotation
            # pos = self.sim.get_base_position("object"+str(i)) + np.random.rand(3)*0.002 * self.noise_obs
            # ori = np.array([0, self.sim.get_base_rotation("object"+str(i))[1], 0])  + np.random.rand(3)*0.002 * self.noise_obs
            pos = self.sim.get_base_position("object"+str(i))
            poses.append(pos)
            ori = np.array(
                [0, self.sim.get_base_rotation("object"+str(i))[1], 0])
            self.sim.set_base_pose("object"+str(i), pos, ori)
            self.reach_state[i] = distance(
                pos, self.goal[3*i:3*i+3]) < self.distance_threshold or self.reach_state[i]
            obs.append(
                self.goal[3*i:3*i+3] if (self.reach_state[i] and self.reach_once) else pos)
            obs.append(ori)
            obs.append(self.sim.get_base_velocity("object"+str(i)))
            obs.append(self.sim.get_base_angular_velocity("object"+str(i)))
        observation = np.array(obs).flatten()
        # update subgoal
        # if subgoal reached
        if self.subgoal_generation:
            for i in range(self.num_blocks):
                if np.linalg.norm(poses[i]-self.subgoals[i]) < self.distance_threshold:
                    self.goal[i*3:i*3+3] = self.final_goal[i*3:i*3+3]
                    self.sim.set_base_pose(
                        'target'+str(i), self.goal[i*3:i*3+3], [0]*3)
        return observation

    def get_achieved_goal(self) -> np.ndarray:
        ag = []
        for i in range(self.num_blocks):
            pos = self.goal[3*i:3*i+3] \
                if (self.reach_state[i] and self.reach_once) else \
                self.sim.get_base_position("object"+str(i))
            ag.append(pos)
        ag = np.array(ag)
        if self.use_musk:
            ag[self.musk_index] = np.zeros(3)
        achieved_goal = (ag).flatten()
        return achieved_goal

    def reset(self, attr_dict={} ,goal=None, obj_pos_dict=None, num_need_handover=None,) -> None:
        self.reach_state = [False]*self.num_blocks
        obj_pos = self._sample_objects()
        if obj_pos_dict != None:  # over write by external command.
            for k, v in obj_pos_dict.items():
                obj_pos[k*3:k*3+3] = v
        self.goal = self._sample_goal(
            obj_pos=obj_pos, num_need_handover=num_need_handover) if goal == None else goal
        if self.subgoal_generation:
            self.final_goal = self.goal
            self.goal = self.subgoals.flatten()
        '''
        #For debug, manually set goal
        obj_pos = np.asarray([-0.3, 0.2, self.object_size/2])
        # obj_pos_0 = np.append(self.get_ee_position0()+np.array([self.object_size*self.block_length/2.5,0,0]), \
        # self.get_ee_position1()-np.array([self.object_size*self.block_length/2.5,0,0]))
        # obj_pos_1 = np.asarray([0.3, -0.1, self.object_size/2])
        # obj_pos_0 = self.get_ee_position0()+np.array([self.object_size*self.block_length/2.5,0,0])
        # obj_pos_1 = self.get_ee_position1()-np.array([self.object_size*self.block_length/2.5,0,0])
        # obj_pos = np.append(obj_pos_0, obj_pos_1)
        self.goal = np.asarray([0.3, 0.2, self.object_size/2])
        # self.final_goal = np.asarray([0.8, -0.06, self.object_size/2, -0.8, 0.06, self.object_size/2])
        # self.subgoals = np.asarray([[0.18, -0.06, self.object_size/2],[-0.18, 0.06, self.object_size/2]])
        '''
        for i in range(self.num_blocks):
            self.sim.set_base_pose(
                "target"+str(i), self.goal[i*3:(i+1)*3], np.array([0.0, 0.0, 0.0, 1.0]))
            self.sim.set_base_pose(
                "object"+str(i), obj_pos[i*3:(i+1)*3], np.array([0.0, 0.0, 0.0, 1.0]))
        # set gravity
        if self.np_random.uniform() < self.has_gravaty_rate:
            self.sim.physics_client.setGravity(0, 0, -9.81)
        else:
            self.sim.physics_client.setGravity(0, 0, 0)
        assert len(
            obj_pos) % self.num_blocks == 0, 'task observation shape linear to num blocks'
        obs = self.get_obs()
        self.obj_obs_size = int(len(obs)/self.num_blocks)

    def _sample_goal(self, obj_pos, num_need_handover=None) -> np.ndarray:
        goals = []
        if self.target_shape == 'tower':
            base_pos = self.np_random.uniform(
                self.goal_range_low, self.goal_range_high)
            base_pos[0] = np.random.choice([-1, 1]) * base_pos[0]
            for i in range(self.num_blocks):
                # z offset for the cube center
                goals.append(np.array([0, 0, self.object_size*i])+base_pos)
            goals = np.array(goals).flatten()
            # goals = np.append(goals, [0]*6) # Note: this one is used to calculate the gripper distance
        elif self.target_shape == 'any':
            self.num_need_handover = 0
            need_handover_goal_idx = []
            positive_side_goal_idx = []
            negative_side_goal_idx = []
            # generate if same side list
            if num_need_handover != None:  # manually set the index need handover
                if_other_side_list = np.zeros(self.num_blocks)
                handover_idx = np.random.choice(
                    np.arange(self.num_blocks), size=num_need_handover, replace=False)
                if_other_side_list[handover_idx] = 1
            elif self.single_side:
                if_other_side_list = np.zeros(self.num_blocks)
            # elif self.use_task_distribution:
            #     if_other_side_list = np.zeros(self.num_blocks)
            #     num_need_handover = np.random.choice(np.arange(self.num_blocks+1), 1, p=self.task_distribution)[0]
            #     handover_idx = np.random.choice(np.arange(self.num_blocks), size=num_need_handover, replace=False)
            #     if_other_side_list[handover_idx] = 1
            else:
                for _ in range(20):
                    if_other_side_list = (self.np_random.uniform(
                        size=self.num_blocks) < self.other_side_rate)
                    if sum(if_other_side_list) <= self.max_num_need_handover:
                        break
            for i in range(self.num_blocks):
                obj_side = (float(obj_pos[i*3] > 0)*2-1)
                if_same_side = if_other_side_list[i]*(-2) + 1
                goal_side = obj_side * if_same_side
                self.num_need_handover += int(if_same_side < 0)
                for _ in range(10):
                    # sample goal
                    if self.reach_once:
                        goal = self.np_random.uniform(
                            self.goal_range_low, self.goal_range_high)
                    else:
                        goal = self.np_random.uniform(
                            self.obj_range_low, self.obj_range_high)
                    goal[0] = goal_side*goal[0]
                    if len(goals) == 0:
                        break
                    # if goal is satisfied, append
                    # elif (np.linalg.norm(goal - obj_pos[i*3:i*3+3])) > self.distance_threshold*1.2:
                    x_size = self.object_size*1.5 if self.use_small_obj else self.object_size*6
                    if min(abs(goals - goal)[..., 0]) > x_size or \
                            min(abs(goals - goal)[..., 1]) > self.object_size*2:
                        break
                if goal_side > 0:
                    positive_side_goal_idx.append(i)
                else:
                    negative_side_goal_idx.append(i)
                if if_same_side < 0:
                    need_handover_goal_idx.append(i)
                goals.append(goal)
            if self.num_need_handover == 0:  # make object in the air to learn pnp
                if len(positive_side_goal_idx) > 0:
                    idx = np.random.choice(positive_side_goal_idx)
                    goals[idx] = np.random.uniform(
                        self.goal_range_low, self.goal_range_high)
                if len(negative_side_goal_idx) > 0:
                    idx = np.random.choice(negative_side_goal_idx)
                    new_goal = np.random.uniform(
                        self.goal_range_low, self.goal_range_high)
                    new_goal[0] = -new_goal[0]
                    goals[idx] = new_goal
            # if self.curriculum_type == 'goal_in_obj':
            # goal in object rate, curriculum trick
            new_idx = np.arange(self.num_blocks)
            np.random.shuffle(new_idx)
            relabel_num = 0
            for j in new_idx[1:]:
                if self.np_random.uniform() > self.goal_not_in_obj_rate:  # get goal to obj
                    goals[j] = obj_pos[j*3:j*3+3]
                    relabel_num += 1
            # if relabel_num == (self.num_blocks - 1):
            #     new_goal = np.random.uniform(self.goal_range_low, self.goal_range_high)
            #     new_goal[0] = np.random.choice([-1,1])*new_goal[0]
            #     goals[new_idx[0]] = new_goal
            if (not self.parallel_robot):  # check if the goal is close to arm base, if true, move away
                for goal in goals:
                    if abs(goal[0]) > 0.6:
                        goal[1] += 0.25 if goal[1] > 0 else (-0.25)
            if self.subgoal_generation:
                self.subgoals = np.array(goals)
                for idx in need_handover_goal_idx:
                    # generate goal rate: 0.5
                    if self.np_random.uniform() < self.subgoal_rate:
                        '''this trick to relabel is useless to learn handover'''
                        # if goal is far, then generate subgoal
                        if self.num_blocks == 1:
                            # one block case, generate subgoal in gap
                            self.subgoals[idx][0] = self.np_random.uniform(
                                low=-0.03, high=0.03)
                            self.subgoals[idx][2] = self.np_random.uniform(
                                low=0.08, high=0.12)
                        else:
                            if goals[idx][0] > 0.2:  # if goal is far
                                self.subgoals[idx][0] = 0.2
                            elif goals[idx][0] < -0.2:
                                self.subgoals[idx][0] = -0.2
                        # clip to make the y range smaller
                        self.subgoals[idx][1] = np.clip(self.subgoals[idx][1],
                                                        -self.goal_xyz_range[1]/2, self.goal_xyz_range[1]/2)
        elif self.target_shape == 'positive_side':
            goals = []
            goal0 = self.np_random.uniform(
                self.goal_range_low, self.goal_range_high)
            while True:
                goal1 = self.np_random.uniform(
                    self.goal_range_low, self.goal_range_high)
                if (abs(goal1 - goal0)[0]) > self.object_size*1.5 or \
                        (abs(goal1 - goal0)[1]) > self.object_size*1.5:
                    break
            if goal0[0] < goal1[0]:
                goals = [goal0, goal1]
            else:
                goals = [goal1, goal0]
        goals = np.array(goals)
        if self.use_musk:
            num_musk = self.num_blocks - self.num_not_musk
            self.musk_index = self.np_random.choice(
                np.arange(self.num_blocks), num_musk, replace=False)
            goals[self.musk_index] = np.zeros(3)
        goals = (goals).flatten()
        return goals

    def _sample_objects(self) -> np.ndarray:
        obj_pos = []
        self.obj_init_side = [0]*self.num_blocks
        num_positive = 0
        num_negative = 0
        for i in range(self.num_blocks):
            # get target object side
            while True:
                pos = self.np_random.uniform(
                    self.obj_range_low, self.obj_range_high)
                if self.single_side:
                    self.obj_init_side[i] = -1
                else:
                    self.obj_init_side[i] = self.np_random.choice([-1, 1])
                if self.exchange_only:
                    if num_negative > 0 and num_positive == 0:
                        self.obj_init_side[i] = 1
                    elif num_positive > 0 and num_negative == 0:
                        self.obj_init_side[i] = -1
                pos[0] = self.obj_init_side[i] * pos[0]
                if len(obj_pos) == 0:
                    break
                elif min(np.linalg.norm(obj_pos - pos, axis=-1)) > self.object_size*4:
                    break
            obj_pos.append(pos)
            if pos[0] < 0:
                num_negative += 1
            else:
                num_positive += 1
        if (not self.parallel_robot):  # check if the goal is close to arm base, if true, move away
            for pos in obj_pos:
                if abs(pos[0]) > 0.6:
                    pos[1] += 0.25 if pos[1] > 0 else (-0.25)
        choosed_block_id = np.random.choice(np.arange(self.num_blocks))
        if self.np_random.uniform() > self.obj_not_in_hand_rate:
            if self.np_random.uniform() > 0.5 or self.single_side:
                obj_pos[choosed_block_id] = self.get_ee_position0() +\
                    np.array([self.np_random.uniform(
                        low=-self.object_size*self.block_length/2.5,
                        high=self.object_size*self.block_length/2.5), 0, 0]) *\
                    (not self.use_small_obj)
            else:
                obj_pos[choosed_block_id] = self.get_ee_position1() +\
                    np.array([self.np_random.uniform(
                        low=-self.object_size*self.block_length/2.5,
                        high=self.object_size*self.block_length/2.5), 0, 0]) *\
                    (not self.use_small_obj)
        return np.array(obj_pos).flatten()

    def is_success(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> Union[np.ndarray, float]:
        # must be vectorized !!
        dists = [
            np.linalg.norm(achieved_goal[..., i * 3:(i + 1) * 3] -
                           desired_goal[..., i * 3:(i + 1) * 3], axis=-1)
            for i in range(self.num_blocks)
        ]
        success = [d < self.distance_threshold for d in dists]
        return float(np.all(success))

    def compute_reward(self, achieved_goal, desired_goal, info: Dict[str, Any]) -> Union[np.ndarray, float]:
        delta = (achieved_goal - desired_goal).reshape(-1, self.num_blocks, 3)
        dist_block2goal = np.linalg.norm(delta, axis=-1)
        rew = -np.sum(dist_block2goal > self.distance_threshold,
                      axis=-1, dtype=float)
        if self.shared_op_space or self.gap_distance == 0:
            ee_dis = info['ee_pos'][0] - info['ee_pos'][1]
            d = np.sqrt(
                0.5*(np.square(ee_dis[0]/0.12) + np.square(ee_dis[1]/0.24)))
            dis_rew = max(1.5-d, 0)
            rew -= dis_rew
        if self.reward_type == 'final':
            # e.g. 3obj: 0-0.5-1-1.5-3
            num_goal_reached = rew+self.num_blocks
            if not num_goal_reached == self.num_blocks:
                rew = -self.num_blocks+num_goal_reached/2
        if len(rew) == 1:
            return rew[0]
        else:  # to process multi dimension input
            return rew

    def _update_obj_goal_range(self):
        self.goal_range_low = np.array(
            [0, -self.goal_xyz_range[1]/2, self.object_size/2])
        self.goal_range_high = np.array(
            self.goal_xyz_range) + self.goal_range_low
        self.obj_range_low = np.array(
            [self.gap_distance/1.2*(not self.shared_op_space), -self.obj_xyz_range[1] / 2, self.object_size/2])
        self.obj_range_high = np.array(self.obj_xyz_range) + self.obj_range_low

    def change(self, config=None):
        if config is None:
            return
        if isinstance(config, (list, np.ndarray)):
            if self.use_task_distribution:
                self.task_distribution = config
            return
        else:
            if self.curriculum_type == 'gravity':
                self.has_gravaty_rate = config
            elif self.curriculum_type == 'os':
                self.other_side_rate = config
            elif self.curriculum_type == 'hand':
                self.obj_not_in_hand_rate = config
            elif self.curriculum_type == 'goal_in_obj':
                self.goal_not_in_obj_rate = config
            elif self.curriculum_type == 'goal_z':
                self.goal_xyz_range[-1] = config*0.2
                self.goal_range_high = np.array(
                    self.goal_xyz_range) + self.goal_range_low
            elif self.curriculum_type == 'num_blocks':
                self.num_blocks = int(config)
                self._max_episode_steps = self.base_ep_len * self.num_blocks
            elif self.curriculum_type == 'hand_range_num_mix':
                # goal space: config 1->1.5  goal 0.4 -> 0.9
                # inhand rate: config 1.5->2 inhand 0.5->1
                # 1-hand05 1.5-hand0 1.6-goal05obj04 1.7-goal07obj06 1.8-goal09obj08 2-2obj goal0.4-0.9 obj0.3-0.8
                self.goal_xyz_range = [np.clip(config-0.6, 0.4, 0.9), 0.3, 0.2]
                self.obj_xyz_range = self.goal_xyz_range.copy()
                self.obj_xyz_range[0] = self.goal_xyz_range[0]-0.1
                self._update_obj_goal_range()
                self.obj_not_in_hand_rate = np.clip(config-1, 0.5, 1)
                self.num_blocks = int(config)
                self._max_episode_steps = self.base_ep_len * self.num_blocks
            elif self.curriculum_type == 'musk':
                if self.num_not_musk < self.num_blocks:
                    self.num_not_musk = int(config*self.num_blocks)+1
            elif self.curriculum_type == 'os_num_mix':
                # 1- os=0 num=1; 1.5- os=0.6 num=1; 2- os=0.6 num=2 ...
                # expand number of block first
                if config < 1.5:
                    self.num_blocks = int(config)
                    self._max_episode_steps = self.base_ep_len * self.num_blocks
                elif config < 2:  # expand otherside rate
                    self.other_side_rate = 0.8 if self.gap_distance == 0 else 0.6  # 50%need two handover
                    self._max_episode_steps = self.base_ep_len * self.num_blocks
                else:  # expand number
                    self.num_blocks = int(config)
                    self._max_episode_steps = self.base_ep_len * self.num_blocks
            elif self.curriculum_type == 'hand_num_mix':
                # 1- hand=0.5 num=1; 1.5- hand=1 num=1; 2- hand=0.8 num=2 ...
                # curriculum step=0.01 bar=0.8
                # expand number of block first
                if config < 1.5:
                    self.obj_not_in_hand_rate = 1.5-config
                    self.num_blocks = int(config)
                    self._max_episode_steps = self.base_ep_len * self.num_blocks
                elif config < 2:  # expand otherside rate
                    self.obj_not_in_hand_rate = 1
                    self._max_episode_steps = self.base_ep_len * self.num_blocks
                else:  # expand number
                    self.other_side_rate = 1
                    self.num_blocks = int(config)
                    self._max_episode_steps = self.base_ep_len * self.num_blocks
            return
