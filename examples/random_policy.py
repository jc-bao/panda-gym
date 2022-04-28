import gym
import panda_gym
import numpy as np
import time
import os
from pybullet_data import getDataPath

# env = gym.make("PandaRearrangeBimanual-v0", render=True, task_kwargs = {
#     'obj_xyz_range':[0.3, 0.4, 0],
#     'num_blocks': 1, # number of blocks
#     'os_rate': 0.6, # init goal in different table
#     'os_num_dist': 'binominal', # other side number distribution 'uniform', 'binominal'
#     'obj_in_hand_rate': 0.2, # init obj in hand
#     'gap_distance': None, # if None, auto set
#     'debug_mode': True, # if show debug info
#     'base_ep_len': 50, 
    # })

env = gym.make("PandaRearrangeBimanual-v0", render=True, task_kwargs={'goal_scale':1,'debug_mode':True, 'obj_in_hand_rate':0, 'gap_distance': 0})
# env = gym.make("PandaRearrangeUnstable-v2", render=True)
# env = gym.make("PandaRelativePNPBimanualObjInHand-v0", render=True)
# env = gym.make("PandaTowerBimanualSharedOpSpace-v0", render=True)
# env = gym.make("PandaPNPBimanual-v0", render=True)
obs = env.reset()
origin_ag = obs['achieved_goal']
done = False

param = 0
total_rew = 0
# env.task.obj_not_in_hand_rate=0
# count = 0
# for i in range(100):
#     obs = env.reset()
#     ag = obs['achieved_goal']
#     g = obs['desired_goal']
#     if (ag[0]>0) != (g[0]>0):
#         count+=1
# print(count/100)
# exit()

for _ in range(200):
    for i in range(env._max_episode_steps):
        action = np.zeros_like(env.action_space.sample())
        # disp0 = [-0.15, 0, 0.05]-env.robot0.get_ee_position()
        # disp1 = [0.15,0,0.05]-env.robot1.get_ee_position()
        # action[:3] = disp0/np.linalg.norm(disp0)*0.1
        # action[:3] = disp0
        # action[4:7] = disp1/np.linalg.norm(disp1)*0.1
        # action[4:7] = disp1
        # print([env.robot0.get_joint_angle(joint=i) for i in range(7)])
        # action[3]=-1
        # action[7]=-1
        # action[0]=1
        # action[1]=-0.2
        # action[4]=-1
        # action[5]=0.2
        obs, reward, done, info = env.step(action)
        # env.reset(panda1_init = [0.2,0.1,0.1])
        # recorder.add_keyframe()
        g = obs['desired_goal']
        ag = obs['achieved_goal']
        # total_rew += reward
        # env.render(mode='human')
        # print(info['unstable_state'], reward)
        # print(reward)
        if i == env._max_episode_steps-1:
            # param = (param + 0.1)
            # env.change([0,1])
            # env.change(1.9)
            origin_ag = obs['achieved_goal']
            # print(total_rew)
            total_rew = 0
            if param < 0.9:
                param += 0.1
            obs = env.reset({'os_rate':param})
            # print(info['dropout'])
env.close()