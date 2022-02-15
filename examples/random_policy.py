import gym
import panda_gym
import numpy as np
import time
import os
from pybullet_data import getDataPath

env = gym.make("PandaTowerBimanualParallelExchange-v2", render=True)
# env = gym.make("PandaTowerBimanualOsNumMix-v1", render=True)
# env = gym.make("PandaRearrangeUnstable-v2", render=True)
# env = gym.make("PandaRelativePNPBimanualObjInHand-v0", render=True)
# env = gym.make("PandaTowerBimanualSharedOpSpace-v0", render=True)
# env = gym.make("PandaPNPBimanual-v0", render=True)
obs = env.reset()
origin_ag = obs['achieved_goal']
done = False

param = 1
total_rew = 0
# env.task.obj_not_in_hand_rate=0
for _ in range(100):
    for i in range(env._max_episode_steps):
        action = (env.action_space.sample())
        # disp0 = [-0.45,0,0.1]-env.robot0.get_ee_position()
        # disp1 = [0.45,0,0.1]-env.robot1.get_ee_position()
        # action[:3] = disp0/np.linalg.norm(disp0)*0.1
        # action[4:7] = disp1/np.linalg.norm(disp1)*0.1
        # action[3]=-1
        # action[7]=-1
        # action[0]=1
        # action[5]=1
        obs, reward, done, info = env.step(action)
        # recorder.add_keyframe()
        g = obs['desired_goal']
        ag = obs['achieved_goal']
        total_rew += reward
        # env.render(mode='human')
        # print(info['unstable_state'], reward)
        if i == env._max_episode_steps-1:
            param = (param + 0.2)
            # env.change(param)
            obs = env.reset()
            origin_ag = obs['achieved_goal']
            # print(total_rew)
            total_rew = 0
env.close()