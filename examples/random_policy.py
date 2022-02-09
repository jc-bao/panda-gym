import gym
import panda_gym
import numpy as np
import time
import os
from pybullet_data import getDataPath

env = gym.make("PandaTowerBimanual-v1", render=True)
# env = gym.make("PandaTowerBimanualOsNumMix-v1", render=True)
# env = gym.make("PandaRearrangeUnstable-v2", render=True)
# env = gym.make("PandaRelativePNPBimanualObjInHand-v0", render=True)
# env = gym.make("PandaTowerBimanualSharedOpSpace-v0", render=True)
# env = gym.make("PandaPNPBimanual-v0", render=True)
# recorder = panda_gym.PyBulletRecorder()
# recorder.register_object(0, getDataPath()+'/franka_panda/panda.urdf')
# recorder.register_object(1, getDataPath()+'/franka_panda/panda.urdf')
obs = env.reset()
origin_ag = obs['achieved_goal']
done = False

param = 1
total_rew = 0
# env.task.obj_not_in_hand_rate=0
for _ in range(10):
    for i in range(env._max_episode_steps):
        action = (env.action_space.sample())
        action[3]=-1
        action[7]=-1
        # action[1]=1
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
            # print(((ag[0]>0)==(g[0]>0) and (ag[0]>0)==(g[0]>0)))
            # env.change(param)
            obs = env.reset()
            origin_ag = obs['achieved_goal']
            # print(total_rew)
            total_rew = 0
env.close()