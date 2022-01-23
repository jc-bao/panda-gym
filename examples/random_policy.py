import gym
import panda_gym
import numpy as np
import time
import os
from pybullet_data import getDataPath

# env = gym.make("PandaTowerBimanualGoalInObj-v2", render=True)
env = gym.make("PandaTowerBimanualNoGapMixCurri-v0", render=True)
# env = gym.make("PandaTowerBimanualSharedOpSpace-v0", render=True)
# env = gym.make("PandaTowerBimanualMusk-v2", render=True)
# recorder = panda_gym.PyBulletRecorder()
# recorder.register_object(0, getDataPath()+'/franka_panda/panda.urdf')
# recorder.register_object(1, getDataPath()+'/franka_panda/panda.urdf')
obs = env.reset()
origin_ag = obs['achieved_goal']
done = False

param = 1
total_rew = 0
for _ in range(100):
    print(env._max_episode_steps, env.task.other_side_rate)
    for i in range(env._max_episode_steps):
        action = (env.action_space.sample())
        obs, reward, done, info = env.step(action)
        # recorder.add_keyframe()
        g = obs['desired_goal']
        ag = obs['achieved_goal']
        total_rew += reward
        # env.render(mode='human')
        if i == env._max_episode_steps-1:
            param = (param + 1)
            # print(((ag[0]>0)==(g[0]>0) and (ag[0]>0)==(g[0]>0)))
            env.change(param)
            obs = env.reset()
            origin_ag = obs['achieved_goal']
            # print(total_rew)
            total_rew = 0
env.close()

# recorder.save('demo.pkl')