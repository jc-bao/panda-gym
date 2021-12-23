import gym
import panda_gym
import numpy as np

env = gym.make("PandaTowerBimanual-v2", render=True)
# env = gym.make("PandaRelativePNPBimanual-v0", render=True)

obs = env.reset()
origin_ag = obs['achieved_goal']
done = False

param = 1
total_rew = 0
for i in range(1000):
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    ag = obs['achieved_goal']
    total_rew += reward
    env.render()
    if done: 
        param = 1 - param
        env.change(param)
        obs = env.reset()
        origin_ag = obs['achieved_goal']
        print(total_rew)
        total_rew = 0
env.close()
