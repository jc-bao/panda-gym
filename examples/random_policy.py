import gym
import panda_gym
import numpy as np
import time

# env = gym.make("PandaTowerBimanual-v2", render=True)
env = gym.make("PandaTowerBimanualInHand-v1", render=True)
# env = gym.make("PandaTowerBimanualMusk-v2", render=True)

obs = env.reset()
origin_ag = obs['achieved_goal']
done = False

param = 0.5
total_rew = 0
for i in range(1000):
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    g = obs['desired_goal']
    total_rew += reward
    env.render()
    if done: 
        obj_0_air = int(abs(g[2]-0.02)>0.001)
        # param += 0.1
        env.change(param)
        obs = env.reset()
        origin_ag = obs['achieved_goal']
        print(total_rew)
        total_rew = 0
env.close()
