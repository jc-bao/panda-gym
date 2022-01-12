import gym
import panda_gym
import numpy as np
import time

# env = gym.make("PandaTowerBimanualGoalInObj-v2", render=True)
# env = gym.make("PandaTowerBimanualInHand-v2", render=True)
env = gym.make("PandaTowerBimanualNumBlocks-v2", render=True)
# env = gym.make("PandaTowerBimanualMusk-v2", render=True)

obs = env.reset()
origin_ag = obs['achieved_goal']
done = False

param = 1
total_rew = 0
for _ in range(10):
    for i in range(env._max_episode_steps):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        g = obs['desired_goal']
        ag = obs['achieved_goal']
        total_rew += reward
        # env.render()
        if i == env._max_episode_steps-1:
            param = min(param + 1, 6)
            # print(((ag[0]>0)==(g[0]>0) and (ag[0]>0)==(g[0]>0)))
            env.change(param)
            obs = env.reset()
            origin_ag = obs['achieved_goal']
            # print(total_rew)
            total_rew = 0
env.close()
