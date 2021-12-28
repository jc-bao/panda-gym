import gym
import panda_gym
import numpy as np

# env = gym.make("PandaTowerBimanualMusk-v2", render=True)
env = gym.make("PandaRelativePNPBimanualObjInHand-v0", render=True)

obs = env.reset()
origin_ag = obs['achieved_goal']
done = False

param = 0
total_rew = 0
for i in range(1000):
    action = env.action_space.sample()
    action[-1]=-1
    action[3]=-1
    obs, reward, done, info = env.step(action)
    # env.reset()
    ag = obs['achieved_goal']
    total_rew += reward
    env.render()
    if done: 
        param += 0.3
        env.change(param)
        obs = env.reset()
        origin_ag = obs['achieved_goal']
        print(total_rew)
        total_rew = 0
env.close()
