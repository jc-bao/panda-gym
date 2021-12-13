import gym
import panda_gym
import time

env = gym.make("PandaTowerBimanual-v2", render=True)

obs = env.reset()
done = False

for _ in range(1000):
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    env.render()
    if done: env.reset()

env.close()
