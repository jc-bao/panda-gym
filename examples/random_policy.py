import gym
import panda_gym
import time

env = gym.make("PandaTowerBimanual-v2", render=True)

obs = env.reset()
done = False

total_rew = 0
for _ in range(1000):
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    total_rew += reward
    print(obs['desired_goal'],obs['achieved_goal'])
    env.render()
    if done: 
        env.reset()
        print(total_rew)
        total_rew = 0
env.close()
