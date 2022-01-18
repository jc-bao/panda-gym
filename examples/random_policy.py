import gym
import panda_gym
import numpy as np
import time

# env = gym.make("PandaTowerBimanualGoalInObj-v2", render=True)
# env = gym.make("PandaTowerBimanualInHand-v2", render=True)
env = gym.make("PandaTowerBimanualSharedOpSpace-v0", render=True)
# env = gym.make("PandaTowerBimanualMusk-v2", render=True)

obs = env.reset()
origin_ag = obs['achieved_goal']
done = False

param = 1
total_rew = 0
for _ in range(100):
    for i in range(env._max_episode_steps):
        action = (env.action_space.sample())
        # action[:3] = 2*(np.array([0, 0.12, 0.1]) - env.robot0.get_ee_position())
        # action[4:7] = 2*(np.array([0, -0.12, 0.1]) - env.robot1.get_ee_position())
        # action[3] = -1
        # action[7] = -1
        # action[0]=-1
        # action[4]=1
        # x distance: 0.12*1.5, y distance: 0.24*1.5
        # print(env.robot1.get_ee_position()-env.robot0.get_ee_position())
        obs, reward, done, info = env.step(action)
        g = obs['desired_goal']
        ag = obs['achieved_goal']
        total_rew += reward
        env.render(mode='human')
        if i == env._max_episode_steps-1:
            param = min(param + 1, 6)
            # print(((ag[0]>0)==(g[0]>0) and (ag[0]>0)==(g[0]>0)))
            env.change(param)
            obs = env.reset()
            origin_ag = obs['achieved_goal']
            # print(total_rew)
            total_rew = 0
env.close()
