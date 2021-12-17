import gym
import panda_gym
import numpy as np

def policy(obs, timestep):
    timestep = timestep%50
    obs = obs['observation']
    robot1_obs = obs[:7]
    robot2_obs = obs[7:14]
    task_obs = obs[14:]
    obj_pos = task_obs[:3]
    robot1_pos = robot1_obs[:3]
    robot2_pos = robot2_obs[:3]
    delta1 = obj_pos + [-0.05,0,0.003] - robot1_pos
    delta2 = obj_pos + [0.05,0,0.003] - robot2_pos
    if timestep<20:
        act1 = np.append(delta1/np.linalg.norm(delta1)*0.3, 1)
        act2 = np.append(delta2/np.linalg.norm(delta2)*0.3, 1)
    if timestep>=20 and timestep < 25:
        act1 = np.array([0]*3+[-1])
        act2 = np.array([0]*3+[-1])
    if timestep>=25 and timestep < 35:
        act1 = np.array([0,0,0.5,-1])
        act2 = np.array([0,0,0.5,-1])
    if timestep>=35:
        act1 = np.array([0,0,0,-1])
        act2 = np.array([0,0,0,-1])
    return np.concatenate((act1, act2))
    # return np.concatenate((act1, [0]*4))

env = gym.make("PandaTowerBimanualGravity-v2", render=True)

obs = env.reset()
done = False

total_rew = 0
rate = 0
for i in range(1000):
    action = env.action_space.sample()
    action = policy(obs, i)
    obs, reward, done, info = env.step(action)
    total_rew += reward
    env.render()
    if done: 
        rate = 1 - rate
        env.change(rate)
        env.reset()
        print(total_rew)
        total_rew = 0
env.close()

