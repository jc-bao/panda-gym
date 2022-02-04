import gym
import panda_gym
import numpy as np

def policy(obs, timestep):
    timestep = timestep%50
    goal_1 = obs['desired_goal'][:3]
    goal_2 = obs['desired_goal'][3:]
    obs = obs['observation']
    robot1_obs = obs[:7]
    robot2_obs = obs[7:14]
    task_obs = obs[14:]
    obj_pos_1 = task_obs[:3]
    obj_pos_2 = task_obs[12:15]
    robot1_pos = robot1_obs[:3]
    robot2_pos = robot2_obs[:3]
    delta1 = goal_1 - obj_pos_1
    delta2 = goal_2 - obj_pos_2
    # delta1 = goal_1 - robot1_pos
    # delta2 = goal_2 - robot2_pos
    if timestep<20:
        act1 = np.append(delta1/(np.linalg.norm(delta1)+0.01), -1)
        act2 = np.append(delta2/(np.linalg.norm(delta2)+0.01), -1)
    elif timestep>=20:
        act1 = np.array([0,0,0,1])
        act2 = np.array([0,0,0,1])
    return np.concatenate((act1, act2))

env = gym.make("PandaTowerBimanualReachOnce-v2", render=True)

obs = env.reset()
done = False

total_rew = 0
for i in range(1000):
    action = env.action_space.sample()
    action = policy(obs, i)
    obs, reward, done, info = env.step(action)
    total_rew += reward
    print(reward)
    if i%50==0: 
        env.reset()
        print(total_rew)
        total_rew = 0
env.close()

