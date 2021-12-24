import gym
import panda_gym
import numpy as np

def policy(obs, timestep):
    goal = obs['desired_goal']
    goal1_pos = goal[:3]
    goal2_pos = goal[3:]
    obs = obs['observation']
    robot1_obs = obs[:7]
    robot2_obs = obs[7:14]
    task_obs = obs[14:]
    obj1_pos = task_obs[:3]
    obj2_pos = task_obs[12:15]
    robot1_pos = robot1_obs[:3]
    robot2_pos = robot2_obs[:3]
    delta1 = obj1_pos + [-0.04,0,0.003] - robot1_pos
    delta2 = obj2_pos + [0.04,0,0.003] - robot2_pos
    delta3 = goal1_pos - obj1_pos
    delta4 = goal2_pos - obj2_pos
    if timestep<40:
        print('reach')
        act1 = np.append(delta1/np.linalg.norm(delta1)*0.3, 1)
        act2 = np.append(delta2/np.linalg.norm(delta2)*0.3, 1)
    if timestep>=40 and timestep < 45:
        print('pick')
        act1 = np.array([0]*3+[-1])
        act2 = np.array([0]*3+[-1])
    if timestep>=45 and timestep < 90:
        print('lift')
        act1 = np.append(delta3/np.linalg.norm(delta3)*0.3, -1)
        act2 = np.append(delta4/np.linalg.norm(delta4)*0.3, -1)
        # act1 = np.array([0,0,0.5,-1])
        # act2 = np.array([0,0,0.5,-1])
    if timestep>=90:
        print('hold')
        act1 = np.array([0,0,0,-1])
        act2 = np.array([0,0,0,-1])
    return np.concatenate((act1, act2))
    # return np.concatenate((act1, [0]*4))

env = gym.make("PandaTowerBimanual-v2", render=True)
total_rew = 0
env.task.other_side_rate = 0
for _ in range(10):
    obs = env.reset()
    for t in range(100):
        action = env.action_space.sample()
        action = policy(obs, t)
        obs, reward, done, info = env.step(action)
        total_rew += reward
        env.render()
        print(reward)
env.close()