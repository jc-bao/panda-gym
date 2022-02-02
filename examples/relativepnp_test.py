import gym
import panda_gym
import numpy as np

def policy(obs, timestep):
    timestep = timestep%50
    goal_abs = obs['desired_goal'][:3]
    goal_relative = obs['desired_goal'][3:]
    obs = obs['observation']
    robot1_obs = obs[:7]
    robot2_obs = obs[7:14]
    task_obs = obs[14:]
    obj_pos = task_obs[:3]
    robot1_pos = robot1_obs[:3]
    robot2_pos = robot2_obs[:3]
    delta1 = -goal_relative/2 + [0,0,0.1] - robot1_pos
    delta2 = goal_relative/2 + [0,0,0.1] - robot2_pos
    delta3 = goal_abs - robot1_pos
    if timestep<20:
        act1 = np.append(delta1/np.linalg.norm(delta1)*0.3, -1)
        act2 = np.append(delta2/np.linalg.norm(delta2)*0.3, -1)
    elif timestep>=20:
        act1 = np.append(delta3/np.linalg.norm(delta3)*0.3, -1)
        act2 = np.array([0,0,0,-1])
    return np.concatenate((act1, act2))
    # return np.concatenate((act1, [0]*4))

env = gym.make("PandaPNPBimanualObjInHand-v0", render=True)

obs = env.reset()
done = False

total_rew = 0
for i in range(1000):
    action = env.action_space.sample()
    action = policy(obs, i)
    obs, reward, done, info = env.step(action)
    total_rew += reward
    env.render()
    print(reward)
    if done: 
        env.reset()
        print(total_rew)
        total_rew = 0
env.close()

