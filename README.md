# panda-bimanual

## Usage

```python
import gym
import panda_gym

env = gym.make('PandaReach-v2', render=True)

obs = env.reset()
done = False
while not done:
    action = env.action_space.sample() # random action
    obs, reward, done, info = env.step(action)

env.close()
```

## TODO 

- [ ] remove orientation lock
- [ ] not allow push down
- [ ] training without goal sample in the gap