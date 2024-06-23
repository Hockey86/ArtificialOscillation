import gymnasium as gym
import numpy as np
from gymnasium import spaces


class AgentEnv(gym.Env):
    """
    action space: 0 is run current plan, # 1 is to come up with a new plan
    """
    def __init__(self, state_bounds, state_init):
        super().__init__()
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Dict({k:spaces.Box(low=v[1], high=v[1], shape=(1,), dtype=v[0]) for k,v in state_bounds.items()})
        self.state_init = state_init
        assert self.observation_space.contains(self.state_init)

    def step(self, action):
        if action==0:
            ...
        elif action==1:
            ...
        #else:
        #    raise ValueError(f'action={action}')

        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        ...
        return self.state_init, info

    def render(self):
        ...

    def close(self):
        ...


if __name__=='__main__':
    from stable_baselines3.common.env_checker import check_env
    env = PubMedEnv()
    check_env(env)

