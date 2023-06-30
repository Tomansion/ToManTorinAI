import gymnasium as gym
import numpy as np
from gymnasium import spaces


class GymnasiumEnv(gym.Env):
    def __init__(self):
        self.observation_space = spaces.Discrete(10)
        self.action_space = spaces.Discrete(2)
        self.action_mask = np.ones(self.action_space.n, dtype=bool)

        self.state = 0

    def reset(self, seed=None, options=None):
        self.state = np.random.randint(0, 10)
        self.update_action_mask()
        return self.state, {}

    def step(self, action):
        if not self.action_mask[action]:
            # raise ValueError("Invalid action selected. The action is masked.")
            pass

        if action == 0:
            self.state -= 1
        elif action == 1:
            self.state += 1

        self.state = np.clip(self.state, 0, 10 - 1)

        reward = -abs(self.state - 5)

        done = self.state == 5

        self.update_action_mask()

        return self.state, reward, done, False, {}

    def update_action_mask(self):
        # Implement your logic to update the action mask based on the current state
        # Example: Disable action 0 if the state is 0, and disable action 1 if the state is 10
        if self.state == 0:
            self.action_mask[0] = False
        elif self.state == 10:
            self.action_mask[1] = False
        else:
            self.action_mask = np.ones(self.action_space.n, dtype=bool)

    def action_masks(self):
        return self.action_mask.tolist()
