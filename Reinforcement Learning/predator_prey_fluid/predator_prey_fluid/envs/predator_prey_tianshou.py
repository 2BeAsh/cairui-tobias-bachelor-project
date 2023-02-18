import gymnasium as gym
from gymnasium import spaces

import torch
from torch import nn

import tianshou as ts
import numpy as np
import os
import pygame



#%% Create the Environment
class PredatorPreyEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}


    def __init__(self, render_mode=None, size=2):
        self.size = size  # Boundary size
        self.window_size = 512  # PyGame window size
        self.max_steps = 1000  # Max allowed steps to catch prey


        # Observations are dictionary with agent's and target's location
        # Each location is a 2d array with x and y coordinate
        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(0, self.size, shape=(2,), dtype=np.float32),
                "target": spaces.Box(0, self.size, shape=(2,), dtype=np.float32)
            }
            )

        #self.observation_space = spaces.Box(0, self.size, shape=(2, 2), dtype=np.float32)

        # One action, the angle value (constant velocity)
        self.action_space = spaces.Box(0, 2*np.pi, shape=(1,), dtype=np.float32)

        self.render_mode = render_mode

        self.window = None
        self.clock = None


    # Translate environment's state into an observation
    def _get_obs(self):
        #return [self._agent_location, self._target_location]
        return {"agent": self._agent_location, "target": self._target_location}


    def _get_info(self):
       #return np.array([np.linalg.norm(self._agent_location - self._target_location, ord=1)])
       return {
            "distance": np.linalg.norm(self._agent_location - self._target_location, ord=1)
        }


    # Reset function
    def reset(self, seed=None, options=None):
        # Fix seed
        super().reset(seed=seed)

        self.max_steps = 1000  # Reset timescale
        # Initial random position
        self._agent_location = self.np_random.uniform(low=0, high=self.size, size=2)

        # Make sure target is not at agent's position - I believe this might be redundant, or changed such that it does not start within epsilon distance
        self._target_location = self._agent_location
        while np.array_equal(self._target_location, self._agent_location):
            self._target_location = self.np_random.uniform(
                low=0, high=self.size, size=2,
            )

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_fame()

        return observation, info


    def step(self, action):
        move_size = 1
        angle_moved = action
        x_move = move_size * np.cos(angle_moved)
        y_move = move_size * np.sin(angle_moved)
        move = np.array([x_move, y_move])

        # Make sure stays inside borders
        self._agent_location = np.clip(
            self._agent_location + move, a_min=0, a_max=self.size)

        # End of episode iff within epsilon of target
        epsilon = 1  # Catch target radius
        dist = np.linalg.norm(self._agent_location - self._target_location, ord=1)
        terminated = dist <= epsilon
        if terminated:
            reward = 1000  # Get 1000 score for catching prey
        else:
            reward = -1  # Lose 1 score for each step taken
        observation = self._get_obs()
        info = self._get_info()

        # Time steps
        self.max_steps -= 1
        if self.max_steps <= 0:
            truncated = True
        else:
            truncated = False

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, truncated, info


    # Rendering
    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()


    def _render_frame(self):  # - VIRKER IKKE FØR FÅR INSTALLERET PYGAME.
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.window_size, self.window_size)
            )
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (
                self.window_size / self.size
        )  # Single grid size

        # Draw target
        pygame.draw.rect(
            canvas,
            (255, 0, 0),
            pygame.Rect(
                pix_square_size * self._target_location,
                (pix_square_size, pix_square_size),
            ),
        )
        # Draw agent
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (self._agent_location + 0.5) * pix_square_size,
            pix_square_size / 3,
        )

        if self.render_mode == "human":
            # Copy drawings from `canvas` to visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # Fix framerates
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

        def close(self):
            if self.window is not None:
                pygame.display.quit()
                pygame.quit()


#%% Test run
from gymnasium.wrappers import FlattenObservation

env = PredatorPreyEnv()
env = FlattenObservation(env)

#%% Test environment uden trainer
episodes = 5
for episode in range(1, episodes+1):
    obs = env.reset()
    done = False
    truncated = False
    score = 0

    while not (done or truncated):
        env.render()
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        score += reward
    print(f"Episode: {episode}, Score: {score}, ended early: {truncated}")

#%% Train with Tianshou

# Vectorized environment
N_env_train = 10
train_envs = ts.env.DummyVectorEnv([lambda: PredatorPreyEnv() for _ in range(N_env_train)])
test_envs = ts.env.DummyVectorEnv([lambda: PredatorPreyEnv() for _ in range(100)])

# Network
class Net(nn.Module):
    def __init__(self, state_shape, action_shape):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(np.prod(state_shape), 128), nn.ReLU(inplace=True),
            nn.Linear(128, 128), nn.ReLU(inplace=True),
            nn.Linear(128, 128), nn.ReLU(inplace=True),
            nn.Linear(128, np.prod(action_shape)),
        )

    def forward(self, obs, state=None, info={}):
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float)  # HER DER ER EN ERROR! Den siger at første værdi i obs ikke har en len().
        batch = obs.shape[0]
        logits = self.model(obs.view(batch, -1))
        return logits, state

state_shape = env.observation_space.shape or env.observation_space.n
action_shape = env.action_space.shape or env.action_space.n
net = Net(state_shape, action_shape)
optim = torch.optim.Adam(net.parameters(), lr=1e-3)

# Policy
policy = ts.policy.DQNPolicy(net, optim, discount_factor=0.9, estimation_step=3, target_update_freq=320)

# Collector
train_collector = ts.data.Collector(policy, train_envs, ts.data.VectorReplayBuffer(20_000, N_env_train), exploration_noise=True)
test_collector = ts.data.Collector(policy, test_envs, exploration_noise=True)

# Train policy with Trainer
result = ts.trainer.offpolicy_trainer(
    policy, train_collector, test_collector,
    max_epoch=10, step_per_epoch=10_000, step_per_collect=10,
    update_per_step=0.1, episode_per_test=100, batch_size=64,
    train_fn=lambda epoch, env_step: policy.set_eps(0.1),
    test_fn=lambda epoch, env_step: policy.set_eps(0.5),
    stop_fn=lambda mean_rewards: mean_rewards >= env.spec.reward_threshold)
print(f"Finished Training! Use {result['duration']}")
