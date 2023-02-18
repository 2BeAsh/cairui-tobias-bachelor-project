import gym
from gym import spaces

import numpy as np
import os
import pygame  #  Problemer med at opdatere Python til en version der kan håndtere Pygame.

#%% Create the Environment
class PredatorPreyEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}


    def __init__(self, render_mode=None, size=1):
        self.size = size  # Boundary size
        self.window_size = 512  # PyGame window size
        self.max_steps = 1000  # Max allowed steps to catch prey
        self.render_mode = render_mode
        self.window = None
        self.clock = None
        # Each location is a 2d array with x and y coordinate
        self.observation_space = spaces.Box(-self.size, self.size, shape=(4,), dtype=np.float32)
        # One action, the angle value (constant velocity)
        self.action_space = spaces.Box(-np.pi, np.pi, shape=(1,), dtype=np.float32)

    # De to funktioner er blot for convinience, så slipper for at skrive/beregne hver gang.
    def _get_obs(self):
        return np.concatenate((self._agent_location, self._target_location))


    def _get_info(self):
        return np.linalg.norm(self._agent_location - self._target_location, ord=1)


    # Reset function
    def reset(self, seed=None, options=None):
        # Fix seed
        super().reset(seed=seed)  # Betyder env.reset(seed=seed)
        self.max_steps = 1000  # Reset timescale
        # Initial random position
        self._agent_location = self.np_random.uniform(low=-self.size, high=self.size, size=(2,))

        # Make sure target is not at agent's position - Tror er overflødig, eller ændrer så ikke starter inden for epsilon afstand af target
        self._target_location = self._agent_location  # Skal have initial array at sammenligne med. Ikke sikkert er nødvendigt når er i kontinuert rum.
        while np.array_equal(self._target_location, self._agent_location):
            self._target_location = self.np_random.uniform(
                low=-self.size, high=self.size, size=(2,),
            )

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_fame()

        return observation, info


    def step(self, action):
        move_size = 0.5
        angle_moved = action
        x_move = move_size * np.cos(angle_moved)
        y_move = move_size * np.sin(angle_moved)
        move = np.array([x_move, y_move]).reshape(2,)

        # Make sure stays inside borders
        self._agent_location = np.clip(
            self._agent_location + move, a_min=-self.size, a_max=self.size)
        # End of episode iff within epsilon of target
        epsilon = 1  # Catch target radius
        dist = self._get_info()
        terminated = dist <= epsilon
        if terminated:  # Lige nu reward primitiv, kan forbedres. Fx dist afhængig
            reward = 1000  # Get 1000 score for catching prey
        else:
            reward = -1  # Lose 1 score for each step taken
        observation = self._get_obs()
        info = dist

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


    def _render_frame(self):
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
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_checker import check_env

from gym.wrappers import FlattenObservation

env = PredatorPreyEnv()
#env = FlattenObservation(env)  # Learners kan ikke lide Dict observation spaces.

#%% Tjek om virker uden trainer
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

# Check om SB3 har problemer med environmentet (det har den...)
print("Environment check:", check_env(env))

#%% Train
log_path = os.path.join("Training", "Logs")
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_path)
model.learn(total_timesteps=10_000)
