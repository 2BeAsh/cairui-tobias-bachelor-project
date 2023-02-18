"""Noter:
Tror fejl "AssertionError: ... singe value, not a tuple" skyldes mismatch mellem observation returned af reset og observation defineret i __init__.
"""



# Imports
import gym
from gym import spaces
from gym.wrappers import FlattenObservation

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_checker import check_env

import numpy as np
import os
import pygame


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
        # Observations are dictionary with agent's and target's location
        # Each location is a 2d array with x and y coordinate
        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(-self.size, self.size, shape=(2,), dtype=np.float32),
                "target": spaces.Box(-self.size, self.size, shape=(2,), dtype=np.float32)
            }
        )
        # One action, the angle value (constant velocity)
        self.action_space = spaces.Box(-1, 1, shape=(1,), dtype=np.float32)


    # De to funktioner er blot for convinience, så slipper for at skrive fulde udtryk hver gang.
    def _get_obs(self):
        return {"agent": self._agent_location, "target": self._target_location}


    def _flatten_obs(self):
        obs_agent = self._agent_location
        obs_target = self._target_location
        return np.concatenate((obs_agent, obs_target))


    def _get_info(self):
        return {
            "distance": np.linalg.norm(self._agent_location - self._target_location, ord=1)
        }


    # Reset function
    def reset(self, seed=None, options=None):
        # Fix seed
        super().reset(seed=seed)  # Betyder env.reset(seed=seed)
        self.max_steps = 1000  # Reset timescale
        # Initial random position
        self._agent_location = self.np_random.uniform(low=-self.size, high=self.size, size=2)

        # Make sure target is not at agent's position - Tror er overflødig, eller ændrer så ikke starter inden for epsilon afstand af target
        self._target_location = self._agent_location
        while np.array_equal(self._target_location, self._agent_location):
            self._target_location = self.np_random.uniform(
                low=-self.size, high=self.size, size=2,
            )

        observation = self._get_obs()  # Tror måske problemet er, at _get_obs() ikke påvirkers af wrappers?? - Nok nærmere noget med info
        #observation = self._flatten_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_fame()

        return observation, info  # TROR MÅSKE FEJLEN ER HER. HVIS INKLUDERER info BLIVER DET TUPPLE. MEN GYM DOC SIGER SKAL HAVE INFO MED??


    def step(self, action):
        move_size = 0.5
        x_move = move_size * np.cos(action * np.pi)
        y_move = move_size * np.sin(action * np.pi)
        move = np.array([x_move, y_move]).reshape(2,)

        # Make sure stays inside borders
        self._agent_location = np.clip(
            self._agent_location + move, a_min=-self.size, a_max=self.size)

        # End of episode iff within epsilon of target
        epsilon = 1  # Catch target radius
        info = self._get_info()
        dist = info["distance"]
        terminated = dist <= epsilon
        if terminated:  # Lige nu reward primitiv, kan forbedres. Fx dist afhængig
            reward = 1000  # Get 1000 score for catching prey
        else:
            reward = -1  # Lose 1 score for each step taken
        observation = self._get_obs()

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


# Observation space wrappers
class RelativePositionWrapper(gym.ObservationWrapper):
    """Change the observation to return relative position of agent and target."""
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(2,))


    def observation(self, obs):
        return obs["target"] - obs["agent"]


class DistanceWrapper(gym.ObservationWrapper):
    """Change the observation to return the distance between agent and target."""
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(1,))


    def observation(self, obs):
        return np.linalg.norm(obs["target"] - obs["agent"], ord=1)


#%% Test run
env = PredatorPreyEnv()
#env = DistanceWrapper(env)
#env = RelativePositionWrapper(env)
env = FlattenObservation(env)  # Learners kan ikke lide Dict observation spaces.


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
#print("Environment check:", check_env(env))

#%% Train
log_path = os.path.join("Training", "Logs")
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_path)
model.learn(total_timesteps=10_000)
