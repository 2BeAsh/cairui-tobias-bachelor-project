""" Notes
Make predator prey custom environment from bottom. 
"""
# Imports
import numpy as np
import os

import gym
from gym import spaces
from gym.wrappers import FlattenObservation

from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy


class PredatorPreyEnv(gym.Env):
    """Custom Environment that follows gym interface."""


    def __init__(self, catch_radius, width=1, height=1, remaining_steps=1000):
        super().__init__()
        self.width = width  # Width and height of learning playground
        self.height = height
        self.remaining_steps = remaining_steps
        self.catch_radius = 0.5
        
        # Define action and observation space
        # Actions: Angle to move
        # Observations: agent and target position        
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,))  # Angle 
        self.observation_space = spaces.Dict({
            "agent": spaces.Box(low=np.array([0, 0]), high=np.array([self.width, self.height]), shape=(2,), dtype=np.float32),
            "target": spaces.Box(low=np.array([0, 0]), high=np.array([self.width, self.height]), shape=(2,), dtype=np.float32),
        })
    

    def _get_dist(self):
        """Helper function that calculates the distance between the agent and the target"""
        return np.linalg.norm(self._agent_position - self._target_position, ord=1)

        
    def reset(self, seed=None):
        # Fix seed and reset values 
        super().reset(seed=seed)
        self.reward = 0

        # Initial positions
        # Agent always starts one corner, target stats random location not within catch radius. 
        self._agent_position = np.array([0, 0], dtype=np.float32)
        self._target_position = self._agent_position
        dist = self._get_dist()

        while dist <= self.catch_radius:  
            self._target_position = self.np_random.uniform(low=np.array([0, 0]), high=np.array([self.width, self.height]), size=2)
            dist = self._get_dist()
        self._target_position = self._target_position.reshape(2,).astype(np.float32)
        observation = {"agent": self._agent_position, "target": self._target_position}

        return observation  # OBS: need info?


    def step(self, action):
        # Do the move
        move_x = np.cos(np.pi * action)
        move_y = np.sin(np.pi * action)
        agent_move = np.array([move_x, move_y]).reshape(2,)        
        self._agent_position = np.clip(a=self._agent_position + agent_move, a_min=[0, 0], a_max=[self.width, self.height])
        self._agent_position = self._agent_position.reshape(2,).astype(np.float32)
        # In the future:
        # Calculate the fluid velocity field and add that to the move. 
        # Have the target move by ... - Diffusion? - Just following the flow?

        # Reward
        dist = self._get_dist()
        if dist <= self.catch_radius:
            self.reward += 1000
            done = True
        else: 
            self.reward -= 1 
            done = False

        # Truncation - Check if ends early 
        self.remaining_steps -= 1
        if self.remaining_steps <= 0:
            truncated = True
        else:
            truncated = False 

        observation = {"agent": self._agent_position, "target": self._target_position}
        info = {}

        return observation, self.reward, done, info


    def _render_to_file(file, filename="pp_render.txt"):
        """Store the distance between agent and target in a file"""
        dist = self._get_dist()

        file = open(filename, "a+")
        file.write(dist)
        file.close()


    def render(self, mode="live", title="None", *kwargs):
        """Renders environment to screen"""

        if mode == "file":
            self._render_to_file(kwargs.get("filename", "pp_render.txt"))


# Check if SB3 likes environment
env = PredatorPreyEnv(catch_radius=0.5)
#env = FlattenObservation(env)
print("SB3 Check:", check_env(env))

# SB3
log_path = os.path.join("Training", "Logs")
model = PPO("MultiInputPolicy", env, verbose=1, tensorboard_log=log_path)
model.learn(total_timesteps=100_000)
