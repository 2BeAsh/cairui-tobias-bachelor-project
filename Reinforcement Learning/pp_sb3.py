""" Notes
Make predator prey custom environment from bottom. 
"""
# Imports
import numpy as np
import os
import pygame 

import gym
from gym import spaces
from gym.wrappers import FlattenObservation

from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy


class PredatorPreyEnv(gym.Env):
    """Custom Environment that follows gym interface."""

    metadata = {"render_modes": ["file", "human", "rgb_array"], "render_fps":4}


    def __init__(self, catch_radius, width=1, height=1, remaining_steps=1000, render_mode=None):
        #super().__init__() - ingen anelse hvorfor jeg gør det?
        # -- Variables --
        # Model 
        self.width = width  # Width and height of learning playground
        self.height = height
        self.remaining_steps = remaining_steps  # Not used atm
        self.catch_radius = catch_radius
        # Rendering
        self.window_size = 512  # PyGame window size
        assert render_mode is None or render_mode in self.metadata["render_modes"]  # Check if given render_mode matches available
        self.render_mode = render_mode
        self.window = None
        self.clock = None
        
        # -- Define action and observation space -- 
        # Actions: Angle to move
        # Observations: agent and target position        
        self.action_space = spaces.Box(low=np.array([-1, -1], dtype=np.float32), high=np.array([1, 1], dtype=np.float32), shape=(2,))  # dx, dy
        self.observation_space = spaces.Dict({
            "agent": spaces.Box(low=np.array([0, 0], dtype=np.float32), high=np.array([self.width, self.height], dtype=np.float32), shape=(2,)),
            "target": spaces.Box(low=np.array([0, 0], dtype=np.float32), high=np.array([self.width, self.height], dtype=np.float32), shape=(2,)),
        })
    

    def _get_dist(self):
        """Helper function that calculates the distance between the agent and the target"""
        return np.linalg.norm(self._agent_position - self._target_position, ord=1)


    def _get_obs(self):
        """Helper function with observation."""
        return {"agent": self._agent_position, "target": self._target_position}


    def _periodic_boundary(self, movement):
        """Helper function that primitively updates movement according to periodic boundaries. Can probably be optimized!"""
        # x-direction
        if movement[0] > self.width: 
            movement[0] -= self.width
        elif movement[0] < 0:
            movement[0] += self.width
        # y-direction
        if movement[1] > self.height:  
            movement[1] -= self.height
        elif movement[1] < 0:
            movement[1] += self.height

        return movement
        

    def reset(self, seed=None):
        # Fix seed and reset values 
        super().reset(seed=seed)

        # Initial positions
        # Agent always starts one corner, target stats random location not within catch radius. 
        self._agent_position = np.array([0., 0.], dtype=np.float32)
        self._target_position = self._agent_position
        dist = self._get_dist()

        while dist <= self.catch_radius:  
            self._target_position = self.np_random.uniform(low=np.array([0, 0], dtype=np.float32), high=np.array([self.width, self.height], dtype=np.float32), size=2)
            dist = self._get_dist()
        self._target_position = self._target_position.reshape(2,).astype(np.float32)
        observation = self._get_obs()

        if self.render_mode == "human":
            self._render_frame()

        return observation  # Need info? Gym says so, SB3 dislikes


    def step(self, action):
        # -- Movement --
        # Agent
        agent_max_move = 1  # Maximum velocity
        move_agent = self._agent_position + action * agent_max_move  # Implicit dt=1
        
        # Prey
        target_max_move = agent_max_move / 4
        move_target = self._target_position + np.random.uniform(low=-1, high=1, size=(2,)) * target_max_move 
        
        # Fluid velocity field
        # INSERT 

        # Box boundaries
        #self._agent_position = np.clip(a=move_agent, a_min=[0, 0], a_max=[self.width, self.height])
        #self._target_position = np.clip(a=move_target, a_min=[0, 0], a_max=[self.width, self.height])

        # Periodic boundaries - SANDSYNLIGVIS OPTIMERES
        self._agent_position = self._periodic_boundary(move_agent)
        self._target_position = self._periodic_boundary(move_target)

        # Get positions in correct shape
        self._agent_position = self._agent_position.reshape(2,).astype(np.float32)
        self._target_position = self._target_position.reshape(2,).astype(np.float32)


        # Reward
        dist = self._get_dist()
        if dist <= self.catch_radius:  # Catch target
            reward = float(100 * self.width * self.height)  # Scale catch reward with size of playground
            done = True
        else: 
            # For some reason must be a float not numpy float??
            reward = float(-dist ** 2)  # Penalises for being far away - Might not be a good idea when flow is introduced. 
            done = False

        # Truncation - Check if ends early  # OBS SB3 DOES NOT LIKE THIS. Might delete.
        self.remaining_steps -= 1
        if self.remaining_steps <= 0:
            truncated = True
        else:
            truncated = False 

        observation = self._get_obs()
        info = {}

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, done, info


    def _render_to_file(self, filename="pp_render.txt"):
        """Store the distance between agent and target in a file"""
        dist = self._get_dist()
        agent = self._agent_position
        target = self._target_position
        write = f"{dist}, {agent}, {target}"
        file = open(filename, "a+")
        file.write(write)
        file.close()


    def render(self, **kwargs):
        """Renders environment to screen"""
        if self.render_mode == "file":
            self._render_to_file(kwargs.get("filename", "pp_render.txt"))
        elif self.render_mode == "rgb_array":
            return self._render_frame()


    def _render_frame(self):
        # First setup PyGame window and Clock
        if (self.window is None) and (self.render_mode == "human"):
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        # Create canvas and make it white
        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))  # White
        
        # Draw target as rectangle
        pix_size = self.catch_radius * self.window_size / self.width  # Ideally, dots would be such that when they overlap, the prey is catched

        #agent_rect = pygame.Rect(left=float(self._target_position[0]), 
        #                         top=float(self._target_position[1]),
        #                         width=float(pix_size),
        #                         height=float(pix_size)
        #)
        
        target_rectangle = [float(self._target_position[0]), float(self._target_position[1]), float(pix_size), float(pix_size)]  # x, y, width, height
        pygame.draw.rect(
            canvas,
            (255, 0, 0),
            target_rectangle
        )

        #pygame.draw.rect(
        #    canvas,
        #    (255, 0, 0),  # Red
        #    pygame.Rect(
        #        pix_size * np.array(self._target_position),
        #        (pix_size, pix_size)
        #        ),
        #    )
        # Draw agent as circle
        pygame.draw.circle(
            canvas,
            (0, 0, 255),  # Color
            (float(self._agent_position[0]), float(self._agent_position[1])),  # x, y
            pix_size / 2,  # Radius
            )

        """ 
        pygame.draw.circle(
            canvas,
            (0, 0, 255),  # Blue
            (self._agent_position + 0.5) * pix_size,
            pix_size / 3
            )
        """

        # Draw canvas on visible window
        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # FPS 
            self.clock.tick(self.metadata["render_fps"])
        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))
    
    
    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
    

def check_model():
    env = PredatorPreyEnv(catch_radius=0.5)
    print("-- SB3 CHECK ENV: --")
    if check_env(env) == None:
        print("   The Environment is compatible with SB3")
    else:
        print(check_env(env))


def train(catch_radius, width, height, remaining_steps, train_total_steps):
    env = PredatorPreyEnv(catch_radius, width, height, remaining_steps)    
    #env = FlattenObservation(env)  - Not necessary?

    # Train with SB3
    log_path = os.path.join("Training", "Logs")
    model = PPO("MultiInputPolicy", env, verbose=1, tensorboard_log=log_path)
    model.learn(total_timesteps=train_total_steps)
    model.save("ppo_predator_prey")


def show_result(catch_radius, width, height, remaining_steps, render_mode):
    """Arguments must match that of the loaded model"""
    # Load model and create environment
    model = PPO.load("ppo_predator_prey")
    env = PredatorPreyEnv(catch_radius, width, height, remaining_steps, render_mode)
    
    obs = env.reset()
    done = False
    truncated = False 
    while not done or truncated:
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        env.render()
    env.close()


# Run the code
# Parameters
catch_radius = 0.1
width = 4
height = 4
remaining_steps = 1000
train_total_steps = int(1e5) 

#check_model()
#train(catch_radius, width, height, remaining_steps, train_total_steps)
show_result(catch_radius, width, height, remaining_steps, render_mode="human")