""" Notes
Make predator prey custom environment from bottom. 
"""
# Imports
import numpy as np
import os
import sys
import pygame 
import matplotlib.pyplot as plt

import gym
from gym import spaces
from gym.wrappers import FlattenObservation

from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

# Get custom module
sys.path.append('Fluid')
import fluid_module as fm


# Environment
class PredatorPreyEnv(gym.Env):
    """Custom Environment that follows gym interface."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps":4}


    def __init__(self, squirmer_radius=0.1, width=1, height=1, legendre_degree=2, render_mode=None):
        #super().__init__() - ingen anelse hvorfor jeg gør det?
        # -- Variables --
        # Model 
        self.width = width  # Width and height of learning playground
        self.height = height
        self.squirmer_radius = squirmer_radius  # Also the radius of the squirmer
        self.legendre_degree = legendre_degree  # Highest mode the squirmer can access
        # Rendering
        self.window_size = 512  # PyGame window size
        assert render_mode is None or render_mode in self.metadata["render_modes"]  # Check if given render_mode matches available
        self.render_mode = render_mode
        self.window = None
        self.clock = None
        
        # -- Define action and observation space -- 
        # Actions: Strength of Legendre Modes
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.legendre_degree,), dtype=np.float32)  #spaces.Box(low=np.array([-1, -1], dtype=np.float32), high=np.array([1, 1], dtype=np.float32), shape=(2,))  # dx, dy
        
        # Observations: agent and target positions, and velocity field on agent
        # The reference frame is at the Squirmer, such that it never moves. Instead, it moves the velocity field which in turn moves the target.
        self.observation_space = spaces.Dict({
            "target": spaces.Box(low=np.array([0, 0], dtype=np.float32), high=np.array([self.width, self.height], dtype=np.float32), shape=(2,)),
            "velocity": spaces.Box(low=-np.inf, high=np.inf, shape=(2,))  # Target's velocity
        })
    

    def _get_dist(self):
        """Helper function that calculates the distance between the agent and the target"""
        return np.linalg.norm(self._target_position, ord=1)


    def _cartesian_to_polar(self):
        r = self._get_dist()
        theta = np.arctan(self._target_position[0] / self._target_position[1])
        return r, theta


    def _get_obs(self):
        """Helper function with observation."""
        return {"target": self._target_position, "velocity": self._target_velocity}


    def _periodic_boundary(self, movement):
        """Helper function that (primitively) updates movement according to periodic boundaries. Can probably be optimized!"""
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
        # Agent always in center 
        self._agent_position = np.array([self.width / 2, self.height / 2], dtype=np.float32) #self.np_random.uniform(low=np.array([0, 0], dtype=np.float32), high=np.array([self.width, self.height], dtype=np.float32), size=2)
        self._agent_position = self._agent_position.reshape(2,).astype(np.float32)
        
        # Target stats random location not within 4 times catch radius.
        self._target_position = self._agent_position
        dist = self._get_dist()
        while dist <= 4 * self.squirmer_radius:  
            self._target_position = self.np_random.uniform(low=np.array([0, 0], dtype=np.float32), high=np.array([self.width, self.height], dtype=np.float32), size=2)
            dist = self._get_dist()
        self._target_position = self._target_position.reshape(2,).astype(np.float32)
        
        # Initial velocity
        self._target_velocity = np.array([0, 0], dtype=np.float32).reshape(2,)

        # Observation
        observation = self._get_obs()

        if self.render_mode == "human":
            self._render_frame()

        return observation  # Need info? Gym says so, SB3 dislikes


    def step(self, action):
        # Agent changes velocity at target's position
        # Target is moved by velocity        
        # Target's only movement is by the Squirmer's influence, does not diffuse
        
        # -- Movement --
        r, theta = self._cartesian_to_polar()
        B = action
        velocity_x, velocity_y = fm.field_cartesian(r, theta, n=self.legendre_degree, B=B, a=self.squirmer_radius)
        velocity = np.array([velocity_x, velocity_y], dtype=np.float32)
        print(velocity)
        # ARTIFICIALLY CAP VELOCITY
        velocity = np.clip(velocity, -1, 1)


        move_target = self._target_position + velocity  # dt = 1
        self._target_position = self._periodic_boundary(move_target).reshape(2,).astype(np.float32)

        # Box boundaries
        #self._agent_position = np.clip(a=move_agent, a_min=[0, 0], a_max=[self.width, self.height])
        #self._target_position = np.clip(a=move_target, a_min=[0, 0], a_max=[self.width, self.height])

        # Periodic boundaries - SANDSYNLIGVIS OPTIMERES
        #self._agent_position = self._periodic_boundary(move_agent)


        # Get positions in correct shape
        #self._agent_position = self._agent_position.reshape(2,).astype(np.float32)

        # Reward
        dist = self._get_dist()
        catch_radius = 1.5 * self.squirmer_radius
        if dist <= catch_radius:  # Catch target
            reward = float(100 * self.width * self.height)  # Scale catch reward with size of playground
            done = True
        else: 
            # For some reason must be a float not numpy float??
            reward = float(-dist ** 2)  # Penalises for being far away - Might not be a good idea when flow is introduced. 
            done = False

        observation = self._get_obs()
        info = {}

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, done, info


    def render(self):
        """Renders environment to screen or prints to file"""
        if self.render_mode == "rgb_array":
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
        pix_size = self.window_size / self.width  # Ideally, dots would be such that when they overlap, the prey is catched
        
        target_rectangle = [float(self._target_position[0] * pix_size), float(self._target_position[1] * pix_size), float(pix_size * self.squirmer_radius / 2), float(pix_size * self.squirmer_radius / 2)]  # x, y, width, height
        pygame.draw.rect(
            canvas,
            (255, 0, 0),
            target_rectangle
        )

        pygame.draw.circle(
            canvas,
            (0, 0, 255),  # Color
            (float(self._agent_position[0] * pix_size), float(self._agent_position[1] * pix_size)),  # x, y  - Maybe needs to add +0.5 to positions?
            pix_size * self.squirmer_radius / 2,  # Radius
            )

        # Draw arrow representing direction of flow
        # Flow right now is constant in x direction
        pygame.draw.polygon(  # Arrow head
            canvas,
            (0, 0, 0),  # Black
            [(self.width / 2 * pix_size, self.height / 2 * pix_size),      # Upper point
             (self.width / 1.8 * pix_size, self.height / 2.2 * pix_size),  # Middle point
             (self.width / 2 * pix_size, self.height / 2.4 * pix_size)],   # Lower point
        )
        pygame.draw.line(
            canvas,
            (0, 0, 0),  # Black
            (self.width / 1.8 * pix_size,self.height / 2.2 * pix_size),  # start point
            (self.width / 2.4 * pix_size, self.height / 2.2 * pix_size),  # end point
            width=2,
        )

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
    env = PredatorPreyEnv()
    #env = FlattenObservation(env)
    print("-- SB3 CHECK ENV: --")
    if check_env(env) == None:
        print("   The Environment is compatible with SB3")
    else:
        print(check_env(env))


def train(squirmer_radius, width, height, legendre_degree, train_total_steps):
    env = PredatorPreyEnv(squirmer_radius, width, height, legendre_degree)    

    # Train with SB3
    log_path = os.path.join("Reinforcement Learning", "Training", "Logs")
    model = PPO("MultiInputPolicy", env, verbose=1, tensorboard_log=log_path)
    model.learn(total_timesteps=train_total_steps)
    model.save("ppo_predator_prey")


def show_result(squirmer_radius, width, height, legendre_degree, render_mode):
    """Arguments must match that of the loaded model"""
    # Load model and create environment
    model = PPO.load("ppo_predator_prey")
    env = PredatorPreyEnv(squirmer_radius, width, height, legendre_degree, render_mode)
    
    obs = env.reset()
    done = False
    reward_list = []
    # Run and render model
    while not done:
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)
        reward_list.append(reward)
        env.render()
    env.close()


# Run the code
# Parameters
squirmer_radius = 0.1
width = 4
height = 4
legendre_degree = 2
train_total_steps = int(1e5) 

#check_model()
#train(squirmer_radius, width, height, legendre_degree, train_total_steps)
show_result(squirmer_radius, width, height, legendre_degree, render_mode="human")
