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

# Load fluid functions
sys.path.append('./Fluid')
import fluid_module as fm


# Environment
class PredatorPreyEnv(gym.Env):
    """Gym environment for a predator-prey system in a fluid. The frame of reference is """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}


    def __init__(self, squirmer_radius=0.1, width=1, height=1, n_legendre=2, m_legendre=2, render_mode=None):
        #super().__init__() - ingen anelse hvorfor jeg havde skrevet det?
        # -- Variables --
        # Model 
        self.width = width  # Width and height of learning playground
        self.height = height
        self.squirmer_radius = squirmer_radius  # The catch radius is based on this
        self.n_legendre = n_legendre  # Highest mode the squirmer can access
        self.m_legendre = m_legendre
        self.dimless_factor = 4 / (3 * self.squirmer_radius ** 3)  # Divide velocities by this to remove dimensions

        # Rendering
        self.window_size = 512  # PyGame window size
        assert render_mode is None or render_mode in self.metadata["render_modes"]  # Check if given render_mode matches available
        self.render_mode = render_mode
        self.window = None
        self.clock = None
        
        # -- Define action and observation space -- 
        # Actions: Strength of Legendre Modes
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)  

        # Observations: target position and velocity field 
        # The reference frame is at the Squirmer, such that it never moves. Instead, it moves the velocity field which in turn moves the target.
        self.observation_space = spaces.Dict({
            "target": spaces.Box(low=np.array([0, 0], dtype=np.float32), high=np.array([self.width, self.height], dtype=np.float32), shape=(2,)),
            "velocity": spaces.Box(low=-np.inf, high=np.inf, shape=(2,))  # Target's velocity
        })
    

    def _get_dist(self):
        """Helper function that calculates the distance between the agent and the target"""
        return np.linalg.norm(self._agent_position - self._target_position, ord=1)


    def _cartesian_to_polar(self):
        r = self._get_dist()
        agent_target_vec = self._agent_position - self._target_position  # Vector pointing from target to agent
        theta = np.arctan(agent_target_vec[0] / agent_target_vec[1])
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
    

    def _reward_time_optimized(self):
        r = self._get_dist()
        d0 = np.min([self.height, self.width]) / 2
        far_away = r > d0
        captured = r < self.catch_radius
        done = False

        if far_away:  # Penalizes further if too large a distance
            gamma = -1000
        elif captured:
            d0 = 0.9  # Artificial, should be multiplied by dimensionless factor
            gamma = 200 / (self.time - d0)  # beta_T approx equal d0, where beta_T approximates the time needed to capture the target, which is the time it takes to move in a straight line
            done = True
        else:
            gamma = 0
        return float(gamma - r), done


    def reset(self, seed=None):
        # Fix seed and reset values 
        super().reset(seed=seed)

        # Time
        self.time = 0

        # Initial positions
        self._agent_position = np.array([self.width / 2, self.height / 2], dtype=np.float32)  # Agent in center
        self._agent_position = self._agent_position.reshape(2,).astype(np.float32)
        
        # Target starts random location not within 4 times catch radius.
        self.catch_radius = 2. * self.squirmer_radius  # Velocities blow up near the squirmer
        self._target_position = self._agent_position
        dist = self._get_dist()
        while dist <= 2 * self.catch_radius:  
            self._target_position = self.np_random.uniform(low=np.array([0, 0], dtype=np.float32), high=np.array([self.width, self.height], dtype=np.float32), size=2)
            dist = self._get_dist()  # Update distance
        self._target_position = self._target_position.reshape(2,).astype(np.float32)
        
        # Initial velocity set to 0
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
        
        # -- Action setup --
        B01, B_tilde11 = action
        B = np.array([[0, B01], 
                      [0, 0],
                      [0, 0]])
        B_tilde = np.array([[0, 0], 
                            [0, B_tilde11],
                            [0, 0]])

        # -- Movement --
        # Convert to polar coordinates and get the cartesian velocity of the flow. 
        # Remove the velocity's dimensions, and add it to the target's position (implicit dt=0)
        # Apply periodic boundaries to the position.
        r, theta = self._cartesian_to_polar()
        velocity_x, velocity_y = fm.field_cartesian(r, theta,
                                                    n=self.n_legendre, m=self.m_legendre,
                                                    a=self.squirmer_radius, B=B, 
                                                    B_tilde=B_tilde)
        velocity = np.array([velocity_x, velocity_y], dtype=np.float32) / self.dimless_factor
        move_target = self._target_position + velocity
        self._target_position = self._periodic_boundary(move_target).reshape(2,).astype(np.float32)

        # -- Reward --
        reward, done = self._reward_time_optimized()

        # -- Update values --
        self.time += 1
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
        
        # Draw target as rectangle - NOTE THIS SHOULD PROBABLY BE CHANGED TO CIRCLE
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
    #env = FlattenObservation(env)  - I though this would be needed, but gives error.
    print("-- SB3 CHECK ENV: --")
    if check_env(env) == None:
        print("   The Environment is compatible with SB3")
    else:
        print(check_env(env))


def train(squirmer_radius, width, height, n_legendre, m_legendre, train_total_steps):
    env = PredatorPreyEnv(squirmer_radius, width, height, n_legendre, m_legendre)    

    # Train with SB3
    log_path = os.path.join("Reinforcement Learning", "Training", "Logs")
    model = PPO("MultiInputPolicy", env, verbose=1, tensorboard_log=log_path)
    model.learn(total_timesteps=train_total_steps)
    model.save("ppo_predator_prey")


def show_result(squirmer_radius, width, height, n_legendre, m_legendre, render_mode):
    """Arguments should match that of the loaded model for correct results"""
    # Load model and create environment
    model = PPO.load("ppo_predator_prey")
    env = PredatorPreyEnv(squirmer_radius, width, height, n_legendre, m_legendre, render_mode)
    
    # Run and render model
    obs = env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)
        env.render()
    env.close()


# -- Run the code --
# Parameters
squirmer_radius = 0.1
width = 5
height = 5
n_legendre = 2
m_legendre = 2
train_total_steps = int(1e5) 

#check_model()
#train(squirmer_radius, width, height, n_legendre, m_legendre, train_total_steps)
show_result(squirmer_radius, width, height, n_legendre, m_legendre, render_mode="human")

# tensorboard --logdir=.
