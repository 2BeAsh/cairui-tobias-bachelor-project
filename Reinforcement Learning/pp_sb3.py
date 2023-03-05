""" Notes
Make predator prey custom environment from bottom.
"""
# Imports
import numpy as np
import os
import sys
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "1"
import pygame
import matplotlib.pyplot as plt

import gym
from gym import spaces
from gym.wrappers import FlattenObservation

from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

# Load custom fluid functions
sys.path.append('./Fluid')
import fluid_module as fm


# Environment
class PredatorPreyEnv(gym.Env):
    """Gym environment for a predator-prey system in a fluid. The frame of reference is """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}


    def __init__(self, squirmer_radius, width, height, n_legendre=2, m_legendre=2, render_mode=None):
        #super().__init__() - ingen anelse om hvorfor jeg havde skrevet det eller hvor det kommer fra?
        # -- Variables --
        # Model
        self.width = width  # Width and height of learning playground
        self.height = height
        self.squirmer_radius = squirmer_radius  # The catch radius is based on this
        self.n_legendre = n_legendre  # Highest mode the squirmer can access
        self.m_legendre = m_legendre
        self.B_max = 1
        self.dimless_factor = 4 * self.B_max / (3 * self.squirmer_radius ** 3)  # Characteristic velocity: Divide velocities by this to remove dimensions
        self.charac_time = self.squirmer_radius ** 4 / (4 * self.B_max) # characteristic time
        self.dt = 1 / self.charac_time
        # Rendering
        self.window_size = 512  # PyGame window size
        assert render_mode is None or render_mode in self.metadata["render_modes"]  # Check if given render_mode matches available
        self.render_mode = render_mode
        self.window = None
        self.clock = None

        # -- Define action and observation space --
        # Actions: Strength of Legendre Modes
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)

        # Observations: Angle and distance between target and agent
        max_distance = np.sqrt(self.width ** 2 + self.height ** 2) / 2  # Agent is located in center of playground, so max distance is the distance from the center to one corner, which is sqrt((width/2)^2 + (height/2)^2)
        self.observation_space = spaces.Dict({
            "distance": spaces.Box(low=0., high=max_distance, shape=(1,), dtype=np.float32),  # Even though is catched at self.catch_radius, the move could put it closer.
            "angle": spaces.Box(low=-np.pi, high=np.pi, shape=(1,), dtype=np.float32)
        })


    def _array_float(self, x, shape):
        """Helper function to input x into a shape sized array with dtype np.float32"""
        return np.array([x], dtype=np.float32).reshape(shape)


    def _get_dist(self):
        """Helper function that calculates the distance between the agent and the target"""
        return np.linalg.norm(self._agent_position - self._target_position, ord=1)


    def _cartesian_to_polar(self):
        # Not sure if correctly calculated!!
        r = self._get_dist()
        agent_target_vec = self._target_position - self._agent_position  # Vector pointing from target to agent
        theta = np.arctan(agent_target_vec[0] / agent_target_vec[1])
        return r, theta


    def _get_obs(self):
        """Helper function with observation."""
        r, theta = self._cartesian_to_polar()
        r_arr = self._array_float(r, shape=(1,))
        theta_arr = self._array_float(theta, shape=(1,))
        return {"distance": r_arr, "angle": theta_arr}


    def _periodic_boundary(self, movement):
        """Helper function that (primitively) updates movement according to periodic boundaries. Assumes the movement does not exceed the boundary by more than the width or height. Can probably be optimized!"""
        # x-direction
        if movement[0] > self.width / 2:
            movement[0] -= self.width / 2
        elif movement[0] < -self.width / 2:
            movement[0] += self.width / 2
        # y-direction
        if movement[1] > self.height / 2:
            movement[1] -= self.height / 2
        elif movement[1] < -self.height / 2:
            movement[1] += self.height / 2

        return self._array_float(movement, shape=(2,))


    def _reward_time_optimized(self):
        r = self._get_dist()
        d0 = self.initial_target_position  # np.min([self.height, self.width]) / 2  # Should be multiplied by dimensionless factor
        far_away = r > d0
        captured = r < self.catch_radius
        done = False

        if far_away:  # Penalizes further if too large a distance
            gamma = -1000
        elif captured:
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
        self._agent_position = self._array_float([0, 0], shape=(2,))  # Agent in center

        # Target starts random location not within 4 times catch radius.
        self.catch_radius = 2. * self.squirmer_radius  # Velocities blow up near the squirmer
        self.initial_target_position = self._agent_position
        dist = self._get_dist()
        while dist <= 2 * self.catch_radius:
            self.initial_target_position = self.np_random.uniform(low=np.array([-self.width/2, -self.height/2], dtype=np.float32), high=np.array([self.width/2, self.height/2], dtype=np.float32), size=2)
            dist = self._get_dist()  # Update distance
        self.initial_target_position = self._array_float(self._target_position, shape=(2,))
        self._target_position = self.initial_target_position

        # Initial velocity set to 0
        self._target_velocity = self._array_float([0, 0], shape=(2,))

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
        B01, B_tilde11 = action * self.B_max
        B = np.array([[0, B01],
                      [0, 0],
                      [0, 0]], dtype=np.float32)
        B_tilde = np.array([[0, 0],
                            [0, B_tilde11],
                            [0, 0]], dtype=np.float32)

        # -- Movement --
        # Convert to polar coordinates and get the cartesian velocity of the flow.
        # Remove the velocity's dimensions, and add it to the target's position (implicit dt=0)
        # Apply periodic boundaries to the position.
        r, theta = self._cartesian_to_polar()  # dimensioner!!! husk
        velocity_x, velocity_y = fm.field_cartesian(r, theta,
                                                    n=self.n_legendre, m=self.m_legendre,
                                                    a=self.squirmer_radius, B=B,
                                                    B_tilde=B_tilde)
        velocity = np.array([velocity_x, velocity_y], dtype=np.float32) / self.dimless_factor
        move_target = self._target_position + velocity * self.dt  # måske problem dimensions
        self._target_position = self._periodic_boundary(move_target)

        # -- Reward --
        reward, done = self._reward_time_optimized()

        # -- Update values --
        self.time += self.dt  # Dimensions
        observation = self._get_obs()
        info = {}

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, done, info


    def _coord_to_pixel(self, position):
        """PyGame window is fixed at (0, 0), but we want (0, 0) to be in the center of the screen. The area is width x height, so all points must be shifted by (width/2, -height/2)"""
        return position + self._array_float([self.width/2, self.height/2], shape=(2,))


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

        # Shift positions and scale their size to the canvas
        pix_size = self.window_size / self.width  # Ideally, dots would be such that when they overlap, the prey is catched - NOTE that this would imply catch_radius=squirmer_radius, or use catch_radius instead of squirmer_radius when drawing
        target_position_draw = self._coord_to_pixel(self._target_position) * pix_size
        agent_position_draw = self._coord_to_pixel(self._agent_position) * pix_size

        # Draw target
        pygame.draw.circle(
            canvas,  # What surface to draw on
            (255, 0, 0),  # Color
            (float(target_position_draw[0]), float(target_position_draw[1])),  # x, y
            float(pix_size * self.squirmer_radius / 2)  # Radius
        )

        # Draw agent
        pygame.draw.circle(
            canvas,
            (0, 0, 255),  # Color
            (float(agent_position_draw[0]), float(agent_position_draw[1])),  # x, y  - Maybe needs to add +0.5 to positions?
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


def check_model(squirmer_radius, width, height, n_legendre, m_legendre):
    env = PredatorPreyEnv(squirmer_radius, width, height, n_legendre, m_legendre)
    #env = FlattenObservation(env)  - I thought this would be needed, but gives an error.
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
width = 2
height = 2
n_legendre = 2
m_legendre = 2
train_total_steps = int(1e5)

check_model(squirmer_radius, width, height, n_legendre, m_legendre)
#train(squirmer_radius, width, height, n_legendre, m_legendre, train_total_steps)
#show_result(squirmer_radius, width, height, n_legendre, m_legendre, render_mode="human")

# tensorboard --logdir=.
