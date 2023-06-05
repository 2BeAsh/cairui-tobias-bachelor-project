# Imports
import numpy as np
import os
import sys
import csv

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.lines as mlines
import matplotlib.colors as colors
from matplotlib.lines import Line2D

import gym
from gym import spaces
from gym.wrappers import FlattenObservation

from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

# Load custom functions
sys.path.append('./Fluid')
import field_velocity
import power_consumption
import bem_two_objects
import bem


def oseen_inverses(N1, squirmer_radius, target_radius, squirmer_position, target_position, epsilon, viscosity):
    """Calculate Oseen tensor inverses"""
    # Without target
    x1, y1, z1, dA = bem.canonical_fibonacci_lattice(N1, squirmer_radius)
    theta = np.arccos(z1 / squirmer_radius)
    phi = np.arctan2(y1, x1)
    x1_stack = np.stack((x1, y1, z1)).T
    A_oseen = bem.oseen_tensor_surface(x1_stack, dA, epsilon, viscosity)
    A_oseen_inv = np.linalg.inv(A_oseen)
    
    # With target
    N2 = int(4 * np.pi * target_radius ** 2 / dA)
    x2, y2, z2, _ = bem.canonical_fibonacci_lattice(N2, target_radius)
    x2_stack = np.stack((x2, y2, z2)).T
    squirmer_position = np.append([0], squirmer_position)  # RL is 2d, Oseen i 3d with x plane zero
    target_position = np.append([0], target_position)
    A_oseen_with = bem_two_objects.oseen_tensor_surface_two_objects(x1_stack, x2_stack, squirmer_position, target_position, dA, epsilon, viscosity)
    A_oseen_with_inv = np.linalg.inv(A_oseen_with)
    
    return x1_stack, N2, theta, phi, A_oseen_inv, A_oseen_with_inv


# Environment
class PredictDirectionEnv(gym.Env):
    """Gym environment for a predator-prey system in a fluid."""


    def __init__(self, N_surface_points, squirmer_radius, target_radius, max_mode, sensor_noise, viscosity, target_initial_position, reg_offset, coord_plane="yz"):
        #super().__init__() - ingen anelse om hvorfor jeg havde skrevet det eller hvor det kommer fra?
        # -- Variables --
        # Model
        self.N_surface_points = N_surface_points
        self.squirmer_radius = squirmer_radius
        self.target_radius = target_radius
        self.max_mode = max_mode # Max available legendre modes. 
        self.sensor_noise = sensor_noise
        self.target_initial_position = target_initial_position
        self.coord_plane = coord_plane
        assert coord_plane in ["xy", "xz", "yz", None]
        self.epsilon = reg_offset  # Width of delta function blobs.
        self.viscosity = viscosity
        
        # Parameters
        self.catch_radius = self.squirmer_radius + self.epsilon
        
        # -- Define action and observation space --
        # Actions: Strength of Legendre Modes
        if max_mode == 4:
            number_of_modes = 45  # Counted from power factors.
        elif max_mode == 3:
            number_of_modes = 27
        elif max_mode == 2: 
            number_of_modes = 13
        action_shape = (number_of_modes,)  # Weight of each mode.
        self.action_space = spaces.Box(low=-1, high=1, shape=action_shape, dtype=np.float32)

        # Observation is vector pointing in direction of average force
        self.observation_space = spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32) 

        # -- Calculate Oseen inverses --
        # Initial Positions
        self._agent_position = self._array_float([0, 0], shape=(2,))  # Agent in center
        self._target_position = self._array_float(self.target_initial_position, shape=(2,))

        self.x1_stack, self.N2, self.theta, self.phi, self.A_oseen_inv, self.A_oseen_with_inv = oseen_inverses(self.N_surface_points, self.squirmer_radius, self.target_radius, 
                                                                                                               self._agent_position, self._target_position, self.epsilon, self.viscosity)
        

    def _array_float(self, x, shape):  # Kan ændres til at shap bare tager x.shape
        """Helper function to input x into a shape sized array with dtype np.float32"""
        return np.array([x], dtype=np.float32).reshape(shape)


    def _average_force_difference(self, mode_array):
        # Find forces with and without target
        ux1, uy1, uz1 = field_velocity.field_cartesian_squirmer(self.max_mode, r=self.squirmer_radius, theta=self.theta, phi=self.phi, 
                                                                squirmer_radius=self.squirmer_radius, mode_array=mode_array,)
        u_comb = np.array([ux1, uy1, uz1]).ravel()
        u_comb_without = np.append(u_comb, np.zeros(6))  # No target
        u_comb_with = np.append(u_comb, np.zeros(12+3*self.N2))
        
        force_without = self.A_oseen_inv @ u_comb_without
        force_with = self.A_oseen_with_inv @ u_comb_with 
        
        # Differences
        N1 = self.N_surface_points
        dfx = force_with[:N1].T - force_without[:N1].T  # NOTE burde man allerede her tage abs()? Relatvant ift støj!?
        dfy = force_with[N1: 2*N1].T - force_without[N1: 2*N1].T
        dfz = force_with[2*N1: 3*N1].T - force_without[2*N1: 3*N1].T

        # Noise
        dfx += np.random.normal(loc=0, scale=self.sensor_noise, size=dfx.size)
        dfy += np.random.normal(loc=0, scale=self.sensor_noise, size=dfy.size)
        dfz += np.random.normal(loc=0, scale=self.sensor_noise, size=dfz.size)

        # Weight and force
        weight = np.sqrt(dfx ** 2 + dfy ** 2 + dfz ** 2)
        f_average = np.sum(weight[:, None] * self.x1_stack, axis=0)
        f_average_norm = f_average / np.linalg.norm(f_average, ord=2)
        return dfx, dfy, dfz, f_average_norm

        
    def _minimal_angle_difference(self, x, y):
        diff1 = x - y
        diff2 = diff1 + 2 * np.pi
        diff3 = diff1 - 2 * np.pi
        return np.min(np.abs([diff1, diff2, diff3]))
    
    
    def _reward(self, mode_array):
        # Calculate angle and average direction of change
        _, _, _, change_direction = self._average_force_difference(mode_array)

        agent_target_vec = self._target_position - self._agent_position  # Vector pointing from target to agent
        if self.coord_plane == "xz":
            angle = np.arctan2(agent_target_vec[0], agent_target_vec[1])
            angle_largest_change = np.arctan2(change_direction[0], change_direction[2])
        elif self.coord_plane == "xy":
            angle = np.arctan2(agent_target_vec[1], agent_target_vec[0])
            angle_largest_change = np.arctan2(change_direction[1], change_direction[0])
        elif self.coord_plane == "yz" or self.coord_plane == None:
            angle = np.arctan2(agent_target_vec[0], agent_target_vec[1])
            angle_largest_change = np.arctan2(change_direction[1], change_direction[2])

        # Reward is based on how close the angles are, closer is better
        angle_difference_norm = self._minimal_angle_difference(angle, angle_largest_change) / np.pi
        reward = 1 - angle_difference_norm
        return angle, angle_largest_change, reward
        

    def reset(self, seed=None):
        # Fix seed and reset values
        super().reset(seed=seed)
                
        # Initial positions
        self._agent_position = self._array_float([0, 0], shape=(2,))  # Agent in center
        self._target_position = self._array_float(self.target_initial_position, shape=(2,))
                
        # Initial observation is no field
        observation = self._array_float([0, 0, 0], shape=(3,))

        return observation


    def step(self, action):
        # -- Action setup --
        # Actions are the available modes.
        mode_array = power_consumption.normalized_modes(action, self.max_mode, self.squirmer_radius, self.viscosity)
                        
        # -- Reward --
        angle, guessed_angle, reward = self._reward(mode_array)

        # -- Update values --
        _, _, _, x_change = self._average_force_difference(mode_array)
        observation = self._array_float(x_change, shape=(3,))
        info = {"angle": angle, "guessed angle": guessed_angle}
        done = True  # Only one time step as the system does not evolve over time
        
        return observation, reward, done, info


def check_model(N_surface_points, squirmer_radius, target_radius, max_mode, sensor_noise, viscosity, target_initial_position, reg_offset, coord_plane):
    env = PredictDirectionEnv(N_surface_points, squirmer_radius, target_radius, max_mode, sensor_noise, viscosity, target_initial_position, reg_offset, coord_plane)
    print("-- SB3 CHECK ENV: --")
    if check_env(env) == None:
        print("   The Environment is compatible with SB3")
    else:
        print(check_env(env))

    
    



# -- Run the code --
if __name__ == "__main__":
    # Model Parameters
    N_surface_points = 1300
    squirmer_radius = 1
    target_radius = 0.25
    tot_radius = squirmer_radius + target_radius
    target_initial_position = [2, 2] / np.sqrt(2)
    max_mode = 2
    viscosity = 1
    sensor_noise = 0.05
    reg_offset = 0.05
    coord_plane = "yz"

    check_model(N_surface_points, squirmer_radius, target_radius, max_mode, sensor_noise, viscosity, target_initial_position, reg_offset, coord_plane)

#"target_radius", "noise", "position", "angle", "else"
# If wants to see reward over time, write the following in cmd in the log directory
# tensorboard --logdir=.