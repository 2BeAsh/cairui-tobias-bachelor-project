# Imports
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.lines as mlines

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


# Environment
class PredatorPreyEnv(gym.Env):
    """Gym environment for a predator-prey system in a fluid."""


    def __init__(self, N_surface_points, squirmer_radius, target_radius, max_mode):
        #super().__init__() - ingen anelse om hvorfor jeg havde skrevet det eller hvor det kommer fra?
        # -- Variables --
        # Model
        self.N_surface_points = N_surface_points
        self.squirmer_radius = squirmer_radius
        self.target_radius = target_radius
        self.max_mode = max_mode # Max available legendre modes. The minus 1 comes from different definitons. 

        # Parameters
        self.regularization_offset = 0.1  # "Width" of blob delta functions.
        self.B_max = 1
        self.viscosity = 1
        self.lab_frame = True
        self.charac_time = 3 * self.squirmer_radius ** 4 / (4 * self.B_max)
        self.epsilon = 0.05  # Extra catch distance, because velocities blow up near squirmer
        self.catch_radius = self.squirmer_radius + self.epsilon
        
        # -- Define action and observation space --
        # Actions: Strength of Legendre Modes
        number_of_modes = 45  # Counted from power factors.
        action_shape = (number_of_modes-1,)  # Weight of each mode. -1 because radius fixed. Skal måske reduceres til færre actions, dvs. færre modes.
        self.action_space = spaces.Box(low=-1, high=1, shape=action_shape, dtype=np.float32)

        # Observation is vector pointing in direction of average force
        self.observation_space = spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32) 


    def _array_float(self, x, shape):
        """Helper function to input x into a shape sized array with dtype np.float32"""
        return np.array([x], dtype=np.float32).reshape(shape)

    
    def _average_direction_change(self, mode_array):
        agent_center = np.concatenate((np.array([0]), self._agent_position))  # Expects 3d vector, but RL is 2d
        target_center = np.concatenate((np.array([0]), self._target_position))
        return bem_two_objects.average_change_direction(self.N_surface_points, self.max_mode, self.squirmer_radius, self.target_radius,
                                                                            agent_center, target_center, mode_array, self.regularization_offset, self.viscosity)
    
    def _reward(self, mode_array):
        # Calculate angle and average direction of change
        agent_target_vec = self._target_position - self._agent_position  # Vector pointing from target to agent
        angle = np.arctan2(agent_target_vec[0], agent_target_vec[1])
        
        change_direction = self._average_direction_change(mode_array)
        angle_largest_change = np.arctan2(change_direction[1], change_direction[2])
        
        # Reward is based on how close the angles are, closer is better
        angle_difference_norm = np.abs((angle - angle_largest_change) / np.pi)
        reward = 1 - angle_difference_norm
        
        return reward
        

    def reset(self, seed=None):
        # Fix seed and reset values
        super().reset(seed=seed)

        # Initial positions
        self._agent_position = self._array_float([0, 0], shape=(2,))  # Agent in center
        tot_radius = self.squirmer_radius + self.target_radius
        self._target_position = self._array_float([0.9*tot_radius, 0.6*tot_radius], shape=(2,))  # Location arbitraty but fixed
        
        # Initial observation is no field
        observation = self._array_float([0, 0, 0], shape=(3,))

        return observation


    def step(self, action):
        # -- Action setup --
        # Actions are the available modes.
        # Modes are equal to n-sphere coordinates divided by the square root of the mode factors
        max_power = 1
        x_n_sphere = power_consumption.n_sphere_angular_to_cartesian(action, max_power)
        mode_factors = power_consumption.constant_power_factor(squirmer_radius, self.viscosity, self.max_mode)
        mode_non_zero = mode_factors.nonzero()
        mode_array = np.zeros_like(mode_factors)
        mode_array[mode_non_zero] = x_n_sphere / np.sqrt(mode_factors[mode_non_zero].ravel())
                        
        # -- Reward --
        reward = self._reward(mode_array)

        # -- Update values --
        observation = self._array_float(self._average_direction_change(mode_array), shape=(3,))
        info = {"action": action}
        done = True  # Only one time step as the system does not evolve over time
        
        return observation, reward, done, info


def check_model(N_surface_points, squirmer_radius, target_radius, max_mode):
    env = PredatorPreyEnv(N_surface_points, squirmer_radius, target_radius, max_mode)
    print("-- SB3 CHECK ENV: --")
    if check_env(env) == None:
        print("   The Environment is compatible with SB3")
    else:
        print(check_env(env))


def train(N_surface_points, squirmer_radius, target_radius, max_mode, train_total_steps):
    env = PredatorPreyEnv(N_surface_points, squirmer_radius, target_radius, max_mode)

    # Train with SB3
    log_path = os.path.join("Reinforcement Learning", "Training", "Logs_direction")
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_path)
    model.learn(total_timesteps=train_total_steps)
    model.save("ppo_predator_prey_direction")


# Skal rettes til denne opsætning
# Til visualisering kunne man plotte de top tre modes (dvs de tre modes som igennem actions blev vægtet højest) over iterationer. 
def plot_mode_vs_time(N_surface_points, squirmer_radius, target_radius, max_mode):
    """Arguments should match that of the loaded model for correct results"""
    # Load model and create environment
    model = PPO.load("ppo_predator_prey_direction")
    env = PredatorPreyEnv(N_surface_points, squirmer_radius, target_radius, max_mode)

    # Run and render model
    obs = env.reset()

    action_list = []
        
    action, _states = model.predict(obs)
    obs, reward, done, info = env.step(action)        
    action_list.append(action)
    
    actions = np.array(action_list)
    
    # Plot mode values over time
    fig_mode, ax_mode = plt.subplots(dpi=200, figsize=(8, 12))
    ax_mode.plot(actions, "--.")
    ax_mode.set(xlabel="Time", ylabel="Mode weight", title="Mode weights against time")
    legend = [r"$B_{01}$", r"$B_{02}$", r"$B_{03}$", r"$B_{04}$", r"$B_{11}$", r"$B_{12}$", r"$B_{13}$", 
              r"$B_{14}$", r"$B_{22}$", r"$B_{23}$", r"$B_{24}$", r"$B_{33}$", r"$B_{34}$", r"$B_{44}$",
                
              r"$B_{tilde,11}$", r"$B_{tilde,12}$", r"$B_{tilde,13}$", r"$B_{tilde,14}$", r"$B_{tilde,22}$", 
              r"$B_{tilde,23}$", r"$B_{tilde,24}$", r"$B_{tilde,33}$", r"$B_{tilde,34}$", r"$B_{tilde,44}$",
              
              r"$C_{02}$", r"$C_{03}$", r"$C_{04}$", r"$C_{12}$", r"$C_{13}$", r"$C_{14}$", 
              r"$C_{22}$", r"$C_{23}$", r"$C_{24}$", r"$C_{33}$", r"$C_{34}$", r"$C_{44}$",
              
              r"$C_{tilde,12}$", r"$C_{tilde,13}$", r"$C_{tilde,14}$", r"$C_{tilde,22}$", 
              r"$C_{tilde,23}$", r"$C_{tilde,24}$", r"$C_{tilde,33}$", r"$C_{tilde,34}$", r"$C_{tilde,44}$",
              ]
    ax_mode.legend(legend, fontsize=5)
    fig_mode.tight_layout()
    plt.show()
    plt.close()
    

# -- Run the code --
# Parameters
N_surface_points = 80
squirmer_radius = 1
target_radius = 1.1
max_mode = 4

train_total_steps = int(2e5)

#check_model(N_surface_points, squirmer_radius, target_radius, max_mode)
train(N_surface_points, squirmer_radius, target_radius, max_mode, train_total_steps)
#plot_mode_vs_time(N_surface_points, squirmer_radius, target_radius, max_mode)


# If wants to see reward over time, write the following in cmd in the log directory
# tensorboard --logdir=.
