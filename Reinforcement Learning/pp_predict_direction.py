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


# Environment
class PredatorPreyEnv(gym.Env):
    """Gym environment for a predator-prey system in a fluid."""


    def __init__(self, N_surface_points, squirmer_radius, target_radius, max_mode, sensor_noise, target_initial_position):
        #super().__init__() - ingen anelse om hvorfor jeg havde skrevet det eller hvor det kommer fra?
        # -- Variables --
        # Model
        self.N_surface_points = N_surface_points
        self.squirmer_radius = squirmer_radius
        self.target_radius = target_radius
        self.max_mode = max_mode # Max available legendre modes. 
        self.sensor_noise = sensor_noise
        self.target_initial_position = target_initial_position
        
        # Parameters
        self.regularization_offset = 0.1  # "Width" of blob delta functions.
        self.B_max = 1
        self.viscosity = 1
        self.lab_frame = True
        self.epsilon = 0.05  # Extra catch distance, because velocities blow up near squirmer
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


    def _array_float(self, x, shape):
        """Helper function to input x into a shape sized array with dtype np.float32"""
        return np.array([x], dtype=np.float32).reshape(shape)

    
    def _minimal_angle_difference(self, x, y):
        diff1 = x - y
        diff2 = diff1 + 2 * np.pi
        diff3 = diff1 - 2 * np.pi
        return np.min(np.abs([diff1, diff2, diff3]))
    
    
    def _average_direction_change(self, mode_array):
        agent_center = np.concatenate((np.array([0]), self._agent_position))  # Expects 3d vector, but RL is 2d
        target_center = np.concatenate((np.array([0]), self._target_position))
        return bem_two_objects.average_change_direction(self.N_surface_points, self.max_mode, self.squirmer_radius, self.target_radius, agent_center, 
                                                        target_center, mode_array, self.regularization_offset, self.viscosity, noise=self.sensor_noise)
    
    def _reward(self, mode_array):
        # Calculate angle and average direction of change
        agent_target_vec = self._target_position - self._agent_position  # Vector pointing from target to agent
        angle = np.arctan2(agent_target_vec[0], agent_target_vec[1])
        
        change_direction = self._average_direction_change(mode_array)
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
        # Modes are equal to n-sphere coordinates divided by the square root of the mode factors
        # max_power = 1
        # phi_n_sphere = action * np.pi / 2
        # phi_n_sphere[-1] = action[-1] * np.pi
        # x_n_sphere = power_consumption.n_sphere_angular_to_cartesian(phi_n_sphere, max_power)  # Makes sure sums to 1
        # power_factors = power_consumption.constant_power_factor(squirmer_radius, self.viscosity, self.max_mode)  # Power factors in front of modes
        # power_non_zero = power_factors.nonzero()
        # mode_array = np.zeros_like(power_factors)
        # mode_array[power_non_zero] = x_n_sphere / np.sqrt(power_factors[power_non_zero].ravel())
        
        mode_array = power_consumption.normalized_modes(action, self.max_mode, self.squirmer_radius, self.viscosity)
                        
        # -- Reward --
        angle, guessed_angle, reward = self._reward(mode_array)

        # -- Update values --
        observation = self._array_float(self._average_direction_change(mode_array), shape=(3,))
        info = {"angle": angle, "guessed angle": guessed_angle}
        done = True  # Only one time step as the system does not evolve over time
        
        return observation, reward, done, info


def check_model(N_surface_points, squirmer_radius, target_radius, max_mode, sensor_noise, target_initial_position):
    env = PredatorPreyEnv(N_surface_points, squirmer_radius, target_radius, max_mode, sensor_noise, target_initial_position)
    print("-- SB3 CHECK ENV: --")
    if check_env(env) == None:
        print("   The Environment is compatible with SB3")
    else:
        print(check_env(env))


def train(N_surface_points, squirmer_radius, target_radius, max_mode, sensor_noise, target_initial_position, viscosity, train_total_steps):
    env = PredatorPreyEnv(N_surface_points, squirmer_radius, target_radius, max_mode, sensor_noise, target_initial_position)

    # Train with SB3
    log_path = os.path.join("Reinforcement Learning", "Training", "Logs_direction")
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_path)
    model.learn(total_timesteps=train_total_steps)
    model_path = os.path.join(log_path, "predict_direction")
    model.save(model_path)
    
    # Save parameters in csv file
    file_path = os.path.join(log_path, "system_parameters.csv")
    with open(file_path, mode="w") as file:
        writer = csv.writer(file, delimiter=",")
        writer.writerow(["Surface Points", "Squirmer Radius", "Target Radius", "Max Mode", "Sensor Noise", 
                         "Target y", "Target z", "Centers Distance", "viscosity", "Train Steps"])
        writer.writerow([N_surface_points, squirmer_radius, target_radius, max_mode, sensor_noise, target_initial_position[0], target_initial_position[1],
                         np.linalg.norm(target_initial_position, ord=2), viscosity, train_total_steps])
    
    
def mode_names(max_mode):
    B_names = []
    B_tilde_names = []
    C_names = []
    C_tilde_names = []
    for i in range(max_mode+1):
        for j in range(i, max_mode+1):
            if j > 0:
                B_str = r"$B_{" + str(i) + str(j) + r"}$"
                B_names.append(B_str)
            if j > 1:
                C_str = r"$C_{" + str(i) + str(j) + r"}$"
                C_names.append(C_str)            
            if i > 0:
                B_tilde_str = r"$\tilde{B}_{" + str(i) + str(j) + r"}$"
                B_tilde_names.append(B_tilde_str)
                if j > 1:
                    C_tilde_str = r"$\tilde{C}_{" + str(i) + str(j) + r"}$"
                    C_tilde_names.append(C_tilde_str)
                    
    return B_names, B_tilde_names, C_names, C_tilde_names


def mode_iteration(N_iter, PPO_number, mode_lengths):
    # Load parameters and model, create environment
    parameters_path = f"Reinforcement Learning/Training/Logs_direction/PPO_{PPO_number}/system_parameters.csv"
    parameters = np.genfromtxt(parameters_path, delimiter=",", skip_header=1)
    N_surface_points = int(parameters[0])
    squirmer_radius = parameters[1]
    target_radius = parameters[2]
    max_mode = int(parameters[3])
    sensor_noise = parameters[4]
    target_y = parameters[5]
    target_z = parameters[6]
    viscosity = parameters[8]
    model_path = f"Reinforcement Learning/Training/Logs_direction/PPO_{PPO_number}/predict_direction"
    
    model = PPO.load(model_path)
    env = PredatorPreyEnv(N_surface_points, squirmer_radius, target_radius, max_mode, sensor_noise, np.array([target_y, target_z]))
    
    # Empty arrays for loop
    B_actions = np.empty((N_iter, mode_lengths[0]))
    B_tilde_actions = np.empty((N_iter, mode_lengths[1]))
    C_actions = np.empty((N_iter, mode_lengths[2]))
    C_tilde_actions = np.empty((N_iter, mode_lengths[3]))

    rewards = np.empty((N_iter))
    guessed_angles = np.empty((N_iter))

    # Run model N_iter times
    obs = env.reset()
    for i in range(N_iter):
        action, _ = model.predict(obs)
        obs, reward, _, info = env.step(action)
        rewards[i] = reward
        guessed_angles[i] = info["guessed angle"]
        mode_array = power_consumption.normalized_modes(action, max_mode, squirmer_radius, viscosity)
                
        B_actions[i, :] = mode_array[0][np.nonzero(mode_array[0])]
        B_tilde_actions[i, :] = mode_array[1][np.nonzero(mode_array[1])]
        C_actions[i, :] = mode_array[2][np.nonzero(mode_array[2])]
        C_tilde_actions[i, :] = mode_array[3][np.nonzero(mode_array[3])]
    
    return B_actions, B_tilde_actions, C_actions, C_tilde_actions, rewards, guessed_angles, parameters


def plot_mode_choice(N_iter, PPO_number):
    """Plot the modes taken at different iterations."""
    # Add more colors
    mpl.rcParams["axes.prop_cycle"] = mpl.cycler(color=['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 
                                                        'purple', 'pink', 'brown', 'orange', 'teal', 'coral', 'lightblue', 
                                                        'lime', 'lavender', 'turquoise', 'darkgreen', 'tan', 'salmon', 'gold'])

    # Names
    B_names, B_tilde_names, C_names, C_tilde_names = mode_names(max_mode)
            
    #B_names = [r"$B_{01}$", r"$B_{02}$", r"$B_{03}$", r"$B_{04}$", r"$B_{11}$", r"$B_{12}$", r"$B_{13}$", 
    #           r"$B_{14}$", r"$B_{22}$", r"$B_{23}$", r"$B_{24}$", r"$B_{33}$", r"$B_{34}$", r"$B_{44}$",]
    #B_tilde_names = [r"$\tilde{B}_{11}$", r"$\tilde{B}_{12}$", r"$\tilde{B}_{13}$", r"$\tilde{B}_{14}$", r"$\tilde{B}_{22}$", 
    #                 r"$\tilde{B}_{23}$", r"$\tilde{B}_{24}$", r"$\tilde{B}_{33}$", r"$\tilde{B}_{34}$", r"$\tilde{B}_{44}$",]
    #C_names = [r"$C_{02}$", r"$C_{03}$", r"$C_{04}$", r"$C_{12}$", r"$C_{13}$", r"$C_{14}$", 
    #           r"$C_{22}$", r"$C_{23}$", r"$C_{24}$", r"$C_{33}$", r"$C_{34}$", r"$C_{44}$",]
    #C_tilde_names = [r"$\tilde{C}_{12}$", r"$\tilde{C}_{13}$", r"$\tilde{C}_{14}$", r"$\tilde{C}_{22}$", 
    #                 r"$\tilde{C}_{23}$", r"$\tilde{C}_{24}$", r"$\tilde{C}_{33}$", r"$\tilde{C}_{34}$", r"$\tilde{C}_{44}$",]
    
    mode_lengths = [len(B_names), len(B_tilde_names), len(C_names), len(C_tilde_names)]
    B_actions, B_tilde_actions, C_actions, C_tilde_actions, rewards, guessed_angles, parameters = mode_iteration(N_iter, PPO_number, mode_lengths)
    
    target_y = parameters[5]
    target_z = parameters[6]
    guessed_angles = guessed_angles * 180 / np.pi
    angle = np.arctan2(target_y, target_z)
    
    # Plot
    def fill_axis(axis, y, marker, label, title):        
        axis.set(xticks=[], title=(title, 7), ylim=(-0.17, 0.17))
        axis.set_title(title, fontsize=7)
        axis.plot(y, marker=marker, ls="--", lw=0.75)
        axis.legend(label, fontsize=4, bbox_to_anchor=(1.05, 1), 
                    loc='upper left', borderaxespad=0.)
        
    # Define axis and fill them
    fig, ax = plt.subplots(nrows=2, ncols=2, dpi=200)
    ax1 = ax[0, 0]
    ax2 = ax[0, 1]
    ax3 = ax[1, 0]
    ax4 = ax[1, 1]

    fill_axis(ax1, B_actions, ".", B_names, title=r"$B$ weights")
    fill_axis(ax2, B_tilde_actions, ".", B_tilde_names, title=r"$\tilde{B}$ weights")
    fill_axis(ax3, C_actions, ".", C_names, title=r"$C$ weights")
    fill_axis(ax4, C_tilde_actions, ".", C_tilde_names, title=r"$\tilde{C}$ weights")
    
    # xticks
    xticks = []
    for reward, angle_guess in zip(rewards, guessed_angles):
        tick_str = f"R: {np.round(reward, 2)}, " + r"$\theta_g$: " + str(np.round(angle_guess, 2))
        xticks.append(tick_str)
        
    # General setup
    ax2.set(yticks=[])
    ax3.set(xlabel="Iteration", xticks=(np.arange(N_iter)))
    ax3.set_xticklabels(xticks, rotation=20, size=5)
    ax4.set(xlabel="Iteration", xticks=(np.arange(N_iter)), yticks=[])
    ax4.set_xticklabels(xticks, rotation=20, size=5)
    fig.suptitle(fr"Mode over iterations, Noise = {sensor_noise}, $\theta =${np.round(angle * 180 / np.pi, 2)}", fontsize=10)
    fig.tight_layout()
    
    # Save and show
    figname = f"noise{parameters[4]}_maxmode{parameters[3]}_targetradius{parameters[2]}_distance{parameters[6]}_trainingsteps{parameters[-1]}.png"            
    plt.savefig("Reinforcement Learning/Recordings/Images/" + figname)
    plt.show()


def plot_mode_iteration_average(N_model_runs, PPO_list, changed_parameter, plot_reward=True):
    assert changed_parameter in ["radius", "noise", "position", "angle"]
    B_names, B_tilde_names, C_names, C_tilde_names = mode_names(max_mode)
    mode_lengths = [len(B_names), len(B_tilde_names), len(C_names), len(C_tilde_names)]
    PPO_len = len(PPO_list)
        
    B_mean = np.empty((PPO_len, mode_lengths[0])) 
    B_tilde_mean = np.empty((PPO_len, mode_lengths[1])) 
    C_mean = np.empty((PPO_len, mode_lengths[2])) 
    C_tilde_mean = np.empty((PPO_len, mode_lengths[3])) 

    B_std = np.empty_like(B_mean) 
    B_tilde_std = np.empty_like(B_tilde_mean)
    C_std = np.empty_like(C_mean)
    C_tilde_std = np.empty_like(C_tilde_mean)
    
    reward_mean = np.empty(PPO_len)
    reward_std = np.empty_like(reward_mean)
    
    changed_parameter_list = np.empty(PPO_len)
    
    for i, PPO in enumerate(PPO_list):
        B_actions, B_tilde_actions, C_actions, C_tilde_actions, rewards, _, parameters = mode_iteration(N_model_runs, PPO, mode_lengths)
        # Mean and std
        B_mean[i, :] = np.mean(B_actions, axis=0)
        B_tilde_mean[i, :] = np.mean(B_tilde_actions, axis=0)
        C_mean[i, :] = np.mean(C_actions, axis=0)
        C_tilde_mean[i, :] = np.mean(C_tilde_actions, axis=0)
        
        B_std[i, :] = np.std(B_actions, axis=0)
        B_tilde_std[i, :] = np.std(B_tilde_actions, axis=0)
        C_std[i, :] = np.std(C_actions, axis=0)
        C_tilde_std[i, :] = np.std(C_tilde_actions, axis=0)
        
        reward_mean[i] = np.mean(rewards)
        reward_std[i] = np.std(rewards) / np.sqrt(len(rewards))
        
        if changed_parameter == "radius":
            changed_parameter_list[i] = parameters[2]  # Target radius
            xlabel = "Target Radius"
        elif changed_parameter == "noise":
            changed_parameter_list[i] = parameters[4]  # Sensor noise
            xlabel = "Sensor Noise"
        elif changed_parameter == "angle":
            changed_parameter_list[i] = np.arctan2(parameters[5], parameters[6])  # arcan ( target y / target z )
            xlabel = "Angel [radian]"
        else:  # Target initial position
            changed_parameter_list[i] = parameters[7]  # Distance between the two centers
            xlabel = "Center-center distance"
            

    def fill_axis(axis, y, sy, mode_name, title):        
        x_vals = changed_parameter_list
        axis.set(title=(title, 7), ylim=(-0.06, 0.06))
        axis.set_title(title, fontsize=7)
        for i in range(y.shape[1]):
            axis.errorbar(x_vals, y[:, i], yerr=sy[:, i], fmt=".--", lw=0.75)
        axis.legend(mode_name, fontsize=4, bbox_to_anchor=(1.05, 1), 
                    loc='upper left', borderaxespad=0.)
    
    
    fig, ax = plt.subplots(nrows=2, ncols=2, dpi=200)
    axB = ax[0, 0]
    axBt = ax[0, 1]
    axC = ax[1, 0]
    axCt = ax[1, 1]
    
    fill_axis(axB, B_mean, B_std, B_names, r"$B$ modes")
    fill_axis(axBt, B_tilde_mean, B_tilde_std, B_tilde_names, title=r"$\tilde{B}$ modes")
    fill_axis(axC, C_mean, C_std, C_names, title=r"$C$ modes")
    fill_axis(axCt, C_tilde_mean, C_tilde_std, C_tilde_names, title=r"$\tilde{C}$ modes")
            
    # General setup
    axBt.set(yticks=[])
    axC.set(xlabel=xlabel)
    axCt.set(xlabel=xlabel, yticks=[])
    fig.suptitle(fr"Average mode values over {xlabel}", fontsize=10)
    fig.tight_layout()
    
    # Save and show
    figname = f"average_modes_maxmode{max_mode}_{xlabel}{changed_parameter_list}.png"
    plt.savefig("Reinforcement Learning/Recordings/Images/" + figname)
    plt.show()
    
    if plot_reward:
        figr, axr = plt.subplots(dpi=200)
        axr.errorbar(changed_parameter_list, reward_mean, yerr=reward_std, fmt=".")
        axr.set(xlabel=xlabel, ylabel="Reward", title="Mean reward")
        figr.tight_layout()
        plt.show()
    
    
# -- Run the code --
# Model Parameters
N_surface_points = 80
squirmer_radius = 1
target_radius = 0.8
tot_radius = squirmer_radius + target_radius
target_initial_position = [0, 1.3*tot_radius]
max_mode = 2
viscosity = 1
sensor_noise = 0.02
train_total_steps = int(3e5)

# Plotting parameters
N_iter = 10
PPO_number = 28  # For which model to load when plotting, after training
PPO_list = [26, 27]

#check_model(N_surface_points, squirmer_radius, target_radius, max_mode, sensor_noise, target_initial_position)
#train(N_surface_points, squirmer_radius, target_radius, max_mode, sensor_noise, target_initial_position, viscosity, train_total_steps)
plot_mode_choice(N_iter, PPO_number)
#plot_mode_iteration_average(N_model_runs=N_iter, PPO_list=PPO_list, changed_parameter="angle")

# If wants to see reward over time, write the following in cmd in the log directory
# tensorboard --logdir=.