# Imports
import numpy as np
import os
import sys
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
        action_shape = (number_of_modes-1,)  # Weight of each mode. -1 because radius fixed.
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
        tot_radius = self.squirmer_radius + self.target_radius
        self._target_position = self._array_float(self.target_initial_position, shape=(2,))
        
        # Initial observation is no field
        observation = self._array_float([0, 0, 0], shape=(3,))

        return observation


    def step(self, action):
        # -- Action setup --
        # Actions are the available modes.
        # Modes are equal to n-sphere coordinates divided by the square root of the mode factors
        max_power = 1
        x_n_sphere = power_consumption.n_sphere_angular_to_cartesian(action, max_power)  # Makes sure sums to 1
        power_factors = power_consumption.constant_power_factor(squirmer_radius, self.viscosity, self.max_mode)  # Power factors in front of modes
        power_non_zero = power_factors.nonzero()
        mode_array = np.zeros_like(power_factors)
        mode_array[power_non_zero] = x_n_sphere / np.sqrt(power_factors[power_non_zero].ravel())
                        
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


def train(N_surface_points, squirmer_radius, target_radius, max_mode, sensor_noise, target_initial_position, train_total_steps):
    env = PredatorPreyEnv(N_surface_points, squirmer_radius, target_radius, max_mode, sensor_noise, target_initial_position)

    # Train with SB3
    log_path = os.path.join("Reinforcement Learning", "Training", "Logs_direction")
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_path)
    model.learn(total_timesteps=train_total_steps)
    model_path = os.path.join(log_path, "predict_direction")
    model.save(model_path)
    
    # Save parameters in csv file
    file_path = os.path.join(log_path, "system_parameters.csv")
    file = open(file_path, "w")
    file_data = f"Surface Points {N_surface_points}, Squirmer Radius {squirmer_radius}, Target radius {target_radius}, Max Mode {max_mode}, Noise {sensor_noise}, Target position {target_initial_position}, Centers distance {np.linalg.norm(target_initial_position, ord=2)}, Train Steps {train_total_steps}"
    file.write(file_data)
    file.close()
    

def plot_mode_choice(N_surface_points, squirmer_radius, target_radius, max_mode, sensor_noise, target_initial_position, N_iter, PPO_number, viscosity):
    """Plot the modes taken at different iterations."""
    # Add more colors
    mpl.rcParams["axes.prop_cycle"] = mpl.cycler(color=['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 
                                                        'purple', 'pink', 'brown', 'orange', 'teal', 'coral', 'lightblue', 
                                                        'lime', 'lavender', 'turquoise', 'darkgreen', 'tan', 'salmon', 'gold'])

    # Names
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
            
    #B_names = [r"$B_{01}$", r"$B_{02}$", r"$B_{03}$", r"$B_{04}$", r"$B_{11}$", r"$B_{12}$", r"$B_{13}$", 
    #           r"$B_{14}$", r"$B_{22}$", r"$B_{23}$", r"$B_{24}$", r"$B_{33}$", r"$B_{34}$", r"$B_{44}$",]
    #B_tilde_names = [r"$\tilde{B}_{11}$", r"$\tilde{B}_{12}$", r"$\tilde{B}_{13}$", r"$\tilde{B}_{14}$", r"$\tilde{B}_{22}$", 
    #                 r"$\tilde{B}_{23}$", r"$\tilde{B}_{24}$", r"$\tilde{B}_{33}$", r"$\tilde{B}_{34}$", r"$\tilde{B}_{44}$",]
    #C_names = [r"$C_{02}$", r"$C_{03}$", r"$C_{04}$", r"$C_{12}$", r"$C_{13}$", r"$C_{14}$", 
    #           r"$C_{22}$", r"$C_{23}$", r"$C_{24}$", r"$C_{33}$", r"$C_{34}$", r"$C_{44}$",]
    #C_tilde_names = [r"$\tilde{C}_{12}$", r"$\tilde{C}_{13}$", r"$\tilde{C}_{14}$", r"$\tilde{C}_{22}$", 
    #                 r"$\tilde{C}_{23}$", r"$\tilde{C}_{24}$", r"$\tilde{C}_{33}$", r"$\tilde{C}_{34}$", r"$\tilde{C}_{44}$",]
    
    B_len = len(B_names)
    B_tilde_len = len(B_tilde_names)
    C_len = len(C_names)
    C_tilde_len = len(C_tilde_names)
    B_actions = np.empty((N_iter, B_len))
    B_tilde_actions = np.empty((N_iter, B_tilde_len))
    C_actions = np.empty((N_iter, C_len))
    C_tilde_actions = np.empty((N_iter, C_tilde_len))
    
    rewards = np.empty((N_iter))
    guessed_angles = np.empty((N_iter))
    # Load model and create environment
    model_path = f"Reinforcement Learning/Training/Logs_direction/PPO_{PPO_number}/predict_direction"
    model = PPO.load(model_path)
    env = PredatorPreyEnv(N_surface_points, squirmer_radius, target_radius, max_mode, sensor_noise, target_initial_position)
    
    # Run model N_iter times
    obs = env.reset()
    for i in range(N_iter):
        action, _ = model.predict(obs)
        obs, reward, _, info = env.step(action)
        rewards[i] = reward
        guessed_angles[i] = info["guessed angle"]
        angle = info["angle"]
        # Get modes
        max_power = 1
        x_n_sphere = power_consumption.n_sphere_angular_to_cartesian(action, max_power)  # Makes sure sums to 1
        power_factors = power_consumption.constant_power_factor(squirmer_radius, viscosity, max_mode)  # Power factors in front of modes
        power_non_zero = power_factors.nonzero()
        mode_array = np.zeros_like(power_factors)
        mode_array[power_non_zero] = x_n_sphere / np.sqrt(power_factors[power_non_zero].ravel())
        
        B_actions[i, :] = mode_array[0][np.nonzero(mode_array[0])]
        B_tilde_actions[i, :] = mode_array[1][np.nonzero(mode_array[1])]
        C_actions[i, :] = mode_array[2][np.nonzero(mode_array[2])]
        C_tilde_actions[i, :] = mode_array[3][np.nonzero(mode_array[3])]
        
    # Plot
    fig, ax = plt.subplots(nrows=2, ncols=2, dpi=200)
    ax1 = ax[0, 0]
    ax2 = ax[0, 1]
    ax3 = ax[1, 0]
    ax4 = ax[1, 1]


    def fill_axis(axis, y, marker, label, title):        
        axis.set(xticks=[], title=(title, 7), ylim=(-0.17, 0.17))
        axis.set_title(title, fontsize=7)
        axis.plot(y, marker=marker, ls="--", lw=0.75)
        axis.legend(label, fontsize=4, bbox_to_anchor=(1.05, 1), 
                    loc='upper left', borderaxespad=0.)


    fill_axis(ax1, B_actions, ".", B_names, title=r"$B$ weights")
    fill_axis(ax2, B_tilde_actions, ".", B_tilde_names, title=r"$\tilde{B}$ weights")
    fill_axis(ax3, C_actions, ".", C_names, title=r"$C$ weights")
    fill_axis(ax4, C_tilde_actions, ".", C_tilde_names, title=r"$\tilde{C}$ weights")
    
    guessed_angles = guessed_angles * 180 / np.pi
    xticks = []
    for reward, guess_angle in zip(rewards, guessed_angles):
        tick_str = f"Reward: {np.round(reward, 2)}\nGuessed Angle: {np.round(guess_angle, 2)}"
        xticks.append(tick_str)
    
    ax2.set(yticks=[])
    ax3.set(xlabel="Iteration", xticks=(np.arange(N_iter)))
    ax3.set_xticklabels(xticks, rotation=20, size=5)
    ax4.set(xlabel="Iteration", xticks=(np.arange(N_iter)), yticks=[])
    ax4.set_xticklabels(xticks, rotation=20, size=5)
    fig.suptitle(f"Mode over iterations, Noise = {sensor_noise}, True angle = {np.round(angle * 180 / np.pi, 2)}", fontsize=10)
    fig.tight_layout()
    
    figname = f"noise{sensor_noise}_maxmode{max_mode}_targetradius{target_radius}.png"            
    plt.savefig("Reinforcement Learning/Recordings/Images/" + figname)
    plt.show()


# -- Run the code --
# Parameters
N_surface_points = 80
squirmer_radius = 1
target_radius = 0.5
tot_radius = squirmer_radius + target_radius
target_initial_position = [1.8*tot_radius, 0]
max_mode = 3
N_iter = 5
viscosity = 1
sensor_noise = 0.01
train_total_steps = int(1.5e5)

PPO_number = 10  # For which model to load when plotting, after training

# -- Sensor noise resultater: --
# Max mode 4:
    # 0.20, 200k skridt: Oscillerer 0.5
    # 0.15, 100k skridt: Oscillerer 0.55
    # 0.10, 160k skridt: ikke konvergeret endnu
    # 0.05, 300k skridt: Konvergerer og oscillerer 0.825
# Max mode 3:
    # 0.1, 200k skridt: 


#check_model(N_surface_points, squirmer_radius, target_radius, max_mode, sensor_noise, target_initial_position)
train(N_surface_points, squirmer_radius, target_radius, max_mode, sensor_noise, target_initial_position, train_total_steps)
#plot_mode_choice(N_surface_points, squirmer_radius, target_radius, max_mode, sensor_noise, target_initial_position, N_iter PPO_number, viscosity)

# If wants to see reward over time, write the following in cmd in the log directory
# tensorboard --logdir=.
