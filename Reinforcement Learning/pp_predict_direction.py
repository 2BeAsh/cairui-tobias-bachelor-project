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
def plot_mode_vs_time(N_surface_points, N_iter, squirmer_radius, target_radius, max_mode):
    """Plot the actions taken at different iterations. Actions correspond to the weight/importance a mode is given.
    Color goes from bright to dark with increasing n and m values."""
        
    B_names = [r"$B_{01}$", r"$B_{02}$", r"$B_{03}$", r"$B_{04}$", r"$B_{11}$", r"$B_{12}$", r"$B_{13}$", 
               r"$B_{14}$", r"$B_{22}$", r"$B_{23}$", r"$B_{24}$", r"$B_{33}$", r"$B_{34}$", r"$B_{44}$",]
    B_tilde_names = [r"$B_{tilde,11}$", r"$B_{tilde,12}$", r"$B_{tilde,13}$", r"$B_{tilde,14}$", r"$B_{tilde,22}$", 
                     r"$B_{tilde,23}$", r"$B_{tilde,24}$", r"$B_{tilde,33}$", r"$B_{tilde,34}$", r"$B_{tilde,44}$",]
    C_names = [r"$C_{02}$", r"$C_{03}$", r"$C_{04}$", r"$C_{12}$", r"$C_{13}$", r"$C_{14}$", 
               r"$C_{22}$", r"$C_{23}$", r"$C_{24}$", r"$C_{33}$", r"$C_{34}$", r"$C_{44}$",]
    C_tilde_names = [r"$C_{tilde,12}$", r"$C_{tilde,13}$", r"$C_{tilde,14}$", r"$C_{tilde,22}$", 
                     r"$C_{tilde,23}$", r"$C_{tilde,24}$", r"$C_{tilde,33}$", r"$C_{tilde,34}$", r"$C_{tilde,44}$",]
    
    B_len = len(B_names)
    B_tilde_len = len(B_tilde_names)
    C_len = len(C_names)
    C_tilde_len = len(C_tilde_names)
    
    B_actions = np.empty((N_iter, B_len))
    B_tilde_actions = np.empty((N_iter, B_tilde_len))
    C_actions = np.empty((N_iter, C_len))
    C_tilde_actions = np.empty((N_iter, C_tilde_len))
    
    # Load model and create environment
    model = PPO.load("ppo_predator_prey_direction")
    env = PredatorPreyEnv(N_surface_points, squirmer_radius, target_radius, max_mode)
    
    # Run model N_iter times
    obs = env.reset()
    for i in range(N_iter):
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)
        B_actions[i, :] = action[: B_len]
        B_tilde_actions[i, :] = action[B_len: B_len+B_tilde_len]
        C_actions[i, :] = action[B_len+B_tilde_len : B_len+B_tilde_len+C_len]
        C_tilde_actions[i, :] = action[-C_tilde_len:]
    
    
    # Loop through each set of N_iter point such that it can get unique color
    cm_norm = colors.Normalize(0, 1)    
    def plot_N_iter_points(mode_values, cmap, marker):
        length = mode_values.shape[1]
        intensity = np.linspace(0.2, 1, length)
        map = cm.ScalarMappable(norm=cm_norm, cmap=cmap)
        lines = []
        for i in range(length):
            col = map.to_rgba(intensity[i])
            line, = plt.plot(np.abs(mode_values[:, i]), marker, lw=1, ls="--", c=col)
            lines.append(line)
        return lines

    # Figure setup    
    fig, ax = plt.subplots(dpi=200, figsize=(8, 10))
    ax.set(xlabel="Time", ylabel="Mode weight", title="Mode weights against time", xlim=(-0.5, N_iter-0.5), xticks=[])

    B_lines = plot_N_iter_points(B_actions, "Reds", "x")
    B_tilde_lines = plot_N_iter_points(B_tilde_actions, "Blues", ".")
    # C_lines = plot_N_iter_points(C_actions, "Greens", "*")
    # C_tilde_lines = plot_N_iter_points(C_tilde_actions, "Oranges", "<")

    # Plot mode values over iterations
    
    
    #B_line = ax.plot(B_actions, "--x", lw=1, )
    #B_tilde_line = ax.plot(B_tilde_actions, "--.", lw=1)

    # Legend
    B_legend = plt.legend([B_lines[i] for i in np.arange(len(B_lines))], B_names, loc="upper left")
    B_tilde_legend = plt.legend([B_tilde_lines[i] for i in np.arange(len(B_tilde_lines))], B_tilde_names, loc="upper right")

    ax.add_artist(B_legend)
    ax.add_artist(B_tilde_legend)
    
    # https://stackoverflow.com/questions/12761806/matplotlib-2-different-legends-on-same-graph
    # https://stackoverflow.com/questions/44911452/attributeerror-unknown-property-cmap 
    

    
    fig.tight_layout()
    plt.show()
    
    
def plot_mode_choice(N_surface_points, N_iter, squirmer_radius, target_radius, max_mode):
    """Plot the actions taken at different iterations. Actions correspond to the weight/importance a mode is given.
    Color goes from bright to dark with increasing n and m values."""
    # Add more colors
    mpl.rcParams["axes.prop_cycle"] = mpl.cycler(color=['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 
                                                        'purple', 'pink', 'brown', 'orange', 'teal', 'coral', 'lightblue', 
                                                        'lime', 'lavender', 'turquoise', 'darkgreen', 'tan', 'salmon', 'gold'])

    B_names = [r"$B_{01}$", r"$B_{02}$", r"$B_{03}$", r"$B_{04}$", r"$B_{11}$", r"$B_{12}$", r"$B_{13}$", 
               r"$B_{14}$", r"$B_{22}$", r"$B_{23}$", r"$B_{24}$", r"$B_{33}$", r"$B_{34}$", r"$B_{44}$",]
    B_tilde_names = [r"$\tilde{B}_{11}$", r"$\tilde{B}_{12}$", r"$\tilde{B}_{13}$", r"$\tilde{B}_{14}$", r"$\tilde{B}_{22}$", 
                     r"$\tilde{B}_{23}$", r"$\tilde{B}_{24}$", r"$\tilde{B}_{33}$", r"$\tilde{B}_{34}$", r"$\tilde{B}_{44}$",]
    C_names = [r"$C_{02}$", r"$C_{03}$", r"$C_{04}$", r"$C_{12}$", r"$C_{13}$", r"$C_{14}$", 
               r"$C_{22}$", r"$C_{23}$", r"$C_{24}$", r"$C_{33}$", r"$C_{34}$", r"$C_{44}$",]
    C_tilde_names = [r"$\tilde{C}_{12}$", r"$\tilde{C}_{13}$", r"$\tilde{C}_{14}$", r"$\tilde{C}_{22}$", 
                     r"$\tilde{C}_{23}$", r"$\tilde{C}_{24}$", r"$\tilde{C}_{33}$", r"$\tilde{C}_{34}$", r"$\tilde{C}_{44}$",]
    
    B_len = len(B_names)
    B_tilde_len = len(B_tilde_names)
    C_len = len(C_names)
    C_tilde_len = len(C_tilde_names)
    
    B_actions = np.empty((N_iter, B_len))
    B_tilde_actions = np.empty((N_iter, B_tilde_len))
    C_actions = np.empty((N_iter, C_len))
    C_tilde_actions = np.empty((N_iter, C_tilde_len))
    
    # Load model and create environment
    model = PPO.load("ppo_predator_prey_direction")
    env = PredatorPreyEnv(N_surface_points, squirmer_radius, target_radius, max_mode)
    
    # Run model N_iter times
    obs = env.reset()
    for i in range(N_iter):
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)
        B_actions[i, :] = action[: B_len]
        B_tilde_actions[i, :] = action[B_len: B_len+B_tilde_len]
        C_actions[i, :] = action[B_len+B_tilde_len : B_len+B_tilde_len+C_len]
        C_tilde_actions[i, :] = action[-C_tilde_len:]

    # Plot
    fig, ax = plt.subplots(nrows=2, ncols=2, dpi=200)
    ax1 = ax[0, 0]
    ax2 = ax[0, 1]
    ax3 = ax[1, 0]
    ax4 = ax[1, 1]

    def fill_axis(axis, y, marker, label, title):        
        axis.set(xticks=[])
        axis.set_title(title, fontsize=7)
        axis.plot(np.abs(y), marker=marker, ls="--", lw=0.75)
        axis.legend(label, fontsize=4, bbox_to_anchor=(1.05, 1), 
                    loc='upper left', borderaxespad=0.)

        
    seperate_letters = True
    if seperate_letters:
        fill_axis(ax1, B_actions, ".", B_names, title=r"$B$ modes")
        fill_axis(ax2, B_tilde_actions, ".", B_tilde_names, title=r"$\tilde{B}$ modes")
        fill_axis(ax3, C_actions, ".", C_names, title=r"$C$ modes")
        fill_axis(ax4, C_tilde_actions, ".", C_tilde_names, title=r"$\tilde{C}$ modes")

    ax2.set(yticks=[])
    ax4.set(yticks=[])
    
    fig.tight_layout()
    plt.show()

# -- Run the code --
# Parameters
N_surface_points = 80
squirmer_radius = 1
target_radius = 1.1
max_mode = 4
N_iter = 5

train_total_steps = int(2e5)

#check_model(N_surface_points, squirmer_radius, target_radius, max_mode)
#train(N_surface_points, squirmer_radius, target_radius, max_mode, train_total_steps)
#plot_mode_vs_time(N_surface_points, N_iter, squirmer_radius, target_radius, max_mode)
plot_mode_choice(N_surface_points, N_iter, squirmer_radius, target_radius, max_mode)


# If wants to see reward over time, write the following in cmd in the log directory
# tensorboard --logdir=.
