""" Notes
- Observation space skal inkludere de modes den tog i sidste time step.
"""
# Imports
import numpy as np
import os
import sys
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "1"
import pygame
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.lines as mlines
import csv

import gym
from gym import spaces
#from gym.wrappers import FlattenObservation

from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv
#from stable_baselines3.common.vec_env import DummyVecEnv
#from stable_baselines3.common.evaluation import evaluate_policy

# Load custom functions
sys.path.append('./Fluid')
import field_velocity
import power_consumption
import bem_two_objects
import bem


# Environment
class PredatorPreyEnv(gym.Env):
    """Gym environment for a predator-prey system in a fluid."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}


    def __init__(self, N_surface_points, squirmer_radius, target_radius, spawn_radius, max_mode, sensor_noise, viscosity, spawn_angle=None, lab_frame=True, render_mode=None, scale_canvas=1):
        #super().__init__() - ingen anelse om hvorfor jeg havde skrevet det eller hvor det kommer fra?
        # -- Variables --
        # Model
        self.N_surface_points = N_surface_points  # Points on squirmer surface
        self.squirmer_radius = squirmer_radius 
        self.target_radius = target_radius
        self.spawn_radius = spawn_radius  # Max distance the target can be spawned away from the agent
        self.max_mode = max_mode  # Max available legendre mode, max n in B_{mn}
        self.spawn_angle = spawn_angle
        self.sensor_noise = sensor_noise
        self.lab_frame = lab_frame  # Choose lab or squirmer frame of reference. If True, in lab frame otherwise in squirmer
        self.viscosity = viscosity

        # Parameters
        self.B_max = 1
        self.charac_velocity = 4 * self.B_max / (3 * self.squirmer_radius ** 3)  # Characteristic velocity: Divide velocities by this to remove dimensions
        self.charac_time = 3 * self.squirmer_radius ** 4 / (4 * self.B_max) # characteristic time
        tau = 1  # Seconds per iteration. 
        self.dt = tau / self.charac_time
        self.epsilon = 0.1  # Width of regularization blobs
        self.extra_catch_radius = 0.1

        # Rendering
        self.scale_canvas = scale_canvas  # Makes canvas this times smaller
        self.window_size = 512  # PyGame window size
        assert render_mode is None or render_mode in self.metadata["render_modes"]  # Check if given render_mode matches available
        self.render_mode = render_mode
        self.window = None
        self.clock = None

        # -- Define action and observation space --
        # Actions: Strength of modes
        if max_mode == 4:
            number_of_modes = 45  # Counted from power factors.
        elif max_mode == 3:
            number_of_modes = 27
        elif max_mode == 2: 
            number_of_modes = 13
        action_shape = (number_of_modes, )
        self.action_space = spaces.Box(low=-1, high=1, shape=action_shape, dtype=np.float32)

        # Observations: average force difference vector and angle predicted at previous time step
        self.observation_space = spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)


    def _array_float(self, x):
        """Helper function to input x into a shape sized array with dtype np.float32"""
        return np.array([x], dtype=np.float32).reshape(np.size(x), )


    def _get_dist(self):
        """Helper function that calculates the distance between the agent and the target"""
        return np.linalg.norm(self._agent_position - self._target_position, ord=2)

    
    def _cartesian_to_polar(self):
        r = self._get_dist()
        agent_target_vec = self._target_position - self._agent_position  # Vector pointing from target to agent
        theta = np.arctan2(agent_target_vec[0], agent_target_vec[1])
        return r, theta


    def _average_force_difference(self, mode_array):
        # RL is 2d but Oseen tensor is 3d, add x=0 dimension
        agent_center = np.append([0], self._agent_position)
        target_center = np.append([0], self._target_position)
        
        # Calculate Oseen tensors
        A_oseen_two_objects = bem_two_objects.oseen_tensor_surface_two_objects(self.x1_stack, self.x2_stack, agent_center, target_center, 
                                                                               self.dA, self.epsilon, self.viscosity)
        A_oseen_one_object = bem.oseen_tensor(self.epsilon, self.dA, self.viscosity, evaluation_points=self.x1_stack)
        
        # Squirmer surface velocity
        ux1, uy1, uz1 = field_velocity.field_cartesian(self.max_mode, r=self.squirmer_radius, 
                                                       theta=self.theta1, phi=self.phi1, 
                                                       squirmer_radius=self.squirmer_radius, 
                                                       mode_array=mode_array,
                                                       lab_frame=self.lab_frame)
        u_stack = np.array([ux1, uy1, uz1]).ravel()
        u_one_object = np.append(u_stack, np.zeros(6))  # No target
        u_two_objects = np.append(u_stack, np.zeros(12+3*self.N2))
        
        # Solve for forces and velocities
        force_one_object = np.linalg.solve(A_oseen_one_object, u_one_object)
        force_two_objects = np.linalg.solve(A_oseen_two_objects, u_two_objects)
        u_squirmer = force_two_objects[-11: -9]  # Only U_y and U_z
        u_target = force_two_objects[-5: -3]
        
        # Force differences - NOTE bør man overhovedet have dfx i RL, når kun bevæger yz planet?
        N1 = self.N_surface_points
        dfx = force_two_objects[:N1].T - force_one_object[:N1].T  # NOTE burde man allerede her tage abs()? Relatvant ift støj!?
        dfy = force_two_objects[N1: 2*N1].T - force_one_object[N1: 2*N1].T
        dfz = force_two_objects[2*N1: 3*N1].T - force_one_object[2*N1: 3*N1].T

        # Gaussian noise
        dfx += np.random.normal(loc=0, scale=self.sensor_noise, size=dfx.size)
        dfy += np.random.normal(loc=0, scale=self.sensor_noise, size=dfy.size)
        dfz += np.random.normal(loc=0, scale=self.sensor_noise, size=dfz.size)

        # Weight and force
        weight = np.sqrt(dfx ** 2 + dfy ** 2 + dfz ** 2)
        f_average = np.sum(weight[:, None] * self.x1_stack, axis=0)
        f_average_norm = f_average / np.linalg.norm(f_average, ord=2)
        
        return u_squirmer, u_target, f_average_norm
        
        
    def _reward(self):
        """Penalize with distance from target. Normalized by starting distance. 
        """
        # NOTE prøv om kan normalisere reward. Evt med: spawn radius / (karaktertistisk v * karakteristisk t)
        dist = self._get_dist()
        too_far_away = dist > 1.5 * self.spawn_radius
        if too_far_away:  # Stop simulation if too far away and penalize hard
            reward = - 1000
            done = True                        
        elif dist > self.catch_radius:
            reward = - dist / self.spawn_radius
            done = False
        else:  # Catched, dist < self.catch_radius
            reward = 1000
            done = True
        return reward, done
        
    
    def reset(self, seed=None):
        # Fix seed and reset values
        super().reset(seed=seed)

        # Initial values
        self.time = 0
        self._agent_position = self._array_float([0, 0])
        self.previous_angle_guess = 0  # NOTE måske angle guess = 0 ikke er smart? Især hvis target faktisk er i 0. Mindre problem hvis target spawnes tilfældigt sted
        observation = self._array_float([0, 0, 0, self.previous_angle_guess])  # No initial force difference field
        
        # Target initial position randomly determined or fixed, depending on spawn_angle is given
        self.catch_radius = self.squirmer_radius + self.target_radius + self.extra_catch_radius

        if self.spawn_angle is None:
            self._target_position = self._agent_position  
            dist = self._get_dist() 
            while dist <= 2 * self.catch_radius:  # While distance between target and agent is too small, find a new initial position for the target
                initial_distance = np.random.uniform(low=0, high=self.spawn_radius)
                initial_angle = np.random.uniform(-1, 1) * np.pi
                self._target_position = self._array_float([initial_distance * np.sin(initial_angle), initial_distance * np.cos(initial_angle)])
                dist = self._get_dist() 
                
        # In case wants specific starting location
        else:  
            self._target_position = self._array_float([self.spawn_radius * np.sin(self.spawn_angle), self.spawn_radius * np.cos(self.spawn_angle)])
        
        self.initial_target_position = 1 * self._target_position  # Needed for reward calculation

        # Surface coordinates squirmer 
        x1, y1, z1, self.dA = bem.canonical_fibonacci_lattice(self.N_surface_points, self.squirmer_radius)
        self.theta1 = np.arccos(z1 / self.squirmer_radius)  # [0, pi]
        self.phi1 = np.arctan2(y1, z1)  # [0, 2pi]
        self.x1_stack = np.stack((x1, y1, z1)).T

        # Surface coordinates target
        self.N2 = int(4 * np.pi * self.target_radius ** 2 / self.dA)  # Ensures object 1 and 2 has same dA
        x2, y2, z2, _ = bem.canonical_fibonacci_lattice(self.N2, self.target_radius)
        self.x2_stack = np.stack((x2, y2, z2)).T

        if self.render_mode == "human":
            self._render_frame()

        return observation  # SB3 Vectorized envs requires only observation, not info, unlike Gym 0.26


    def step(self, action):
        # Action to normalized modes
        mode_array = power_consumption.normalized_modes(action, self.max_mode, self.squirmer_radius, self.viscosity)
        
        # Velocities and update movement
        u_squirmer, u_target, f_average = self._average_force_difference(mode_array)  
        self._target_position += u_target * self.dt 
        if self.lab_frame:
            self._agent_position += u_squirmer * self.dt
        
        # Update values 
        reward, done = self._reward()
        observation = self._array_float(np.append(f_average, self.previous_angle_guess))
        self.previous_angle_guess = np.arctan2(f_average[1], f_average[2]) / np.pi  # Must be between -1 and 1
        self.time += self.dt
        info = {"modes": mode_array[np.nonzero(mode_array)], "time": self.time, "target": self._target_position, "agent": self._agent_position}

        if self.render_mode == "human":
            self._render_frame()
        
        return observation, reward, done, info


    def _coord_to_pixel(self, position):
        """PyGame window is fixed at (0, 0), but we want (0, 0) to be in the center of the screen. The area is width x height, so all points must be shifted by (width/2, -height/2)"""
        return position + self._array_float([self.spawn_radius, self.spawn_radius]) * self.scale_canvas


    def render(self):
        """Renders environment to screen"""
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
        pix_size = self.window_size / (2 * self.spawn_radius * self.scale_canvas)
        target_position_draw = self._coord_to_pixel(self._target_position) * pix_size 
        agent_position_draw = self._coord_to_pixel(self._agent_position) * pix_size

        # Draw target
        pygame.draw.circle(
            canvas,  # What surface to draw on
            (255, 0, 0),  # Color
            (float(target_position_draw[0]), float(target_position_draw[1])),  # x, y coordinate
            float(pix_size * (self.target_radius + self.extra_catch_radius))  # Radius
        )

        # Draw agent
        pygame.draw.circle(
            canvas,
            (0, 0, 255),  # Color
            (float(agent_position_draw[0]), float(agent_position_draw[1])),  # x, y  - Maybe needs to add +0.5 to positions?
            pix_size * self.squirmer_radius ,  # Radius
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


def check_model(N_surface_points, squirmer_radius, target_radius, spawn_radius, max_mode, sensor_noise, viscosity, spawn_angle, lab_frame, render_mode, scale_canvas):
    env = PredatorPreyEnv(N_surface_points, squirmer_radius, target_radius, spawn_radius, max_mode, sensor_noise, viscosity, spawn_angle, lab_frame, render_mode, scale_canvas)
    print("-- SB3 CHECK ENV: --")
    if check_env(env) == None:
        print("   The Environment is compatible with SB3")
    else:
        print(check_env(env))


def train(N_surface_points, squirmer_radius, target_radius, spawn_radius, max_mode, sensor_noise, viscosity, spawn_angle, lab_frame, train_total_steps):
    env = PredatorPreyEnv(N_surface_points, squirmer_radius, target_radius, spawn_radius, max_mode, sensor_noise, viscosity, spawn_angle, lab_frame, render_mode=None)
    # env = Monitor(env)
    # env = SubprocVecEnv([lambda: env])
    
    # Train with SB3
    log_path = os.path.join("RL", "Training", "Logs")
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_path)
    model.learn(total_timesteps=train_total_steps)
    model_path = os.path.join(log_path, "ppo_predator_prey")
    model.save(model_path)
    
    # Save parameters in csv file
    file_path = os.path.join(log_path, "system_parameters.csv")
    with open(file_path, mode="w") as file:
        writer = csv.writer(file, delimiter=",")
        writer.writerow(["Surface Points ", "Squirmer Radius ", "Target Radius ", "Max Mode ", "Sensor Noise ", 
                         "Spawn radius ", "Spawn angle ", "viscosity ", "Train Steps "])
        writer.writerow([N_surface_points, squirmer_radius, target_radius, max_mode, sensor_noise, spawn_radius, spawn_angle,
                        viscosity, train_total_steps])


def run_environment(PPO_number):
    # Load parameters and model, create environment
    parameters_path = f"RL/Training/Logs/PPO_{PPO_number}/system_parameters.csv"
    parameters = np.genfromtxt(parameters_path, delimiter=",", skip_header=1)
    N_surface_points = int(parameters[0])
    squirmer_radius = parameters[1]
    target_radius = parameters[2]
    max_mode = int(parameters[3])
    sensor_noise = parameters[4]
    spawn_radius = parameters[5]
    spawn_angle = parameters[6]
    viscosity = parameters[7]
    model_path = f"RL/Training/Logs/PPO_{PPO_number}/ppo_predator_prey"
    
    mode_values_list = []
    time_list = []
    target_pos_list = []
    agent_pos_list = []
    reward_list = []
    
    # Run model
    model = PPO.load(model_path)
    env = PredatorPreyEnv(N_surface_points, squirmer_radius, target_radius, spawn_radius, max_mode, sensor_noise, viscosity, spawn_angle, lab_frame=True, render_mode=None, scale_canvas=1)    
    
    obs = env.reset()
    done = False
    
    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, info = env.step(action)
        mode_values = 1 * info["modes"]  # Make sure does not replace previous values
        time = 1 * info["time"]
        target_pos = 1 * info["target"]
        agent_pos = 1 * info["agent"]
        
        mode_values_list.append(mode_values)
        time_list.append(time)
        target_pos_list.append(target_pos)
        agent_pos_list.append(agent_pos)
        reward_list.append(reward)
    return parameters, mode_values_list, time_list, target_pos_list, agent_pos_list, reward_list


def animation(PPO_number, lab_frame, render_mode, scale_canvas):
    # Load model and create environment
    parameters_path = f"RL/Training/Logs/PPO_{PPO_number}/system_parameters.csv"
    parameters = np.genfromtxt(parameters_path, delimiter=",", skip_header=1)
    N_surface_points = int(parameters[0])
    squirmer_radius = parameters[1]
    target_radius = parameters[2]
    max_mode = int(parameters[3])
    sensor_noise = parameters[4]
    spawn_radius = parameters[5]
    spawn_angle = parameters[6]
    viscosity = parameters[7]
    
    model_path = os.path.join("RL", "Training", "Logs", f"PPO_{PPO_number}", "ppo_predator_prey")
    model = PPO.load(model_path)    
    env = PredatorPreyEnv(N_surface_points, squirmer_radius, target_radius, spawn_radius, max_mode, sensor_noise, viscosity, spawn_angle, lab_frame, render_mode, scale_canvas)

    # Run and render model
    obs = env.reset()
    done = False
        
    while not done:
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)
        env.render()
        
    env.close()
        

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


def plot_info(PPO_number):
    # Add more plot colors
    mpl.rcParams["axes.prop_cycle"] = mpl.cycler(color=['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 
                                                        'purple', 'pink', 'brown', 'orange', 'teal', 'coral', 'lightblue', 
                                                        'lime', 'lavender', 'turquoise', 'darkgreen', 'tan', 'salmon', 'gold'])
    # Get values and names
    parameters, mode_list, time_list, target_pos_list, agent_pos_list, reward_list = run_environment(PPO_number)
    squirmer_radius = parameters[1]
    target_radius = parameters[2]
    spawn_radius = parameters[5]
    B_names, B_tilde_names, C_names, C_tilde_names = mode_names(max_mode)
    mode_names_list = B_names + B_tilde_names + C_names + C_tilde_names
    modes = np.array(mode_list)
    time = np.array(time_list)
    target_pos = np.array(target_pos_list)
    agent_pos = np.array(agent_pos_list)
    reward_arr = np.array(reward_list)
    
    # -- Plot mode values over time --
    fig_mode, ax_mode = plt.subplots(dpi=150, figsize=(8, 6))
    ax_mode.plot(time, modes, "--.")
    ax_mode.set(xlabel="Time", ylabel="Mode values", title="Mode values against time")
    ax_mode.legend(mode_names_list, fontsize=4, bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.)
    fig_mode.tight_layout()
    plt.show()
    plt.close()
    
    # -- Plot target and agent position over time --
    # NOTE ville være godt hvis man kan få deres radius til at passe. Potentielt patch circles? - Kræver nok loop
    fig_pos, ax_pos = plt.subplots(dpi=150, figsize=(6, 6))
    color_target = cm.Reds(np.linspace(0, 1, len(target_pos[:, 0])))  # Colors gets darker with time. 
    color_agent = cm.Blues(np.linspace(0, 1, len(agent_pos[:, 0])))
    
    def add_circle_to_axis(axis, coord, radius, color, label):
        circle = plt.Circle(coord, radius, facecolor="none", edgecolor=color, fill=False, label=label)
        axis.add_patch(circle)
    
    for i in range(len(agent_pos)):
        add_circle_to_axis(ax_pos, target_pos[i, :], target_radius, color_target[i], label="Target")
        add_circle_to_axis(ax_pos, agent_pos[i, :], squirmer_radius, color_agent[i], label="Agent")
    
    #ax_pos.scatter(target_pos[:, 0], target_pos[:, 1], s=15, label="Target", facecolor="none", edgecolors=color_target)
    #ax_pos.scatter(agent_pos[:, 0], agent_pos[:, 1], s=8000, label="Agent", facecolor="none", edgecolors=color_agent)
    ax_pos.set(xlabel="x", ylabel="y", title="Agent and target position over time", xlim=(-spawn_radius, spawn_radius), ylim=(-spawn_radius, spawn_radius))
    # Making a good looking legend
    agent_legend_marker = mlines.Line2D(xdata=[], ydata=[], marker=".", markersize=12, linestyle="none", fillstyle="none", color=color_agent[-1], label="Agent")
    target_legend_marker = mlines.Line2D(xdata=[], ydata=[], marker=".", markersize=12, linestyle="none", fillstyle="none", color=color_target[-1], label="Target")
    ax_pos.legend(handles=[agent_legend_marker, target_legend_marker])
    fig_pos.tight_layout()
    plt.show()
    plt.close()
                
    # -- Plot reward over time --
    fig_reward, ax_reward = plt.subplots(dpi=150, figsize=(8, 6))
    ax_reward.plot(time, reward_arr, ".--", label="Reward")
    ax_reward.set(xlabel="Time", ylabel="Reward", title="Agent Reward against time")
    ax_reward.legend()
    
    fig_reward.tight_layout()     
    plt.show()


# -- Run the code --
# Parameters
N_surface_points = 80
squirmer_radius = 1
target_radius = 0.8
tot_radius = squirmer_radius + target_radius
spawn_radius = 3 * tot_radius
max_mode = 2  
sensor_noise = 0.1
viscosity = 1
spawn_angle = np.pi / 4
lab_frame = True
render_mode = "human"
scale_canvas = 1.4  # Makes everything on the canvas a factor smaller / zoomed out

PPO_number = 9
train_total_steps = int(1.5e5)

if __name__ == "__main__":  # Required for SubprocVecEnv
    #check_model(N_surface_points, squirmer_radius, target_radius, spawn_radius, max_mode, sensor_noise, viscosity, spawn_angle, lab_frame, render_mode, scale_canvas)
    #train(N_surface_points, squirmer_radius, target_radius, spawn_radius, max_mode, sensor_noise, viscosity, spawn_angle, lab_frame, train_total_steps)
    #animation(PPO_number, N_surface_points, squirmer_radius, target_radius, spawn_radius, max_mode, sensor_noise, viscosity, spawn_angle, lab_frame, render_mode, scale_canvas)
    plot_info(PPO_number)


# If wants to see reward over time, write the following in cmd in the log directory
# tensorboard --logdir=.
