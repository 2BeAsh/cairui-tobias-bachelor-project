""" Notes
- Observation space skal inkludere de modes den tog i sidste time step.
"""
# Imports
import numpy as np
import os
import csv
import sys
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "1"
import pygame

import gym
from gym import spaces

# Stable baselines 3
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

# Matplotlib
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.lines as mlines

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


    def __init__(self, squirmer_radius, spawn_radius, max_mode, viscosity, cap_modes, spawn_angle=None, render_mode=None, scale_canvas=1, squirmer_frame=True):
        #super().__init__() - ingen anelse om hvorfor jeg havde skrevet det eller hvor det kommer fra?
        # -- Variables --
        # Model
        self.squirmer_radius = squirmer_radius 
        self.spawn_radius = spawn_radius  # Max distance the target can be spawned away from the agent
        self.max_mode = max_mode  # Max available legendre mode, max n in B_{mn}
        self.spawn_angle = spawn_angle
        self.viscosity = viscosity
        self.squirmer_frame = squirmer_frame
        self.cap_modes = cap_modes
        assert cap_modes in ["uncapped", "constant"]

        # Parameters
        self.B_max = 1
        self.charac_velocity = 4 * self.B_max / (3 * self.squirmer_radius ** 3)  # Characteristic velocity: Divide velocities by this to remove dimensions
        self.charac_time = 3 * self.squirmer_radius ** 4 / (4 * self.B_max) # characteristic time
        tau = 1 / 3 # Seconds per iteration. 
        self.dt = tau #/ self.charac_time
        self.extra_catch_radius = 0.2
        self.catch_radius = self.squirmer_radius + self.extra_catch_radius
        
        # Rendering
        self.scale_canvas = scale_canvas  # Makes canvas this times smaller
        self.window_size = 512  # PyGame window size
        assert render_mode is None or render_mode in self.metadata["render_modes"]  # Check if given render_mode matches available
        self.render_mode = render_mode
        self.window = None
        self.clock = None

        # -- Define action and observation space --
        # Actions: Strength of modes
        
        if self.cap_modes == "constant":  # Capped, sum(modes) = 1. Only two modes, realistic case
            action_shape = (1,)  # alpha, distribution of the two modes
        else:  # Uncapped, Zhu2022 case, allows sqrt(2) speed
            action_shape = (2,)  # The two modes
        self.action_space = spaces.Box(low=-1, high=1, shape=action_shape, dtype=np.float32)

        # Observations: radius and angle to target
        self.observation_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)  # (distance, angle). 


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

   
    def _get_obs(self):
        """Helper function which convertes values into observation space values (between -1 and 1).
        Get the distance and angle, and then convert them to values between -1 and 1.
        For the angle divide by pi. 
        For the distance, shift the values by half of its max (ideally the max would be infinity, but as we stop the simulation at spawn_radius it is chosen) and divide by the max
        """
        r, theta = self._cartesian_to_polar()
        # Normalize distance and angle
        upper_dist = 1.5 * self.spawn_radius  # why 1.5?
        r_unit = (r - upper_dist / 2) / upper_dist
        theta_unit = theta / np.pi
        return self._array_float([r_unit, theta_unit])        


    def _reward_time_optimized(self):
        r = self._get_dist()
        too_far_away = r > self.spawn_radius
        captured = r < self.catch_radius
        done = False
        if too_far_away:  # Stop simulation and penalize hard if goes too far away
            reward = -1000
            done = True
        elif captured:  # Reward for how fast the target was catched
            reward = 200 / (self.time - self.d0) 
            if self.time - self.d0 < 0:
                print("Time diff: ", self.time - self.d0)
                reward = 3000
            done = True
        else:
            reward = -r
        return float(reward), done


    def reset(self, seed=None):
        # Fix seed and reset values
        super().reset(seed=seed)

        # Initial values
        self.time = 0
        self._agent_position = self._array_float([0, 0])
        self._target_velocity = self._array_float([0, 0])

        # Target starts random location not too close to agent
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

        self.d0 = (self._get_dist() - self.catch_radius) / 1   # Shortest distance between squirmer surface and target. Because can move faster than allowed when uncapped, reward can become negative, thus makes d0 smaller by arbitrary factor (here 1.3)

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
        B = np.zeros((self.max_mode+1, self.max_mode+1))
        B_tilde = np.zeros_like(B)
        C = np.zeros_like(B)
        C_tilde = np.zeros_like(B)
        if self.cap_modes == "constant":  # Realistic case, no sqrt(2) speed allowed
            B[0, 1] = np.sin(action * np.pi)
            B_tilde[1, 1] = np.cos(action * np.pi)
        else:  # Uncapped, Zhu2022, allows sqrt(2) speed
            B[0, 1] = action[0] / self.B_max
            B_tilde[1, 1] = action[1] / self.B_max        
        mode_array = np.array([B, B_tilde, C, C_tilde])    
            
        # -- Movement --
        # Convert to polar coordinates and get the cartesian velocity of the flow.
        r, theta = self._cartesian_to_polar()
        if self.squirmer_frame:  # Train using squirmer frame, plot using lab frame
            _, velocity_y, velocity_z = field_velocity.field_cartesian_squirmer(max_mode=self.max_mode, r=r, theta=theta, phi=np.pi/2, squirmer_radius=self.squirmer_radius, mode_array=mode_array)
        else:
            _, velocity_y, velocity_z = field_velocity.field_cartesian(max_mode=self.max_mode, r=r, theta=theta, phi=np.pi/2, squirmer_radius=self.squirmer_radius, mode_array=mode_array)
            
        target_velocity = self._array_float([velocity_y, velocity_z]) #/ self.charac_velocity
        self._target_position = self._target_position + target_velocity * self.dt 

        B_01 = mode_array[0, 0, 1]
        B_tilde_11 = mode_array[1, 1, 1]
        mode_info = [B_01, B_tilde_11]        
        if not self.squirmer_frame:  # Lab frame for plotting
            squirmer_velocity = np.array([B_tilde_11, -B_01], dtype=np.float32) 
            self._agent_position = self._agent_position + squirmer_velocity * self.dt 
            
        # -- Reward --
        reward, done = self._reward_time_optimized()

        # -- Update values --
        self.time += self.dt
        observation = self._get_obs()
        info = {"modes": mode_info, "time": self.time, "target": self._target_position, "agent": self._agent_position}

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
            float(pix_size * self.extra_catch_radius)  # Radius
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


def train(squirmer_radius, spawn_radius, max_mode, viscosity, cap_modes, spawn_angle, train_total_steps):
    log_path = os.path.join("RL", "Training", "Logs_zhu", cap_modes)
    env = PredatorPreyEnv(squirmer_radius, spawn_radius, max_mode, viscosity, cap_modes, spawn_angle, render_mode=None)
    #env = make_vec_env(lambda: env, n_envs=1) #wrapper_class=SubprocVecEnv)
                       
    # Train with SB3
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_path)
    model.learn(total_timesteps=train_total_steps)
    model_path = os.path.join(log_path, "ppo_predator_prey")
    model.save(model_path)
    
    # Save parameters in csv file
    file_path = os.path.join(log_path, "system_parameters.csv")
    with open(file_path, mode="w") as file:
        writer = csv.writer(file, delimiter=",")
        writer.writerow(["squirmer_radius", "spawn_radius", "max_mode", "viscosity", 
                         "mode_cap", "spawn_angle", "train_steps"])
        writer.writerow([squirmer_radius, spawn_radius, max_mode, viscosity, cap_modes, spawn_angle, train_total_steps])


def pygame_animation(PPO_number, cap_modes, render_mode, scale_canvas, squirmer_frame=False):
    """Show Pygame animation"""
    # Load model and create environment
    parameters_path = f"RL/Training/Logs_zhu/{cap_modes}/PPO_{PPO_number}/system_parameters.csv"
    parameters = np.genfromtxt(parameters_path, delimiter=",", names=True, dtype=None, encoding='UTF-8') #=[np.float32, np.float32, int, np.float32, str, np.float32, bool, int])
    squirmer_radius = parameters["squirmer_radius"]
    spawn_radius = parameters["spawn_radius"]
    max_mode = parameters["max_mode"]
    viscosity = parameters["viscosity"]
    cap_modes = parameters["mode_cap"]
    spawn_angle = parameters["spawn_angle"]
    model_path = f"RL/Training/Logs_zhu/{cap_modes}/PPO_{PPO_number}/ppo_predator_prey"
    
    model = PPO.load(model_path)
    env = PredatorPreyEnv(squirmer_radius, spawn_radius, max_mode, viscosity, cap_modes, spawn_angle, render_mode=render_mode, scale_canvas=scale_canvas, squirmer_frame=squirmer_frame)

    # Run and render model
    obs = env.reset()
    done = False
        
    while not done:
        action, _ = model.predict(obs)
        obs, _, done, _ = env.step(action)
        env.render()
        
    env.close()


def path_mode_plot(PPO_list, cap_modes):
    """Plot path taken by both predator and prey, then in seperate plot show the mode values over time."""    
    # For each entry in PPO_list, find the squirmer and target positions, and modes values over time. Plot them on seperate axis
    xy_width = 8
    
    def run_model(PPO_number):
        # Load parameters and model, create environment
        parameters_path = f"RL/Training/Logs_zhu/{cap_modes}/PPO_{PPO_number}/system_parameters.csv"
        parameters = np.genfromtxt(parameters_path, delimiter=",", names=True, dtype=None, encoding='UTF-8')
        global SQUIRMER_RADIUS
        SQUIRMER_RADIUS = parameters["squirmer_radius"]
        spawn_radius = parameters["spawn_radius"]
        max_mode = parameters["max_mode"]
        viscosity = parameters["viscosity"]
        spawn_angle = parameters["spawn_angle"]
        model_path = f"RL/Training/Logs_zhu/{cap_modes}/PPO_{PPO_number}/ppo_predator_prey"
        
        model = PPO.load(model_path)
        env = PredatorPreyEnv(SQUIRMER_RADIUS, spawn_radius, max_mode, viscosity, cap_modes, spawn_angle, render_mode=None, squirmer_frame=False)
        
        # Empty arrays for loop
        B_vals = []    
        Bt_vals = []
        time_vals = []
        agent_coord = []
        target_coord = []
        
        # Run model
        obs = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs)
            obs, reward, done, info = env.step(action)
            modes = info["modes"]
            B_vals.append(modes[0])
            Bt_vals.append(modes[1])
            time_vals.append(info["time"])
            agent_coord.append(info["agent"])
            target_coord.append(info["target"])
        
        return np.array(B_vals), np.array(Bt_vals), np.array(time_vals), np.array(agent_coord), np.array(target_coord)

        
    def fill_position_axis(axis, coord_agent, coord_target):
        color_target = cm.Reds(np.linspace(0, 1, len(coord_agent[:, 0])))  # Colors gets darker with time. 
        color_agent = cm.Blues(np.linspace(0, 1, len(coord_agent[:, 0])))
        
        # Plot agent
        for i in range(len(coord_agent)):
            pass
            circle = plt.Circle(coord_agent[i, :], SQUIRMER_RADIUS, facecolor="none", edgecolor=color_agent[i], fill=False)
            axis.add_patch(circle)
        # Plot target
        axis.scatter(coord_target[:, 0], coord_target[:, 1], s=2, c=color_target)            
        
        axis.set(xlabel=r"$y$", yticks=[], xlim=(-xy_width, xy_width), ylim=(-xy_width, xy_width))# xlim=(0, 5), ylim=(-3, 2))
        # Making a good looking legend
        agent_legend_marker = mlines.Line2D(xdata=[], ydata=[], marker=".", markersize=12, linestyle="none", fillstyle="none", color=color_agent[-1], label="Squirmer")
        target_legend_marker = mlines.Line2D(xdata=[], ydata=[], marker=".", markersize=12, linestyle="none", color=color_target[-2], label="Target")
        axis.legend(handles=[agent_legend_marker, target_legend_marker], fontsize=6)
    
    
    def fill_mode_axis(axis, time, B, Bt):
        axis.plot(time, B, "--.", label=r"$B_{01}$", color="green")
        axis.plot(time, Bt, "--.", label=r"$\tilde{B}_{11}$", color="blue")
        axis.legend(fontsize=6)
        axis.set(xlabel="Time", ylim=(-1.1, 1.1), yticks=[])
        
        
    fig, ax = plt.subplots(nrows=2, ncols=len(PPO_list), dpi=200)        
    for i in range(len(PPO_list)):
        B, Bt, time, agent_coord, target_coord = run_model(PPO_list[i])
        fill_position_axis(ax[0, i], agent_coord, target_coord)
        fill_mode_axis(ax[1, i], time, B, Bt)      
    
    ax[0, 0].set(ylabel=r"$z$")
    ax[0, 0].set_yticks(ticks=np.arange(-xy_width, xy_width+1, 4))
    ax[1, 0].set(ylabel=r"Mode Values")
    ax[1, 0].set_yticks(ticks=[-1, 0, 1])
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":  
    def check_model(squirmer_radius, spawn_radius, max_mode, viscosity, cap_modes, spawn_angle, render_mode, scale_canvas):
        env = PredatorPreyEnv(squirmer_radius, spawn_radius, max_mode, viscosity, cap_modes, spawn_angle, render_mode, scale_canvas)
        print("-- SB3 CHECK ENV: --")
        if check_env(env) == None:
            print("   The Environment is compatible with SB3")
        else:
            print(check_env(env))
    
    
    # Parameters
    squirmer_radius = 1
    spawn_radius = 4.5
    max_mode = 2 
    viscosity = 1
    cap_modes = "uncapped"  # Options: "uncapped", "constant",
    spawn_angle = - np.pi / 4

    check_model(squirmer_radius, spawn_radius, max_mode, viscosity, cap_modes, spawn_angle, render_mode=None, scale_canvas=1)

# If wants to see reward over time, write the following in cmd in the log directory
# tensorboard --logdir=.
