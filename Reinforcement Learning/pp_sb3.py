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
import matplotlib.cm as cm
import matplotlib.lines as mlines

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
import power_factor as pf


# Environment
class PredatorPreyEnv(gym.Env):
    """Gym environment for a predator-prey system in a fluid. The frame of reference is """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}


    def __init__(self, squirmer_radius, spawn_radius, legendre_modes, scale_canvas=1, const_angle=None, cap_modes=True, lab_frame=True, render_mode=None):
        #super().__init__() - ingen anelse om hvorfor jeg havde skrevet det eller hvor det kommer fra?
        # -- Variables --
        # Model
        self.squirmer_radius = squirmer_radius  # The catch radius and more is based on this
        self.spawn_radius = spawn_radius  # Max distance the target can be spawned away from the agent
        self.const_angle = const_angle
        self.cap_modes = cap_modes  # If True, will have sum(modes) <= 1, if False: uncapped, if "constant" will have sum(modes)=1
        self.legendre_modes = legendre_modes - 1  # Max available legendre modes. The minus 1 comes from different definitons. 
        self.lab_frame = lab_frame  # Choose lab or squirmer frame of reference. If True, in lab frame otherwise in squirmer
        self.B_max = 1
        self.charac_velocity = 4 * self.B_max / (3 * self.squirmer_radius ** 3)  # Characteristic velocity: Divide velocities by this to remove dimensions
        self.charac_time = 3 * self.squirmer_radius ** 4 / (4 * self.B_max) # characteristic time
        tau = 0.3  # Seconds per iteration. 
        self.dt = tau / self.charac_time
        
        # Rendering
        self.scale_canvas = scale_canvas  # Makes canvas this times smaller
        self.window_size = 512  # PyGame window size
        assert render_mode is None or render_mode in self.metadata["render_modes"]  # Check if given render_mode matches available
        self.render_mode = render_mode
        self.window = None
        self.clock = None

        # -- Define action and observation space --
        # Actions: Strength of Legendre Modes
        assert cap_modes in [True, False, "constant"]
        if cap_modes:  # Capped, sum(modes) <= 1
            action_shape=(3,)  # 3 parameters that determines strength of modes
        elif cap_modes == "constant":  # Capped, sum(modes) = 1
            action_shape = (1,)  # alpha, distribution of the two modes
        else:  # Uncapped
            action_shape = (2,)  # The two modes
        self.action_space = spaces.Box(low=-1, high=1, shape=action_shape, dtype=np.float32)

        # Observations: distance and angle between target and agent. Angle is measured from the vertical axis
        self.observation_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)  # (distance, angle). 


    def _array_float(self, x, shape):
        """Helper function to input x into a shape sized array with dtype np.float32"""
        return np.array([x], dtype=np.float32).reshape(shape)


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
        upper_dist = 1.5 * self.spawn_radius
        r_unit = (r - upper_dist / 2) / upper_dist
        theta_unit = theta / np.pi
        return self._array_float([r_unit, theta_unit], shape=(2,))        


    def _reward_time_optimized(self):
        r = self._get_dist()
        d0 = np.linalg.norm(self.initial_target_position, ord=2)  # Time it takes to move from initial position to target if travelling in a straight line, in time units
        too_far_away = r > self.spawn_radius
        captured = r < self.catch_radius
        done = False
        # Fix d0, evt. bare fjern
        if too_far_away:  # Stop simulation and penalize hard if goes too far away
            gamma = -1000
            done = True
        elif captured:  # Believe there is a problem with self.time - d0, as sometimes get massive negative reward when catches.
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

        # Target starts random location not within a factor times catch radius. catch_radius is a factor of squirmer_radius
        self.epsilon = 0.05  # Extra catch distance
        self.catch_radius = self.squirmer_radius + self.epsilon  # Velocities blow up near the squirmer.

        if self.const_angle is None:
            self._target_position = self._agent_position  
            dist = self._get_dist()
            while dist <= 2 * self.catch_radius:  # While distance between target and agent is too small, find a new initial position for the target
                initial_distance = np.random.uniform(low=0, high=self.spawn_radius)
                initial_angle = np.random.uniform(-1, 1) * np.pi
                self._target_position = self._array_float([initial_distance * np.cos(initial_angle), initial_distance * np.sin(initial_angle)], shape=(2,))
                dist = self._get_dist()  # Update distance
        
        else:  # In case wants specific starting location
            self._target_position = self._array_float([self.spawn_radius * np.cos(self.const_angle), self.spawn_radius * np.sin(self.const_angle)], shape=(2,))

        self.initial_target_position = 1 * self._target_position  # Needed for reward calculation

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
        if self.cap_modes:
            B_01, B_tilde_11  = (action[1] + 1) / 2 * np.sin(action[0] * np.pi), (action[2] + 1) / 2 * np.cos(action[0] * np.pi)
            mode_info = [B_01, B_tilde_11]
        elif self.cap_modes == "constant":
            B_01 = np.sin(action * np.pi)
            B_tilde_11 = np.cos(action * np.pi)
            mode_inf0 = [B_01, B_tilde_11]
        else:
            B_01 = action[0] / self.B_max
            B_tilde_11 = action[1] / self.B_max        
            mode_info = [B_01, B_tilde_11]
        
        # -- Movement --
        # Convert to polar coordinates and get the cartesian velocity of the flow.

        r, theta = self._cartesian_to_polar()
        B = np.zeros((self.legendre_modes+1, self.legendre_modes+1))
        B_tilde = np.zeros_like(B)
        C = np.zeros_like(B)
        C_tilde = np.zeros_like(B)
        B[0, 1] = B_01
        B_tilde[1, 1] = B_tilde_11
        velocity_y, velocity_z = fm.field_cartesian(N=self.legendre_modes, r=r, 
                                                    theta=theta, a=self.squirmer_radius, 
                                                    B=B, B_tilde=B_tilde, 
                                                    C=C, C_tilde=C_tilde, 
                                                    lab_frame=self.lab_frame)
        velocity = np.array([velocity_y, velocity_z], dtype=np.float32) / self.charac_velocity
        self._target_position = self._target_position + velocity * self.dt 
        
        if self.lab_frame:
            squirmer_velocity = np.array([B_tilde_11, -B_01], dtype=np.float32) 
            self._agent_position = self._agent_position + squirmer_velocity  * self.dt 
            
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
        return position + self._array_float([self.spawn_radius, self.spawn_radius], shape=(2,)) * self.scale_canvas


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
            float(pix_size * self.epsilon)  # Radius
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


def check_model(squirmer_radius, spawn_radius, legendre_modes, scale_canvas, start_angle, cap_modes, lab_frame):
    env = PredatorPreyEnv(squirmer_radius, spawn_radius, legendre_modes, scale_canvas, start_angle, cap_modes, lab_frame)
    print("-- SB3 CHECK ENV: --")
    if check_env(env) == None:
        print("   The Environment is compatible with SB3")
    else:
        print(check_env(env))


def train(squirmer_radius, spawn_radius, legendre_modes, scale_canvas, start_angle, cap_modes, lab_frame, train_total_steps):
    env = PredatorPreyEnv(squirmer_radius, spawn_radius, legendre_modes, scale_canvas, start_angle, cap_modes, lab_frame)

    # Train with SB3
    log_path = os.path.join("Reinforcement Learning", "Training", "Logs")
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_path)
    model.learn(total_timesteps=train_total_steps)
    model.save("ppo_predator_prey")


def show_result(squirmer_radius, spawn_radius, legendre_modes, scale_canvas, start_angle, cap_modes, lab_frame, render_mode):
    """Arguments should match that of the loaded model for correct results"""
    # Load model and create environment
    model = PPO.load("ppo_predator_prey")
    env = PredatorPreyEnv(squirmer_radius, spawn_radius, legendre_modes, scale_canvas, start_angle, cap_modes, lab_frame, render_mode)

    # Run and render model
    obs = env.reset()
    done = False
        
    while not done:
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)
        env.render()
        
    env.close()
        

def plot_info(squirmer_radius, spawn_radius, legendre_modes, scale_canvas, start_angle, cap_modes, lab_frame):
    """Arguments should match that of the loaded model for correct results"""
    # Load model and create environment
    model = PPO.load("ppo_predator_prey")
    env = PredatorPreyEnv(squirmer_radius, spawn_radius, legendre_modes, scale_canvas, start_angle, cap_modes, lab_frame)

    # Run and render model
    obs = env.reset()
    done = False
    
    mode_list = []
    time_list = []
    target_pos_list = []
    agent_pos_list = []
    reward_list = []
    
    while not done:
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)        
        mode_list.append(info["modes"])
        time_list.append(info["time"])
        target_pos_list.append(info["target"])
        agent_pos_list.append(info["agent"])
        reward_list.append(reward)
        
    modes = np.array(mode_list)
    time = np.array(time_list)
    target_pos = np.array(target_pos_list)
    agent_pos = np.array(agent_pos_list)
    reward_arr = np.array(reward_list)
    
    # Plot mode values over time
    fig_mode, ax_mode = plt.subplots(dpi=150, figsize=(8, 6))
    ax_mode.plot(time, modes, "--.")
    ax_mode.set(xlabel="Time", ylabel="Mode values", title="Mode values against time")
    ax_mode.legend([r"$B_{01}$", r"$\tilde{B}_{11}$"])
    fig_mode.tight_layout()
    plt.show()
    plt.close()
    
    # Plot target and agent position over time
    fig_pos, ax_pos = plt.subplots(dpi=150, figsize=(3, 6))
    color_target = cm.Reds(np.linspace(0, 1, len(target_pos[:, 0])))  # Colors gets darker with time. 
    color_agent = cm.Blues(np.linspace(0, 1, len(agent_pos[:, 0])))
    ax_pos.scatter(target_pos[:, 0], target_pos[:, 1], s=15, label="Target", facecolor="none", edgecolors=color_target)
    ax_pos.scatter(agent_pos[:, 0], agent_pos[:, 1], s=8000, label="Agent", facecolor="none", edgecolors=color_agent)
    ax_pos.set(xlabel="x", ylabel="y", xlim=(-1, 1), title="Agent and target position over time")
    # Making a good looking legend
    agent_legend_marker = mlines.Line2D(xdata=[], ydata=[], marker=".", markersize=10, linestyle="none", fillstyle="none", color="navy", label="Agent")
    target_legend_marker = mlines.Line2D(xdata=[], ydata=[], marker=".", markersize=5, linestyle="none", fillstyle="none", color="firebrick", label="Target")
    ax_pos.legend(handles=[agent_legend_marker, target_legend_marker])
    fig_pos.tight_layout()
    plt.show()
    plt.close()
                
    # Plot reward over time
    fig_reward, ax_reward = plt.subplots(dpi=150, figsize=(8, 6))
    ax_reward.plot(time, reward_arr, ".--", label="Reward")
    ax_reward.set(xlabel="Time", ylabel="Reward", title="Agent Reward against time")
    ax_reward.legend()
    
    fig_reward.tight_layout()     
    plt.show()

# -- Run the code --
# Parameters
squirmer_radius = 1
spawn_radius = 5
legendre_modes = 2  # DOES NOT WORK FOR >2
scale_canvas = 1.4  # Makes everything factor smaller / zoomed out
start_angle = np.pi/ 2
cap_modes = True  # True, False, "constant"
lab_frame = True

train_total_steps = int(8e5)
render_mode = "human"

#check_model(squirmer_radius, spawn_radius, legendre_modes, scale_canvas, start_angle, cap_modes, lab_frame)
#train(squirmer_radius, spawn_radius, legendre_modes, scale_canvas, start_angle, cap_modes, lab_frame, train_total_steps)
#show_result(squirmer_radius, spawn_radius, legendre_modes, scale_canvas, start_angle, cap_modes, lab_frame, render_mode)
plot_info(squirmer_radius, spawn_radius, legendre_modes, scale_canvas, start_angle, cap_modes, lab_frame)


# tensorboard --logdir=.
