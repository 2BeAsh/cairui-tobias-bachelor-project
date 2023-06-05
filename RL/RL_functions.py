"""This file is meant for running the functions in the other RL files."""
import numpy as np
import sys
import os
import csv

# Matplotlib
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.lines as mlines

# Stable baselines 3
from stable_baselines3 import PPO


# Own files
import pp_predict_direction as direction
import pp_sb3_zhu2022 as zhu
sys.path.append('./Fluid')
import field_velocity
import power_consumption
import bem_two_objects
import bem


# --- Zhu function definitions ---
def zhu_train(squirmer_radius, spawn_radius, max_mode, viscosity, cap_modes, spawn_angle, train_total_steps):
    log_path = os.path.join("RL", "Training", "Logs_zhu")
    env = zhu.PredatorPreyEnv(squirmer_radius, spawn_radius, max_mode, viscosity, cap_modes, spawn_angle, render_mode=None)
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


def zhu_pygame_animation(PPO_number, render_mode, scale_canvas):
    """Show Pygame animation"""
    # Load model and create environment
    parameters_path = f"RL/Training/Logs_zhu/PPO_{PPO_number}/system_parameters.csv"
    parameters = np.genfromtxt(parameters_path, delimiter=",", names=True, dtype=None, encoding='UTF-8') #=[np.float32, np.float32, int, np.float32, str, np.float32, bool, int])
    squirmer_radius = parameters["squirmer_radius"]
    spawn_radius = parameters["spawn_radius"]
    max_mode = parameters["max_mode"]
    viscosity = parameters["viscosity"]
    cap_modes = parameters["mode_cap"]
    spawn_angle = parameters["spawn_angle"]
    model_path = f"RL/Training/Logs_zhu/PPO_{PPO_number}/ppo_predator_prey"
    
    model = PPO.load(model_path)
    env = zhu.PredatorPreyEnv(squirmer_radius, spawn_radius, max_mode, viscosity, cap_modes, spawn_angle, render_mode=render_mode, scale_canvas=scale_canvas)

    # Run and render model
    obs = env.reset()
    done = False
        
    while not done:
        action, _ = model.predict(obs)
        obs, _, done, _ = env.step(action)
        env.render()
        
    env.close()


def zhu_path_mode_plot(PPO_list):
    """Plot path taken by both predator and prey, then in seperate plot show the mode values over time."""
    # ---- NOTE TO DO ----
    # One legend for each row, placed to the right
    
    # For each entry in PPO_list, find the squirmer and target positions, and modes values over time. Plot them on seperate axis
    xy_width = 5
    
    def run_model(PPO_number):
        # Load parameters and model, create environment
        parameters_path = f"RL/Training/Logs_zhu/PPO_{PPO_number}/system_parameters.csv"
        parameters = np.genfromtxt(parameters_path, delimiter=",", names=True, dtype=None, encoding='UTF-8')
        squirmer_radius = parameters["squirmer_radius"]
        spawn_radius = parameters["spawn_radius"]
        max_mode = parameters["max_mode"]
        viscosity = parameters["viscosity"]
        cap_modes = parameters["mode_cap"]
        spawn_angle = parameters["spawn_angle"]
        model_path = f"RL/Training/Logs_zhu/PPO_{PPO_number}/ppo_predator_prey"
        
        model = PPO.load(model_path)
        env = zhu.PredatorPreyEnv(squirmer_radius, spawn_radius, max_mode, viscosity, cap_modes, spawn_angle, render_mode=None)
        
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
            circle = plt.Circle(coord_agent[i, :], squirmer_radius, facecolor="none", edgecolor=color_agent[i], fill=False)
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



# --- Direction Prediction function definitions ---
def direction_train(N_surface_points, squirmer_radius, target_radius, max_mode, sensor_noise, viscosity, target_initial_position, reg_offset, coord_plane, train_total_steps, subfolder=None):
    env = direction.PredictDirectionEnv(N_surface_points, squirmer_radius, target_radius, max_mode, sensor_noise, viscosity, target_initial_position, reg_offset, coord_plane)

    # Train with SB3
    log_path = os.path.join("RL", "Training", "Logs_direction")
    if subfolder != None:
        log_path = os.path.join(log_path, subfolder)
    model_path = os.path.join(log_path, "predict_direction")
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_path)
    model.learn(total_timesteps=train_total_steps)
    model.save(model_path)
    
    # Save parameters in csv file
    file_path = os.path.join(log_path, "system_parameters.csv")
    with open(file_path, mode="w") as file:
        writer = csv.writer(file, delimiter=",")
        writer.writerow(["surface_points", "squirmer_radius", "target_radius", "max_mode ", "sensor_noise", 
                         "target_x1 ", "target_x2 ", "centers_distance", "viscosity", "regularization_offset", "coordinate_plane", "train_steps"])
        writer.writerow([N_surface_points, squirmer_radius, target_radius, max_mode, sensor_noise, target_initial_position[0], target_initial_position[1],
                         np.linalg.norm(target_initial_position, ord=2), viscosity, reg_offset, coord_plane, train_total_steps])
        

def mode_names(max_mode):
    """Get the name of the modes given the max mode in strings."""
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


def direction_mode_iteration(N_iter, PPO_number, mode_lengths, subfolder=None):
    """Run environment N_iter times with training data from PPO_number directory."""
    # Load parameters and model, create environment
    if subfolder != None:
        parameters_path = f"RL/Training/Logs_direction/{subfolder}/PPO_{PPO_number}/system_parameters.csv"
        model_path = f"RL/Training/Logs_direction/{subfolder}/PPO_{PPO_number}/predict_direction"
    else:
        parameters_path = f"RL/Training/Logs_direction/PPO_{PPO_number}/system_parameters.csv"
        model_path = f"RL/Training/Logs_direction/PPO_{PPO_number}/predict_direction"

    parameters = np.genfromtxt(parameters_path, delimiter=",", names=True, dtype=None, encoding='UTF-8')
    N_surface_points = int(parameters["surface_points"])
    squirmer_radius = parameters["squirmer_radius"]
    target_radius = parameters["target_radius"]
    max_mode = parameters["max_mode"]
    sensor_noise = parameters["sensor_noise"]
    target_x1 = parameters["target_x1"]
    target_x2 = parameters["target_x2"]
    viscosity = parameters["viscosity"]
    reg_offset = parameters["regularization_offset"]
    coord_plane = parameters["coordinate_plane"]
    
    if coord_plane not in ["xy", "yz", "xz", None]:  # Backwards compatability when only yz plane was allowed
        coord_plane = "yz"
    
    model = PPO.load(model_path)
    env = direction.PredictDirectionEnv(N_surface_points, squirmer_radius, target_radius, max_mode, sensor_noise, viscosity, np.array([target_x1, target_x2]), reg_offset, coord_plane)
    
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


def direction_mode_choice_plot(N_iter, PPO_number, subfolder=None):
    """Plot the modes taken at different iterations."""
    # Add more colors
    matplotlib.rcParams["axes.prop_cycle"] = matplotlib.cycler(color=['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 
                                                        'purple', 'pink', 'brown', 'orange', 'teal', 'coral', 'lightblue', 
                                                        'lime', 'lavender', 'turquoise', 'darkgreen', 'tan', 'salmon', 'gold'])

    # Names
    B_names, B_tilde_names, C_names, C_tilde_names = mode_names(max_mode_direction)
    mode_lengths = [len(B_names), len(B_tilde_names), len(C_names), len(C_tilde_names)]
    B_actions, B_tilde_actions, C_actions, C_tilde_actions, rewards, guessed_angles, parameters = direction_mode_iteration(N_iter, PPO_number, mode_lengths, subfolder)
    
    target_x1 = parameters["target_x1"]
    target_x2 = parameters["target_x2"]
    sensor_noise = parameters["sensor_noise"]
    guessed_angles = guessed_angles * 180 / np.pi
    angle = np.arctan2(target_x1, target_x2)
    
    # Plot
    def fill_axis(axis, y, marker, label, title):        
        axis.set(xticks=[], title=(title, 7), ylim=(-1, 1))
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
    figname = f"noise{parameters['sensor_noise']}_maxmode{parameters['max_mode']}_targetradius{parameters['target_radius']}_distance{parameters['centers_distance']}_trainingsteps{parameters['train_steps']}.png"            
    plt.savefig("RL/Recordings/Images/" + figname)
    plt.show()


def direction_mode_iteration_average_plot(N_model_runs, PPO_list, changed_parameter, plot_reward=True, subfolder=None):
    assert changed_parameter in ["target_radius", "noise", "position", "angle", "else"]
    B_names, B_tilde_names, C_names, C_tilde_names = mode_names(max_mode_direction)
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
    
    for i, PPO_val in enumerate(PPO_list):
        B_actions, B_tilde_actions, C_actions, C_tilde_actions, rewards, _, parameters = direction_mode_iteration(N_model_runs, PPO_val, mode_lengths, subfolder)
        # Mean and std
        B_mean[i, :] = np.mean(B_actions, axis=0)
        B_tilde_mean[i, :] = np.mean(B_tilde_actions, axis=0)
        C_mean[i, :] = np.mean(C_actions, axis=0)
        C_tilde_mean[i, :] = np.mean(C_tilde_actions, axis=0)
        
        B_std[i, :] = np.std(B_actions, axis=0) / np.sqrt(N_model_runs - 1)
        B_tilde_std[i, :] = np.std(B_tilde_actions, axis=0) / np.sqrt(N_model_runs - 1)
        C_std[i, :] = np.std(C_actions, axis=0) / np.sqrt(N_model_runs - 1)
        C_tilde_std[i, :] = np.std(C_tilde_actions, axis=0) / np.sqrt(N_model_runs - 1)
        
        reward_mean[i] = np.mean(rewards)
        reward_std[i] = np.std(rewards) / np.sqrt(N_model_runs - 1)
        
        if changed_parameter == "target_radius":
            changed_parameter_list[i] = parameters["target_radius"]  # Target radius
            xlabel = "Target Radius"
        elif changed_parameter == "noise":
            changed_parameter_list[i] = parameters["sensor_noise"]  # Sensor noise
            xlabel = "Sensor Noise"
        elif changed_parameter == "angle":
            changed_parameter_list[i] = np.arctan2(parameters["target_x1"], parameters["target_x2"]) * 180 / np.pi  # arcan ( target y / target z ). Unit: Degrees
            xlabel = "Angle"
        elif changed_parameter == "position":  # Target initial position
            changed_parameter_list[i] = parameters["centers_distance"]  # Distance between the two centers
            xlabel = "Center-center distance"
        else:
            x = parameters["centers_distance"]/(parameters["target_radius"]+ parameters["squirmer_radius"])
            changed_parameter_list[i] = x
            xlabel = "else"
            

    def fill_axis(axis, y, sy, mode_name, title):        
        x_vals = changed_parameter_list
        sort_idx = np.argsort(x_vals)
        x_sort = x_vals[sort_idx]
        axis.set(title=(title, 7), ylim=(-0.5, 0.5))
        axis.set_title(title, fontsize=7)
        for i in range(y.shape[1]):
            y_sort = y[:, i][sort_idx]
            sy_sort = sy[:, i][sort_idx]
            axis.errorbar(x_sort, y_sort, yerr=sy_sort, fmt=".--", lw=0.75)
        axis.legend(mode_name, fontsize=4, bbox_to_anchor=(1.05, 1), 
                    loc='upper left', borderaxespad=0.)
        axis.grid()
    
    
    fig, ax = plt.subplots(nrows=2, ncols=2, dpi=200)
    axB = ax[0, 0]
    axBt = ax[0, 1]
    axC = ax[1, 0]
    axCt = ax[1, 1]
    
    fill_axis(axB, np.abs(B_mean), B_std, B_names, r"$B$ modes")
    fill_axis(axBt, np.abs(B_tilde_mean), B_tilde_std, B_tilde_names, title=r"$\tilde{B}$ modes")
    fill_axis(axC, np.abs(C_mean), C_std, C_names, title=r"$C$ modes")
    fill_axis(axCt, C_tilde_mean, C_tilde_std, C_tilde_names, title=r"$\tilde{C}$ modes")
            
    # General setup
    axBt.set_yticklabels([])
    axC.set(xlabel=xlabel)
    axCt.set(xlabel=xlabel, yticklabels=[])
    fig.suptitle(fr"Average mode values over {xlabel}", fontsize=10)
    fig.tight_layout()
    
    # Save and show
    figname = f"average_modes_maxmode{max_mode_direction}_{xlabel}{changed_parameter_list}.png"
    #plt.savefig("RL/Recordings/Images/" + figname)
    plt.show()
    
    if plot_reward:
        figr, axr = plt.subplots(dpi=200)
        axr.errorbar(changed_parameter_list, reward_mean, yerr=reward_std, fmt=".")
        axr.set(xlabel=xlabel, ylabel="Reward", title="Mean reward")
        figr.tight_layout()
        plt.show()



# --- Run the code ---
# - Common Parameters -
squirmer_radius = 1
viscosity = 1

# - Zhu parameters - 
spawn_radius_zhu = 4.5
max_mode_zhu = 1
cap_modes_zhu = "uncapped"  # Options: "uncapped", "constant",
spawn_angle_zhu = - np.pi / 4
render_mode = "human"
scale_canvas = 1.4  # Makes everything on the canvas a factor smaller / zoomed out
train_total_steps_zhu = int(3.5e5)
PPO_number_zhu = 4
PPO_list_zhu = [1, 2, 3, 4]


# - Zhu functions -
#zhu_train(squirmer_radius, spawn_angle_zhu, max_mode_zhu, viscosity, cap_modes_zhu, spawn_angle_zhu, train_total_steps_zhu)
#zhu_pygame_animation(PPO_number_zhu, render_mode, scale_canvas)
#zhu_path_mode_plot(PPO_list_zhu)


# - Predict direction parameters -
# Model Parameters
N_surface_points = 1300
target_radius = 0.25
tot_radius = squirmer_radius + target_radius
target_initial_position_direction = [2, 2] / np.sqrt(2)
max_mode_direction = 2
sensor_noise = 0.05
reg_offset = 0.05
coord_plane = "yz"
train_total_steps_direction = int(1e5)

# - Predict direction functions -
# Angle train
#direction_train(N_surface_points, squirmer_radius, target_radius, max_mode_direction, sensor_noise, viscosity, target_initial_position_direction, reg_offset, coord_plane, target_initial_position_direction, subfolder="angle")

# Center distance train
#direction_train(N_surface_points, squirmer_radius, target_radius, max_mode_direction, sensor_noise, viscosity, target_initial_position_direction, reg_offset, coord_plane, target_initial_position_direction, subfolder="center_distance")

# Target Radius train
#direction_train(N_surface_points, squirmer_radius, target_radius, max_mode_direction, sensor_noise, viscosity, target_initial_position_direction, reg_offset, coord_plane, target_initial_position_direction, subfolder="target_radius")

# Sensor noise train
#direction_train(N_surface_points, squirmer_radius, target_radius, max_mode_direction, sensor_noise, viscosity, target_initial_position_direction, reg_offset, coord_plane, target_initial_position_direction, subfolder="sensor_noise")

#direction_mode_choice_plot(N_iter=10, PPO_number=1, subfolder="target_radius")
#direction_mode_iteration_average_plot(N_model_runs=10, PPO_list=[1, 2, 3], changed_parameter="target_radius", plot_reward=True, subfolder="target_radius")
#Changed parameter: "target_radius", "noise", "position", "angle", "else"