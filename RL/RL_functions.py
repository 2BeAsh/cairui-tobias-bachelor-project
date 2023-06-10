"""This file is meant for running the functions in the other RL files."""
import numpy as np
import sys

# Own files
import pp_predict_direction as direction
import pp_sb3_zhu2022 as zhu
sys.path.append('./Fluid')
import field_velocity
import power_consumption
import bem_two_objects
import bem

# --- Direction Prediction function definitions ---
# --- Run the code ---
# - Common Parameters -
squirmer_radius = 1
viscosity = 1

# -- Zhu parameters --
spawn_radius_zhu = 6
max_mode_zhu = 1
cap_modes_zhu = "constant"  # Options: "uncapped", "constant",
spawn_angle_zhu = - np.pi / 4
render_mode = "human"
scale_canvas = 1.4  # Makes everything on the canvas a factor smaller / zoomed out
train_total_steps_zhu = int(4e5)
PPO_number_zhu = 2
PPO_list_zhu = [4, 3, 1, 2]

# -- Zhu functions --
#zhu.train(squirmer_radius, spawn_radius_zhu, max_mode_zhu, viscosity, cap_modes_zhu, spawn_angle_zhu, train_total_steps_zhu, squirmer_frame=False)
#zhu.pygame_animation(PPO_number_zhu, cap_modes_zhu, render_mode, scale_canvas)
#zhu.path_mode_plot(PPO_list_zhu, cap_modes_zhu)

# -- Predict direction parameters --
# Model Parameters
N_surface_points = 700
target_radius = 0.4
target_initial_position = [2, 2] / np.sqrt(2)
distance = 2
max_mode_direction = 2
sensor_noise = 0.18
reg_offset = 0.05
coord_plane = "xz"
train_total_steps_direction = int(3.5e5)

# Variables
target_radius_variable = 0.6
target_initial_position_variable = [2.2, 2.2] / np.sqrt(2)
angle_variable = 3*np.pi/4
sensor_noise_variable = 0.14

# - Predict direction functions -
# Angle train
direction.train(N_surface_points, squirmer_radius, target_radius, max_mode_direction, sensor_noise, viscosity, distance * np.array([np.sin(angle_variable), np.cos(angle_variable)]), reg_offset, coord_plane, train_total_steps_direction, subfolder="angle")

# Center distance train - Cairui
#direction.train(N_surface_points, squirmer_radius, target_radius_variable, max_mode_direction, sensor_noise, viscosity, target_initial_position_variable, reg_offset, coord_plane, train_total_steps_direction, subfolder="center_distance")
#direction.train(N_surface_points, squirmer_radius, target_radius_variable, max_mode_direction, sensor_noise, viscosity, target_initial_position_variable, reg_offset, coord_plane, train_total_steps_direction, subfolder="center_distance")

# Target Radius train - Tobias Stationær
#direction.train(N_surface_points, squirmer_radius, target_radius_variable, max_mode_direction, sensor_noise, viscosity, target_initial_position, reg_offset, coord_plane, train_total_steps_direction, subfolder="target_radius")

# Sensor noise train -cx Tobias Bærbar
#direction.train(N_surface_points, squirmer_radius, target_radius, max_mode_direction, sensor_noise_variable, viscosity, target_initial_position, reg_offset, coord_plane, train_total_steps_direction, subfolder="sensor_noise")

# - Plot - 
# Guide til plot:
    # 1) Vælg hvilken parameter som bruges i changed_parameter. subfolder er altid lig changed_parameter og skal ikke ændres
    # 2) Vælg hvilke filer skal bruges i PPO_list
    # 3) Hvis plot_modes_one_graph bruges, skal indices angives. Dette er hvilke grafer der skal plottes, så B_idx=[0, 2] betyder B01 og B02 plottes (fordi B modes er B01, B11, B02 ...)

changed_parameter = "angle"  #Changed parameter: "target_radius", "sensor_noise", "distance", "angle", "else"
subfolder = changed_parameter  # Does not work for "else"
PPO_list = [12,13,14,15,16,17,18,]
#direction.mode_choice_plot(max_mode_direction, N_iter=10, PPO_number=1, subfolder=subfolder)
direction.mode_iteration_average_plot(max_mode_direction, N_model_runs=10, PPO_list=PPO_list, changed_parameter=changed_parameter, plot_reward=True, subfolder=subfolder)
direction.plot_modes_one_graph(B_idx=[0, 1, 4], Bt_idx=[0, 1], C_idx=[0], Ct_idx=[], 
                               max_mode=max_mode_direction, N_model_runs=10, PPO_list=PPO_list, 
                               changed_parameter=changed_parameter, subfolder=subfolder)

# tensorboard --logdir=.