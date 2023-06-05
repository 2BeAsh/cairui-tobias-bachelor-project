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
zhu.train(squirmer_radius, spawn_angle_zhu, max_mode_zhu, viscosity, cap_modes_zhu, spawn_angle_zhu, train_total_steps_zhu)
zhu.pygame_animation(PPO_number_zhu, render_mode, scale_canvas)
zhu.path_mode_plot(PPO_list_zhu)

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
#direction.train(N_surface_points, squirmer_radius, target_radius, max_mode_direction, sensor_noise, viscosity, target_initial_position_direction, reg_offset, coord_plane, target_initial_position_direction, subfolder="angle")

# Center distance train
#direction.train(N_surface_points, squirmer_radius, target_radius, max_mode_direction, sensor_noise, viscosity, target_initial_position_direction, reg_offset, coord_plane, target_initial_position_direction, subfolder="center_distance")

# Target Radius train
#direction.train(N_surface_points, squirmer_radius, target_radius, max_mode_direction, sensor_noise, viscosity, target_initial_position_direction, reg_offset, coord_plane, target_initial_position_direction, subfolder="target_radius")

# Sensor noise train
#direction.train(N_surface_points, squirmer_radius, target_radius, max_mode_direction, sensor_noise, viscosity, target_initial_position_direction, reg_offset, coord_plane, target_initial_position_direction, subfolder="sensor_noise")

#direction.mode_choice_plot(max_mode_direction, N_iter=10, PPO_number=1, subfolder="target_radius")
#direction.mode_iteration_average_plot(max_mode_direction, N_model_runs=10, PPO_list=[1, 2, 3], changed_parameter="target_radius", plot_reward=True, subfolder="target_radius")
#Changed parameter: "target_radius", "noise", "position", "angle", "else"
# tensorboard --logdir=.