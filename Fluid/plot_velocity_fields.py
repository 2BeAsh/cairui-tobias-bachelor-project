import numpy as np
import field_velocity
import matplotlib.pyplot as plt 
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Parameters and Mode setup
a = 1
max_length = 2  # Points radius
max_mode = 1
B = np.zeros((max_mode+1, max_mode+1))
B_tilde = np.zeros_like(B)
C = np.zeros_like(B)
C_tilde = np.zeros_like(B)
B[0, 1] = 1
#B_tilde[1, 1] = 1
mode_array = np.array([B, B_tilde, C, C_tilde])

def generate_velocity_field(mode_array, max_length):
    # Coordinates
    N_points = 100
    z = np.linspace(-max_length, max_length, N_points)
    y = np.linspace(-max_length, max_length, N_points)
    Y, Z = np.meshgrid(y, z)
    R = np.sqrt(Z**2 + Y**2)
    Theta = np.arctan(Y/Z)
    phi = np.pi / 2

    # Loop through each 
    u_y = np.empty((N_points, N_points))
    u_z = np.empty_like(u_y)
    for i in range(N_points):
        for j in range(N_points):
            _, u_y_val, u_z_val = field_velocity.field_cartesian(max_mode, R[i,j], Theta[i,j], phi, a, mode_array)
            u_y[i, j] = u_y_val
            u_z[i, j] = u_z_val

    # Remove velocity values inside squirmer
    R = R.flatten()
    u_y[np.where(R<a)]=0
    u_z[np.where(R<a)]=0
    velocity_magnitude = np.sqrt(u_y ** 2 + u_z ** 2)
    return Y, Z, u_y, u_z, velocity_magnitude


def plot_velocity_field(mode_array_list, max_length, title_list):
    def fill_axis(axis, mode_array, title):
        Y, Z, u_y, u_z, velocity_magnitude = generate_velocity_field(mode_array, max_length)
        axis.set(xlim=(-max_length, max_length), ylim=(-max_length, max_length),
                 xlabel="y", ylabel="z", title=title)
        axis.streamplot(Y, Z, u_y, u_z, density=1, color="k")
        contour = axis.contourf(Y, Z, velocity_magnitude)
        circle1 = plt.Circle((0, 0), a, color='white', edgecolor="k")
        axis.add_patch(circle1)
        return contour


    # -- Plot counterf and streamplot -- 
    fig, ax = plt.subplots(nrows=2, ncols=2, dpi=200)
    ax00 = ax[0, 0]
    ax01 = ax[0, 1]
    ax10 = ax[1, 0]
    ax11 = ax[1, 1]

    # Plot on each axis
    fill_axis(ax00, mode_array_list[0], title_list[0])
    contour_ax01 = fill_axis(ax01, mode_array_list[1], title_list[1])
    fill_axis(ax10, mode_array_list[2], title_list[2])
    contour_ax11 = fill_axis(ax11, mode_array_list[3], title_list[3])
    
    # Remove unnecessary ticks
    ax00.set(xticks=[])
    ax01.set(xticks=[], yticks=[])
    ax11.set(yticks=[])
    
    # Colorbars
    divider_ax01 = make_axes_locatable(ax01)
    divider_ax11 = make_axes_locatable(ax11)
    cax01 = divider_ax01.append_axes("right", size="5%", pad=0.05)
    cax11 = divider_ax11.append_axes("right", size="5%", pad=0.05)
    cbar_ax01 = plt.colorbar(contour_ax01, cax=cax01)
    cbar_ax11 = plt.colorbar(contour_ax11, cax=cax11)
    cbar_ax01.set_label("Velocity strength")
    cbar_ax11.set_label("Velocity strength")
    
    #cbar_ax01 = plt.colorbar(contour_ax01)
    #cbar_ax11 = plt.colorbar(contour_ax11)
    plt.show()