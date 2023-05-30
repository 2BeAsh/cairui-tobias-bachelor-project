import numpy as np
import matplotlib.pyplot as plt
import bem

# Parameters
viscosity = 1
squirmer_radius = 1
max_mode = 2

# Mode array
B = np.zeros((max_mode+1, max_mode+1))
Bt = np.zeros_like(B)
C = np.zeros_like(B)
Ct = np.zeros_like(B)
B[0, 1] = 1
B[1, 1] = 1
Bt[1, 1] = 1
mode_array = np.array([B, Bt, C, Ct])
B01 = mode_array[0, 0, 1]
B11 = mode_array[0, 1, 1]
Bt11 = mode_array[1, 1, 1]


def velocity_difference(N_surface_points, regularization_offset):
    """Calculate procentwise difference between analytical center of mass velocity and numerical center of mass found with the Oseen tensor"""
    # Analytical center of mass velocity
    u_anal = 4 / (3 * squirmer_radius ** 3) * np.array([B11, Bt11, -B01]) 
    
    # Numerical center of mass velocity
    force = bem.force_on_sphere(N_surface_points, max_mode, squirmer_radius, mode_array, regularization_offset, viscosity, lab_frame=True)
    u_num = force[-6:-3]
    
    # Procentwise difference
    p_diff = (u_num - u_anal) / u_anal * 100
    # print("U analytisk og Numerisk, N=", N_surface_points)
    # print(u_anal)
    # print(u_num)
    # print("")
    return p_diff


def plot_N_comparison(N_surface_points_list, regularization_offset):
    # Get data
    p_diff_arr = np.empty((len(N_surface_points_list), 3))
    for i, N in enumerate(N_surface_points_list):
        p_diff_arr[i, :] = velocity_difference(N, regularization_offset)

    # Plot
    fig, ax = plt.subplots(dpi=200)
    ax.plot(N_surface_points_list, p_diff_arr, ".--")
    ax.set(xlabel=r"$N$", ylabel=r"$U_{err} (\%)$", title=f"Regularization offset = {regularization_offset}, Squirmer radius = {squirmer_radius}, Viscosity = {viscosity}")
    ax.legend(labels=[r"$U_x$", r"$U_y$", r"$U_z$"])
    #ax.text(0.1, 0.9, s=f"Regularization offset = {regularization_offset}", transform=ax.transAxes)
    fig.tight_layout()
    plt.show()
    

def plot_regularization_comparison(regularization_offset_list, N_surface_points):
    # Get data
    p_diff_arr = np.empty((len(regularization_offset_list), 3))
    for i, eps in enumerate(regularization_offset_list):
        p_diff_arr[i, :] = velocity_difference(N_surface_points, eps)

    # Plot
    fig, ax = plt.subplots(dpi=200)
    ax.plot(regularization_offset_list, p_diff_arr, ".--")
    ax.axhline(y=0, ls="dashed", alpha=0.7, color="grey")
    ax.set(xlabel=r"Regularization offset", ylabel=r"$U_{err} (\%)$", title=f"N = {N_surface_points}, Squirmer radius = {squirmer_radius}, Viscosity = {viscosity}")
    ax.legend(labels=[r"$U_x$", r"$U_y$", r"$U_z$"])
    fig.tight_layout()
    plt.show()
    
    
if __name__ == "__main__":
    # Plot N
    N_values = np.arange(50, 330, 20)
    reg_offset = 0.79
    plot_N_comparison(N_values, reg_offset)
    
    # Plot regu
    N = 80
    eps_vals = np.linspace(0.0001, 1.5, 15) 
    plot_regularization_comparison(eps_vals, N)
