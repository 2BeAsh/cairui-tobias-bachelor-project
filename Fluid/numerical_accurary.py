import numpy as np
import matplotlib.pyplot as plt
import field_velocity as fv
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
B[0, 1] = 1 / np.sqrt(2)
B[1, 1] = 1 / np.sqrt(2)
Bt[1, 1] = 1 / np.sqrt(2)
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
    p_diff = u_num / u_anal 
    return p_diff


def field_velocity_fractions(N_surface_points, regularization_offset):
        # -- Parametre --
        a = squirmer_radius
        N = N_surface_points  # Punkter på overflade
        eps = regularization_offset
                
        # Surface points
        x_surface, y_surface, z_surface, dA = bem.canonical_fibonacci_lattice(N, a)
        coord_surface = np.stack([x_surface, y_surface, z_surface]).T
        theta = np.arctan2(np.sqrt(x_surface ** 2 + y_surface ** 2), z_surface)
        phi = np.arctan2(y_surface, x_surface)
        
        # Surface velocity in squirmer frame
        u_surface_x, u_surface_y, u_surface_z = fv.field_cartesian_squirmer(max_mode, a, theta, phi, a, mode_array)
        u_surface = np.stack([u_surface_x, u_surface_y, u_surface_z])
        u_surface = np.append(u_surface, np.zeros(6)).T  # Add Force = 0 = Torque
        
        # Surface forces
        A_surface = bem.oseen_tensor_surface(coord_surface, dA, eps, viscosity)
        F_surface = np.linalg.solve(A_surface, u_surface)
        U_num = F_surface[-6: -3]
        rot_num = F_surface[-3: ]
        
        # Velocity field arbitrary points
        point_width = 3
        x_point = np.linspace(-point_width, point_width, 30)
        y_point = 1 * x_point
        X_mesh, Y_mesh = np.meshgrid(x_point, y_point)
        X = X_mesh.ravel()        
        Y = Y_mesh.ravel()
        Z = 0 * X
        coord_eval = np.stack((X, Y, Z)).T
        r2 = np.sum(coord_eval**2, axis=1)
        
        A_point = bem.oseen_tensor(coord_surface, coord_eval, eps, dA, viscosity)
        u_point = A_point @ F_surface
        u_point = u_point[: -6]
        u_point = np.reshape(u_point, (len(u_point)//3, 3), order="F")

        # Convert to lab frame:
        u_point += U_num


        # -- Analytical field --
        theta_point = np.arctan2(np.sqrt(X ** 2 + Y ** 2), Z)
        phi_point = np.arctan2(Y, X)
        u_anal_x, u_anal_y, u_anal_z = fv.field_cartesian(max_mode, np.sqrt(r2), theta_point, phi_point, a, mode_array)
        u_anal_x[r2 < a ** 2] = 0
        u_anal_y[r2 < a ** 2] = 0

        # Remove inside squirmer
        u_point[r2 < a ** 2, :] = 0

        # -- Mean of fractions --
        u_point_x = u_point[:, 0]
        u_point_y = u_point[:, 1]
        
        u_point_x_nz = u_point_x[u_point_x != 0]
        u_point_y_nz = u_point_y[u_point_y != 0]

        u_anal_x_nz = u_anal_x[u_anal_x != 0]
        u_anal_y_nz = u_anal_y[u_anal_y != 0]

        mean_frac_x = np.mean(u_point_x_nz / u_anal_x_nz)
        mean_frac_y = np.mean(u_point_y_nz / u_anal_y_nz)
        mean_frac_arr = np.stack((mean_frac_x, mean_frac_y)).T
        
        std_frac_x = np.std(u_point_x_nz / u_anal_x_nz)
        std_frac_y = np.std(u_point_y_nz / u_anal_y_nz)
        std_frac_arr = np.stack((std_frac_x, std_frac_y)).T

        U_anal = 4 / (3 * a ** 3)
        U_frac = U_num[:2] / U_anal  # Ignore z
        return U_frac, mean_frac_arr, std_frac_arr
    

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
    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.legend(labels=[r"$U_x$", r"$U_y$", r"$U_z$"])
    fig.tight_layout()
    plt.show()



def plot_regularization_offset_field_velocity(regularization_offset_list, N_surface_points):
    mean_frac_arr = np.empty((len(regularization_offset_list), 2))
    std_frac_arr = np.empty_like(mean_frac_arr)
    U_arr = np.empty((len(regularization_offset_list), 2))

    for i, eps in enumerate(regularization_offset_list):
        U, mean, std = field_velocity_fractions(N_surface_points, eps)
        U_arr[i] = U * 100
        mean_frac_arr[i, :] = mean * 100
        std_frac_arr[i, :] = std * 100
    
    # Plot
    fig, ax = plt.subplots(dpi=200)
    for i in range(2):
        ax.errorbar(regularization_offset_list, mean_frac_arr[:, i], yerr=0, fmt=".--")
    ax.plot(regularization_offset_list, U_arr, "x--", label=r"$U_{num}/U_{anal}$")
    ax.set(xlabel="Regularization offset", ylabel="Percentage of", title=f"N = {N_surface_points}, Squirmer radius = {squirmer_radius}, Viscosity = {viscosity}")
    ax.legend([r"$E(u_{num}^x/u_{anal}^x)$", r"$E(u_{num}^y/u_{anal}^y)$", r"$U_{num}^x/U_{anal}^x$", r"$U_{num}^y/U_{anal}^y$"])
    fig.tight_layout()
    plt.show()
    
    
if __name__ == "__main__":
    # Plot regu
    N = 300
    eps_vals = np.linspace(0.01, 0.1, 25) 
    #plot_regularization_comparison(eps_vals, N)

    # Plot field strength fractions
    #plot_regularization_offset_field_velocity(eps_vals, N)

    # Plot N
    N_values = np.arange(10, 350, 10)
    reg_offset = 0.01
    plot_N_comparison(N_values, reg_offset)
    
