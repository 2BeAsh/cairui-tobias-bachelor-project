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
    force = bem.force_on_sphere(N_surface_points, max_mode, squirmer_radius, mode_array, regularization_offset, viscosity)
    u_num = force[-6:-3]
    
    # Procentwise difference
    p_diff = np.abs(u_anal - u_num) 
    return p_diff


def field_velocity_fractions(N_surface_points, regularization_offset):
        # -- Parametre --
        a = squirmer_radius
        N = N_surface_points  # Punkter på overflade
        eps = regularization_offset
        dA = 4 * np.pi * a ** 2 / N
                
        F_surface, coord_surface = bem.force_on_sphere(N_surface_points, max_mode, squirmer_radius, mode_array, eps, viscosity, return_points=True)
        U_num = F_surface[-6: -3]
        
        # Velocity field in points in xy plane
        point_width = 50
        x_point = np.linspace(-point_width, point_width, 100)
        y_point = 1 * x_point
        X_mesh, Y_mesh = np.meshgrid(x_point, y_point)
        X = X_mesh.ravel()        
        Y = Y_mesh.ravel()
        Z = 0 * X
        coord_eval = np.stack((X, Y, Z)).T
        r2 = np.sum(coord_eval**2, axis=1)  # For removal of points inside squirmer
        
        A_point = bem.oseen_tensor(coord_surface, coord_eval, eps, dA, viscosity)
        u_point = A_point @ F_surface
        u_point = u_point[: -6]
        u_point = np.reshape(u_point, (len(u_point)//3, 3), order="F")

        # -- Analytical field --
        theta_point = np.arctan2(np.sqrt(X ** 2 + Y ** 2), Z)
        phi_point = np.arctan2(Y, X)
        u_anal_x, u_anal_y, u_anal_z = fv.field_cartesian(max_mode, np.sqrt(r2), theta_point, phi_point, a, mode_array)

        # Remove velocities inside squirmer
        u_anal_x[r2 < a ** 2] = 0
        u_anal_y[r2 < a ** 2] = 0
        u_point[r2 < a ** 2, :] = 0
        
        # -- Mean of fractions --
        u_point_x = u_point[:, 0]
        u_point_y = u_point[:, 1]
        
        # Get non zero velocities
        u_point_x_nz = u_point_x[u_point_x != 0]
        u_point_y_nz = u_point_y[u_point_y != 0]

        u_anal_x_nz = u_anal_x[u_anal_x != 0]
        u_anal_y_nz = u_anal_y[u_anal_y != 0]

        # Compute mean of each point's fraction
        mean_frac_x = np.mean(-(u_point_x_nz-u_anal_x_nz) / u_anal_x_nz)
        mean_frac_y = np.mean(u_point_y_nz / u_anal_y_nz)
        mean_frac_arr = np.stack((mean_frac_x, mean_frac_y)).T
        
        std_frac_x = np.std(-(u_point_x_nz-u_anal_x_nz)/ u_anal_x_nz)
        std_frac_y = np.std(u_point_y_nz / u_anal_y_nz)
        std_frac_arr = np.stack((std_frac_x, std_frac_y)).T / np.sqrt(len(u_point_x_nz) - 1)
        
        # Compute analytical translatorial velocity
        U_anal = 4 / (3 * a ** 3) * np.sqrt(1**2 + 0*1**2)
        U_frac = np.sqrt(U_num[1]**2 + 0*U_num[0]**2 )/ U_anal  # Ignore z
        return U_frac, mean_frac_arr, std_frac_arr 
    

def plot_N_comparison(N_surface_points_list, regularization_offset):
    # Get data
    p_diff_arr = np.empty((len(N_surface_points_list), 2))
    for i, N in enumerate(N_surface_points_list):
         _,p_diff_arr[i, :], _= field_velocity_fractions(N, regularization_offset) #velocity_difference(N, regularization_offset) #
        

    # Plot
    fig, ax = plt.subplots(dpi=200)
    ax.plot(N_surface_points_list, p_diff_arr, ".--")
    ax.set(xlabel=r"$N$", ylabel=r"$U_{err} (\%)$", title=f"Regularization offset = {regularization_offset}, Squirmer radius = {squirmer_radius}, Viscosity = {viscosity}")
    ax.legend(labels=[r"$U_x$", r"$U_y$", r"$U_z$"])
    #ax.text(0.1, 0.9, s=f"Regularization offset = {regularization_offset}", transform=ax.transAxes)
    fig.tight_layout()
    plt.show()
    

def plot_regularization_comparison(regularization_offset_list, N_surface_points_list, log=False):
    # Start plot
    marker_list = [".", "^", "P", "x", "d"]
    color_list = ["b", "g", "r", "c", "m"]

    fig, ax = plt.subplots(dpi=200)
    ax.set(xlabel=r"Regularization offset", ylabel=r"$U_{err}^x (\%)$")
    
    # Get data
    for i, N in enumerate(N_surface_points_list):
        p_diff_arr = np.empty((len(regularization_offset_list), 1))
        for j, eps in enumerate(regularization_offset_list):
            p_diff_arr[j], _, _ = velocity_difference(N, eps)
            
        ax.plot(regularization_offset_list, p_diff_arr, marker=marker_list[i], color=color_list[i], label=fr"$N = $ {N}")
        
    # Plot
    if log:
        ax.set_xscale("log")
        ax.set_yscale("log")

    #ax.legend(labels=[r"$U_x$", r"$U_y$", r"$U_z$"])
    ax.legend()
    ax.axhline(y=0, ls="dashed", alpha=0.7, color="grey")
    fig.tight_layout()
    plt.show()


def plot_field_velocity_average(regularization_offset_list, N_surface_points_list, log=False):
    # Start figure
    marker_list = [".", "^", "P", "x", "d"]
    color_list = ["b", "g", "r", "c", "m"]
    fig, ax = plt.subplots(dpi=200)

    # Create data
    for i, N in enumerate(N_surface_points_list):
        mean_frac_arr = np.empty((len(regularization_offset_list)))
        std_frac_arr = np.empty_like(mean_frac_arr)
        U_arr = np.empty((len(regularization_offset_list)))
        for j, eps in enumerate(regularization_offset_list):
            U, mean, std = field_velocity_fractions(N, eps)
            #U_arr[j] = U[0]  # x coordinate
            mean_frac_arr[j] = mean[0]
            std_frac_arr[j] = std[0]
        # Plot
        ax.errorbar(regularization_offset_list, mean_frac_arr, yerr=std_frac_arr, fmt=marker_list[i]+"--", color=color_list[i], label=fr"$N = $ {N}")

    ax.axhline(y=1, ls="dashed", color="grey", alpha=0.5)    
    ax.set(xlabel="Regularization offset", ylabel=r"E$(u_{num}^x/u_{ana}^x)$", title=r"Average of $u_{num}^x/u_{ana}^x$ in each point")
    if log:
        ax.set_xscale("log")
        ax.set_yscale("log")
    ax.legend()
    fig.tight_layout()
    plt.show()
    
    
def cd_plot(N_surface_points_list, regularization_offset_list):
    fig, ax = plt.subplots(dpi=200)
    #fig1, ax1 = plt.subplots(dpi=200)

    marker_list = [".", "^", "P", "x", "d"]
    color_list = ["b", "g", "r", "c", "m"]

     # Get data
    for j, regularization_offset in enumerate(regularization_offset_list):
        print(j)
        p_diff_arr = np.empty((len(N_surface_points_list), 1))
        p_er_bar = np.empty((len(N_surface_points_list), 2))

        for i, N in enumerate(N_surface_points_list):
            p_diff_arr[i], _,_= field_velocity_fractions(N, regularization_offset) #velocity_difference(N, regularization_offset) #
            #print(p_er_bar[i,:])
    # Plot
        ax.plot(N_surface_points_list, p_diff_arr[:], marker = marker_list[j], color=color_list[j], label = fr"$\epsilon$ = {regularization_offset:.3f}")
        #ax.errorbar(N_surface_points_list, p_diff_arr[:, 0], yerr=p_er_bar[:, 0],marker = marker_list[j], color=color_list[j], label = fr"$\epsilon$ = {regularization_offset:.3f}")
        ax.legend(fontsize=8) #labels=[r"$U_x$", r"$U_y$", r"$U_z$"]
        ax.set(xlabel=r"$N$", title=fr"")
        ax.set_ylabel(ylabel=r"$u_{err}(\%)$", fontsize=11)
        ax.axhline(y=1, ls="dashed", color="grey", alpha=0.5)
        #ax.set_yscale('log')

        #ax.text(0.1, 0.9, s=f"Regularization offset = {regularization_offset}", transform=ax.transAxes)
        fig.tight_layout()

         #ax.plot(N_surface_points_list, p_diff_arr[:], marker = marker_list[j], color=color_list[j], label = fr"$\epsilon$ = {regularization_offset:.3f}")
        #ax1.errorbar(N_surface_points_list, p_diff_arr[:, 0], yerr=p_er_bar[:, 0],marker = marker_list[j], color=color_list[j], label = fr"$\epsilon$ = {regularization_offset:.3f}")
        #ax1.legend(fontsize=8) #labels=[r"$U_x$", r"$U_y$", r"$U_z$"]
        #ax1.set(xlabel=r"$N$", title=f"")
        #ax1.set_ylabel(ylabel=r"$u_{err}(\%)$", fontsize=11)
        #ax.set_xscale("log")
        #ax.set_yscale("log")
        #ax1.axhline(y=0, ls="dashed", color="grey", alpha=0.5)
        #ax.set_yscale('log')

        #ax.text(0.1, 0.9, s=f"Regularization offset = {regularization_offset}", transform=ax.transAxes)
        #fig1.tight_layout()
    plt.show()
    
    
if __name__ == "__main__":
    N_vals = np.array([50, 250, 500, 1000])
    #eps_vals = np.linspace(0.001, 0.1, 30)
    eps_vals = np.logspace(-4, -1, 25) 

    # Plot reg. offset
    #plot_regularization_comparison(eps_vals, N_vals, log=True)

    #Plot field strength fractions
    plot_field_velocity_average(eps_vals, N_vals, log=False)

    # Plot N
    N_values = np.logspace(1.55, 3.1, 36, dtype=np.int32) 
    reg_offset = 0.01
    #plot_N_comparison(N_values, reg_offset)
    
    # CD plot
    #eps_vals = np.array([ 0.0015, 0.002 , 0.003, 0.005, 0.008])
    #cd_plot(N_values, eps_vals)
    
