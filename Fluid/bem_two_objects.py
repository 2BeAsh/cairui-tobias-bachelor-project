import numpy as np
import field_velocity as fv
import boundary_element_method as bem


def difference_vectors(x_1, x_2=None):
    if x_2 is None:
        dx = x_1[:, 0][:, None] - x_1[:, 0][None, :]
        dy = x_1[:, 1][:, None] - x_1[:, 1][None, :]
        dz = x_1[:, 2][:, None] - x_1[:, 2][None, :]
    else:
        dx = x_1[:, 0][:, None] - x_2[:, 0][None, :]
        dy = x_1[:, 1][:, None] - x_2[:, 1][None, :]
        dz = x_1[:, 2][:, None] - x_2[:, 2][None, :]
    return dx, dy, dz


def torque_condition(x_stacked):
    """The torque sums to 0 condition for the ROWS. To get the column, transpose the result

    Args:
        x_stacked (N, 3)-array: array with the (x, y, z) coordinates

    Returns:
        (3, 3N)-array: Torque condition along rows.
    """
    x = x_stacked[:, 0]
    y = x_stacked[:, 1]
    z = x_stacked[:, 2]
    N = len(x)
    torque_arr = np.zeros((3, 3*N))
    torque_arr[0, N: 2*N] = -z
    torque_arr[0, 2*N: 3*N] = y
    torque_arr[1, : N] = z
    torque_arr[1, 2*N: 3*N] = -x
    torque_arr[2, : N] = -y
    torque_arr[2, N: 2*N] = x
    return torque_arr


def oseen_tensor_surface_two_objects(x_1, x_2, x1_center, x2_center, dA, regularization_offset, viscosity):
    """Find Oseen tensor for all points on the surface of the squirmer, another object and their interaction with each other

    Args:
        x_1 ((N, 3)-array): Coordinates for each point on the first object's surface centered around (0,0,0)
        x_1 ((N, 3)-array): Coordinates for each point on the second object's surface centered around (0,0,0)
        center_vector (3, 1)-array: Vector that points from object 1's center to object 2's center
        epsilon (float): The regularization offset
        dA (float): Area of each patch, assumed the same for all points
        viscosity (float): Viscosity of the fluid.

    Returns:
        (6N1+12, 6N2+12)-array: Oseen tensor S, N1=len(x_1), N2=len(x_2)
    """
    oseen_factor = dA / (8 * np.pi * viscosity)
    eps = regularization_offset
    N1 = len(x_1)
    N2 = len(x_2)

    # Difference vectors, first object 1 to 1, then 2 to 2 and then 1 to 2
    dx12, dy12, dz12 = difference_vectors(x_1 + x1_center, x_2 + x2_center)
    r12 = np.sqrt(dx12 ** 2 + dy12 ** 2 + dz12 ** 2)
    r12_eps_cube = np.sqrt(r12 ** 2 + eps ** 2) ** 3
    
    # -- Construct S matrices --  Skulle man bruge Sparse matricer?
    S = np.zeros((3*(N1+N2)+12, 3*(N1+N2)+12))
    
    # Consists of four submatrices. Each are symmetrical.
    # Upper left (3*N1. 3*N1) is Object 1 to itself
    # Upper right (3*N1, 3*N2) is cross terms
    # Lower left (3*N2, 3*N1) is cross terms - Transposed of upper right
    # Lower right (3*N2, 3*N2) is Object 2 to itself
    
    # Upper left and lower right can be found using previously defined functions
    S[: 3*N1, : 3*N1] = bem.oseen_tensor(eps, dA, viscosity, x_1)[:-6, :-6]  # Cut off last 6 rows and columns from force=0=torque condition
    S[3*N1: 3*(N1+N2), 3*N1: 3*(N1+N2)] = bem.oseen_tensor(eps, dA, viscosity, x_2)[:-6, :-6]
    
    # Upper right and lower left
    S_12 = np.zeros((3*N1, 3*N2))

    # Central part: S11 S22 S33
    d11 = [dx12, dy12, dz12]
    for i in range(3):
        S_12[i*N1 : (i+1)*N1, i*N2 : (i+1)*N2] = (r12 ** 2 + 2 * eps ** 2 + d11[i] ** 2) / r12_eps_cube
        
    # Right part: S12 S13 S23
    S_12[: N1, N2: 2*N2] = dx12 * dy12 / r12_eps_cube
    S_12[: N1, 2*N2: 3*N2] = dx12 * dz12 / r12_eps_cube
    S_12[N1: 2*N1, 2*N2: 3*N2] = dz12 * dy12 / r12_eps_cube
    
    # Left part: S21 S31, S32
    S_12[N1: 2*N1, : N2] = dx12 * dy12 / r12_eps_cube
    S_12[2*N1: 3*N1, : N2] = dx12 * dz12 / r12_eps_cube
    S_12[2*N1: 3*N1, N2: 2*N2] = dz12 * dy12 / r12_eps_cube
    
    S_12 *= oseen_factor

    # Update S
    S[: 3*N1, 3*N1: 3*(N1+N2)] = S_12  # Upper right
    S[3*N1: 3*(N1+N2), : 3*N1] = S_12.T  # Lower left
    
    # -- Conditions --
    # Force
    # First define the one-matricies for both objects, then update S according to them
    force_obj1 = np.zeros((3, 3*N1))
    force_obj2 = np.zeros((3, 3*N2))
    for i in range(3):
        force_obj1[i, i*N1: (i+1)*N1] = 1
        force_obj2[i, i*N2: (i+1)*N2] = 1
    S[-12: -9, : 3*N1] = force_obj1
    S[: 3*N1, -12: -9] = force_obj1.T
    S[-6: -3, 3*N1: 3*(N1+N2)] = force_obj2
    S[3*N1: 3*(N1+N2), -6: -3] = force_obj2.T
    
    # Torque
    # Object 1 row then column
    torque_obj1 = torque_condition(x_1)
    S[-9: -6, : 3*N1] = torque_obj1
    S[:3*N1, -9: -6] = torque_obj1.T
    # Object 2 row then column
    torque_obj2 = torque_condition(x_2)  # Distance from center to surface, not the center-shifted data
    S[-3: , 3*N1: 3*(N1+N2)] = torque_obj2
    S[3*N1: 3*(N1+N2), -3:] = torque_obj2.T
    
    return S
    

def oseen_tensor_two_objects(x_1, x_2, x_eval, x1_center, x2_center, dA, regularization_offset, viscosity):
    """Two objects, calculate Oseen tensor in points outside of surface.

    Args:
        x_1 ((N, 3)-array): Coordinates for each point on the first object's surface centered around (0,0,0)
        x_1 ((N, 3)-array): Coordinates for each point on the second object's surface centered around (0,0,0)
        center_vector (3, 1)-array: Vector that points from object 1's center to object 2's center
        epsilon (float): The regularization offset
        dA (float): Area of each patch, assumed the same for all points
        viscosity (float): Viscosity of the fluid.

    Returns:
        (6N1+12, 6N2+12)-array: Oseen tensor S, N1=len(x_1), N2=len(x_2)
    """
    N1 = len(x_1)
    N2 = len(x_2)
    N_eval = len(x_eval)
    
    S = np.zeros((3*N_eval, 3*(N1+N2)))
    S[:, : 3*N1] = bem.oseen_tensor(regularization_offset, dA, viscosity, evaluation_points=x_eval, source_points=x_1+x1_center)[:-6, :-6]  # Remove Force=0=Torque rows and columns
    S[:, 3*N1 : 3*(N1+N2)] = bem.oseen_tensor(regularization_offset, dA, viscosity, evaluation_points=x_eval, source_points=x_2+x2_center)[:-6, :-6] 
    return S
    

def force_surface_two_objects(N1, max_mode, squirmer_radius, radius_obj2, x1_center, x2_center, B, B_tilde, C, C_tilde, regularization_offset, viscosity, lab_frame=True, return_points=False):
    """Calculates the force vectors at N_sphere points on a sphere with radius squirmer_radius. 

    Args:
        N_sphere (int): Amount of points on the sphere. 
        max_mode (int): The max Legendre mode available.
        squirmer_radius (float): Radius of squirmer.
        B ((max_mode, max_mode)-array): Matrix with the B mode values
        B_tilde ((max_mode, max_mode)-array)): Matrix with the B_tilde mode values
        C ((max_mode, max_mode)-array)): Matrix with the C mode values
        C_tilde ((max_mode, max_mode)-array)): Matrix with the C_tilde mode values
        regularization_offset (float): epsilon that "blobs" the delta function at singularities
        viscosity (float): Viscosity of the fluid.
        lab_frame (bool, optional): Wheter the velocities are in lab (True) or squirmer frame (False). Defaults to True.

    Returns:
        (3(N1+N2)+12, 1)-array): Forces on the sphere. First N values are the x part, the next N the y and the last N the z part of the forces, then same for object 2 and +12 comes from conditions
    """
    # Get coordinates to points on the two surfaces
    x1, y1, z1, dA = bem.canonical_fibonacci_lattice(N1, squirmer_radius)
    theta = np.arccos(z1 / squirmer_radius)  # [0, pi]
    phi = np.arctan2(y1, x1)  # [0, 2*pi]
    x1_stacked = np.stack((x1, y1, z1)).T  
    
    N2 = int(4 * np.pi * radius_obj2 ** 2 / dA)  # Ensures object 1 and 2 has same dA
    x2, y2, z2, _ = bem.canonical_fibonacci_lattice(N2, radius_obj2)
    x2_stacked = np.stack((x2, y2, z2)).T  

    A_oseen = oseen_tensor_surface_two_objects(x1_stacked, x2_stacked, x1_center, x2_center,
                                               dA, regularization_offset, viscosity)
    # Get velocities in each of the points for both objects
    ux1, uy1, uz1 = fv.field_cartesian(N=max_mode, r=squirmer_radius, 
                                       theta=theta, phi=phi, 
                                       a=squirmer_radius, 
                                       B=B, B_tilde=B_tilde, 
                                       C=C, C_tilde=C_tilde, 
                                       lab_frame=lab_frame)
    u_comb = np.array([ux1, uy1, uz1]).ravel()  
    u_comb = np.append(u_comb, np.zeros(12 + 3*N2))  # 2*6 zeros from Forces=0=Torqus + 3N2 zeros as Object 2 no own velocity
    
    # Solve for the forces, A_oseen @ forces = u_comb
    force_arr = np.linalg.solve(A_oseen, u_comb)
    
    if return_points:
        return force_arr, x1_stacked, x2_stacked
    return force_arr 


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from matplotlib.pyplot import Line2D
    from matplotlib.animation import FuncAnimation

    # -- Test functions --
    def test_2obj_surface():
        eps = 0.1
        viscosity = 1
        N1 = 50
        max_mode = 2
        squirmer_radius = 1
        radius_obj2 = 0.5
        x1_center = np.zeros(3)
        x2_center = np.array([2 * squirmer_radius, 2 * squirmer_radius, 2 * squirmer_radius])
        B = np.zeros((max_mode+1, max_mode+1))
        B_tilde = np.zeros_like(B)
        C = np.zeros_like(B)
        C_tilde = np.zeros_like(B)
        B[1, 1] = 0
            
        force_surface_two_objects(N1, max_mode, squirmer_radius, radius_obj2, x1_center, x2_center, B, B_tilde, C, C_tilde, eps, viscosity, lab_frame=True)    


    def test_2obj_point():
        # Choose parameters
        eps = 0.1
        viscosity = 1
        N1 = 50
        max_mode = 3
        squirmer_radius = 1
        radius_obj2 = 0.4
        x1_center = np.array([0, 0, 0]) 
        x2_center = np.array([0, squirmer_radius+radius_obj2, 0])
        B = np.zeros((max_mode+1, max_mode+1))
        B_tilde = np.zeros_like(B)
        C = np.zeros_like(B)
        C_tilde = np.zeros_like(B)
        B[1, 1] = 1
        
        # Force
        force_with_condition, x1_surface, x2_surface = force_surface_two_objects(N1, max_mode, squirmer_radius, radius_obj2, x1_center, x2_center, B, B_tilde, C, C_tilde, eps, viscosity, lab_frame=True, return_points=True)
        translation = force_with_condition[-12: -6]
        rotation = force_with_condition[-6:]
        force = force_with_condition[:-12]  # Remove translation and rotation
        
        # Evaluation points
        evaluation_points = np.linspace(-4, 4, 30)
        X, Y = np.meshgrid(evaluation_points, evaluation_points)
        x_e = X.ravel()
        y_e = Y.ravel()
        z_e = 0 * X.ravel()
        x_e_stack = np.stack((x_e, y_e, z_e)).T
        
        # Oseen tensor in evaluation points
        dA = 4 * np.pi * squirmer_radius ** 2 / N1
        A_e = oseen_tensor_two_objects(x1_surface, x2_surface, x_e_stack, x1_center, x2_center, dA, eps, viscosity)
        
        # Velocity
        v_e = A_e @ force
        v_e = np.reshape(v_e, (len(v_e)//3, 3), order="F")
        
        # Remove values inside squirmer
        r2_obj1 = np.sum((x_e_stack-x1_center)**2, axis=1)
        r2_obj2 = np.sum((x_e_stack-x2_center)**2, axis=1)
        v_e[r2_obj1 < squirmer_radius ** 2, :] = 0
        v_e[r2_obj2 < radius_obj2 ** 2, :] = 0
        
        # -- Plot --
        fig, ax = plt.subplots(dpi=150, figsize=(6, 6))
        # Add circles
        circle_obj1 = plt.Circle(x1_center[:2], squirmer_radius, color="b", alpha=0.5)  # No need for z component
        circle_obj2 = plt.Circle(x2_center[:2], radius_obj2, color="g", alpha=0.5)
        ax.add_patch(circle_obj1)
        ax.add_patch(circle_obj2)
        # Add arrows
        ax.quiver(x_e_stack[:, 0], x_e_stack[:, 1], v_e[:, 0], v_e[:, 1], color="red")
        ax.set(xlabel="x", ylabel="y", title="Squirmer field two objects")
        # Write translation and rotation on image
        text_min = np.min(x_e)
        text_max = np.max(x_e)
        ax.text(text_min, text_max, s=f"U={np.round(translation, 4)}", fontsize=10)
        ax.text(text_min, text_max-0.3, s=f"$\omega$={np.round(rotation, 4)}", fontsize=10)
        ax.legend(["Squirmer", "Target"], loc="lower right")
        fig.tight_layout()
        plt.savefig("fluid/images/nytnavn.png")
        plt.show()


    def plot_force_distance():
        """Plots the surface forces on the target as the distance between squirmer and target is increased."""
        # Choose parameters
        eps = 0.1
        viscosity = 1
        N1 = 50
        max_mode = 3
        squirmer_radius = 1
        radius_obj2 = 0.4
        x1_center = np.array([0, 0, 0])
        B = np.zeros((max_mode+1, max_mode+1))
        B_tilde = np.zeros_like(B)
        C = np.zeros_like(B)
        C_tilde = np.zeros_like(B)
        B[1, 1] = 1
        dA = 4 * np.pi * squirmer_radius ** 2 / N1
        N2 = int(4 * np.pi * radius_obj2 ** 2 / dA)

        # Force
        # Loop through different x2_center vectors
        x2_start_values = np.arange(squirmer_radius+radius_obj2, 4, 0.3)
        
        force_obj1 = np.empty((3*N1, len(x2_start_values)))
        force_obj2 = np.empty((3*N2, len(x2_start_values)))
        
        for i, x2_val in enumerate(x2_start_values):
            x2_center = np.array([x2_val, 0, 0])
            force_with_condition = force_surface_two_objects(N1, max_mode, squirmer_radius, radius_obj2, x1_center, x2_center, B, B_tilde, C, C_tilde, eps, viscosity, lab_frame=True)
            force_obj1[:, i] = force_with_condition[:3*N1]
            force_obj2[:, i] = force_with_condition[3*N1: 3*(N1+N2)]
        
        fx_plot = force_obj1[:N1, :][::5].T  # First N1 points are x values, only plot every 5th point, transpose such that shapes match.
        fy_plot = force_obj1[N1: 2*N1, :][::5].T
        fz_plot = force_obj1[2*N1: 3*N1, :][::5].T

        fx2_plot = force_obj2[:N2, :][::5].T  # First N2 points are x values, only plot every 5th point, transpose such that shapes match.
        fy2_plot = force_obj2[N2: 2*N2, :][::5].T
        fz2_plot = force_obj2[2*N2: 3*N2, :][::5].T
        
        # Plot forces obj1 
        fig, ax = plt.subplots(dpi=150, figsize=(8, 6))
        ax.plot(x2_start_values, fx_plot, ".--", c="blue")
        ax.plot(x2_start_values, fy_plot, ".--", c="red")
        ax.plot(x2_start_values, fz_plot, ".--", c="green")
        ax.set(xlabel="x distance squirmer-target center", ylabel="Force on squirmer", title="Squirmer force on every 5th point against distance")
        # Legend
        legend_elements = [Line2D([0], [0], color="blue", ls="dashdot", lw=3, label=r"f_x"),
                        Line2D([0], [0], color="red", ls="dashdot", lw=3, label=r"f_y"),
                        Line2D([0], [0], color="green", ls="dashdot", lw=3, label=r"f_z")]
        ax.legend(handles=legend_elements)
        fig.tight_layout()
        plt.show()

        # Plot forces obj2
        fig2, ax2 = plt.subplots(dpi=150, figsize=(8, 6))
        ax2.plot(x2_start_values, fx2_plot, ".--", c="blue")
        ax2.plot(x2_start_values, fy2_plot, ".--", c="red")
        ax2.plot(x2_start_values, fz2_plot, ".--", c="green")
        ax2.set(xlabel="x distance squirmer-target center", ylabel="Force on target", title="Target force on every 5th point against distance")
        ax2.legend(handles=legend_elements)
        fig2.tight_layout()
        plt.show()

    
    def plot_force_quiver_both(influence):
        # Choose parameters
        eps = 0.1
        viscosity = 1
        N1 = 480
        max_mode = 3
        squirmer_radius = 1
        x1_center = np.array([0, 0, 0])

        if influence == "giga":
            radius_obj2 = 1.5
            x2_center = np.array([0.2, squirmer_radius+radius_obj2, 0])
        elif influence == "high":
            radius_obj2 = 0.8
            x2_center = np.array([0.2, squirmer_radius+radius_obj2, 0])
        elif influence == "mid":
            radius_obj2 = 0.8
            x2_center = np.array([0.2, squirmer_radius+radius_obj2+0.5, 0])
        else: # low
            radius_obj2 = 0.5
            x2_center = np.array([0.2, squirmer_radius+radius_obj2+0.5, 0])
        
        B = np.zeros((max_mode+1, max_mode+1))
        B_tilde = np.zeros_like(B)
        C = np.zeros_like(B)
        C_tilde = np.zeros_like(B)
        B[1, 1] = 1
        dA = 4 * np.pi * squirmer_radius ** 2 / N1
        N2 = int(4 * np.pi * radius_obj2 ** 2 / dA)

        # Force
        force_with_condition, x1_surface, x2_surface = force_surface_two_objects(N1, max_mode, squirmer_radius, radius_obj2, x1_center, x2_center, B, B_tilde, C, C_tilde, eps, viscosity, lab_frame=True, return_points=True)
        force_obj1 = force_with_condition[:3*N1]
        force_obj2 = force_with_condition[3*N1: 3*(N1+N2)]
        
        fx_plot = force_obj1[:N1].T  # First N1 points are x values, only plot every 5th point, transpose such that shapes match.
        fy_plot = force_obj1[N1: 2*N1].T
        fz_plot = force_obj1[2*N1: 3*N1].T
        
        fx2_plot = force_obj2[:N2].T  # First N2 points are x values, only plot every 5th point, transpose such that shapes match.
        fy2_plot = force_obj2[N2: 2*N2].T
        fz2_plot = force_obj2[2*N2: 3*N2].T
        
        # Plot forces 
        ax = plt.figure(dpi=200, figsize=(6, 6)).add_subplot(projection="3d")
        ax.quiver(x1_surface[:, 0][::5]+x1_center[0], x1_surface[:, 1][::5]+x1_center[1], x1_surface[:, 2][::5]+x1_center[2], 
                fx_plot[::5], fy_plot[::5], fz_plot[::5], 
                length=0.05, label="Squirmer, scale=0.05")
        ax.quiver(x2_surface[:, 0]+x2_center[0], x2_surface[:, 1]+x2_center[1], x2_surface[:, 2]+x2_center[2], 
                fx2_plot, fy2_plot, fz2_plot,
                length=0.15, color="r", label="Target, scale=0.15")
        ax.set(xlabel="x", ylabel="y", zlabel="z", xlim=(-2.3, 2.3), ylim=(-2.3, 2.3), zlim=(-2.3, 2.3), title=fr"$B_{11}$ mode both objects, {influence} influence")#xticks=[], yticks=[], zticks=[])
        ax.legend(fontsize=7)
        plt.show()


    def plot_force_quiver_squirmer(influence):
        # Choose parameters
        eps = 0.1
        viscosity = 1
        N1 = 200
        max_mode = 3
        squirmer_radius = 1
        x1_center = np.array([0, 0, 0])

        if influence == "giga":
            radius_obj2 = 1.5
            x2_center = np.array([0.2, squirmer_radius+radius_obj2, 0])
        elif influence == "high":
            radius_obj2 = 0.8
            x2_center = np.array([0.2, squirmer_radius+radius_obj2, 0])
        elif influence == "mid":
            radius_obj2 = 0.8
            x2_center = np.array([0.2, squirmer_radius+radius_obj2+0.5, 0])
        else: # low
            radius_obj2 = 0.5
            x2_center = np.array([0.2, squirmer_radius+radius_obj2+0.5, 0])
        
        B = np.zeros((max_mode+1, max_mode+1))
        B_tilde = np.zeros_like(B)
        C = np.zeros_like(B)
        C_tilde = np.zeros_like(B)
        B[1, 1] = 1

        # Force
        force_with_condition, x1_surface, _ = force_surface_two_objects(N1, max_mode, squirmer_radius, radius_obj2, x1_center, x2_center, B, B_tilde, C, C_tilde, eps, viscosity, lab_frame=True, return_points=True)
        force_obj1 = force_with_condition[:3*N1]
        
        fx_plot = force_obj1[:N1].T  # First N1 points are x values, transpose such that shapes match.
        fy_plot = force_obj1[N1: 2*N1].T
        fz_plot = force_obj1[2*N1: 3*N1].T
        
        # Plot forces 
        x_quiv = x1_surface[:, 0] + x1_center[0]
        y_quiv = x1_surface[:, 1] + x1_center[1]
        z_quiv = x1_surface[:, 2] + x1_center[2]
        
        ax = plt.figure(dpi=200, figsize=(6, 6)).add_subplot(projection="3d")
        ax.quiver(x_quiv, y_quiv, z_quiv, 
                fx_plot, fy_plot, fz_plot, length=0.05)
        center_dist = np.round(np.linalg.norm([x1_center, x2_center], ord=2), 2)
        ax.scatter(x2_center[0], x2_center[1], x2_center[2], s=radius_obj2*100, c="red", label=f"target, radius={radius_obj2}, center dist={center_dist}")
        ax.set(xlabel="x", ylabel="y", zlabel="z", xlim=(-1.2, 2.5), ylim=(-1.2, 2.5), zlim=(-1.2, 2.5), title=fr"$B_{11}$ mode only squirmer, {influence} influence")#xticks=[], yticks=[], zticks=[])
        ax.legend(fontsize=8, loc="upper center")
        figname = "fluid/images/" + "squirmer_force_field_" + influence + ".png"
        plt.savefig(figname)
        plt.show()


    def plot_force_quiver_target(influence):
        # Choose parameters
        eps = 0.1
        viscosity = 1
        N1 = 480
        max_mode = 3
        squirmer_radius = 1
        x1_center = np.array([0, 0, 0])

        if influence == "giga":
            radius_obj2 = 1.5
            x2_center = np.array([0.2, squirmer_radius+radius_obj2, 0])
        elif influence == "high":
            radius_obj2 = 0.8
            x2_center = np.array([0.2, squirmer_radius+radius_obj2, 0])
        elif influence == "mid":
            radius_obj2 = 0.8
            x2_center = np.array([0.2, squirmer_radius+radius_obj2+0.5, 0])
        else: # low
            radius_obj2 = 0.5
            x2_center = np.array([0.2, squirmer_radius+radius_obj2+0.5, 0])
            
        B = np.zeros((max_mode+1, max_mode+1))
        B_tilde = np.zeros_like(B)
        C = np.zeros_like(B)
        C_tilde = np.zeros_like(B)
        B[1, 1] = 1
        dA = 4 * np.pi * squirmer_radius ** 2 / N1
        N2 = int(4 * np.pi * radius_obj2 ** 2 / dA)

        # Force
        force_with_condition, _, x2_surface = force_surface_two_objects(N1, max_mode, squirmer_radius, radius_obj2, x1_center, x2_center, B, B_tilde, C, C_tilde, eps, viscosity, lab_frame=True, return_points=True)
        force_obj2 = force_with_condition[3*N1: 3*(N1+N2)]
        
        fx2_plot = force_obj2[:N2].T  # First N2 points are x values, only plot every 5th point, transpose such that shapes match.
        fy2_plot = force_obj2[N2: 2*N2].T
        fz2_plot = force_obj2[2*N2: 3*N2].T
        
        # Plot forces 
        ax = plt.figure(dpi=200, figsize=(6, 6)).add_subplot(projection="3d")
        ax.quiver(x2_surface[:, 0]+x2_center[0], x2_surface[:, 1]+x2_center[1], x2_surface[:, 2]+x2_center[2], 
                fx2_plot, fy2_plot, fz2_plot,
                length=0.2, color="r")
        ax.set(xlabel="x", ylabel="y", zlabel="z", xlim=(-0.5, 0.5), ylim=(1, 2), zlim=(0, 1), title=fr"$B_{11}$ mode only target, {influence} influence")#xticks=[], yticks=[], zticks=[])
        plt.show()


    def quiver_data_to_segments(X, Y, Z, u, v, w, length=1):
        segments = (X, Y, Z, X+v*length, Y+u*length, Z+w*length)
        segments = np.array(segments).reshape(6,-1)
        return [[[x, y, z], [u, v, w]] for x, y, z, u, v, w in zip(*list(segments))]
    
    
    def animate_force(radius_obj2):
        # Choose parameters
        eps = 0.1
        viscosity = 1
        N1 = 200
        max_mode = 3
        squirmer_radius = 1
        x1_center = np.array([0, 0, 0])
        # Modes
        B = np.zeros((max_mode+1, max_mode+1))
        B_tilde = np.zeros_like(B)
        C = np.zeros_like(B)
        C_tilde = np.zeros_like(B)
        B[1, 1] = 1

        # Force
        total_radius = squirmer_radius+radius_obj2
        x2_center_values = np.arange(total_radius+0.05, 1.5*total_radius, total_radius/14)[::-1]
        num_x2_vals = len(x2_center_values)
        fx = np.empty((N1, num_x2_vals))
        fy = np.empty_like(fx)
        fz = np.empty_like(fx)
                    
        for i, x2_center_y in enumerate(x2_center_values):
            x2_center = np.array([0.2, x2_center_y, 0])
        
            force_with_condition, x1_surface, _ = force_surface_two_objects(N1, max_mode, squirmer_radius, radius_obj2, x1_center, x2_center, B, B_tilde, C, C_tilde, eps, viscosity, lab_frame=True, return_points=True)
            fx[:, i] = force_with_condition[:N1].T
            fy[:, i] = force_with_condition[N1: 2*N1].T
            fz[:, i] = force_with_condition[2*N1: 3*N1].T        
        
        x_quiv = x1_surface[:, 0]+x1_center[0]
        y_quiv = x1_surface[:, 1]+x1_center[1]
        z_quiv = x1_surface[:, 2]+x1_center[2]
    
        # Animation
        fig = plt.figure(figsize=(6, 6), dpi=200)
        ax = fig.add_subplot(projection="3d")
        ax.set(xlim=(-1.2, 2.5), ylim=(-1.2, 2.5), zlim=(-1.2, 2.5), xlabel="x", ylabel="y", zlabel="z")
        quiv_length = 0.05
        
        # Initial image
        ax.quiver(x_quiv, y_quiv, z_quiv, 
                  fx[:, 0], fy[:, 0], fz[:, 0], 
                  length=quiv_length)
        point, = ax.plot(x2_center[0], x2_center_values[0], ls="", marker="o", color="r")
        ax.legend([f"Target radius={radius_obj2}", f"Squirmer Radius={squirmer_radius}"], fontsize=8)
        title = ax.set_title("Force field")
        
        # Update function
        def animate(i):
            # Quiver
            ax.collections = []  # Easiest way to do animation is just to redraw the plot over and over again. 
            ax.quiver(x_quiv, y_quiv, z_quiv, 
                      fx[:, i], fy[:, i], fz[:, i], 
                      length=quiv_length)
            # Point
            point.set_data(x2_center[0], x2_center_values[i])
            point.set_3d_properties(x2_center[2])
            # Text
            title.set_text(f"Squirmer force field, time={i}")
        
        anim = FuncAnimation(fig, animate, interval=500, frames=len(x2_center_values))
        plt.draw()
        plt.show()
        video_name = f"fluid/images/squirmer_force_field_radius{radius_obj2}.gif"
        anim.save(video_name, writer="Pillow")
        

    def spherical_harmonic_test(radius_obj2, lm_pair):
        from spherical_harmonics import spherical_harmonic_representation as SHR
        # Choose parameters
        eps = 0.1
        viscosity = 1
        N1 = 200
        max_mode = 3
        squirmer_radius = 1
        x1_center = np.array([0, 0, 0])
        # Modes
        B = np.zeros((max_mode+1, max_mode+1))
        B_tilde = np.zeros_like(B)
        C = np.zeros_like(B)
        C_tilde = np.zeros_like(B)
        B[1, 1] = 1

        # Force
        total_radius = squirmer_radius+radius_obj2
        x2_center = np.array([0, total_radius, 0.4])
        
        force_with_condition, x1_surface, _ = force_surface_two_objects(N1, max_mode, squirmer_radius, radius_obj2, x1_center, x2_center, B, B_tilde, C, C_tilde, eps, viscosity, lab_frame=True, return_points=True)
        force = force_with_condition[: 3*N1]        

        # Plot coefficients against distance.
        
        
    
    #test_2obj_point()
    #plot_force_distance()
    #influence_list = ["low", "mid", "high", "giga"]
    #for influence in ["mid"]:
        #plot_force_quiver_both(influence)
        #plot_force_quiver_squirmer(influence)
        #plot_force_quiver_target(influence)

    #animate_force(radius_obj2=1.5)
    spherical_harmonic_test(0.8, [[1, 0], [2, 0]])