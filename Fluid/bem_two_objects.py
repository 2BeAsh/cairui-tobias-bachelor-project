import numpy as np
import field_velocity as fv
import bem


def oseen_tensor_surface_two_objects(x_1, x_2, x1_center, x2_center, dA, regularization_offset, viscosity):
    """Find Oseen tensor for all points on the surface of the squirmer, another object and their interaction with each other

    Args:
        x_1 ((N, 3)-array): Coordinates for each point on the first object's surface centered around (0,0,0)
        x_2 ((N, 3)-array): Coordinates for each point on the second object's surface centered around (0,0,0)
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
    dx12, dy12, dz12 = bem.difference_vectors(x_1 + x1_center, x_2 + x2_center)
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
    S[: 3*N1, : 3*N1] = bem.oseen_tensor_surface(x_1, dA, eps, viscosity)[:-6, :-6]  # Cut off last 6 rows and columns from force=0=torque condition
    S[3*N1: 3*(N1+N2), 3*N1: 3*(N1+N2)] = bem.oseen_tensor_surface(x_2, dA, eps, viscosity)[:-6, :-6]
    
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
    # Force for both objects, 
    force_obj1 = np.zeros((3, 3*N1))
    force_obj2 = np.zeros((3, 3*N2))
    for i in range(3):
        force_obj1[i, i*N1: (i+1)*N1] = 1
        force_obj2[i, i*N2: (i+1)*N2] = 1

    torque_obj1 = bem.torque_condition(x_1)
    torque_obj2 = bem.torque_condition(x_2)  # Distance from center to surface, not the center-shifted data
    
    # Rows
    S[-12: -9, : 3*N1] = force_obj1
    S[-6: -3, 3*N1: 3*(N1+N2)] = force_obj2
    S[-9: -6, : 3*N1] = torque_obj1
    S[-3: , 3*N1: 3*(N1+N2)] = torque_obj2
    
    # Columns    
    S[: 3*N1, -12: -9] = -force_obj1.T  
    S[3*N1: 3*(N1+N2), -6: -3] = -force_obj2.T 
    S[: 3*N1, -9: -6] = -torque_obj1.T  
    S[3*N1: 3*(N1+N2), -3:] = -torque_obj2.T 
    
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
    S[:, : 3*N1] = bem.oseen_tensor(x_1+x1_center, x_eval, regularization_offset, dA, viscosity)[:-6, :-6]  # Remove Force=0=Torque rows and columns
    S[:, 3*N1 : 3*(N1+N2)] = bem.oseen_tensor(x_2+x2_center, x_eval, regularization_offset, dA, viscosity)[:-6, :-6] 
    return S
    

def force_surface_two_objects(N1, max_mode, squirmer_radius, radius_obj2, x1_center, x2_center, mode_array, regularization_offset, viscosity, return_points=False):
    """Calculates the force vectors at N_sphere points on a sphere with radius squirmer_radius. 

    Args:
        N1 (int): Amount of object 1 points.
        max_mode (int): The max Legendre mode available.
        squirmer_radius (float): Radius of squirmer.
        mode_array: list of mode arrays 
        regularization_offset (float): epsilon that "blobs" the delta function at singularities
        viscosity (float): Viscosity of the fluid.
        lab_frame (bool, optional): Wheter the velocities are in lab (True) or squirmer frame (False). Defaults to True.

    Returns:
        (3(N1+N2)+12, 1)-array): Forces on the sphere. First N values are the x part, the next N the y and the last N the z part of the forces, then same for object 2 and +12 comes from conditions
    """
    # Get coordinates to points on the two surfaces
    x1, y1, z1, dA = bem.canonical_fibonacci_lattice(N1, squirmer_radius)
    x1_stacked = np.stack((x1, y1, z1)).T  
    theta = np.arctan2(np.sqrt(x1 ** 2 + y1 ** 2), z1)  # [0, pi]
    phi = np.arctan2(y1, x1)  # [0, 2*pi]
    
    N2 = int(4 * np.pi * radius_obj2 ** 2 / dA)  # Ensures object 1 and 2 has same dA
    x2, y2, z2, _ = bem.canonical_fibonacci_lattice(N2, radius_obj2)
    x2_stacked = np.stack((x2, y2, z2)).T  

    A_oseen = oseen_tensor_surface_two_objects(x1_stacked, x2_stacked, x1_center, x2_center,
                                               dA, regularization_offset, viscosity)
    # Get velocities in each point on squirmer
    ux1, uy1, uz1 = fv.field_cartesian_squirmer(max_mode, r=squirmer_radius, theta=theta, phi=phi, squirmer_radius=squirmer_radius, mode_array=mode_array)
    u_comb = np.stack([ux1, uy1, uz1])
    u_comb = np.append(u_comb, np.zeros(12 + 3*N2)).T  # 2*6 zeros from Forces=0=Torqus + 3N2 zeros as Object 2 no own velocity
    
    # Solve for the forces, A_oseen @ forces = u_comb
    force_arr = np.linalg.solve(A_oseen, u_comb)
    
    if return_points:
        return force_arr, x1_stacked, x2_stacked
    return force_arr 

        
def average_change_direction(N, max_mode, squirmer_radius, radius_obj2, x1_center, x2_center, mode_array, regularization_offset, viscosity, noise):
    # Force difference
    force, x_vec, _ = force_surface_two_objects(N, max_mode, squirmer_radius, radius_obj2, x1_center, x2_center, 
                                      mode_array, regularization_offset, viscosity, return_points=True)
    force_no_target = bem.force_on_sphere(N, max_mode, squirmer_radius, mode_array, regularization_offset, viscosity)

    dfx = force[:N] - force_no_target[:N]
    dfy = force[N: 2*N] - force_no_target[N: 2*N]
    dfz = force[2*N: 3*N] - force_no_target[2*N: 3*N]
    
    # Gaussian Noise
    dfx += np.random.normal(loc=0, scale=noise, size=dfx.size)
    dfy += np.random.normal(loc=0, scale=noise, size=dfy.size)
    dfz += np.random.normal(loc=0, scale=noise, size=dfz.size)
    # Weights
    weight = np.sqrt(dfx ** 2 + dfy ** 2 + dfz ** 2)
    f_average = np.sum(weight[:, None] * x_vec, axis=0)
    f_average_norm = f_average / np.linalg.norm(f_average, ord=2)  
    return f_average_norm


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from matplotlib.pyplot import Line2D
    from matplotlib.animation import FuncAnimation
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    # -- Test functions --
    def plot_2_objects_velocity_field():
        import matplotlib.pyplot as plt
        # Choose parameters
        squirmer_radius = 1
        radius_obj2 = 0.5
        N1 = 300
        eps = 0.05
        viscosity = 1
        max_mode = 2
        x1_center = np.array([0, 0, 0])  # NOTE feltet afhænger af hvor man sætter squirmer!
        x2_center = np.array([-1.55, 0.3, 0])
        B = np.zeros((max_mode+1, max_mode+1))
        Bt = np.zeros_like(B)
        C = np.zeros_like(B)
        Ct = np.zeros_like(B)
        B[1,1] = 1
        mode_array = np.array([B, Bt, C, Ct])
        
        # Force
        force_with_condition, x1_surface, x2_surface = force_surface_two_objects(N1, max_mode, squirmer_radius, radius_obj2, x1_center, 
                                                                                 x2_center, mode_array, eps, viscosity, return_points=True)
        U_num = force_with_condition[-12: -9]
        U_num_2 = force_with_condition[-6: -3]
        rot_num = force_with_condition[-9: -6]
        force = force_with_condition[:-12]  # Remove translation and rotation
                
        # Evaluation points
        N_eval = 30
        evaluation_points = np.linspace(-3, 3, N_eval)
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
        
        # Lab frame
        squirmer_frame = False
        target_frame = True
        if squirmer_frame:
            v_e -= U_num
            title = "Squirmer"
        elif target_frame:
            v_e -= U_num_2
            title = "Target "
        else:  # Lab
            title = "Lab"
        
        title += f" frame, N1 = {N1}, Reg. offset = {eps}"
        # Remove values inside squirmer
        r2_obj1 = np.sum((x_e_stack-x1_center)**2, axis=1)
        r2_obj2 = np.sum((x_e_stack-x2_center)**2, axis=1)
        v_e[r2_obj1 < squirmer_radius ** 2, :] = 0
        v_e[r2_obj2 < radius_obj2 ** 2, :] = 0
        
        # -- Plot --
        fig, ax = plt.subplots(dpi=150)
        ax.set(xlabel="x", ylabel="y", title=title)

        # Quiver & Streamplot
        ax.quiver(x_e_stack[:, 0], x_e_stack[:, 1], v_e[:, 0], v_e[:, 1], color="red")
        #ax.streamplot(X, Y, v_e[:, 0].reshape((N_eval, N_eval)), v_e[:, 1].reshape((N_eval, N_eval)), density=2)
        
        # Countour plot
        velocity_magnitude = np.sqrt(v_e[:, 0].reshape((N_eval, N_eval))**2 + v_e[:, 1].reshape((N_eval, N_eval))**2)
        #contour_velocity = ax.contourf(X, Y, velocity_magnitude,  vmin=0, vmax=2.5, levels=16, cmap='Blues')
        
        # Colorbars  
        #divider = make_axes_locatable(ax)
        #cax = divider.append_axes("right", size="5%", pad=0.05)
       # cbar = plt.colorbar(contour_velocity, cax=cax)

        # Add circles
        circle_obj1 = plt.Circle(x1_center[:2], squirmer_radius, color="b", alpha=0.5)  # No need for z component
        circle_obj2 = plt.Circle(x2_center[:2], radius_obj2, color="g", alpha=0.5)
        ax.add_patch(circle_obj1)
        ax.add_patch(circle_obj2)
        
        # Add text
        ax.text(x=0.05, y=0.95, s=f"U_n={np.round(U_num, 4)}", fontsize=10, transform=ax.transAxes, color="blue")
        ax.legend(["Squirmer", "Target"], loc="lower right")
        fig.tight_layout()
        #plt.savefig("fluid/images/two_objects_velocity_field.png")
        plt.show()
    
    
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
        mode_array = np.array([B, B_tilde, C, C_tilde])
        # Force
        total_radius = squirmer_radius + radius_obj2
        x2_center_values = np.arange(total_radius+0.05, 1.5*total_radius, total_radius/14)[::-1]
        num_x2_vals = len(x2_center_values)
        fx = np.empty((N1, num_x2_vals))
        fy = np.empty_like(fx)
        fz = np.empty_like(fx)
                    
        for i, x2_center_y in enumerate(x2_center_values):
            x2_center = np.array([0.2, x2_center_y, 0])

            force_with_condition, x1_surface, _ = force_surface_two_objects(N1, max_mode, squirmer_radius, radius_obj2, x1_center, x2_center, 
                                                                            mode_array, eps, viscosity, lab_frame=True, return_points=True)
            fx[:, i] = force_with_condition[:N1].T
            fy[:, i] = force_with_condition[N1: 2*N1].T
            fz[:, i] = force_with_condition[2*N1: 3*N1].T        
        
        x_quiv = x1_surface[:, 0] + x1_center[0]
        y_quiv = x1_surface[:, 1] + x1_center[1]
        z_quiv = x1_surface[:, 2] + x1_center[2]
    
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

    
    def force_difference():
        # Choose parameters
        eps = 0.05
        viscosity = 1
        N1 = 700
        max_mode = 2
        squirmer_radius = 1
        radius_obj2 = 0.4
        total_radius = squirmer_radius+radius_obj2
        x1_center = np.array([0, 0, 0])
        x2_center = np.array([0, 2, 0])
        sensor_noise = 0
        # Modes
        B = np.zeros((max_mode+1, max_mode+1))
        B_tilde = np.zeros_like(B)
        C = np.zeros_like(B)
        C_tilde = np.zeros_like(B)
        B_tilde[1, 1] = 0.75

        # Force                            
        force_with_condition, x1_surface, _ = force_surface_two_objects(N1, max_mode, squirmer_radius, radius_obj2, x1_center, x2_center, np.array([B, B_tilde, C, C_tilde]), eps, viscosity, return_points=True)
        fx = force_with_condition[:N1].T
        fy = force_with_condition[N1: 2*N1].T
        fz = force_with_condition[2*N1: 3*N1].T        
        
        # Force difference 
        force_no_target = bem.force_on_sphere(N1, max_mode, squirmer_radius, np.array([B, B_tilde, C, C_tilde]), eps, viscosity)
        fx_no = force_no_target[:N1].T
        fy_no = force_no_target[N1: 2*N1].T
        fz_no = force_no_target[2*N1: 3*N1].T        

        fx_diff = fx - fx_no
        fy_diff = fy - fy_no
        fz_diff = fz - fz_no
        
        x_change = average_change_direction(N1, max_mode, squirmer_radius, radius_obj2, x1_center, x2_center, np.array([B, B_tilde, C, C_tilde]), eps, viscosity, sensor_noise)
        
        x_quiv = x1_surface[:, 0] + x1_center[0]
        y_quiv = x1_surface[:, 1] + x1_center[1]
        z_quiv = x1_surface[:, 2] + x1_center[2]
    
        difference = True
    
        # Plot
        fig = plt.figure(figsize=(8, 8), dpi=200)
        ax = fig.add_subplot(projection="3d")
        #ax.set(xlabel="x", ylabel="y", zlabel="z")

        if difference:
            ax.quiver(x_quiv, y_quiv, z_quiv, fx_diff, fy_diff, fz_diff, color="b")
            ax.quiver(x1_center[0], x1_center[1], x1_center[2],
                      x_change[0], x_change[1], x_change[2], color="green")
            ax.set(xlim=(-squirmer_radius, squirmer_radius), ylim=(-squirmer_radius, squirmer_radius), zlim=(-squirmer_radius, squirmer_radius), xticks=[], yticks=[], zticks=[])

        else:
            ax.quiver(x_quiv, y_quiv, z_quiv, fx, fy, fz, color="b", length=0.05)
            ax.plot(x2_center[0], x2_center[1], x2_center[2], "ro", markersize=10)
            ax.set(xlim=(-1.2, 2.5), ylim=(-1.2, 2.5), zlim=(-1.2, 2.5))
            ax.legend([f"Target radius={radius_obj2}", f"Squirmer Radius={squirmer_radius}"], fontsize=8)
        ax.set_title("Force field")
        plt.show()      
                
        
    #force_difference()
    #plot_2_objects_velocity_field()
    #test_2obj_point()
    