import numpy as np
import field_velocity as fv


def canonical_fibonacci_lattice(N, radius):
    """Calculate N points uniformly distributed on the surface of a sphere with given radius, using the canonical Fibonacci Lattice method.

    Args:
        N (int): Number of points
        radius (float): Radius of sphere.

    Returns:
        Tupple of three 1d-arrays and a float: Returns cartesian coordinates of the points distributed on the spherical surface, and the approximate area each point are given.
        x, y, z, area 
    """
    # Find spherical coordinates. 
    # Radius is contant, theta determined by golden ratio and phi is found using the Inverse Transform Method.
    offset = 0.5
    indices = np.arange(N)
    golden_ratio = (1 + np.sqrt(5)) / 2 
    theta = 2 * np.pi * indices / golden_ratio
    phi = np.arccos(1 - 2 * (indices + offset) / N)
    
    # Convert to cartesian
    x = radius * np.sin(phi) * np.cos(theta)
    y = radius * np.sin(phi) * np.sin(theta)
    z = radius * np.cos(phi)
    # The area of each patch is approximated by surface area divided by number of points
    dA = 4 * np.pi * radius ** 2 / N
    return x, y, z, dA 
    
    
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
    
    
def torque_condition_with_minus(x_stacked):
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
    torque_arr[0, N: 2*N] = z
    torque_arr[0, 2*N: 3*N] = -y
    torque_arr[1, : N] = -z
    torque_arr[1, 2*N: 3*N] = x
    torque_arr[2, : N] = y
    torque_arr[2, N: 2*N] = -x
    return torque_arr
    
    
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


def oseen_tensor_surface(coord, dA, regularization_offset, viscosity):  # Mangler F = 0 = Torque
    """Find Oseen tensor for all multiple points on the surface of the squirmer.

    Args:
        x (3d array of floats): coordinates of all points
        epsilon (float): The regularization offset
        dA (float): Area of each patch, assumed the same for all points
        viscosity (float): Viscosity of the fluid.

    Returns:
        (3N+6, 3N+6)-array: Oseen tensor S, N is number of points i.e. len(x), 
    """
    N = len(coord)
    eps = regularization_offset
    oseen_factor = dA / (8 * np.pi * viscosity) 
    
    # Difference vector
    dx, dy, dz = difference_vectors(coord)
    r = np.sqrt(dx**2 + dy**2 + dz**2)  # Symmetrical, could save memory?
    r_epsilon_cubed = np.sqrt(r**2 + eps**2) ** 3
        
    # Find the S matrices
    S_diag = np.zeros((3*N+6, 3*N+6))  # "Diagonal" refers to the NxN matrices S11, S22 and S33 
    
    # Central matrices
    d = [dx, dy, dz]
    for i in range(3):
        S_diag[i*N: (i+1)*N, i*N: (i+1)*N] = (r ** 2 + 2 * eps ** 2 + d[i] ** 2) / r_epsilon_cubed
    
    # Off diag    
    S_off_diag = np.zeros_like(S_diag)
    S_off_diag[0:N, N:2*N] = dx * dy / r_epsilon_cubed  # Element wise multiplication
    S_off_diag[0:N, 2*N:3*N] = dx * dz / r_epsilon_cubed
    S_off_diag[N:2*N, 2*N:3*N] = dy * dz / r_epsilon_cubed
    
    S = (S_diag + S_off_diag + S_off_diag.T) * oseen_factor 
    
    # Force and torque
    # Force
    force_arr = np.zeros((3, 3*N))
    for i in range(3):
        force_arr[i, i*N: (i+1)*N] = 1  # -1
    S[-6: -3, : 3*N] = force_arr
    S[: 3*N, -6: -3] = -force_arr.T
    
    # Torque
    torque_arr = torque_condition(coord)
    S[-3:, : 3*N] = torque_arr
    S[: 3*N, -3:] = -torque_arr.T
    
    return S 


def oseen_tensor(x_sphere, x_eval, regularization_offset, dA, viscosity):    
    # Skal klart opdateres/optimeres. Kan forhåbentlig definere r = x - xi, og ri*rj = r[:, None] * r[None, :]
    N_eval = np.shape(x_eval)[0]  # Eval is points evaluated outside squirmer
    N_sphere = np.shape(x_sphere)[0]
    eps = regularization_offset 
    oseen_factor = dA / (8 * np.pi * viscosity)
    
    dx, dy, dz = difference_vectors(x_eval, x_sphere)
    r = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
    r_epsilon_cubed = np.sqrt(r ** 2 + eps ** 2) ** 3
    
    S = np.zeros((3*N_eval+6, 3*N_sphere+6))  # three coordinates, +3 from force +3 from torque
    # The centermost matrices: S11 S22 S33    
    d = [dx, dy, dz]
    for i in range(3):
        S[i*N_eval: (i+1)*N_eval, i*N_sphere: (i+1)*N_sphere] = (r ** 2 + 2 * eps **2 + d[i] ** 2) / r_epsilon_cubed
    
    # Right part: S12 S13 S23
    S[0:N_eval, N_sphere:2*N_sphere] = dx * dy / r_epsilon_cubed
    S[0:N_eval, 2*N_sphere:3*N_sphere] = dx * dz / r_epsilon_cubed
    S[N_eval:2*N_eval, 2*N_sphere:3*N_sphere] = dz * dy / r_epsilon_cubed
    
    # Left part: S21 S31, S32
    S[N_eval:2*N_eval, 0:N_sphere] = dx * dy / r_epsilon_cubed
    S[2*N_eval:3*N_eval, 0:N_sphere] = dx * dz / r_epsilon_cubed
    S[2*N_eval:3*N_eval, N_sphere:2*N_sphere] = dz * dy / r_epsilon_cubed
    
    S *= oseen_factor
    
    # Forces = 0, U and torque
    # Forces
    force_eval = np.zeros((3*N_eval, 3))
    force_sphere = np.zeros((3, 3*N_sphere))
    for i in range(3):
        force_eval[i*N_eval: (i+1)*N_eval, i] = -1 
        force_sphere[i, i*N_sphere: (i+1)*N_sphere] = 1     
    S[-6: -3, : 3*N_sphere] = force_sphere
    S[: 3*N_eval, -6: -3] = force_eval
    
    # Torque
    S[-3: , :3*N_sphere] = torque_condition(x_sphere)
    S[:3*N_eval, -3:] = -torque_condition(x_eval).T
        
    return S 


def force_on_sphere(N_sphere, max_mode, squirmer_radius, mode_array, regularization_offset, viscosity, lab_frame=True):
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
        (3N_sphere, 1)-array): Forces on the sphere. First N values are the x part, the next N the y and the last N the z part of the forces.
    """
    # Get the A matrix from the Oseen tensor
    x_sphere, y_sphere, z_sphere, dA = canonical_fibonacci_lattice(N_sphere, squirmer_radius)
    theta_sphere = np.arccos(z_sphere / squirmer_radius) 
    phi_sphere = np.arctan2(y_sphere, x_sphere)
    x_e = np.stack((x_sphere, y_sphere, z_sphere)).T 
    A_oseen = oseen_tensor_surface(x_e, dA, regularization_offset, viscosity)
    # Get velocities in each of the points
    u_x, u_y, u_z = fv.field_cartesian_squirmer(max_mode, r=squirmer_radius, theta=theta_sphere, phi=phi_sphere, squirmer_radius=squirmer_radius, mode_array=mode_array)
    u_comb = np.array([u_x, u_y, u_z]).ravel()  # 6 zeros from Forces=0=Torque
    u_comb = np.append(u_comb, np.zeros(6))
    
    # Solve for the forces, A_oseen @ forces = u_comb
    force_arr = np.linalg.solve(A_oseen, u_comb)
    return force_arr 

    
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    def test_oseen_field_cartesian():
        # Parametre
        N = 100
        r = 1.
        eps = 0.01
        viscosity = 1
        max_mode = 2
        B = np.zeros((max_mode+1, max_mode+1))
        B_tilde = np.zeros_like(B)
        C = np.zeros_like(B)
        C_tilde = np.zeros_like(B)
        B[1, 1] = 1
        B_tilde[1, 1] = 0
        
        # x og v
        x, y, z, dA = canonical_fibonacci_lattice(N, r)
        x_surface = np.stack((x, y, z)).T
        theta = np.arccos(z / r)
        phi = np.arctan2(y, x)
        u_x, u_y, u_z = fv.field_cartesian_squirmer(max_mode, r, theta, phi, r, np.array([B, B_tilde, C, C_tilde]))
        v = np.stack((u_x, u_y, u_z)).T
        v = np.reshape(v, -1, order="F")
        v = np.append(v, np.zeros(6))

        # Få Oseen matrix på overfladen og løs for kræfterne
        A = oseen_tensor_surface(x_surface, dA, eps, viscosity)
        #print(A[:, [-6, -5, -4]])
        #A[:-6, :-6] *= N
        F = np.linalg.solve(A, v)
        U = F[-6:-3]  # Skal vises i plot
        ang_freq = F[-3:]
        
        # Evaluer i punkter uden for kuglen
        x_e = np.linspace(-3 * r, 3 * r, 25)
        X, Y = np.meshgrid(x_e, x_e)
        x_e = X.ravel()
        y_e = Y.ravel()
        z_e = 0 * X.ravel()
        x_e = np.stack((x_e, y_e, z_e)).T
        
        # Get Oseen and solve for velocities using earlier force
        A_e = oseen_tensor(x_surface, x_e, eps, dA, viscosity)
        #A_e[:-6, :-6] *= N
        v_e = A_e @ F
        v_e = v_e[:-6]  # Remove Forces
        v_e = np.reshape(v_e, (len(v_e)//3, 3), order="F")

        # Remove values inside squirmer
        r2 = np.sum(x_e**2, axis=1)
        v_e[r2 < r ** 2, :] = 0
        
        # -- Compare to known field --
        # The known velocity field
        y_field = 1 * x_e + 0.1  # +0.1 for at undgå r = 0
        z_field = 1 * x_e + 0.1
        Z_field, Y_field = np.meshgrid(z_field, y_field)
        R_field = np.sqrt(Z_field**2 + Y_field**2)
        Theta = np.arctan2(Y_field, Z_field) + np.pi
        Phi = np.ones(np.shape(Theta)) * np.pi/2

        ux_field, uy_field, uz_field = fv.field_cartesian_squirmer(max_mode, R_field.flatten(), Theta.flatten(), Phi.flatten(), r, np.array([B, B_tilde, C, C_tilde]))
        ux_field = ux_field.reshape(np.shape(R_field))
        uy_field = uy_field.reshape(np.shape(R_field))
        uz_field = uz_field.reshape(np.shape(R_field))
        mask_squirmer = R_field < r
        ux_field[mask_squirmer] = 0
        uy_field[mask_squirmer] = 0
        uz_field[mask_squirmer] = 0
        
        # -- Plot --
        fig, ax = plt.subplots(ncols=1, nrows=1, dpi=150, figsize=(6,6))
        ax_oseen = ax
        ax_oseen.quiver(x_e[:, 0], x_e[:, 1], v_e[:, 0], v_e[:, 1], color="red")
        
        # Center of mass velocity
        U_anal = 4 / (3 * r ** 3) * np.array([B[1, 1], B_tilde[1, 1], -B[0, 1]])
        ax_oseen.quiver(0, 0, U_anal[0], U_anal[1], label=r"$U_{analytical}$")
        ax_oseen.quiver(0, 0, U[0], U[1], label=r"$U_{num}$", color="b")  # Center of Mass velocity numerical
        
        # Axis setup
        ax_oseen.set(xlabel="x", ylabel="y", title=r"With Conditions, Squirmer field, lab frame, $\tilde{B}_{11}$")
        text_min = np.min(x_e)
        text_max = np.max(x_e)
        ax_oseen.text(text_min, text_max, s=f"U_n={np.round(U, 6)}", fontsize=12)
        ax_oseen.text(text_min, text_max-0.3, s=fr"$U_a$={np.round(U_anal, 6)}", fontsize=12)
        U_anal_size = np.linalg.norm(U_anal, ord=2)
        U_num_size = np.linalg.norm(U, ord=2)
        ax_oseen.text(text_min, text_max-0.6, s=fr"U factor difference={np.round(U_anal_size / U_num_size, 2)}", fontsize=12)
        
        ax_oseen.legend()
        #ax_field.quiver(Y_field, Z_field, uy_field, uz_field, color="blue", label="Original Field")
        #ax_field.set(xlabel="y", ylabel="z")
        #ax_field.legend(loc="upper center")  
        #plt.savefig("fluid/images/condition_squirmerfield_labframe_B_tilde11.png")
        plt.show()


    def plot_only_squirmer_field():
        # -- Parametre --
        a = 1  # Squirmer radius
        N = 500  # Punkter på overflade
        eps = 0.1
        viscosity = 1
        max_mode = 2
        B = np.zeros((max_mode+1, max_mode+1))
        Bt = np.zeros_like(B)
        C = np.zeros_like(B)
        Ct = np.zeros_like(B)
        B[1, 1] = 1 
        mode_array = np.array([B, Bt, C, Ct])
        
        # Surface points
        x_surface, y_surface, z_surface, dA = canonical_fibonacci_lattice(N, a)
        coord_surface = np.stack([x_surface, y_surface, z_surface]).T
        theta = np.arctan2(np.sqrt(x_surface ** 2 + y_surface ** 2), z_surface)
        phi = np.arctan2(y_surface, x_surface)
        
        # Surface velocity in squirmer frame
        u_surface_x, u_surface_y, u_surface_z = fv.field_cartesian_squirmer(max_mode, a, theta, phi, a, mode_array)
        u_surface = np.stack([u_surface_x, u_surface_y, u_surface_z])
        u_surface = np.append(u_surface, np.zeros(6)).T  # Add Force = 0 = Torque
        
        # Surface forces
        A_surface = oseen_tensor_surface(coord_surface, dA, eps, viscosity)
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
        
        # Oseen tensor 
        A_point = oseen_tensor(coord_surface, coord_eval, eps, dA, viscosity)
        
        # Velocity
        u_point = A_point @ F_surface
        u_point = u_point[: -6]
        u_point = np.reshape(u_point, (len(u_point)//3, 3), order="F")

        # Convert to lab frame:
        lab_frame = True
        if lab_frame:
            u_point += U_num
            title = "Lab Frame"
            # Analytical field 
            theta_point = np.arctan2(np.sqrt(X ** 2 + Y ** 2), Z)
            phi_point = np.arctan2(Y, X)
            u_anal_x, u_anal_y, u_anal_z = fv.field_cartesian(max_mode, np.sqrt(r2), theta_point, phi_point, a, mode_array)
            u_anal_x[r2 < a ** 2] = 0
            u_anal_y[r2 < a ** 2] = 0
            
        else:
            title = "Squirmer Frame"            
            
        title += f", Reg. offset = {eps}, N surface points = {N}"
        
        # Remove inside squirmer
        u_point[r2 < a ** 2, :] = 0
        
        u_x_num_non_zero = u_point[:, 0][np.nonzero(u_point[:, 0])]
        #u_anal_x_non_zero = u_anal_x[np.nonzero(u_anal_x)]
        #frac_mean = np.mean(u_anal_x_non_zero / u_x_num_non_zero)
        
        # Plot
        fig, ax = plt.subplots(dpi=200, figsize=(8, 8))
        ax.quiver(X_mesh, Y_mesh, u_point[:, 0], u_point[:, 1], label="Numerical")
        #ax.quiver(X_mesh, Y_mesh, u_anal_x, u_anal_y, color="red", label="Analytical", scale=5)
        #ax.quiver(X_mesh, Y_mesh, u_anal_x-u_point[:, 0], u_anal_y-u_point[:, 1])
        #ax.streamplot(X_mesh, Y_mesh, u_point[:, 0].reshape(X_mesh.shape), u_point[:, 1].reshape(Y_mesh.shape))
        
        ax.set(xlabel="x", ylabel="y", title=title)
        ax.legend()
        
        # Text
        U_anal = 4 / (3 * a ** 3)
        U_num_str =  r"$U_{num}=$" + str(np.round(U_num, 4))
        U_anal_str =  r"$U_{anal}=$" + str(np.round(U_anal, 4))
        U_frac_str = r"$U_{num}/U_{anal}$" + str(np.round(U_num / U_anal, 4))
        ax.text(x=0.05, y=0.95, s=U_num_str, transform=ax.transAxes, color="blue")
        ax.text(x=0.05, y=0.90, s=U_anal_str, transform=ax.transAxes, color="blue")
        ax.text(x=0.05, y=0.85, s=U_frac_str, transform=ax.transAxes, color="blue")
        plt.show()
        
        # print("Mean velocity field fraction")
        # print(np.mean(u_anal_x_non_zero / u_x_num_non_zero), " +- ", np.std(u_anal_x_non_zero / u_x_num_non_zero))
        # print("Translational field fraction")
        # print(U_num / U_anal)

        
    plot_only_squirmer_field()
    #test_oseen_field_cartesian()

