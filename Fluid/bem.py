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
    area = 4 * np.pi * radius ** 2 / N
    return x, y, z, area 
    

def oseen_tensor_surface(x, y, z, regularization_offset, dA, viscosity):  # Mangler F = 0 = Torque
    """Find Oseen tensor for all multiple points on the surface of the squirmer.

    Args:
        x (1d array of floats): x-coordinate of all points
        y (1d array of floats): y-coordinate of all points
        z (1d array of floats): z-coordinate of all points
        epsilon (float): The regularization offset
        dA (float): Area of each patch, assumed the same for all points
        viscosity (float): Viscosity of the fluid.

    Returns:
        (3N, 3N)-array: Oseen tensor S, N is number of points i.e. len(x)
    """
    oseen_factor = dA / (8 * np.pi * viscosity)
    N = np.shape(x)[0]
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dz = z[:, None] - z[None, :]
    r = np.sqrt(dx**2 + dy**2 + dz**2)  # Symmetrical, could save memory?
    r_epsilon_cubed = np.sqrt(r**2 + regularization_offset**2) ** 3
        
    # Find the S matrices
    S_diag = np.zeros((3*N, 3*N))  # "Diagonal" refers to the NxN matrices S11, S22 and S33 
    S_diag[:N, :N] = (dx ** 2 + r ** 2 + 2 * regularization_offset ** 2)
    S_diag[N:2*N, N:2*N] = (dy ** 2 + r ** 2 + 2 * regularization_offset ** 2)
    S_diag[2*N:3*N, 2*N:3*N] = (dz ** 2 + r ** 2 + 2 * regularization_offset ** 2)
    
    S_off_diag = np.zeros_like(S_diag)
    S_off_diag[0:N, N:2*N] = dx * dy # Element wise multiplication
    S_off_diag[0:N, 2*N:3*N] = dx * dz
    S_off_diag[N:2*N, 2*N:3*N] = dy * dz
    
    S = (S_diag + S_off_diag + S_off_diag.T) * oseen_factor / r_epsilon_cubed  # NOTE r_eps er kun N,N men skal være 3N-3N
    return S 


def oseen_tensor(regularization_offset, dA, viscosity, evaluation_points, source_points=None):
    epsilon = regularization_offset 
    oseen_factor = dA / (8 * np.pi * viscosity)
    
    # Skal klart opdateres/optimeres. Kan forhåbentlig definere r = x - xi, og ri*rj = r[:, None] * r[None, :]
    N_points = np.shape(evaluation_points)[0]
    x_points = evaluation_points[:, 0]
    y_points = evaluation_points[:, 1]
    z_points = evaluation_points[:, 2]      
    if source_points is None:
        source_points = evaluation_points
    N_sphere = np.shape(source_points)[0]
    x_sphere = source_points[:, 0]
    y_sphere = source_points[:, 1]
    z_sphere = source_points[:, 2]  
    
    # Need difference between each point and all sphere points, done using broadcasting
    dx = x_points[:, None] - x_sphere[None, :]
    dy = y_points[:, None] - y_sphere[None, :]
    dz = z_points[:, None] - z_sphere[None, :]
    r = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
    r_epsilon_cubed = np.sqrt(r ** 2 + epsilon ** 2) ** 3
    
    S = np.zeros((3*N_points+6, 3*N_sphere+6))  # three coordinates, +3 from force +3 from torque
    
    # The centermost matrices: S11 S22 S33
    S[0:N_points, 0:N_sphere] = (r ** 2 + 2 * epsilon ** 2 + dx ** 2) / r_epsilon_cubed
    S[N_points:2*N_points, N_sphere:2*N_sphere] = (r ** 2 + 2 * epsilon ** 2 + dy ** 2) / r_epsilon_cubed
    S[2*N_points:3*N_points, 2*N_sphere:3*N_sphere] = (r ** 2 + 2 * epsilon ** 2 + dz ** 2) / r_epsilon_cubed
    
    # Right part: S12 S13 S23
    S[0:N_points, N_sphere:2*N_sphere] = dx * dy / r_epsilon_cubed
    S[0:N_points, 2*N_sphere:3*N_sphere] = dx * dz / r_epsilon_cubed
    S[N_points:2*N_points, 2*N_sphere:3*N_sphere] = dz * dy / r_epsilon_cubed
    
    # Left part: S21 S31, S32
    S[N_points:2*N_points, 0:N_sphere] = dx * dy / r_epsilon_cubed
    S[2*N_points:3*N_points, 0:N_sphere] = dx * dz / r_epsilon_cubed
    S[2*N_points:3*N_points, N_sphere:2*N_sphere] = dz * dy / r_epsilon_cubed
    
    S *= oseen_factor
    
    # Forces = 0, U and torque
    for i in range(3):
        S[i-6, i*N_sphere:(i+1)*N_sphere] = 1 
        S[i*N_points:(i+1)*N_points, i-6] = 1
    
    # Torque
    # Row
    S[-3, N_sphere:2*N_sphere] = -z_sphere
    S[-3, 2*N_sphere:3*N_sphere] = y_sphere
    S[-2, 0:N_sphere] = z_sphere
    S[-2, 2*N_sphere:3*N_sphere] = -x_sphere
    S[-1, 0:N_sphere] = -y_sphere
    S[-1, N_sphere:2*N_sphere] = x_sphere
    
    # Column - Måske man kunne sige den er transponeret af Row?
    S[0:N_points, -2] = z_points
    S[0:N_points, -1] = -y_points
    S[N_points:2*N_points, -3] = -z_points
    S[N_points:2*N_points, -1] = x_points
    S[2*N_points:3*N_points, -3] = y_points    
    S[2*N_points:3*N_points, -2] = -x_points
    
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
    x_sphere, y_sphere, z_sphere, area = canonical_fibonacci_lattice(N_sphere, squirmer_radius)
    theta_sphere = np.arccos(z_sphere / squirmer_radius) 
    phi_sphere = np.arctan2(y_sphere, x_sphere)
    x_e = np.stack((x_sphere, y_sphere, z_sphere)).T  # NOTE Brokker sig ikke selvom jeg ændrer på om () eller ej, samt transponeret eller ej
    A_oseen = oseen_tensor(regularization_offset, area, viscosity, evaluation_points=x_e)
    # Get velocities in each of the points
    u_x, u_y, u_z = fv.field_cartesian(max_mode, r=squirmer_radius, 
                                       theta=theta_sphere, phi=phi_sphere, 
                                       squirmer_radius=squirmer_radius,
                                       mode_array=mode_array,
                                       lab_frame=lab_frame)
    u_comb = np.array([u_x, u_y, u_z]).ravel()  # 6 zeros from Forces=0=Torque
    u_comb = np.append(u_comb, np.zeros(6))
    # Solve for the forces, A_oseen @ forces = u_comb
    force_arr = np.linalg.solve(A_oseen, u_comb)
    return force_arr 

    
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    def test_oseen_field_cartesian():
        # Parametre
        N = 150
        r = 1.
        eps = 0.1
        viscosity = 1
        max_mode = 2
        B = np.zeros((max_mode+1, max_mode+1))
        B_tilde = np.zeros_like(B)
        C = np.zeros_like(B)
        C_tilde = np.zeros_like(B)
        B[1, 1] = 1
        B_tilde[1, 1] = 1
        
        # x og v
        x, y, z, area = canonical_fibonacci_lattice(N, r)
        theta = np.arccos(z / r)
        phi = np.arctan2(y, x)
        u_x, u_y, u_z = fv.field_cartesian(max_mode, r, theta, phi, r, B, B_tilde, C, C_tilde)

        x_surface = np.stack((x, y, z)).T
        v = np.stack((u_x, u_y, u_z)).T
        v = np.reshape(v, -1, order="F")
        v = np.append(v, np.zeros(6))

        # Få Oseen matrix på overfladen og løs for kræfterne
        import time
        t1 = time.time()
        A = oseen_tensor(regularization_offset=eps, dA=area, viscosity=viscosity,
                        evaluation_points=x_surface)
        F = np.linalg.solve(A, v)
        print(time.time() - t1)
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
        A_e = oseen_tensor(regularization_offset=eps, dA=area, viscosity=viscosity,
                        evaluation_points=x_e, source_points=x_surface)
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

        ux_field, uy_field, uz_field = fv.field_cartesian(max_mode, R_field.flatten(), Theta.flatten(), Phi.flatten(), 
                                                    r, B, B_tilde, C, C_tilde, lab_frame=True)
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
        ax_oseen.set(xlabel="x", ylabel="y", title=r"With Conditions, Squirmer field, lab frame, $\tilde{B}_{11}$")
        text_min = np.min(x_e)
        text_max = np.max(x_e)
        ax_oseen.text(text_min, text_max, s=f"U={np.round(U, 4)}", fontsize=12)
        ax_oseen.text(text_min, text_max-0.3, s=f"$\omega$={np.round(ang_freq, 4)}", fontsize=12)
        
        #ax_field.quiver(Y_field, Z_field, uy_field, uz_field, color="blue", label="Original Field")
        #ax_field.set(xlabel="y", ylabel="z")
        #ax_field.legend(loc="upper center")  
        plt.savefig("fluid/images/condition_squirmerfield_labframe_B_tilde11.png")
        plt.show()


    def test_oseen_given_field():
        N = 250
        r = 1.
        eps = 0.1
        viscosity = 1
        x, y, z, area = canonical_fibonacci_lattice(N, r)
        theta = np.arccos(z / r)
        phi = np.arctan2(y, x)
        # Boundary Conditions
        vx = 1 + 0 * x
        vy = 0 * x
        vz = 0 * x
        
        # Stack
        x = np.stack((x, y, z)).T
        v = np.stack((vx, vy, vz)).T
        v = np.reshape(v, -1, order="F")
        v = np.append(v, np.zeros(6))  # From force=0=Torque

        # Få Oseen matrix på overfladen og løs for kræfterne
        A = oseen_tensor(regularization_offset=eps, dA=area, viscosity=viscosity,
                        evaluation_points=x)
        F = np.linalg.solve(A, v)
        U = F[-6:-3]
        ang_freq = F[-3:]
        # Evaluate i punkter uden for kuglen
        x_e = np.linspace(-3 * r, 3 * r, 25)
        X, Y = np.meshgrid(x_e, x_e)
        x_e = X.ravel()
        y_e = Y.ravel()
        z_e = 0 * X.ravel()
        x_e = np.stack((x_e, y_e, z_e)).T
        
        # Get Oseen and solve for velocities using earlier force
        A_e = oseen_tensor(regularization_offset=eps, dA=area, viscosity=viscosity,
                        evaluation_points=x_e, source_points=x)
        v_e = A_e @ F
        v_e = v_e[:-6]  # Remove Forces
        v_e = np.reshape(v_e, (len(v_e)//3, 3), order="F")

        # Return to squirmer frame by subtracting boundary conditions
        #v_e[:, 0] -= 1
        #v_e[:, 1] -= 1

        # Remove values inside squirmer
        r2 = np.sum(x_e**2, axis=1)
        v_e[r2 < r ** 2, :] = 0
        
        # -- Plot --
        fig, ax = plt.subplots(ncols=1, nrows=1, dpi=150, figsize=(6,6))
        ax_oseen = ax
        ax_oseen.quiver(x_e[:, 0], x_e[:, 1], v_e[:, 0], v_e[:, 1], color="red")
        ax_oseen.set(xlabel="x", ylabel="y", title=r"With Conditions, Artificial field $v_x=1$, lab frame")
        text_min = np.min(x_e)
        text_max = np.max(x_e)
        ax_oseen.text(text_min, text_max, s=f"U={np.round(U, 4)}", fontsize=12)
        ax_oseen.text(text_min, text_max-0.3, s=f"$\omega$={np.round(ang_freq, 4)}", fontsize=12)

        plt.savefig("fluid/images/condition_artificialfield_labframe.png")
        plt.show()
        
        
    def test_oseen_given_force():
        eps = 0.1
        viscosity = 1
        xx = np.linspace(-2, 2, 25)
        X, Y = np.meshgrid(xx, xx)
        X = X.ravel()
        Y = Y.ravel()
        Z = 0 * X
        
        # Source point and force
        xi = np.zeros(3).reshape(1, -1)
        F = np.zeros(9)  # Rotation and U makes bigger
        F[0] = 1.  # Force in x
        #F[-6] = 2.  # Background flow in x
        #F[-1] = -0.5  # Rotation along z-axis
        
        x_e = np.stack((X, Y, Z)).T
        O = oseen_tensor(regularization_offset=eps, dA=0.5, viscosity=viscosity, 
                            evaluation_points=x_e, source_points=xi)
        v = O @ F
        v = v[:-6]  # Remove force and torque
        v = np.reshape(v, (len(v)//3, 3), order='F')
            
        vx = v[:, 0]
        vy = v[:, 1]
        
        fig, ax = plt.subplots(dpi=150, figsize=(6, 6))
        ax.quiver(X, Y, vx, vy, color="red")
        ax.set(xlabel="x", ylabel="y", title="With Conditions, Artificial 'force' $F_x$, lab frame")
        plt.savefig("fluid/images/condition_artificialforce_labframe.png")
        plt.show()        

    test_oseen_field_cartesian()
    #test_oseen_given_field()
    #test_oseen_given_force()

