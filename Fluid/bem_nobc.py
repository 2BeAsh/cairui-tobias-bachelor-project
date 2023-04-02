import numpy as np
import matplotlib.pyplot as plt
import field_velocity as fv
from boundary_element_method import canonical_fibonacci_lattice


def oseen_tensor_nobc(regularization_offset, dA, viscosity, evaluation_points, source_points=None):
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
    
    S = np.zeros((3*N_points, 3*N_sphere))  # three coordinates, +3 from force +3 from torque
    
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
    
    return S * oseen_factor


def force_on_sphere_nobc(N_sphere, distance_squirmer, max_mode, theta, phi, squirmer_radius, B, B_tilde, C, C_tilde, regularization_offset, viscosity, lab_frame=True):
    """Calculates the force vectors at N_sphere points on a sphere with radius squirmer_radius. 

    Args:
        N_sphere (int): Amount of points on the sphere. 
        distance_squirmer (float): Euclidean distance from squirmer centrum to desired point.
        max_mode (int): The max Legendre mode available.
        theta (float): vertial angle between.
        phi (_type_): Horizontal angle
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
    x_sphere, y_sphere, z_sphere, theta_sphere, phi_sphere, area = canonical_fibonacci_lattice(N_sphere, squirmer_radius)
    x_e = np.stack((x_sphere, y_sphere, z_sphere))
    A_oseen = oseen_tensor_nobc(regularization_offset, area, viscosity, evaluation_points=x_e)
    # Get velocities in each of the points
    u_x, u_y, u_z = fv.field_cartesian(N=max_mode, r=squirmer_radius, 
                                    theta=theta_sphere, phi=phi_sphere, 
                                    a=squirmer_radius, 
                                    B=B, B_tilde=B_tilde, 
                                    C=C, C_tilde=C_tilde, 
                                    lab_frame=lab_frame)
    u_comb = np.array([u_x, u_y, u_z]).ravel()  # 6 zeros from Forces=0=Torque
    # Solve for the forces, A_oseen @ forces = u_comb
    force_arr = np.linalg.solve(A_oseen, u_comb)
    return force_arr, u_comb 


def test_oseen_field_cartesian_nobc():
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
    #B_tilde[1, 1] = 1
    
    # x og v
    x, y, z, theta, phi, area = canonical_fibonacci_lattice(N, r)
    u_x, u_y, u_z = fv.field_cartesian(max_mode, r, theta, phi, r, B, B_tilde, C, C_tilde)

    x_surface = np.stack((x, y, z)).T
    v = np.stack((u_x, u_y, u_z)).T
    v = np.reshape(v, -1, order="F")

    # Få Oseen matrix på overfladen og løs for kræfterne
    A = oseen_tensor_nobc(regularization_offset=eps, dA=area, viscosity=viscosity,
                     evaluation_points=x_surface)
    F = np.linalg.solve(A, v)
    
    # Evaluer i punkter uden for kuglen
    x_e = np.linspace(-3 * r, 3 * r, 25)
    X, Y = np.meshgrid(x_e, x_e)
    x_e = X.ravel()
    y_e = Y.ravel()
    z_e = 0 * X.ravel()
    x_e = np.stack((x_e, y_e, z_e)).T
    
    # Get Oseen and solve for velocities using earlier force
    A_e = oseen_tensor_nobc(regularization_offset=eps, dA=area, viscosity=viscosity,
                     evaluation_points=x_e, source_points=x_surface)
    v_e = A_e @ F
    v_e = np.reshape(v_e, (len(v_e)//3, 3), order="F")

    # Remove values inside squirmer
    r2 = np.sum(x_e**2, axis=1)
    v_e[r2 < r ** 2, :] = 0
    
    # -- Plot --
    fig, ax = plt.subplots(ncols=1, nrows=1, dpi=150, figsize=(6,6))
    ax.quiver(x_e[:, 0], x_e[:, 1], v_e[:, 0], v_e[:, 1], color="red")
    ax.set(xlabel="x", ylabel="y", title=r"Without Conditions, Squirmer field, lab frame, ${B}_{11}$")  # B_{11}\,
    plt.savefig("fluid/images/nocondition_squirmerfield_labframe_B_11.png")
    plt.show()


def test_oseen_given_field_nobc():
    N = 250
    r = 1.
    eps = 0.1
    viscosity = 1
    x, y, z, _, _, area = canonical_fibonacci_lattice(N, r)
    
    # Boundary Conditions
    vx = 1 + 0 * x
    vy = 0 * x
    vz = 0 * x
    
    # Stack
    x = np.stack((x, y, z)).T
    v = np.stack((vx, vy, vz)).T
    v = np.reshape(v, -1, order="F")

    # Få Oseen matrix på overfladen og løs for kræfterne
    A = oseen_tensor_nobc(regularization_offset=eps, dA=area, viscosity=viscosity,
                     evaluation_points=x)
    F = np.linalg.solve(A, v)

    # Evaluate i punkter uden for kuglen
    x_e = np.linspace(-3 * r, 3 * r, 25)
    X, Y = np.meshgrid(x_e, x_e)
    x_e = X.ravel()
    y_e = Y.ravel()
    z_e = 0 * X.ravel()
    x_e = np.stack((x_e, y_e, z_e)).T
    
    # Get Oseen and solve for velocities using earlier force
    A_e = oseen_tensor_nobc(regularization_offset=eps, dA=area, viscosity=viscosity,
                     evaluation_points=x_e, source_points=x)
    v_e = A_e @ F
    v_e = np.reshape(v_e, (len(v_e)//3, 3), order="F")

    # Return to squirmer frame by subtracting boundary conditions
    #v_e[:, 0] -= 1
    #v_e[:, 1] -= 1

    # Remove values inside squirmer
    r2 = np.sum(x_e**2, axis=1)
    v_e[r2 < r ** 2, :] = 0
    
    # -- Plot --
    fig, ax = plt.subplots(ncols=1, nrows=1, dpi=150, figsize=(6,6))
    ax.quiver(x_e[:, 0], x_e[:, 1], v_e[:, 0], v_e[:, 1], color="red")
    ax.set(xlabel="x", ylabel="y", title=r"Without Conditions, Artificial field $v_x=1$, lab frame")
    plt.savefig("fluid/images/nocondition_artificialfield_labframe.png")
    plt.show()
    
    
def test_oseen_given_force_nobc():
    eps = 0.1
    viscosity = 1
    xx = np.linspace(-2, 2, 25)
    X, Y = np.meshgrid(xx, xx)
    X = X.ravel()
    Y = Y.ravel()
    Z = 0 * X
    
    # Source point and force
    xi = np.zeros(3).reshape(1, -1)
    F = np.zeros(3)
    F[0] = 1.  # Force in x
    
    x_e = np.stack((X, Y, Z)).T
    O = oseen_tensor_nobc(regularization_offset=eps, dA=0.5, viscosity=viscosity, 
                        evaluation_points=x_e, source_points=xi)
    v = O @ F
    v = np.reshape(v, (len(v)//3, 3), order='F')
        
    vx = v[:, 0]
    vy = v[:, 1]
    
    fig, ax = plt.subplots(dpi=150, figsize=(6, 6))
    ax.quiver(X, Y, vx, vy, color="red")
    ax.set(xlabel="x", ylabel="y", title=r"Without Conditions, Artificial $F_x$, lab frame")
    plt.savefig("fluid/images/nocondition_artificialforce_labframe.png")
    plt.show()
    
    
if __name__ == "__main__":  
    test_oseen_field_cartesian_nobc()
    #test_oseen_given_field_nobc()
    #test_oseen_given_force_nobc()
    