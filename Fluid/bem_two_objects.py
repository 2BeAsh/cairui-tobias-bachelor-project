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


def torque_condition(x_tupple):
    """The torque sums to 0 condition for the ROWS. To get the column, transpose the result

    Args:
        x_tupple (Tupple of 1darrays): Tuple of the (x, y, z) coordinates

    Returns:
        (3, 3N)-array: Torque condition along rows.
    """
    x, y, z = x_tupple
    N = len(x)
    torque_arr = np.zeros((3, 3*N))
    torque_arr[0, N: 2*N] = -z
    torque_arr[0, 2*N: 3*N] = y
    torque_arr[1, : N] = z
    torque_arr[1, 2*N: 3*N] = -x
    torque_arr[2, : N] = -y
    torque_arr[2, N: 2*N] = x
    return torque_arr


def oseen_tensor_surface_two_objects(x_1, x_2, center_vector, dA, regularization_offset, viscosity):
    """Find Oseen tensor for all points on the surface of the squirmer, another object and their interaction with each other

    Args:
        x_1 ((N, 3)-array): Coordinates for each point on the first object's surface centered around (0,0,0)
        x_1 ((N, 3)-array): Coordinates for each point on the second object's surface centered around (0,0,0)
        center_vector (3, 1)-array: Vector that points from object 1's center to object 2's center
        epsilon (float): The regularization offset
        dA (float): Area of each patch, assumed the same for all points
        viscosity (float): Viscosity of the fluid.

    Returns:
        (6N, 6N)-array: Oseen tensor S, N is number of points i.e. len(x)
    """
    oseen_factor = dA / (8 * np.pi * viscosity)
    eps = regularization_offset
    N1 = np.shape(x_1)[0]  # Object 1 number of points
    N2 = np.shape(x_2)[0]

    # Difference vectors, first object 1 to 1, then 2 to 2 and then 1 to 2
    #dx11, dy11, dz11 = difference_vectors(x_1)
    #dx22, dy22, dz22 = difference_vectors(x_2)  # Internally doesn't need to change reference
    dx12, dy12, dz12 = difference_vectors(x_1, x_2 + center_vector)  # Change reference to be object 1's.
    
    # -- Construct S matrices --  - Skulle man bruge Sparse matricer?
    # Consists of four submatrices. Each are symmetrical.
    # Upper left (3*N1. 3*N1) is Object 1 to itself
    # Upper right (3*N1, 3*N2) is cross terms
    # Lower left (3*N2, 3*N1) is cross terms - Transposed of upper right
    # Lower right (3*N2, 3*N2) is Object 2 to itself
    
    S = np.zeros((3*(N1+N2)+12, 3*(N1+N2)+12))
    # Upper left and lower right can be found using previously defined functions

    S[: 3*N1, : 3*N1] = bem.oseen_tensor(eps, dA, viscosity, x_1)[:-6, :-6]  # Cut off boundary condition
    S[3*N1: 3*(N1+N2), 3*N1: 3*(N1+N2)] = bem.oseen_tensor(eps, dA, viscosity, x_2)[:-6, :-6]
    
    # Upper right and lower left
    S_12 = np.zeros((3*N1, 3*N2))
    r12 = np.sqrt(dx12**2 + dy12**2 + dz12**2)
    r12_eps_cube = np.sqrt(r12 ** 2 + eps ** 2) ** 3

    # Central part: S11 S22 S33
    S_12[: N1, : N2] = (r12 ** 2 + 2 * eps ** 2 + dx12 ** 2) / r12_eps_cube
    S_12[N1: 2*N1, N2: 2*N2] = (r12 ** 2 + 2 * eps ** 2 + dy12 ** 2) / r12_eps_cube
    S_12[2*N1: 3*N1, 2*N2: 3*N2] = (r12 ** 2 + 2 * eps ** 2 + dz12 ** 2) / r12_eps_cube
    
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
    
    # Conditions
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
    

def oseen_tensor_point_two_objects(source_points_1, source_points_2, evaluation_points, dA_1, dA_2, regularization_offset, viscosity):
    oseen_factor = dA_1 / (8 * np.pi * viscosity)
    oseen_factor = dA_2 / (8 * np.pi * viscosity)
    eps = regularization_offset
    N = np.shape(source_points_1)[0]  # Along columns
    M = np.shape(evaluation_points)[0]  # Along rows
    print(N, M)
    # Difference vectors, first object 1 to evaluation, then object 2 to evaluation
    dx1, dy1, dz1, r1, r1_eps_cube = difference_vectors(eps, source_points_1, evaluation_points)
    dx2, dy2, dz2, r2, r2_eps_cube = difference_vectors(eps, source_points_2, evaluation_points)
    dx = [dx1, dx2]
    dy = [dy1, dy2]
    dz = [dz1, dz2]
    r = [r1, r2]
    r_eps_cube = [r1_eps_cube, r2_eps_cube]
    oseen_factor = [oseen_factor, oseen_factor]
    
    # -- Construct S matrices --  - Skulle man bruge Sparse matricer?
    # Loop through both objects. 
    S = np.zeros((3*M+12, 6*N+12))  # 3 coordinates per evaluation point, 2 objects each with 3N points and 6 lines of conditions
    for i in range(2):
        # The centermost matrices: S11 S22 S33
        S[0:M, i*N:(i+1)*N] = (r[i] ** 2 + 2 * eps ** 2 + dx[i] ** 2) / r_eps_cube[i] * oseen_factor[i]
        S[M:2*M, (i+1)*N:(i+2)*N] = (r[i] ** 2 + 2 * eps ** 2 + dy[i] ** 2) / r_eps_cube[i] * oseen_factor[i]
        S[2*M:3*M, (i+2)*N:(i+3)*N] = (r[i] ** 2 + 2 * eps ** 2 + dz[i] ** 2) / r_eps_cube[i] * oseen_factor[i]
        
        # Right part: S12 S13 S23
        S[0:M, (i+1)*N:(i+2)*N] = dx[i] * dy[i] / r_eps_cube[i] * oseen_factor[i]
        S[0:M, (i+2)*N:(i+3)*N] = dx[i]* dz[i] / r_eps_cube[i] * oseen_factor[i]
        S[M:2*M, (i+2)*N:(i+3)*N] = dz[i] * dy[i] / r_eps_cube[i] * oseen_factor[i]
        
        # Left part: S21 S31, S32
        S[M:2*M, 0:N] = dx[i] * dy[i] / r_eps_cube[i] * oseen_factor[i]
        S[2*M:3*M, 0:N] = dx[i] * dz[i] / r_eps_cube[i] * oseen_factor[i]
        S[2*M:3*M, N:2*N] = dz[i] * dy[i] / r_eps_cube[i] * oseen_factor[i]
            
    # -- Conditions -- 
    # Force
    force_matrix = np.zeros((3, 6*N))  # NOTE check om er rigtig!!!
    for i in range(3):
        force_matrix[i-3, i*N: (i+1)*N] = 1
    S[-12: -9, :6*N] = force_matrix  # Row 
    S[: 6*N, -12: -9] = force_matrix.T  # Column
    S[-6: -3, : 6*N] = force_matrix
    S[: 6*N, -6: -3] = force_matrix.T
    
    # Torque
    torque_1 = torque_condition((dx1, dy1, dz1))
    torque_2 = torque_condition((dx2, dy2, dz2))
    S[-9: -6, : 3*N] = torque_1  # Row
    S[:3*N, -9: -6] = torque_1.T  # Column
    S[-3:, 3*N: 6*N] = torque_2
    S[3*N: 6*N, -3:] = torque_2.T
        
    return S 


def force_on_sphere_two_objects(N1, max_mode, squirmer_radius, radius_obj2, center_vector, B, B_tilde, C, C_tilde, regularization_offset, viscosity, lab_frame=True):
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
        (6N_sphere+12, 1)-array): Forces on the sphere. First N values are the x part, the next N the y and the last N the z part of the forces, then same for object 2 and +12 comes from conditions
    """
    # Get coordinates to points on the two surfaces
    x_sphere_1, y_sphere_1, z_sphere_1, dA = bem.canonical_fibonacci_lattice(N1, squirmer_radius)
    theta_sphere = np.arccos(z_sphere_1 / squirmer_radius) 
    phi_sphere = np.arctan2(y_sphere_1, x_sphere_1)
    x_surface_1 = np.stack((x_sphere_1, y_sphere_1, z_sphere_1)).T  
    
    N2 = 4 * np.pi * radius_obj2 ** 2 // dA  # Ensures object 1 and 2 has same dA
    x_sphere_2, y_sphere_2, z_sphere_2, _ = bem.canonical_fibonacci_lattice(N2, radius_obj2 / 2)
    x_surface_2 = np.stack((x_sphere_2, y_sphere_2, z_sphere_2)).T  

    A_oseen = oseen_tensor_surface_two_objects(x_surface_1, x_surface_2, center_vector,
                                               dA, regularization_offset, viscosity)
    # Get velocities in each of the points for both objects
    ux1, uy1, uz1 = fv.field_cartesian(N=max_mode, r=squirmer_radius, 
                                       theta=theta_sphere, phi=phi_sphere, 
                                       a=squirmer_radius, 
                                       B=B, B_tilde=B_tilde, 
                                       C=C, C_tilde=C_tilde, 
                                       lab_frame=lab_frame)

    u_comb = np.array([ux1, uy1, uz1, np.zeros(12 + 3*N2)]).ravel()  # 2*6 zeros from Forces=0=Torqus + 3N2 zeros as Object 2 no own velocity
    # Solve for the forces, A_oseen @ forces = u_comb
    force_arr = np.linalg.solve(A_oseen, u_comb)
    return force_arr 


def test_oseen_two_objects_surface():
    eps = 0.1
    viscosity = 1
    N1 = 50
    max_mode = 2
    squirmer_radius = 1
    radius_obj2 = 0.5
    center_vector = np.array([2 * squirmer_radius, 2 * squirmer_radius, 2 * squirmer_radius])
    B = np.zeros((max_mode+1, max_mode+1))
    B_tilde = np.zeros_like(B)
    C = np.zeros_like(B)
    C_tilde = np.zeros_like(B)
    B[1, 1] = 1
        
    force_on_sphere_two_objects(N1, max_mode, squirmer_radius, radius_obj2, center_vector, B, B_tilde, C, C_tilde, eps, viscosity, lab_frame=True)    


def test_oseen_given_force_two_objects():
    eps = 0.1
    viscosity = 1
    xx = np.linspace(-2, 2, 25)
    X, Y = np.meshgrid(xx, xx)
    X = X.ravel()
    Y = Y.ravel()
    Z = 0 * X
    
    # Source point and force
    xi1 = np.zeros(3).reshape(-1, 1)
    xi2 = np.zeros_like(xi1)
    xi2[0] += 1
    # F is Fx, Fy, Fz and Ux, Uy, Uz, and wx, wy, wz for each object, so 6N+12 = 30 long
    F = np.zeros(30)
    F[0] = 1.  # Fx1
    F[3] = -1.  # Fx2
    
    x_e = np.stack((X, Y, Z)).T
    O = oseen_tensor_point_two_objects(xi1, xi2, x_e, dA_1=0.5, dA_2=0.25, regularization_offset=eps, viscosity=viscosity)
    v = O @ F
    print(v.shape)
    v = v[:-12]  # Remove force and torque  NOTE har fjernet 3 mindre...
    print("v fjernet 12", v.shape)
    v = np.reshape(v, (len(v)//6, 6), order='F')  # Fordel de seks hastigheder
    print(v.shape) 
    vx1 = v[:, 0]
    vy1 = v[:, 1]
    print(vx1.shape)
    vx2 = v[:, 3]
    vy2 = v[:, 4]
    print(X.shape)
    fig, ax = plt.subplots(dpi=150, figsize=(6, 6))
    ax.quiver(X, Y, vx1, vy1, color="red")
    ax.quiver(X, Y, vx2, vy2, color="blue")
    ax.set(xlabel="x", ylabel="y", title="With Conditions, Artificial 'force' $F_x$, lab frame")
    plt.show()


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    test_oseen_two_objects_surface()
    
    