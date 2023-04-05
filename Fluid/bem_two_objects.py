import numpy as np
import field_velocity as fv
from boundary_element_method import canonical_fibonacci_lattice


def difference_vectors(eps, x_1, x_2=None):
    if x_2 is None:
        dx = x_1[0][:, None] - x_1[0][None, :]
        dy = x_1[1][:, None] - x_1[1][None, :]
        dz = x_1[2][:, None] - x_1[2][None, :]
    else:
        dx = x_1[0][:, None] - x_2[0][None, :]
        dy = x_1[1][:, None] - x_2[1][None, :]
        dz = x_1[2][:, None] - x_2[2][None, :]
    r = np.sqrt(dx**2 + dy**2 + dz**2)  # Symmetrical, could save memory? 
    r_eps_cubed = np.sqrt(r**2 + eps**2) ** 3
    return dx, dy, dz, r, r_eps_cubed
    

def oseen_tensor_surface_two_objects(x_1, x_2, dA_1, dA_2, regularization_offset, viscosity):
    """Find Oseen tensor for all points on the surface of the squirmer, another object and their interaction with each other

    Args:
        x_1 ((N, 3)-array): Coordinates for each point on the first object's surface
        x_1 ((N, 3)-array): Coordinates for each point on the second object's surface
        epsilon (float): The regularization offset
        dA (float): Area of each patch, assumed the same for all points
        viscosity (float): Viscosity of the fluid.

    Returns:
        (6N, 6N)-array: Oseen tensor S, N is number of points i.e. len(x)
    """
    oseen_factor_1 = dA_1 / (8 * np.pi * viscosity)
    oseen_factor_2 = dA_2 / (8 * np.pi * viscosity)
    oseen_factor_12 = (dA_1 + dA_2) / 2 / (8 * np.pi * viscosity)  # NOTE måske det skal gøres anderledes end den her approksimation
    eps = regularization_offset
    N = np.shape(x_1)[0]
    
    # Difference vectors, first object 1 to 1, then 2 to 2 and then 1 to 2
    dx11, dy11, dz11, r11, r11_eps_cube = difference_vectors(eps, x_1)
    dx22, dy22, dz22, r22, r22_eps_cube = difference_vectors(eps, x_2)
    dx12, dy12, dz12, r12, r12_eps_cube = difference_vectors(eps, x_1, x_2)
    d11 = [dx11, dy11, dz11]
    d22 = [dx22, dy22, dz22]
    d12 = [dx12, dy12, dz12]
    
    # Find the S matrices - Skulle man bruge Sparse matricer?
    # Diagonal - "Diagonal" refers to the centered NxN matrices 
    S_diag = np.zeros((6*N+12, 6*N+12))
    S_off_diag = np.zeros_like(S_diag)        
    for i in range(3): 
        S_diag[i*N: (i+1)*N] = (d11[i] ** 2 + r11 + 2 * eps ** 2) / r11_eps_cube * oseen_factor_1 # 1 to 1
        S_diag[(i+3)*N: (i+4)*N] = (d22[i] ** 2 + r22 + 2 * eps ** 2) / r22_eps_cube * oseen_factor_2  # 2 to 2
        S_off_diag[i*N: (i+4)*N] = (d12[i] ** 2 + r12 + 2 * eps ** 2) / r12_eps_cube * oseen_factor_12  # 1 to 2
            
    # Off diagonal  - Må sgu kunne lave et loop eller lignende?!?
    # 1 to 1    
    S_off_diag[: N, N: 2*N] = dx11 * dy11 / r11_eps_cube * oseen_factor_1
    S_off_diag[: N, 2*N: 3*N] = dz11 * dx11 / r11_eps_cube * oseen_factor_1
    S_off_diag[N: 2*N, 2*N: 3*N] = dy11 * dz11 / r11_eps_cube * oseen_factor_1
    
    # 1 to 2 - Shift columns by 3N
    S_off_diag[: N, 4*N: 5*N] = dx12 * dy12 / r12_eps_cube * oseen_factor_12
    S_off_diag[: N, 5*N: 6*N] = dz12 * dx12 / r12_eps_cube * oseen_factor_12
    S_off_diag[: 2*N, 5*N: 6*N] = dy12 * dz12 / r12_eps_cube * oseen_factor_12
    S_off_diag[:3*N, 3*N: 6*N] += S_off_diag.T[:3*N, 3*N: 6*N]

    # 2 to 2 - Shift rows and columns by 3N
    S_off_diag[3*N: 4*N, 4*N: 5*N] = dx22 * dy22 / r22_eps_cube * oseen_factor_2
    S_off_diag[3*N: 4*N, 5*N: 6*N] = dz22 * dx22 / r22_eps_cube * oseen_factor_2
    S_off_diag[4*N: 5*N, 5*N: 6*N] = dy22 * dz22 / r22_eps_cube * oseen_factor_2
    
    S = (S_diag + S_off_diag + S_off_diag.T)
    
    # Force
    force_matrix = np.zeros((3, 3))
    for i in range(3):
        force_matrix[i-6, i*N: (i+1)*N] = 1
    S[-12: -9, :6*N] = force_matrix
    S[:6*N, -12: -9] = force_matrix.T
    S[-6:, : 6*N] = force_matrix
    S[: 6*N, -6:] = force_matrix.T
    
    # Torque
    # Row
    
    # column = row.T
    
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