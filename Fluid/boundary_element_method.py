import numpy as np
import matplotlib.pyplot as plt


def canonical_fibonacci_lattice(N, radius):
    """Calculate N points uniformly distributed on the surface of a sphere with given radius, using the canonical Fibonacci Lattice method.

    Args:
        N (int): Number of points
        radius (float): Radius of sphere.

    Returns:
        Tupple of three 1d-arrays and a float: Returns cartesian coordinates of the points distributed on the spherical surface, and the approximate area each point are given.
        x, y, z, theta, phi, area 
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
    return x, y, z, theta, phi, area 


def discretized_sphere(N, radius):
    """Calculate N points uniformly distributed on the surface of a sphere with given radius.
    The calculation is done using a modified version of the canonical Fibonacci Lattice.
    From: http://extremelearning.com.au/how-to-evenly-distribute-points-on-a-sphere-more-effectively-than-the-canonical-fibonacci-lattice/
    And inspired by: https://stackoverflow.com/questions/9600801/evenly-distributing-n-points-on-a-sphere 

    Args:
        N (int): Number of points
        radius (float): Radius of sphere.

    Returns:
        Tupple of three 1d-arrays and a float: Returns cartesian coordinates of the points distributed on the spherical surface, and the approximate area each point are given.
    """
    # Find best index offset based on N. 
    if N < 80:
        offset = 2.66
    elif N < 1e3:
        offset = 3.33
    elif N < 4e4:
        offset = 10
    else:  # N > 40_000
        offset = 25 
    # Place the first two points and top and bottom of the sphere (0, 0, radius) and (0, 0, -radius)
    x = np.empty(N)
    y = np.empty_like(x)
    z = np.empty_like(x)
    x[:2] = 0
    y[:2] = 0
    z[0] = radius
    z[1] = -radius
        
    # Find spherical coordinates. 
    # Radius is contant, theta determined by golden ratio and phi is found using the Inverse Transform Method.
    indices = np.arange(N-2)  # Two first points already placed, thus minus 2
    golden_ratio = (1 + np.sqrt(5)) / 2 
    theta = 2 * np.pi * indices / golden_ratio
    phi = np.arccos(1 - 2 * (indices + offset) / (N - 1 + 2 * offset))
    
    # Convert to cartesian
    x[2:] = radius * np.sin(phi) * np.cos(theta)
    y[2:] = radius * np.sin(phi) * np.sin(theta)
    z[2:] = radius * np.cos(phi)
    # The area of each patch is approximated by surface area divided by number of points
    area = 4 * np.pi * radius ** 2 / N
    return x, y, z, theta, phi, area
    

def oseen_tensor_surface(x, y, z, regularization_offset, dA, viscosity):
    """Find Oseen tensor for all multiple points on the surface of the squirmer.

    Args:
        x (1d array of floats): x-coordinate of all points
        y (1d array of floats): y-coordinate of all points
        z (1d array of floats): z-coordinate of all points
        epsilon (float): The regularization offset
        dA (float): Area of each patch, assumed the same for all points
        viscosity (float): Viscosity of the fluid.

    Returns:
        _type_: _description_
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
    
    S = (S_diag + S_off_diag + S_off_diag.T) * oseen_factor / r_epsilon_cubed
    return S 


def oseen_tensor(regularization_offset, dA, viscosity, evaluation_points, source_points=None):
    epsilon = regularization_offset 
    oseen_factor = dA / (8 * np.pi * viscosity)
    
    # Skal klart opdateres. Kan forhåbentlig definere r = x - xi, og ri*rj = r[:, None] * r[None, :]
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
    
    S = np.zeros((3*N_points, 3*N_sphere))  # three coordinates so multiply by three
    
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


def test_oseen_given_field():
    N = 250
    r = 1.
    eps = 0.1
    viscosity = 1
    x, y, z, _, _, area = canonical_fibonacci_lattice(N, r)
    
    # Boundary Conditions
    vx = 1 + 0 * x
    vy = 1 + 0 * x
    vz = 0 * x
    
    # Stack
    x = np.stack((x, y, z)).T
    v = np.stack((vx, vy, vz)).T
    v = np.reshape(v, -1, order="F")
    
    # Få Oseen matrix på overfladen og løs for kræfterne
    A = oseen_tensor(regularization_offset=eps, dA=area, viscosity=viscosity,
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
    A_e = oseen_tensor(regularization_offset=eps, dA=area, viscosity=viscosity,
                     evaluation_points=x_e, source_points=x)
    v_e = A_e @ F
    v_e = np.reshape(v_e, (len(v_e)//3, 3), order="F")
    
    # Return to lab frame by subtracting boundary conditions
    v_e[:, 0] -= 1
    v_e[:, 1] -= 1

    # Remove values inside squirmer
    r2 = np.sum(x_e**2, axis=1)
    v_e[r2 < r ** 2, :] = 0
    
    # Plot
    plt.quiver(x_e[:, 0], x_e[:, 1], v_e[:, 0], v_e[:, 1])
    plt.show()
    
    
def test_oseen_given_force():
    eps = 0.1
    viscosity = 1
    xx = np.linspace(-1, 1, 25)
    X, Y = np.meshgrid(xx, xx)
    X = X.ravel()
    Y = Y.ravel()
    Z = 0 * X
    
    # Source point and force
    xi = np.zeros(3).reshape(1, -1)
    F = np.zeros(3)
    F[0] = 1.
    
    x_e = np.stack((X, Y, Z)).T
    print(x_e.shape)
    O = oseen_tensor(regularization_offset=eps, dA=0.5, viscosity=viscosity, 
                        evaluation_points=x_e, source_points=xi)
    v = O @ F
    v = np.reshape(v, (len(v)//3, 3), order='F')
        
    vx = v[:, 0]
    vy = v[:, 1]
    
    plt.quiver(X, Y, vx, vy)
    plt.show()
    
if __name__ == "__main__":
    test_oseen_given_field()
    test_oseen_given_force()
