import numpy as np
from scipy.special import lpmn

# Benyttes ikke
def field_polar_lab(r, theta, phi, B_11, B_tilde_11, B_01):
    """
    r: afstand fra centrum af squirmer
    theta, phi: polar vinkler. phi=pi/2 hvis ser på y og z akse som vi har gjort indtil videre
    
    B_11, B_tilde_11, B_01: Tal mellem -1 og 1. IKKE en matrix men et ENKEL tal. ET tal IKKE Matrix
    
    return hastighed i polare koordinater i LAB frame
    """
    u_r = 4 / (3 * r ** 3) * (B_11 * np.sin(theta) * np.cos(phi) + B_tilde_11 * np.sin(theta) * np.sin(phi) - B_01 * np.cos(theta))
    u_theta = - 2 / (3 * r ** 3) * (B_11 * np.cos(theta) * np.cos(phi) + B_tilde_11 * np.cos(theta) * np.sin(phi) - B_01 * np.sin(theta))
    u_phi = 2 / (3 * r ** 3) * (B_11 * np.sin(phi) - B_tilde_11 * np.cos(phi))
    return u_r, u_theta, u_phi

# Benyttes ikke
def field_cartesian_squirmer(r, theta, phi, a, B_11, B_tilde_11, B_01):
    """
    r: afstand fra centrum af squirmer
    theta, phi: polar vinkler. phi=pi/2 hvis ser på y og z akse som vi har gjort indtil videre
    
    LÆG MÆRKE TIL at denne funktion tager én yderligere parameter som er a:
    a: radius af squirmer.
    
    B_11, B_tilde_11, B_01: Tal mellem -1 og 1. IKKE en matrix men et ENKEL tal. ET tal IKKE Matrix
    
    return hastighed i kartetiske koordinater i SQUIRMER frame
    """
    u_r, u_theta, u_phi = field_polar_lab(r, theta, phi, B_11, B_tilde_11, B_01)
    
    u_z = np.cos(theta) * u_r - np.sin(theta) * u_theta
    u_y = u_r * np.sin(theta) * np.sin(phi) + u_theta * np.cos(theta) * np.sin(phi) + u_phi * np.cos(phi)
    u_x = u_r * np.sin(theta) * np.cos(phi) + u_theta * np.cos(theta) * np.cos(phi) - u_phi * np.sin(phi)
    
    u_z += B_01 * 4 / (3 * a ** 3)
    u_y += -B_tilde_11 * 4 / (3 * a ** 3)
    u_x += -B_11 * 4 / (3 * a ** 3)
    return u_x, u_y, u_z


def field_polar(N, r, theta, phi, a, B, B_tilde, C, C_tilde):
    """Calculate the field in polar coordinates

    Args:
        N (int larger than 1): Max possible mode
        r (float): Distance between target and agent (prey and squirmer)
        theta (float): Angle between vertical axis z and target
        phi (float): Angle between horizontal axis and target
        a (float): Squirmer radius
        B ((N+1, N+1)-array): Modes
        B_tilde ((N+1, N+1)-array): Modes
        C ((N+1, N+1)-array)): Modes
        C_tilde ((N+1, N+1)-array): Modes

    Returns:
        u_r (float): 
            Velocity in the radial direction
        u_theta (float):
            Angular velocity        
    """
    # NOTE
    # Must be changed to handle forces, which requires theta and phi to be vectors. 
    
    LP, LP_deriv = lpmn(N, N, np.cos(theta))
    r_first_term = 4 / (3 * r ** 3) * (B[1, 1] * np.sin(theta) * np.cos(phi) 
                                       + B_tilde[1, 1] * np.sin(theta) * np.sin(phi) 
                                       - B[0, 1] * np.cos(theta))
    theta_first_term = - 2 / (3 * r ** 3) * (B[1, 1] * np.cos(theta) * np.cos(phi)
                                             + B_tilde[1, 1] * np.cos(theta) * np.sin(phi)
                                             + B[0, 1] * np.sin(theta))
    phi_first_term = 2 / (3 * r ** 3) * (B[1, 1] * np.sin(phi)
                                         - B_tilde[1, 1] * np.cos(phi))
    u_r = r_first_term
    u_theta = theta_first_term 
    u_phi = phi_first_term
    for n in np.arange(2, N+1):  # Sum starts at n=2 and ends at N
        m_arr = np.arange(n+1)  # Since endpoints are not included, all must +1
        #mask = (slice(n+1), slice(n))
        LP_arr = LP[:n+1, n]
        LP_deriv_arr = LP_deriv[:n+1, n]
        B_arr = B[:n+1, n]
        B_tilde_arr = B_tilde[:n+1, n]
        C_arr = C[:n+1, n]
        C_tilde_arr = C_tilde[:n+1, n]
        # Array with velocities for each m can be summed to give the total value of the inner sum.
        u_r_arr = ((n + 1) * LP_arr / r ** (n + 2)
                   * ((r / a) ** 2 - 1)
                   * (B_arr * np.cos(m_arr * phi) + B_tilde_arr * np.sin(m_arr * phi)) 
        )
        u_theta_arr = (np.sin(theta) * LP_deriv_arr
                   * ((n - 2) / (n * a ** 2 * r ** n) - 1 / r ** (n + 2))
                   * (B_arr * np.cos(m_arr * phi) + B_tilde_arr * np.sin(m_arr * phi))
                   + m_arr * LP_arr / (r ** (n + 1) * np.sin(theta)) 
                   * (C_tilde_arr * np.cos(m_arr * phi) - C_arr * np.sin(m_arr * phi))
        )
        u_phi_arr = (np.sin(theta) * LP_deriv_arr / r ** (n + 1)
                     * (C * np.cos(m_arr * phi) + C_tilde * np.sin(m_arr * phi))
                     - m_arr * LP_arr / np.sin(theta)
                     * ((n - 2) / (n * a ** 2 * r ** n) - 1 / r ** (n + 2))
                     * (B_tilde * np.cos(m_arr * phi) - B * np.sin(m_arr * phi))
        )
        u_r += np.sum(u_r_arr)
        u_theta += np.sum(u_theta_arr)
        u_phi += np.sum(u_phi_arr)
     
    return u_r, u_theta, u_phi
    
    
def field_cartesian(N, r, theta, phi, a, B, B_tilde, C, C_tilde, lab_frame=True):
    """Convert polar velocities to cartesian
    
    Args:
        N (int larger than 1): Max possible mode
        r (float): Distance between target and agent (prey and squirmer)
        theta (float): Angle between vertical axis z and target
        phi (float): Angle between horizontal axis and target
        a (float): Squirmer radius
        B ((N+1, N+1)-array): Modes
        B_tilde ((N+1, N+1)-array): Modes
        C ((N+1, N+1)-array)): Modes
        C_tilde ((N+1, N+1)-array): Modes
        lab_frame (Bool): If chooses lab or squirmer frame

    Returns:
        u_r (float): 
            Velocity in the radial direction
        u_theta (float):
            Angular velocity        
    """
    u_r, u_theta, u_phi = field_polar(N, r, theta, phi, a, B, B_tilde, C, C_tilde)
    
    u_z = np.cos(theta) * u_r - np.sin(theta) * u_theta
    u_y = u_r * np.sin(theta) * np.sin(phi) + u_theta * np.cos(theta) * np.sin(phi) + u_phi * np.cos(phi)
    u_x = np.sin(theta) * np.cos(phi) * u_r + np.cos(theta) * np.cos(phi) * u_theta - np.sin(phi) * u_phi
    if not lab_frame:  # Convert to squirmer frame
            u_z += B[0, 1] * 4 / (3 * a ** 3) 
            u_y += -B_tilde[1, 1] * 4 / (3 * a ** 3)
            # u_x is unchanged, as the modes for this is unused
    
    return u_x, u_y, u_z


# Forces
def canonical_fibonacci_lattice(N, radius):
    """Calculate N points uniformly distributed on the surface of a sphere with given radius, using the canonical Fibonacci Lattice method.

    Args:
        N (int): Number of points
        radius (float): Radius of sphere.

    Returns:
        Tupple of three 1d-arrays and a float: Returns cartesian coordinates of the points distributed on the spherical surface, and the approximate area each point are given.
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
    

def oseen_tensor(x, y, z, epsilon, dA, viscosity):
    """Find Oseen tensor for all multiple points.

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
    N = np.shape(x)[0]
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dz = z[:, None] - z[None, :]
    r = np.sqrt(dx**2 + dy**2 + dz**2)  # Symmetrical, could save memory?
    
    # Expand r to match S - OPTIMIZEABLE
    r_expanded = np.repeat(r, 3, axis=0)
    r_expanded = np.repeat(r_expanded, 3, axis=1)
    r_epsilon = np.sqrt(r_expanded**2 + epsilon**2)
    
    S_diag = np.zeros((3*N, 3*N))  # 3 variables, so 3N in each direction
    S_diag[:N, :N] = dx
    S_diag[N:2*N, N:2*N] = dy
    S_diag[2*N:3*N, 2*N:3*N] = dz
    
    S_off_diag = np.zeros_like(S_diag)
    S_off_diag[0:N, N:2*N] = dx * dy  # Element wise multiplication
    S_off_diag[0:N, 2*N:3*N] = dx * dz
    S_off_diag[N:2*N, 2*N:3*N] = dy * dz
    
    S = ((r_expanded**2 + 2 * epsilon**2) * S_diag
         + S_off_diag
         + S_off_diag.T
    ) / r_epsilon ** 3
    
    A = S * dA / (4 * np.pi * viscosity)
    return A


def force_on_sphere(N_sphere, distance_squirmer, max_mode, theta, phi, squirmer_radius, B, B_tilde, C, C_tilde, regularization_offset, viscosity, lab_frame=True):
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
    assert np.array([N_sphere, distance_squirmer, max_mode, squirmer_radius, regularization_offset, viscosity]).all() > 0
    # Get the A matrix from the Oseen tensor
    x_sphere, y_sphere, z_sphere, theta_sphere, phi_sphere, area = discretized_sphere(N_sphere, squirmer_radius)
    A_oseen = oseen_tensor(x_sphere, y_sphere, z_sphere, regularization_offset, area, viscosity)
    # Get velocities in each of the points
    u_x, u_y, u_z = field_cartesian(max_mode, distance_squirmer=squirmer_radius, 
                                    theta=theta_sphere, phi=phi_sphere, 
                                    squirmer_radius=squirmer_radius, 
                                    B=B, B_tilde=B_tilde, 
                                    C=C, C_tilde=C_tilde, 
                                    lab_frame=lab_frame)
    u_comb = np.array([u_x, u_y, u_z]).flatten()

    # Solve for the forces, A_oseen @ forces = u_comb
    force_arr = np.linalg.solve(A_oseen, u_comb)
    return force_arr
    

    
    
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    N_sphere = 10
    distance_squirmer = 1
    max_mode = 2
    theta = np.pi / 3
    phi = np.pi / 2
    squirmer_radius = 1
    B = np.random.uniform(size=(max_mode+1, max_mode+1))
    B_tilde = B / 2
    C = B / 3
    C_tilde = B / 4
    regularization_offset = 0.05
    viscosity = 1
    force_on_sphere(N_sphere, distance_squirmer, max_mode, theta, phi, squirmer_radius, B, B_tilde, C, C_tilde, regularization_offset, viscosity, lab_frame=True)
    
"""     
    N = 2
    r = 3
    a = 1
    theta = 0

    #vals, diff = lpmn(N, N, np.cos(theta))
    #print(diff)
    #u_r, u_theta = field_polar(N, r, theta, a, B, B_tilde, C, C_tilde)
    #print(u_r, u_theta)
    x, y, z,_ = discretized_sphere(N=1000, radius=1)
    fig = plt.figure(dpi=200)
    ax = fig.add_subplot(projection="3d")
    ax.scatter(x, y, z)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    plt.show()
    plt.close()
    
    x_can, y_can, z_can, _ = canonical_fibonacci_lattice(N=1000, radius=1)
    fig_1 = plt.figure(dpi=200)
    ax_1 = fig_1.add_subplot(projection="3d")
    ax_1.scatter(x_can, y_can, z_can)
    ax_1.set_xlabel("x")
    ax_1.set_ylabel("y")
    ax_1.set_zlabel("z")
    plt.show()
    plt.close() """
    