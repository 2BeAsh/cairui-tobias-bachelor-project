import numpy as np


def legendre_poly(n, m, x):
    """Associated Legendre Polynomials for n<=4

    Args:
        n (int): n mode
        m (int): m mode
        x (1d array of floats): The x-values the polynomials are evaluated at.

    Returns:
        (1d array of floats size x): P_n^m(x)
    """
    if n == 1 and m == 0:
        return np.cos(x)
    elif n == 2 and m == 0:
        return 1 / 2 * (3 * np.cos(x) ** 2 - 1)
    elif n == 3 and m == 0:
        return 1 / 2 * (5 * np.cos(x) ** 3 - 3 * np.cos(x))
    elif n == 4 and m == 0:
        return 1 / 8 * (35 * np.cos(x) ** 4 - 30 * np.cos(x) ** 2 + 3)
    elif n == 1 and m == 1:
        return -np.sin(x)
    elif n == 2 and m == 1:
        return -3 * np.cos(x) * np.sin(x)
    elif n == 2 and m == 2:
        return 3 * np.sin(x) ** 2
    elif n == 3 and m == 1:
        return 3/2 * (1 - 5 * np.cos(x)**2)*(1 - np.cos(x)**2)**(1/2)
    elif n == 3 and m == 2:
        return 15 * np.cos(x) * (1 - np.cos(x) ** 2)
    elif n == 3 and m == 3:
        return -15 * (1 - np.cos(x) ** 2) ** (3 / 2)
    elif n == 4 and m == 1:
        return 5/2 * np.cos(x) * (3 - 7 * np.cos(x) ** 2) * ( 1 - np.cos(x) ** 2) ** (1 / 2)
    elif n == 4 and m == 2:
        return 15/2 * (7 * np.cos(x) ** 2 - 1) * (1 - np.cos(x) ** 2) 
    elif n == 4 and m == 3:
        return -105 * np.cos(x) * (1 - np.cos(x) ** 2) ** (3 / 2)
    elif n == 4 and m == 4:
        return 105*(1 - np.cos(x)**2)**2


def legendre_poly_m_sin(n, m, x):
    """Associated Legendre Polynomials times m divided by sin(x) for n<=4

    Args:
        n (int): n mode
        m (int): m mode
        x (1d array of floats): The x-values the polynomials are evaluated at.

    Returns:
        (1d array of floats size x): P_n^m(x) * m / sin(x)
    """
    if n == 1 and m == 0:
        return np.cos(x)
    elif n == 2 and m == 0:
        return m
    elif n == 3 and m == 0:
        return m
    elif n == 4 and m == 0:
        return m
    elif n == 1 and m == 1:
        return -1 * m
    elif n == 2 and m == 1:
        return -3 * np.cos(x) * m
    elif n == 2 and m == 2:
        return 3 * (1 - np.cos(x) ** 2) ** (1 / 2) * m
    elif n == 3 and m == 1:
        return 3/2 * (1 - 5 * np.cos(x)**2) * m
    elif n == 3 and m == 2:
        return 15 * np.cos(x) * (1 - np.cos(x) ** 2) ** (1 / 2) * m
    elif n == 3 and m == 3:
        return -15 * (1 - np.cos(x) ** 2) * m
    elif n == 4 and m == 1:
        return 5/2 * np.cos(x) * (3 - 7 * np.cos(x) ** 2) * m
    elif n == 4 and m == 2:
        return 15/2 * (7 * np.cos(x) ** 2 - 1) * (1 - np.cos(x) ** 2) ** (1 / 2) * m
    elif n == 4 and m == 3:
        return -105 * np.cos(x) * (1 - np.cos(x) ** 2) * m 
    elif n == 4 and m == 4:
        return 105*(1 - np.cos(x)**2) ** (3 / 2) * m


def legendre_poly_deriv_sin(n, m, x):
    """First derivative of Associated Legendre Polynomials times sin(x) for n<=4

    Args:
        n (int): n mode
        m (int): m mode
        x (1d array of floats): The x-values the polynomials are evaluated at.

    Returns:
        (1d array of floats size x): P'_n^m(x) * sin(x)
    """
    if n == 1 and m == 0:
        return 1 * np.sin(x)
    elif n == 2 and m == 0: #done
        return 3  * np.cos(x) * np.sin(x)
    elif n == 3 and m == 0: #done
        return 1 / 2 * (-3 + 3 * 5 * np.cos(x) ** 2) * np.sin(x)
    elif n == 4 and m == 0: #done
        return (17.5 * np.cos(x)**3 - 7.5 * np.cos(x)) * np.sin(x)
    elif n == 1 and m == 1: #done
        return np.cos(x)
    elif n == 2 and m == 1: #done
        return 3 * np.cos(x)**2 - 3 * (1 - np.cos(x)**2) 
    elif n == 2 and m == 2: #done
        return -6 * np.cos(x) * np.sin(x)
    elif n == 3 and m == 1:
        return - np.cos(x)*(1.5 - 7.5 * np.cos(x)**2)  - 15 * np.cos(x) * (1 - np.cos(x) ** 2) 
    elif n == 3 and m == 2:
        return (15 - 45 * np.cos(x)**2) * np.sin(x) 
    elif n == 3 and m == 3:
        return 45 * np.cos(x) * (1 - np.cos(x)**2 )
    elif n == 4 and m == 1:
        return (-2.5 * np.cos(x) **2 * (3 - 7 * np.cos(x) ** 2) 
                - 35 * np.cos(x) ** 2 * (1 - np.cos(x) ** 2) 
                + 2.5 * (1 - np.cos(x) ** 2) * (3 - 7 * np.cos(x) ** 2))
    elif n == 4 and m == 2:
        return (105 * np.cos(x) * (1 - np.cos(x) ** 2) - 2 * np.cos(x) * (52.5 * np.cos(x) ** 2 - 7.5)) * np.sin(x)
    elif n == 4 and m == 3:
        return 315 * np.cos(x)**2 * (1 - np.cos(x) ** 2)  - 105 * (1 - np.cos(x) ** 2)
    elif n == 4 and m == 4:
        return -420 * np.cos(x) * (1 - np.cos(x) ** 2) * np.sin(x)


def field_polar(N, r, theta, phi, a, B, B_tilde, C, C_tilde):
    """Calculate the field in polar coordinates

    Args:
        N (int larger than 1): Max possible mode
        r (float): Distance between target and agent (prey and squirmer)
        theta (1d array of floats): Angle between vertical axis z and target
        phi (1d array of floats): Angle between horizontal axis and target. Must have same size as theta
        a (float): Squirmer radius
        B ((N+1, N+1)-array): Modes
        B_tilde ((N+1, N+1)-array): Modes
        C ((N+1, N+1)-array)): Modes
        C_tilde ((N+1, N+1)-array): Modes

    Returns:
        u_r (1d array of float size theta): 
            Velocity in the radial direction
        u_theta (1d array of float size thet):
            Angular velocity in theta
        u_phi (1d array of float size thet):
            Angular velocity in phi      
    """ 
    # Lower than N=2 values
    u_r = 4 / (3 * r ** 3) * (B[1, 1] * np.sin(theta) * np.cos(phi) 
                              + B_tilde[1, 1] * np.sin(theta) * np.sin(phi) 
                              - B[0, 1] * np.cos(theta))
    u_theta = - 2 / (3 * r ** 3) * (B[1, 1] * np.cos(theta) * np.cos(phi)
                                    + B_tilde[1, 1] * np.cos(theta) * np.sin(phi)
                                    + B[0, 1] * np.sin(theta))
    u_phi = 2 / (3 * r ** 3) * (B[1, 1] * np.sin(phi)
                                - B_tilde[1, 1] * np.cos(phi))

    # Calculate associated Legendre polynomials for all possible m and n values, evaluated in all theta values
    LP = np.empty(shape=(N-2+1, N+1, len(theta)))  # n values, m values, theta values
    LP_sin_m = np.empty_like(LP)
    LP_deriv_sin = np.empty_like(LP)
    for n in range(2, N+1):
        for m in range(N+1):
            LP[n-2, m, :] = legendre_poly(n, m, theta)  # ALP "Associated Legende Polynomial"
            LP_sin_m[n-2, m, :] = legendre_poly_m_sin(n, m, theta)  # ALP times m times sin(theta)
            LP_deriv_sin[n-2, m, :] = legendre_poly_deriv_sin(n, m, theta)  # Derivative of ALP times sin(theta)
    
    # Expand phi dimensions for matrix multiplication
    phi = np.expand_dims(phi, axis=0)
    
    # Sum from n=2 to N
    for n in np.arange(2, N+1):
        m_arr = np.expand_dims(np.arange(n+1), axis=1)
        # Set up modes and ALP matrices
        LP_matrix = LP[n-2, :n+1, :]  # Minimum n value in LP is n=2
        LP_sin_m_matrix = LP_sin_m[n-2, :n+1, :]
        LP_deriv_sin_matrix = LP_deriv_sin[n-2, :n+1, :]
        B_arr = np.expand_dims(B[:n+1, n], axis=1)  # Extra dimension needed for proper multiplication
        B_tilde_arr = np.expand_dims(B_tilde[:n+1, n], axis=1)
        C_arr = np.expand_dims(C[:n+1, n], axis=1)
        C_tilde_arr = np.expand_dims(C_tilde[:n+1, n], axis=1)
        
        u_r_arr = ((n + 1) / r ** (n + 2) 
                   * ((r / a) ** 2 - 1)
                   * LP_matrix
                   * (B_arr * np.cos(m_arr @ phi) + B_tilde_arr * np.sin(m_arr @ phi))
        )
        u_theta_arr = (((n - 2) / (n * a ** 2 * r ** n) - 1 / r ** (n + 2))
                       * LP_deriv_sin_matrix
                       * (B_arr * np.cos(m_arr @ phi) + B_tilde_arr * np.sin(m_arr @ phi))
                       + 1 / r ** (n + 1) * LP_sin_m_matrix
                       * (C_tilde_arr * np.cos(m_arr @ phi) + C_arr * np.sin(m_arr @ phi))
        )
        u_phi_arr = (1 / r ** (n + 1) * LP_deriv_sin_matrix
                     * (C_arr * np.cos(m_arr @ phi) + C_tilde_arr * np.sin(m_arr @ phi))
                     - ((n - 2) / (n * a ** 2 * r ** n) - 1 / r ** (n + 2))
                     * LP_sin_m_matrix
                     * (B_tilde_arr * np.cos(m_arr @ phi) - B_arr * np.sin(m_arr @ phi))
        )
        u_r += np.sum(u_r_arr, axis=0)
        u_theta += np.sum(u_theta_arr, axis=0)
        u_phi += np.sum(u_phi_arr, axis=0)
                
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
        u_x (float): 
            Velocity in the x direction
        u_y (float): 
            Velocity in the y direction
        u_z (float): 
            Velocity in the z direction            
            """
    u_r, u_theta, u_phi = field_polar(N, r, theta, phi, a, B, B_tilde, C, C_tilde)
    u_z = np.cos(theta) * u_r - np.sin(theta) * u_theta
    u_y = u_r * np.sin(theta) * np.sin(phi) + u_theta * np.cos(theta) * np.sin(phi) + u_phi * np.cos(phi)
    u_x = np.sin(theta) * np.cos(phi) * u_r + np.cos(theta) * np.cos(phi) * u_theta - np.sin(phi) * u_phi
    # Convert to squirmer frame
    if not lab_frame:
            u_z += B[0, 1] * 4 / (3 * a ** 3) 
            u_y += -B_tilde[1, 1] * 4 / (3 * a ** 3)
            # u_x is unchanged, as the modes for this is unused
            print("hallo")
    
    return u_x, u_y, u_z

    
if __name__ ==  "__main__":
    import matplotlib.pyplot as plt
    N_sphere = 10
    distance_squirmer = 1
    max_mode = 4
    theta = np.array([0, 0.5, 1, 1.5]) * np.pi
    phi = np.array([0, 0.5, 1, 1.5]) * np.pi * 1.5 + 0.1
    squirmer_radius = 1
    B = np.random.uniform(size=(max_mode+1, max_mode+1))
    B_tilde = B / 2
    C = B / 3
    C_tilde = B / 4
    regularization_offset = 0.05
    viscosity = 1
    
    print(field_polar(max_mode, distance_squirmer, theta, phi, squirmer_radius, B, B_tilde, C, C_tilde))

    #force_on_sphere(N_sphere, distance_squirmer, max_mode, theta, phi, squirmer_radius, B, B_tilde, C, C_tilde, regularization_offset, viscosity, lab_frame=True)
    