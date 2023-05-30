import numpy as np
import associated_legendre_polynomials as alp


def field_polar(max_mode, r, theta, phi, squirmer_radius, mode_array):
    """Calculate the field in polar coordinates

    Args:
        max_mode (int larger than 1): Max possible n value
        r (float): Distance between target and agent (prey and squirmer)
        theta (1d array of floats): Angle between vertical axis z and target
        phi (1d array of floats): Angle between horizontal axis and target. Must have same size as theta
        a (float): Squirmer radius
        B ((max_mode+1, max_mode+1)-array): Modes
        B_tilde ((max_mode+1, max_mode+1)-array): Modes
        C ((max_mode+1, max_mode+1)-array)): Modes
        C_tilde ((max_mode+1, max_mode+1)-array): Modes

    Returns:
        u_r (1d array of float size theta): 
            Velocity in the radial direction
        u_theta (1d array of float size thet):
            Angular velocity in theta
        u_phi (1d array of float size thet):
            Angular velocity in phi      
    """ 
    # Unpack modes
    B = mode_array[0, :, :]
    B_tilde = mode_array[1, :, :]
    C = mode_array[2, :, :]
    C_tilde = mode_array[3, :, :]
    
    a = squirmer_radius
    
    # Lower than n=2 values
    u_r = 4 / (3 * r ** 3) * (B[1, 1] * np.sin(theta) * np.cos(phi) 
                              + B_tilde[1, 1] * np.sin(theta) * np.sin(phi) 
                              - B[0, 1] * np.cos(theta))
    u_theta = - 2 / (3 * r ** 3) * (B[1, 1] * np.cos(theta) * np.cos(phi)
                                    + B_tilde[1, 1] * np.cos(theta) * np.sin(phi)
                                    + B[0, 1] * np.sin(theta))
    u_phi = 2 / (3 * r ** 3) * (B[1, 1] * np.sin(phi)
                                - B_tilde[1, 1] * np.cos(phi))

    # Calculate associated Legendre polynomials for all possible m and n values, evaluated in all theta values
    LP = np.zeros((max_mode+1, max_mode+1, np.size(theta)))  # n values, m values, theta values
    LP_sin_m = np.zeros_like(LP)
    LP_deriv_sin = np.zeros_like(LP)
    for n in range(2, max_mode+1):
        for m in range(max_mode+1):
            LP[m, n, :] = alp.associated_legendre_poly(n, m, theta)
            LP_sin_m[m, n, :] = alp.associated_legendre_poly_m_sin(n, m, theta)  # ALP times m times sin(theta)
            LP_deriv_sin[m, n, :] = alp.associated_legendre_poly_deriv_sin(n, m, theta)  # Derivative of ALP times sin(theta)
    
    # Expand phi dimensions for matrix multiplication
    phi = np.expand_dims(phi, axis=0)
    
    # Sum from n=2 to max_mode
    for n in np.arange(2, max_mode+1):
        m_arr = np.expand_dims(np.arange(n+1), axis=1)
        # Set up modes and ALP matrices
        LP_matrix = LP[:n+1, n, :]  # m, n, theta
        LP_sin_m_matrix = LP_sin_m[:n+1, n, :]
        LP_deriv_sin_matrix = LP_deriv_sin[:n+1, n, :]
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
    
    
def field_cartesian(max_mode, r, theta, phi, squirmer_radius, mode_array, lab_frame=True):
    """Convert polar velocities to cartesian
    
    Args:
        max_mode (int larger than 1): Max possible mode
        r (float): Distance between target and agent (prey and squirmer)
        theta (float): Angle between vertical axis z and target
        phi (float): Angle between horizontal axis and target
        squirmer_radius (float): Squirmer radius
        B ((max_mode+1, max_mode+1)-array): Modes
        B_tilde ((max_mode+1, max_mode+1)-array): Modes
        C ((max_mode+1, max_mode+1)-array)): Modes
        C_tilde ((max_mode+1, max_mode+1)-array): Modes
        lab_frame (Bool): If chooses lab or squirmer frame

    Returns:
        u_x (float): 
            Velocity in the x direction
        u_y (float): 
            Velocity in the y direction
        u_z (float): 
            Velocity in the z direction            
            """
    u_r, u_theta, u_phi = field_polar(max_mode, r, theta, phi, squirmer_radius, mode_array)
    u_z = np.cos(theta) * u_r - np.sin(theta) * u_theta
    u_y = u_r * np.sin(theta) * np.sin(phi) + u_theta * np.cos(theta) * np.sin(phi) + u_phi * np.cos(phi)
    u_x = np.sin(theta) * np.cos(phi) * u_r + np.cos(theta) * np.cos(phi) * u_theta - np.sin(phi) * u_phi
    # Convert to squirmer frame
    if not lab_frame:
            u_z += B[0, 1] * 4 / (3 * squirmer_radius ** 3) 
            u_y += -B_tilde[1, 1] * 4 / (3 * squirmer_radius ** 3)
            # u_x is unchanged, as the modes for this is unused
    
    return u_x, u_y, u_z

    
if __name__ ==  "__main__":
    import matplotlib.pyplot as plt
    N_sphere = 80
    distance_squirmer = 1
    max_mode = 2
    theta = np.array([0, 0.5, 1, 1.5, 2]) * np.pi
    phi = np.array([0, 0.5, 1, 1.5, 2]) * np.pi * 1.5 + 0.1
    squirmer_radius = 1
    B = np.zeros((max_mode+1, max_mode+1))
    B_tilde = np.zeros_like(B)
    C = np.zeros_like(B)
    C_tilde = np.zeros_like(B)
    B[0, 1] = 1
    mode_array = np.array([B, B_tilde, C, C_tilde])
    
    regularization_offset = 0.05
    viscosity = 1
    
    print(field_polar(max_mode, distance_squirmer, theta, phi, squirmer_radius, mode_array))
