import numpy as np
from scipy.special import lpmn


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


def field_polar_lab_updated(N, r, theta, a, B, B_tilde, C, C_tilde):
    """Calculate the field in polar coordinates

    Args:
        N (int larger than 1): Max possible mode
        r (float): Distance between target and agent (prey and squirmer)
        theta (float): Angle between vertical axis z and target
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
    phi = np.pi / 2
    LP, LP_deriv = lpmn(N, N, np.cos(theta))
    r_first_term = 4 / (3 * r ** 3) * (B[1, 1] * np.sin(theta) * np.cos(phi) 
                                       + B_tilde[1, 1] * np.sin(theta) * np.sin(phi) 
                                       - B[0, 1] * np.cos(theta))
    theta_first_term = - 2 / (3 * r ** 3) * (B[1, 1] * np.cos(theta) * np.cos(phi)
                                             + B_tilde[1, 1] * np.cos(theta) * np.sin(phi)
                                             + B[0, 1] * np.sin(theta))
    u_r = r_first_term
    u_theta = theta_first_term 
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
        u_r += np.sum(u_r_arr)
        u_theta += np.sum(u_theta_arr)
     
    return u_r, u_theta
    
    
def field_cartesian_lab_updated(N, r, theta, a, B, B_tilde, C, C_tilde):
    """Convert polar velocities to cartesian
    
    Args:
        N (int larger than 1): Max possible mode
        r (float): Distance between target and agent (prey and squirmer)
        theta (float): Angle between vertical axis z and target
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
    phi = np.pi / 2
    u_phi = 0
    u_r, u_theta = field_polar_lab_updated(N, r, theta, a, B, B_tilde, C, C_tilde)
    
    u_z = np.cos(theta) * u_r - np.sin(theta) * u_theta
    u_y = u_r * np.sin(theta) * np.sin(phi) + u_theta * np.cos(theta) * np.sin(phi) + u_phi * np.cos(phi)
    
    return u_y, u_z




if __name__ == "__main__":
    N = 2
    r = 3
    a = 1
    B = np.random.uniform(size=(N+1, N+1))
    B_tilde = B / 2
    C = B / 3
    C_tilde = B / 4
    theta = np.pi/2
    #print(np.arange(10))
    #print(np.arange(10)[:2])
    vals, diff = lpmn(N, N, np.cos(theta))
    #print(vals[:N, N])
    u_r, u_theta = field_polar_lab_updated(N, r, theta, a, B, B_tilde, C, C_tilde)
    print(u_r, u_theta)