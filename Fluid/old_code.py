import numpy as np
# From fluid_module
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


def field_polar_snip():
    pass
"""         m_arr = np.arange(n+1)  # Since endpoints are not included, all must +1
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
        u_phi += np.sum(u_phi_arr) """