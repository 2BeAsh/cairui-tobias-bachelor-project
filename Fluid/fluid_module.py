import numpy as np


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
    
    #u_z += B_01 * 4 / (3 * a ** 3)
    #u_y += -B_tilde_11 * 4 / (3 * a ** 3)
    #u_x += -B_11 * 4 / (3 * a ** 3)
    return u_x, u_y, u_z
    

if __name__ == "__main__":
    from scipy.special import lpmv, lpmn


    def P_legendre(n, m):
        pass

    def field_polar_no_loop(distance, squirmer_radius, N, B, B_tilde):
        """
        distance (float): Distance between agent and target
        squirmer_radius (float): Radius of the squirmer agent
        N (int): Upper limit of n-sum
        B 
        """
        n_vals = np.arange(1, N+1)
        phi = np.pi / 2
        u_r = 0
        u_theta = 0
        for m in range(N):
            m_vals = np.arange(m)
            indices = tuple(zip(m_vals, n_vals-1))  # n values and indices are shifted by 1
            u_r += np.sum(
                (n_vals + 1) * P_legendre(n_vals, m_vals) / np.power(distance, n_vals+2)
                * ((distance / squirmer_radius) ** 2 - 1)
                * (B[indices] * np.cos(m_vals * phi) + B_tilde[indices] * np.sin(m_vals * phi))
            )
            u_theta += (
                
            )


    xx = np.cos(np.linspace(0, 2 * np.pi, 10))
    m = 1
    n = 2
    theta = np.pi/2
    val, diffval = lpmn(m, n, np.cos(theta))
    #print(diffval)
    #print(legendre_poly_prime(n, m, theta))