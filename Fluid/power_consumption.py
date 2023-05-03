import numpy as np
from scipy.special import factorial


def n_sphere_angular_to_cartesian(phi, radius):
    """ Convert n-sphere angles to cartesian coordinates. 
    Formula: https://i.stack.imgur.com/CFSsT.png 
    Taken from: https://stackoverflow.com/questions/20133318/n-sphere-coordinate-system-to-cartesian-coordinate-system

    Args:
        phi (ndarray): Angular coordinates, phi_1, phi_2, ..., phi_n-1
        radius (float): Radius of the sphere
    """
    # Add extra term to phi
    phi_expanded = np.concatenate((np.array([2*np.pi]), phi))  # using 2pi saves one operation in the cosine part
    
    # Sine factors with cumprod. First term is set to 1
    sine = np.sin(phi_expanded)
    sine[0] = 1
    sine_cumprod = np.cumprod(sine)
    
    # Cosine
    cosine = np.cos(phi_expanded)
    cosine_roll = np.roll(cosine, -1)
    
    return radius * sine_cumprod * cosine_roll
    

def constant_power_factor(squirmer_radius, viscosity):
    """The factor in front of the mode terms which ensures that the power is constant

    Args:
        squirmer_radius (_type_): _description_
        viscosity (_type_): _description_

    Returns:
        _type_: _description_
    """
    max_mode = 4  # Higest value for n
    n = np.arange(max_mode+1)  # Even though n cannot be 0, easier to keep track of indices and later code assumes starts at 0
    m = np.arange(max_mode+1)
    mode_factors = np.zeros((4, len(n), len(m)))  # [Mode, n, m]
    
    n1_common_factor = 64 / (3 * squirmer_radius ** 5) * np.pi * viscosity
    n2 = n[2:]
    m0_common_factor = 4 * n2 * (n2 + 1) * np.pi * viscosity / (squirmer_radius ** (2*n2 + 1))
    m1 = m[1:]
    n2m1_common_factor = 2 * n2[:, None] * (n2[:, None] + 1) * factorial(n2[:, None]+m1[None, :]) * np.pi * n2[:, None] / (squirmer_radius ** (2*n2[:, None] + 1) * factorial(n2[:, None]-m1[None, :]))
    n2m1_common_factor = np.tril(n2m1_common_factor, k=1)
        
    # n == 1
    mode_factors[0, 1, :2] = n1_common_factor  # B10 and #B11
    mode_factors[1, 1, 1] = n1_common_factor  # B_tilde11
    # m == 0, n>=2
    mode_factors[0, 2:, 0] = 4 / (n2 ** 2 * squirmer_radius ** 2) * m0_common_factor  # B n>=2 m=0
    mode_factors[2, 2:, 0] = (n2 + 2) / (2 * n2 + 1) * m0_common_factor  # C n>=2 m=0
    # m>=1, n>=2
    mode_factors[0, 2:, 1:] = 4 / (np.pi ** 2 * squirmer_radius) * n2m1_common_factor
    mode_factors[1, 2:, 1:] = mode_factors[0, 2:, 1:]
    mode_factors[2, 2:, 1:] = (n2[:, None] + 2) / (2 * n2[:, None] + 1) * n2m1_common_factor
    mode_factors[3, 2:, 1:] = mode_factors[2, 2:, 1:]

    # Remove all cases where m>n. Upper triangular part shifted by 1 sat to zero, as n start at 1 and m at 0
    mode_factors = np.tril(mode_factors, k=1)
    return mode_factors


if __name__ == "__main__":
    factors = constant_power_factor(squirmer_radius=1.5, viscosity=1)
    print(np.array_str(factors, precision=2, suppress_small=True))  # Easier to compare when printing fewer decimalts    
