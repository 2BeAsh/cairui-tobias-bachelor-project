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
    

def constant_power_factor(squirmer_radius, viscosity, max_mode):
    """The factor in front of the mode terms which ensures that the power is constant

    Args:
        squirmer_radius (_type_): _description_
        viscosity (_type_): _description_
        max_mode (int): Highest value for n, highest mode

    Returns:
        _type_: _description_
    """
    n = np.arange(max_mode+1)  # Even though n cannot be 0, easier to keep track of indices and later code assumes starts at 0
    m = np.arange(max_mode+1)
    mode_factors = np.zeros((4, len(m), len(n)))  # [Mode, m, n]
    
    n1_common_factor = 64 / (3 * squirmer_radius ** 5) * np.pi * viscosity
    n2 = n[2:]
    m0_common_factor = 4 * n2 * (n2 + 1) * np.pi * viscosity / (squirmer_radius ** (2*n2 + 1))
    m1 = m[1:]
    with np.errstate(divide='ignore'):  # Removes the divide by zero right after, so no need for warning
        n2m1_common_factor = (2 * n2[None, :] * (n2[None, :] + 1) * factorial(n2[None, :]+m1[:, None]) * np.pi * viscosity 
                            / (squirmer_radius ** (2*n2[None, :] + 1) * factorial(n2[None, :]-m1[:, None])))
    n2m1_common_factor = np.triu(n2m1_common_factor, k=-1)  # remove m > n cases
        
    # n == 1
    mode_factors[0, :2, 1] = n1_common_factor  # B01 and #B11
    mode_factors[1, 1, 1] = n1_common_factor  # B_tilde11
    if max_mode > 1:
        # m == 0, n>=2
        mode_factors[0, 0, 2:] = 4 / (n2 ** 2 * squirmer_radius ** 2) * m0_common_factor  # B n>=2 m=0
        mode_factors[2, 0, 2:] = (n2 + 2) / (2 * n2 + 1) * m0_common_factor  # C n>=2 m=0
        # m>=1, n>=2
        mode_factors[0, 1:, 2:] = 4 / (n2[None, :] ** 2 * squirmer_radius) * n2m1_common_factor  # B
        mode_factors[1, 1:, 2:] = mode_factors[0, 1:, 2:]  # B tilde
        mode_factors[2, 1:, 2:] = (n2[None, :] + 2) / (2 * n2[None, :] + 1) * n2m1_common_factor  # C tilde
        mode_factors[3, 1:, 2:] = mode_factors[2, 1:, 2:]  # C tilde

    # Remove all cases where m>n. Upper triangular part shifted by 1 sat to zero, as n start at 1 and m at 0
    mode_factors = np.triu(mode_factors)
    return mode_factors


def normalized_modes(modes, max_mode, squirmer_radius, viscosity):
    mode_factors = constant_power_factor(squirmer_radius, viscosity, max_mode)
    non_zero_idx = np.nonzero(mode_factors)
    #mode_normalized = modes / np.sqrt(power_total)
    mode_factors[non_zero_idx] = modes / np.sqrt(modes ** 2 @ mode_factors[non_zero_idx])
    return mode_factors
    

if __name__ == "__main__":
    phi_vals = np.ones(45-1)
    x_n_sphere = n_sphere_angular_to_cartesian(phi_vals, radius=1)
    factors = constant_power_factor(squirmer_radius=1.1, viscosity=1, max_mode=2)
    factors_non_zero = factors[np.nonzero(factors)]
    
    # -- N-sphere power --
    #print(np.array_str(factors, precision=2, suppress_small=True))
    #print(np.array_str(factors_non_zero, precision=2, suppress_small=True))  # Easier to compare when printing fewer decimalts    
    #print(np.array_str(x_n_sphere, precision=3, suppress_small=True))
    #print(np.array_str(x_n_sphere / np.sqrt(factors_non_zero), precision=3, suppress_small=True))
    
    # -- Normalized power --
    modes = normalized_modes(np.ones(45), max_mode=4, squirmer_radius=1, viscosity=1)
    print(np.array_str(modes, precision=6, suppress_small=True))
