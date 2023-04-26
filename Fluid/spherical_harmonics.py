import numpy as np
from scipy.special import sph_harm


def expansion_coordinates_lm(l, m, func_val, phi, theta):
    """Numerically integrate the expansion coefficient expression.  

    Args:
        l (array_like): degree of spherical harmonics, l => 0
        m (array_like): order of spherical harmonics, |m| <= l. NOTE should probably only be used for m == 0
        func_val (ndarray): Function values on sphere evaluated at each (theta, phi) pair
        theta (ndarray): Longitudinal coordinate; must be in [0, pi]
        phi (ndarray): Colatiduninal/Polar coordinate; must be in [0, 2*pi]

    Returns:
        array_like: expansion coefficient for each l and m given.
    """
    N = len(theta)
    Y_lm_real = sph_harm(m, l, phi, theta).real  # Note the different definitions of angle and m & l.
    return 2 * (np.pi / N) ** 2 * np.sum(Y_lm_real * func_val * np.sin(theta))
    
    
def spherical_harmonic_representation(lm_pair, func_val, x_surface, radius):
    """Represent scalar field on a sphere by spherical harmonics. 

    Args:
        lm_pair ((n, 2)darray): List of (l, m) values
        func_val (ndarray): Scalar field values on sphere
        theta (ndarray): Longitudinal coordinate; must be in [0, pi]
        phi (ndarray): Colatiduninal/Polar coordinate; must be in [0, 2*pi]

    Returns:
        ndarray: The scalar field represented by the lm_pair spherical harmonics
    """
    x = x_surface[:, 0]
    y = x_surface[:, 1]
    z = x_surface[:, 2]
    theta = np.arccos(z / radius)  # [0, pi]
    phi = np.arctan2(y, x)  # [0, 2*pi]
    N = len(x)
    sph_har_arr = np.empty((len(lm_pair), N))
    expa_coeff_arr = np.empty_like(sph_har_arr)
    
    for i, lm in enumerate(lm_pair):
        l = lm[0]
        m = lm[1]
        sph_har_arr[i, :] = sph_harm(m, l, phi, theta).real
        expa_coeff_arr[i, :] = expansion_coordinates_lm(l, m, func_val, phi, theta)
    
    return np.sum(sph_har_arr * expa_coeff_arr, axis=0)
    
    
if __name__ == "__main__":
    lm_pair = [[0, 0], [1, 0], [2, 0]]
    theta = np.linspace(0, np.pi, 20)
    phi = np.linspace(0, 2*np.pi, 20)
    force = 3 * theta + 5 * phi
    
    print(spherical_harmonic_representation(lm_pair, force, theta, phi))
    