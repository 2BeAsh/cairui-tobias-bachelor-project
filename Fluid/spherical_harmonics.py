import numpy as np
from scipy.special import sph_harm


def expansion_coefficient_ml(m, l, func_val, phi, theta):
    """Numerically integrate the expansion coefficient expression.  

    Args:
        l (array_like): degree of spherical harmonics, l => 0
        m (array_like): order of spherical harmonics, |m| <= l. NOTE might only be used for m == 0
        func_val (ndarray): Function values on sphere evaluated at each (theta, phi) pair
        theta (ndarray): Longitudinal coordinate; must be in [0, pi]
        phi (ndarray): Colatiduninal/Polar coordinate; must be in [0, 2*pi]

    Returns:
        array_like: expansion coefficient for each l and m given.
    """
    N = len(theta)
    Y_ml = np.conjugate(sph_harm(m, l, phi, theta))  # Note the different definitions of angle and m & l.
    # print("Spherical Harmonics")
    # print(Y_ml)
    # print("Forces")
    # print(func_val)
    # print("Sin theta")
    # print(np.sin(theta))
    # print("Product")
    # print(Y_ml * func_val * np.sin(theta))
    # print("Sum")
    # print(np.sum(Y_ml * func_val * np.sin(theta)))
    return 2 * (np.pi / N) ** 2 * np.sum(Y_ml * func_val * np.sin(theta))
    
    
def spherical_harmonic_representation(ml_pair, func_val, x_surface, radius):
    """Represent scalar field on a sphere by spherical harmonics. 

    Args:
        ml_pair ((n, 2)darray): List of (m, l) values
        func_val (ndarray): Scalar field values on sphere
        theta (ndarray): Longitudinal coordinate; must be in [0, pi]
        phi (ndarray): Colatiduninal/Polar coordinate; must be in [0, 2*pi]

    Returns:
        ndarray: The scalar field represented by the ml_pair spherical harmonics
    """
    x = x_surface[:, 0]
    y = x_surface[:, 1]
    z = x_surface[:, 2]
    theta = np.arccos(z / radius)  # [0, pi]
    phi = np.arctan2(y, x)  # [0, 2*pi]
    N = len(x)
    sph_har_arr = np.empty((len(ml_pair), N), dtype=complex)
    expa_coeff_arr = np.empty_like(sph_har_arr)
    
    for i, ml in enumerate(ml_pair):
        m = ml[0]
        l = ml[1]
        sph_har_arr[i, :] = sph_harm(m, l, phi, theta)
        expa_coeff_arr[i, :] = expansion_coefficient_ml(m, l, func_val, phi, theta)
    return np.sum((sph_har_arr * expa_coeff_arr).real, axis=0)
    
    
if __name__ == "__main__":
    ml_pair = [[0, 0], [0, 1], [0, 2]]
    x_surface = np.arange(21).reshape(-1, 3)
    force = 3 * x_surface[:, 0] + 0.5 * x_surface[:, 1] + 3 * x_surface[:, 2]
    radius = np.max(x_surface)
    
    print(spherical_harmonic_representation(ml_pair, force, x_surface, radius))
    