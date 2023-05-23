import numpy as np
from scipy.special import sph_harm


def real_sph_harm(m, l, phi, theta):
    """Real expression of spherical harmonics defined by its complex analgoue. 
    Formula: https://en.wikipedia.org/wiki/Spherical_harmonics#Real_form
    Args:
        m (_type_): _description_
        l (_type_): _description_
        phi (_type_): _description_
        theta (_type_): _description_

    Returns:
        _type_: _description_
    """
    if m == 0:
        return sph_harm(m, l, phi, theta)
    elif m > 0: 
        return 1 / np.sqrt(2) * (sph_harm(-m, l, phi, theta) 
                                 + sph_harm(m, l, phi, theta))  # Condon-Shortley phase included in sph_harm.        
    else:  # m < 0
        return 1j / np.sqrt(2) * (sph_harm(m, l, phi, theta) 
                                  - sph_harm(-m, l, phi, theta))


def expansion_coefficient_ml(m, l, func_val, phi, theta):
    """Numerically integrate the expansion coefficient expression.  

    Args:
        l (array_like): degree of spherical harmonics, l >= 0
        m (array_like): order of spherical harmonics, |m| <= l.
        func_val (ndarray): Function values on sphere evaluated at each (theta, phi) pair
        theta (ndarray): Longitudinal coordinate; must be in [0, pi]
        phi (ndarray): Colatiduninal/Polar coordinate; must be in [0, 2*pi]

    Returns:
        array_like: expansion coefficient for given l and m.
    """
    N = len(theta)    
    Y_lm = real_sph_harm(m, l, phi, theta)
    return 2 * (np.pi / N) ** 2 * np.sum(Y_lm * func_val * np.sin(theta))
    
    
def spherical_harmonic_representation(ml_pair, func_val, x_surface, radius):
    """Represent scalar field on a sphere by spherical harmonics. 
    Formula: https://en.wikipedia.org/wiki/Spherical_harmonics#Spherical_harmonics_expansion

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
    sph_har_arr = np.empty((len(ml_pair), N), dtype=complex)  # NOTE bør egentlig være reel.
    expa_coeff_arr = np.empty_like(sph_har_arr)
    
    for i, ml in enumerate(ml_pair):
        m = ml[0]
        l = ml[1]
        sph_har_arr[i, :] = real_sph_harm(m, l, phi, theta)
        expa_coeff_arr[i, :] = expansion_coefficient_ml(m, l, func_val, phi, theta)
    return np.sum(sph_har_arr * expa_coeff_arr, axis=0)
    
    
if __name__ == "__main__":
    ml_pair = [[0, 0], [0, 1], [0, 2], [-1, 1]]
    x_surface = np.arange(21).reshape(-1, 3)
    force = 3 * x_surface[:, 0] + 0.5 * x_surface[:, 1] + 3 * x_surface[:, 2]
    radius = np.max(x_surface)
    
    phi = np.linspace(0, 2*np.pi, len(force))
    theta = np.linspace(0, np.pi, len(force))
    print(expansion_coefficient_ml(1, 1, force, phi, theta))
    #print(spherical_harmonic_representation(ml_pair, force, x_surface, radius))
    