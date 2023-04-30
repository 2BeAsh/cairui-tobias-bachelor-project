import numpy as np

def angular_to_cartesian(phi, radius):
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
    