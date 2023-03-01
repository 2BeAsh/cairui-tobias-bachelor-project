import numpy as np
import matplotlib.pyplot as plt


# Functions
def legendre_poly(n, x):
    if n == 1:
        return np.cos(x)
    elif n == 2:
        return 1 / 2 * (3 * np.cos(x) ** 2 - 1)
    elif n == 3:
        return 1 / 2 * (5 * np.cos(x) ** 3 - 3 * np.cos(x))
    elif n == 4:
        return 1 / 8 * (35 * np.cos(x) ** 4 - 30 * np.cos(x) **2 + 3)
    elif n == 5:
        return 1 / 8 * (63 * np.cos(x) ** 5 - 70 * np.cos(x) ** 3 + 15 * np.cos(x))
    

def legendre_poly_prime(n, x):
    if n == 1:
        return 1
    elif n == 2:
        return 1 / 2 * 3 * 2 * np.cos(x)
    elif n == 3:
        return 1 / 2 * (-3 + 3 * 5 * np.cos(x) ** 2)
    elif n == 4:
        return 1 / 8 * (-2 * 30 * np.cos(x) + 35 * 4 * np.cos(x) ** 3)
    elif n == 5:
        return 1 / 8 * (63 * 5 * np.cos(x) ** 4 - 70 * 3 * np.cos(x) ** 2 + 15)


def field_polar(r, theta, n, B, a):
    """Velocity field in polar coordinates
    r: afstand fra centrum
    theta: vinkel mellem punkt 
    n: Legendre orden. Integer mellem 0 og 5
    B: Float mellem -1 og 1, hvor kraftig feltet skabt er.
    a: radius af squirmer.
    """
    u_r = 0
    for i in range(1, n+1):
        u_r +=  (i+1)* legendre_poly(i , theta)/r**(i+2)*(r**2/a**2-1)*B[i-1]
        
    u_theta = 0
    for i in range(1, n+1):  # Might be optimized by making vectorized function and summing up elements
        u_theta += np.sin(theta) * legendre_poly_prime(i, theta) * ((i - 2) / (i * a ** 2 * r ** i) - 1 / (r ** (i + 2))) * B[i-1]
        
    return u_r, u_theta


def field_cartesian(r, theta, n, B, a):
    """Velocity field in cartesian coordinates
    r: afstand fra centrum
    theta: vinkel mellem punkt 
    n: Legendre orden. Integer mellem 0 og 5
    B: Float mellem -1 og 1, hvor kraftig feltet skabt er.
    a: radius af squirmer.
    """
    u_r, u_theta = field_polar(r, theta, n, B, a)
    u_z = np.cos(theta) * u_r - np.sin(theta) * u_theta
    u_y = u_r * np.sin(theta) + u_theta*np.cos(theta)
    return u_y, u_z



def field_polar_lab(r, theta, n, B, a):
    """Velocity field in polar coordinates
    r: afstand fra centrum
    theta: vinkel mellem punkt 
    n: Legendre orden. Integer mellem 0 og 5
    B: Float mellem -1 og 1, hvor kraftig feltet skabt er.
    a: radius af squirmer.
    """
    u_r = -4 * np.cos(theta) / (3 * r ** 3) * B[0] 
    for i in range(2, n+1):
        u_r+= (i+1)*legendre_poly(i, theta)/(r**(i+2)) * (r**2/a**2 -1) * B[i-1]
        
    u_theta=-2*np.sin(theta)/(3*r**3)*B[0]
    for i in range(2, n+1):
        u_theta += np.sin(theta)*legendre_poly_prime(i, theta)*((i-2)/(i*a**2*r**i) - 1/(r**(i+2))) * B[i-1]
    return u_r, u_theta   


def field_cartesian_lab(r, theta, n, B, a):
    """Velocity field in cartesian coordinates, input must be polar coordinates
    r: afstand fra centrum
    theta: vinkel mellem punkt 
    n: Legendre orden. Integer mellem 0 og 5
    B: Float mellem -1 og 1, hvor kraftig feltet skabt er.
    a: radius af squirmer.
    """
    u_r, u_theta=field_polar_lab(r, theta, n, B, a)
    u_z = np.cos(theta) * u_r - np.sin(theta) * u_theta
    u_y = u_r*np.sin(theta) + u_theta*np.cos(theta)
    return u_y, u_z

