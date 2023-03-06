import numpy as np
from scipy.special import lpmv, lpmn
import matplotlib.pyplot as plt


# Functions
def legendre_poly(n, m, x):
    if n==1 and m==0:
        return np.cos(x)
    elif n==2 and m==0:
        return 1 / 2 * (3 * np.cos(x) ** 2 - 1)
    elif n==3 and m==0:
        return 1 / 2 * (5 * np.cos(x) ** 3 - 3 * np.cos(x))
    elif n==4 and m==0:
        return 1 / 8 * (35 * np.cos(x) ** 4 - 30 * np.cos(x) ** 2 + 3)
    elif n==1 and m==1:
        return -np.sin(x)
    elif n==2 and m==1:
        return -3 * np.cos(x) * np.sin(x)
    elif n==2 and m==2:
        return 0
    

def legendre_poly_prime(n, m, x):
    if n==1 and m==0:
        return 1
    elif n==2 and m==0:
        return 1 / 2 * 3 * 2 * np.cos(x)
    elif n==3 and m==0:
        return 1 / 2 * (-3 + 3 * 5 * np.cos(x) ** 2)
    elif n==4 and m==0: 
        return 1/8 * (-2 * 30 * np.cos(x) + 35 * 4 * np.cos(x) ** 3)
    elif n==1 and m==1:
        return np.cos(x) / np.sin(x)
    elif n==2 and m==1:
        return -3 * (np.sin(x) - 1 / np.sin(x))
    elif n==2 and m==2:
        return 0


def field_polar(r, theta, n, m, a, B, B_tilde, C, C_tilde):
    """Velocity field in polar coordinates
    r: afstand fra centrum
    theta: vinkel mellem punkt 
    n: Legendre orden. Integer mellem 0 og 2
    m: legendre orden integer mellem 0 og 2
    B: Float mellem -1 og 1, hvor kraftig feltet skabt er. B er en n gange m matrix. 
    B_tilde,C, C_tilde samme som B beskriver andre modes: rotation osv...
    a: radius af squirmer.
    """
    u_r = 0
    u_theta = 0
    for i in range(1, n+1):
        for j in range(0, m):
            u_r +=  ((i + 1) * legendre_poly(i, j, theta) / r ** (i+2) * (r ** 2 / a ** 2 - 1)
                     * (np.cos(j*np.pi/2) * B[i, j] + np.sin(j*np.pi/2) * B_tilde[i, j]))
            
            u_theta += (np.sin(theta) * legendre_poly_prime(i, j, theta)
                        * ((i - 2) / (i * a ** 2 * r ** i) - 1 / (r ** (i + 2))) 
                        * (np.cos(j*np.pi/2) * B[i, j] + np.sin(j*np.pi/2) * B_tilde[i, j])
                        + j * legendre_poly(i, j, theta) / (r ** (i + 1) * np.sin(theta)) 
                        * (np.cos(j*np.pi/2) * C[i, j] - np.sin(j*np.pi/2) * C_tilde[i, j])) 
            
    return u_r, u_theta


def field_cartesian(r, theta, n, m, a, B, B_tilde, C=np.array([[0, 0], [0, 0], [0, 0]]), C_tilde=np.array([[0, 0], [0, 0], [0, 0]])):
    u_r, u_theta = field_polar(r, theta, n, m, a, B, B_tilde, C, C_tilde)
    u_z = np.cos(theta) * u_r - np.sin(theta) * u_theta
    u_y = u_r * np.sin(theta) + u_theta * np.cos(theta)
    return u_y, u_z



# virker ikke
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




# NOT IN USE, UNDER CONSTRUCTION
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

# %%
