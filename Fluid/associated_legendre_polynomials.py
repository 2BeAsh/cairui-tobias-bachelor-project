import numpy as np


def associated_legendre_poly(n, m, x):
    """Associated Legendre Polynomials for n<=4

    Args:
        n (int): n mode
        m (int): m mode
        x (1d array of floats): The x-values the polynomials are evaluated at.

    Returns:
        (1d array of floats size x): P_n^m(x)
    """
    assert n <= 4
    if n == 1 and m == 0:
        return np.cos(x)
    elif n == 2 and m == 0:
        return 1 / 2 * (3 * np.cos(x) ** 2 - 1)
    elif n == 3 and m == 0:
        return 1 / 2 * (5 * np.cos(x) ** 3 - 3 * np.cos(x))
    elif n == 4 and m == 0:
        return 1 / 8 * (35 * np.cos(x) ** 4 - 30 * np.cos(x) ** 2 + 3)
    elif n == 1 and m == 1:
        return -np.sin(x)
    elif n == 2 and m == 1:
        return -3 * np.cos(x) * np.sin(x)
    elif n == 2 and m == 2:
        return 3 * np.sin(x) ** 2
    elif n == 3 and m == 1:
        return 3/2 * (1 - 5 * np.cos(x)**2)*(1 - np.cos(x)**2)**(1/2)
    elif n == 3 and m == 2:
        return 15 * np.cos(x) * (1 - np.cos(x) ** 2)
    elif n == 3 and m == 3:
        return -15 * (1 - np.cos(x) ** 2) ** (3 / 2)
    elif n == 4 and m == 1:
        return 5/2 * np.cos(x) * (3 - 7 * np.cos(x) ** 2) * ( 1 - np.cos(x) ** 2) ** (1 / 2)
    elif n == 4 and m == 2:
        return 15/2 * (7 * np.cos(x) ** 2 - 1) * (1 - np.cos(x) ** 2) 
    elif n == 4 and m == 3:
        return -105 * np.cos(x) * (1 - np.cos(x) ** 2) ** (3 / 2)
    elif n == 4 and m == 4:
        return 105*(1 - np.cos(x)**2)**2


def associated_legendre_poly_m_sin(n, m, x):
    """Associated Legendre Polynomials times m divided by sin(x) for n<=4

    Args:
        n (int): n mode
        m (int): m mode
        x (1d array of floats): The x-values the polynomials are evaluated at.

    Returns:
        (1d array of floats size x): P_n^m(x) * m / sin(x)
    """
    assert n <= 4
    if n == 1 and m == 0:
        return np.cos(x)
    elif n == 2 and m == 0:
        return m
    elif n == 3 and m == 0:
        return m
    elif n == 4 and m == 0:
        return m
    elif n == 1 and m == 1:
        return -1 * m
    elif n == 2 and m == 1:
        return -3 * np.cos(x) * m
    elif n == 2 and m == 2:
        return 3 * (1 - np.cos(x) ** 2) ** (1 / 2) * m
    elif n == 3 and m == 1:
        return 3/2 * (1 - 5 * np.cos(x)**2) * m
    elif n == 3 and m == 2:
        return 15 * np.cos(x) * (1 - np.cos(x) ** 2) ** (1 / 2) * m
    elif n == 3 and m == 3:
        return -15 * (1 - np.cos(x) ** 2) * m
    elif n == 4 and m == 1:
        return 5/2 * np.cos(x) * (3 - 7 * np.cos(x) ** 2) * m
    elif n == 4 and m == 2:
        return 15/2 * (7 * np.cos(x) ** 2 - 1) * (1 - np.cos(x) ** 2) ** (1 / 2) * m
    elif n == 4 and m == 3:
        return -105 * np.cos(x) * (1 - np.cos(x) ** 2) * m 
    elif n == 4 and m == 4:
        return 105*(1 - np.cos(x)**2) ** (3 / 2) * m


def associated_legendre_poly_deriv_sin(n, m, x):
    """First derivative of Associated Legendre Polynomials times sin(x) for n<=4

    Args:
        n (int): n mode
        m (int): m mode
        x (1d array of floats): The x-values the polynomials are evaluated at.

    Returns:
        (1d array of floats size x): P'_n^m(x) * sin(x)
    """
    assert n <= 4
    if n == 1 and m == 0:
        return 1 * np.sin(x)
    elif n == 2 and m == 0: #done
        return 3  * np.cos(x) * np.sin(x)
    elif n == 3 and m == 0: #done
        return 1 / 2 * (-3 + 3 * 5 * np.cos(x) ** 2) * np.sin(x)
    elif n == 4 and m == 0: #done
        return (17.5 * np.cos(x)**3 - 7.5 * np.cos(x)) * np.sin(x)
    elif n == 1 and m == 1: #done
        return np.cos(x)
    elif n == 2 and m == 1: #done
        return 3 * np.cos(x)**2 - 3 * (1 - np.cos(x)**2) 
    elif n == 2 and m == 2: #done
        return -6 * np.cos(x) * np.sin(x)
    elif n == 3 and m == 1:
        return - np.cos(x)*(1.5 - 7.5 * np.cos(x)**2)  - 15 * np.cos(x) * (1 - np.cos(x) ** 2) 
    elif n == 3 and m == 2:
        return (15 - 45 * np.cos(x)**2) * np.sin(x) 
    elif n == 3 and m == 3:
        return 45 * np.cos(x) * (1 - np.cos(x)**2 )
    elif n == 4 and m == 1:
        return (-2.5 * np.cos(x) **2 * (3 - 7 * np.cos(x) ** 2) 
                - 35 * np.cos(x) ** 2 * (1 - np.cos(x) ** 2) 
                + 2.5 * (1 - np.cos(x) ** 2) * (3 - 7 * np.cos(x) ** 2))
    elif n == 4 and m == 2:
        return (105 * np.cos(x) * (1 - np.cos(x) ** 2) - 2 * np.cos(x) * (52.5 * np.cos(x) ** 2 - 7.5)) * np.sin(x)
    elif n == 4 and m == 3:
        return 315 * np.cos(x)**2 * (1 - np.cos(x) ** 2)  - 105 * (1 - np.cos(x) ** 2)
    elif n == 4 and m == 4:
        return -420 * np.cos(x) * (1 - np.cos(x) ** 2) * np.sin(x)
