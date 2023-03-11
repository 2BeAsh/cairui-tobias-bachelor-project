import numpy as np
import matplotlib.pyplot as plt
from  scipy.special import factorial


def mode_basic(a, eta):
    return 64 * np.pi * eta / (3 * a ** 5)


def mode_b_0n(n, a, eta):
    """
    n skal være mindst 2
    """
    return 4 * n * (n + 1) * np.pi * eta / (a ** (2 * n + 1)) * 4 / (n ** 2 * a ** 2)


def mode_c_0n(n, a, eta):
    """
    n skal være mindst 2
    """
    return 4 * n * (n + 1) * np.pi * eta / (a ** (2 * n + 1)) * (n + 2) / (2 * n + 1)


def mode_b_mn(n, m, a, eta):
    """
    n skal være mindst 2
    m skal være mindst 1
    """
    return 2 * n * (n + 1) * factorial(n + m) * np.pi * eta / (a ** (2 * n + 1) * factorial(n-m)) * 4 / (n ** 2 * a)


def mode_c_mn(n,m, a, eta):
    """
    n skal være mindst 2
    m skal være mindst 1
    """
    return 2 * n * (n + 1) * factorial(n + m) * np.pi * eta / (a ** (2 * n + 1) * factorial(n-m)) * (n + 2) / (2 * n + 1)