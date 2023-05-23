import numpy as np
from scipy.special import factorial

a = 1.5
eta = 1

B01 = 64 / (3 * a ** 5) * np.pi * eta
B11 = B01
Bt11 = B01
#B02 = 4 * 2 * (2 + 1) * np.pi * eta / a ** (2 * 2 + 1) * 4 / (2 ** 2 * a ** 2)
B12 = 2 * 2 * (2 + 1) * factorial(2 + 1) * np.pi * eta / (a ** (2 * 2 + 1) * factorial(2 - 1)) * 4 / (2 ** 2 * a)
Bt12 = B12
