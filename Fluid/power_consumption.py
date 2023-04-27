import numpy as np
from scipy.special import factorial

def constant_power_factor(squirmer_radius, viscosity):
    """The factor in front of the mode terms which ensures that the power is constant

    Args:
        squirmer_radius (_type_): _description_
        viscosity (_type_): _description_

    Returns:
        _type_: _description_
    """
    max_mode = 4
    n = np.arange(1, max_mode+1)
    m = np.arange(max_mode+1)
    mode_factors = np.zeros((len(n), len(m), 4))

    n1_common_factor = 64 / (3 * squirmer_radius ** 5) * np.pi * viscosity
    n2 = n[1:]
    m0_common_factor = 4 * n2 * (n2 + 1) * np.pi * viscosity / (squirmer_radius ** (2*n2 + 1))
    m1 = m[1:]
    n2m1_common_factor = 2 * n2[:, None] * (n2[:, None] + 1) * factorial(n2[:, None]+m1[None, :]) * np.pi * n2[:, None] / (squirmer_radius ** (2*n2[:, None] + 1) * factorial(n2[:, None]-m1[None, :]))
    n2m1_common_factor = np.tril(n2m1_common_factor)
        
    # n == 1
    mode_factors[0, :, :2] = n1_common_factor
    # m == 0, n>=2
    mode_factors[1:, 0, 0] = 4 / (n2 ** 2 * squirmer_radius ** 2) * m0_common_factor
    mode_factors[1:, 0, 2] = (n2 + 2) / (2 * n2 + 1) * m0_common_factor
    # m>=1, n>=2
    mode_factors[1:, 1:, 0] = 4 / (np.pi ** 2 * squirmer_radius) * n2m1_common_factor
    mode_factors[1:, 1:, 1] = mode_factors[1:, 1:, 0]
    mode_factors[1:, 1:, 2] = (n2[:, None] + 2) / (2 * n2[:, None] + 1) * n2m1_common_factor
    mode_factors[1:, 1:, 3] = mode_factors[1:, 1:, 2]

    # print("Mode factor over 11 B")
    # print(mode_factors[1:, 1:, 0])
    # print("")
    # print("")
    # Remove all cases where m>0
    mode_factors = np.tril(mode_factors)
    return mode_factors


if __name__ == "__main__":
    factors = constant_power_factor(1, 1)
    print(factors)
    print("")
    print(factors[:, :, 0])
    
