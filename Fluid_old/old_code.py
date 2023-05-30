import numpy as np
# From fluid_module
def field_polar_lab(r, theta, phi, B_11, B_tilde_11, B_01):
    """
    r: afstand fra centrum af squirmer
    theta, phi: polar vinkler. phi=pi/2 hvis ser på y og z akse som vi har gjort indtil videre
    
    B_11, B_tilde_11, B_01: Tal mellem -1 og 1. IKKE en matrix men et ENKEL tal. ET tal IKKE Matrix
    
    return hastighed i polare koordinater i LAB frame
    """
    u_r = 4 / (3 * r ** 3) * (B_11 * np.sin(theta) * np.cos(phi) + B_tilde_11 * np.sin(theta) * np.sin(phi) - B_01 * np.cos(theta))
    u_theta = - 2 / (3 * r ** 3) * (B_11 * np.cos(theta) * np.cos(phi) + B_tilde_11 * np.cos(theta) * np.sin(phi) - B_01 * np.sin(theta))
    u_phi = 2 / (3 * r ** 3) * (B_11 * np.sin(phi) - B_tilde_11 * np.cos(phi))
    return u_r, u_theta, u_phi


def field_cartesian_squirmer(r, theta, phi, a, B_11, B_tilde_11, B_01):
    """
    r: afstand fra centrum af squirmer
    theta, phi: polar vinkler. phi=pi/2 hvis ser på y og z akse som vi har gjort indtil videre
    
    LÆG MÆRKE TIL at denne funktion tager én yderligere parameter som er a:
    a: radius af squirmer.
    
    B_11, B_tilde_11, B_01: Tal mellem -1 og 1. IKKE en matrix men et ENKEL tal. ET tal IKKE Matrix
    
    return hastighed i kartetiske koordinater i SQUIRMER frame
    """
    u_r, u_theta, u_phi = field_polar_lab(r, theta, phi, B_11, B_tilde_11, B_01)
    
    u_z = np.cos(theta) * u_r - np.sin(theta) * u_theta
    u_y = u_r * np.sin(theta) * np.sin(phi) + u_theta * np.cos(theta) * np.sin(phi) + u_phi * np.cos(phi)
    u_x = u_r * np.sin(theta) * np.cos(phi) + u_theta * np.cos(theta) * np.cos(phi) - u_phi * np.sin(phi)
    
    u_z += B_01 * 4 / (3 * a ** 3)
    u_y += -B_tilde_11 * 4 / (3 * a ** 3)
    u_x += -B_11 * 4 / (3 * a ** 3)
    return u_x, u_y, u_z


def field_polar_snip():
    pass
"""         m_arr = np.arange(n+1)  # Since endpoints are not included, all must +1
        #mask = (slice(n+1), slice(n))
        LP_arr = LP[:n+1, n]
        LP_deriv_arr = LP_deriv[:n+1, n]
        B_arr = B[:n+1, n]
        B_tilde_arr = B_tilde[:n+1, n]
        C_arr = C[:n+1, n]
        C_tilde_arr = C_tilde[:n+1, n]
        # Array with velocities for each m can be summed to give the total value of the inner sum.
        u_r_arr = ((n + 1) * LP_arr / r ** (n + 2)
                   * ((r / a) ** 2 - 1)
                   * (B_arr * np.cos(m_arr * phi) + B_tilde_arr * np.sin(m_arr * phi)) 
        )
        u_theta_arr = (np.sin(theta) * LP_deriv_arr
                   * ((n - 2) / (n * a ** 2 * r ** n) - 1 / r ** (n + 2))
                   * (B_arr * np.cos(m_arr * phi) + B_tilde_arr * np.sin(m_arr * phi))
                   + m_arr * LP_arr / (r ** (n + 1) * np.sin(theta)) 
                   * (C_tilde_arr * np.cos(m_arr * phi) - C_arr * np.sin(m_arr * phi))
        )
        u_phi_arr = (np.sin(theta) * LP_deriv_arr / r ** (n + 1)
                     * (C * np.cos(m_arr * phi) + C_tilde * np.sin(m_arr * phi))
                     - m_arr * LP_arr / np.sin(theta)
                     * ((n - 2) / (n * a ** 2 * r ** n) - 1 / r ** (n + 2))
                     * (B_tilde * np.cos(m_arr * phi) - B * np.sin(m_arr * phi))
        )
        u_r += np.sum(u_r_arr)
        u_theta += np.sum(u_theta_arr)
        u_phi += np.sum(u_phi_arr) """
        
               
# From boundary_element_method
def discretized_sphere(N, radius):
    """Calculate N points uniformly distributed on the surface of a sphere with given radius.
    The calculation is done using a modified version of the canonical Fibonacci Lattice.
    From: http://extremelearning.com.au/how-to-evenly-distribute-points-on-a-sphere-more-effectively-than-the-canonical-fibonacci-lattice/
    And inspired by: https://stackoverflow.com/questions/9600801/evenly-distributing-n-points-on-a-sphere 

    Args:
        N (int): Number of points
        radius (float): Radius of sphere.

    Returns:
        Tupple of three 1d-arrays and a float: Returns cartesian coordinates of the points distributed on the spherical surface, and the approximate area each point are given.
    """
    # Find best index offset based on N. 
    if N < 80:
        offset = 2.66
    elif N < 1e3:
        offset = 3.33
    elif N < 4e4:
        offset = 10
    else:  # N > 40_000
        offset = 25 
    # Place the first two points and top and bottom of the sphere (0, 0, radius) and (0, 0, -radius)
    x = np.empty(N)
    y = np.empty_like(x)
    z = np.empty_like(x)
    x[:2] = 0
    y[:2] = 0
    z[0] = radius
    z[1] = -radius
        
    # Find spherical coordinates. 
    # Radius is contant, theta determined by golden ratio and phi is found using the Inverse Transform Method.
    indices = np.arange(N-2)  # Two first points already placed, thus minus 2
    golden_ratio = (1 + np.sqrt(5)) / 2 
    theta = 2 * np.pi * indices / golden_ratio
    phi = np.arccos(1 - 2 * (indices + offset) / (N - 1 + 2 * offset))
    
    # Convert to cartesian
    x[2:] = radius * np.sin(phi) * np.cos(theta)
    y[2:] = radius * np.sin(phi) * np.sin(theta)
    z[2:] = radius * np.cos(phi)
    # The area of each patch is approximated by surface area divided by number of points
    area = 4 * np.pi * radius ** 2 / N
    return x, y, z, theta, phi, area



# Fra Prediction træning plot
# Opdeler i fire subplots, enten efter n eller mode
def plot_action_choice(N_surface_points, N_iter, squirmer_radius, target_radius, max_mode, sensor_noise, PPO_number, seperate_modes=True):
    """Plot the actions taken at different iterations. Actions correspond to the weight/importance a mode is given.
    Color goes from bright to dark with increasing n and m values."""
    # Add more colors
    mpl.rcParams["axes.prop_cycle"] = mpl.cycler(color=['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 
                                                        'purple', 'pink', 'brown', 'orange', 'teal', 'coral', 'lightblue', 
                                                        'lime', 'lavender', 'turquoise', 'darkgreen', 'tan', 'salmon', 'gold'])

    # Names
    B_names = []
    B_tilde_names = []
    C_names = []
    C_tilde_names = []
    for i in range(max_mode+1):
        for j in range(i, max_mode+1):
            B_str = r"$B_{" + str(i) + str(j) + r"}$"
            B_names.append(B_str)
            
            if i > 0:
                B_tilde_str = r"$\tilde{B}_{" + str(i) + str(j) + r"}$"
                C_str = r"$C_{" + str(i) + str(j) + r"}$"
                B_tilde_names.append(B_tilde_str)
                C_names.append(C_str)
            elif i > 1:
                C_tilde_str = r"$\tilde{C}_{" + str(i) + str(j) + r"}$"
                C_tilde_names.append(C_tilde_str)
            
    B_names = [r"$B_{01}$", r"$B_{02}$", r"$B_{03}$", r"$B_{04}$", r"$B_{11}$", r"$B_{12}$", r"$B_{13}$", 
               r"$B_{14}$", r"$B_{22}$", r"$B_{23}$", r"$B_{24}$", r"$B_{33}$", r"$B_{34}$", r"$B_{44}$",]
    B_tilde_names = [r"$\tilde{B}_{11}$", r"$\tilde{B}_{12}$", r"$\tilde{B}_{13}$", r"$\tilde{B}_{14}$", r"$\tilde{B}_{22}$", 
                     r"$\tilde{B}_{23}$", r"$\tilde{B}_{24}$", r"$\tilde{B}_{33}$", r"$\tilde{B}_{34}$", r"$\tilde{B}_{44}$",]
    C_names = [r"$C_{02}$", r"$C_{03}$", r"$C_{04}$", r"$C_{12}$", r"$C_{13}$", r"$C_{14}$", 
               r"$C_{22}$", r"$C_{23}$", r"$C_{24}$", r"$C_{33}$", r"$C_{34}$", r"$C_{44}$",]
    C_tilde_names = [r"$\tilde{C}_{12}$", r"$\tilde{C}_{13}$", r"$\tilde{C}_{14}$", r"$\tilde{C}_{22}$", 
                     r"$\tilde{C}_{23}$", r"$\tilde{C}_{24}$", r"$\tilde{C}_{33}$", r"$\tilde{C}_{34}$", r"$\tilde{C}_{44}$",]
    
    B_len = len(B_names)
    B_tilde_len = len(B_tilde_names)
    C_len = len(C_names)
    C_tilde_len = len(C_tilde_names)
    
    B_actions = np.empty((N_iter, B_len))
    B_tilde_actions = np.empty((N_iter, B_tilde_len))
    C_actions = np.empty((N_iter, C_len))
    C_tilde_actions = np.empty((N_iter, C_tilde_len))
    
    rewards = np.empty((N_iter))
    
    # Load model and create environment
    model = PPO.load("ppo_predator_prey_direction")
    env = PredatorPreyEnv(N_surface_points, squirmer_radius, target_radius, max_mode, sensor_noise)
    
    # Run model N_iter times
    obs = env.reset()
    for i in range(N_iter):
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)
        B_actions[i, :] = action[: B_len]
        B_tilde_actions[i, :] = action[B_len: B_len+B_tilde_len]
        C_actions[i, :] = action[B_len+B_tilde_len : B_len+B_tilde_len+C_len]
        C_tilde_actions[i, :] = action[-C_tilde_len:]
        rewards[i] = reward
        
    # Plot
    fig, ax = plt.subplots(nrows=2, ncols=2, dpi=200)
    ax1 = ax[0, 0]
    ax2 = ax[0, 1]
    ax3 = ax[1, 0]
    ax4 = ax[1, 1]


    def fill_axis(axis, y, marker, label, title):        
        axis.set(xticks=[], title=(title, 7))
        axis.set_title(title, fontsize=7)
        axis.plot(np.abs(y), marker=marker, ls="--", lw=0.75)
        axis.legend(label, fontsize=4, bbox_to_anchor=(1.05, 1), 
                    loc='upper left', borderaxespad=0.)


    if seperate_modes:
        fill_axis(ax1, B_actions, ".", B_names, title=r"$B$ weights")
        fill_axis(ax2, B_tilde_actions, ".", B_tilde_names, title=r"$\tilde{B}$ weights")
        fill_axis(ax3, C_actions, ".", C_names, title=r"$C$ weights")
        fill_axis(ax4, C_tilde_actions, ".", C_tilde_names, title=r"$\tilde{C}$ weights")
        figname = f"mode_weight_seperate_mode_noise{sensor_noise}.png"
    else:  # Seperate n
        n1_names = [r"$B_{01}$", r"$B_{11}$", 
                    r"$\tilde{B}_{01}$"]
        n2_names = [r"$B_{02}$", r"$B_{12}$", r"$B_{22}$", 
                    r"$\tilde{B}_{12}$", r"$\tilde{B}_{22}$",
                    r"$C_{02}$", r"$C_{12}$", r"$C_{22}$",
                    r"$\tilde{C}_{12}$", r"$\tilde{C}_{22}$"]
        n3_names = [r"$B_{03}$", r"$B_{13}$", r"$B_{23}$", r"$B_{33}$",
                    r"$\tilde{B}_{13}$", r"$\tilde{B}_{23}$", r"$\tilde{B}_{33}$", 
                    r"$C_{03}$", r"$C_{13}$", r"$C_{23}$", r"$C_{33}$",
                    r"$\tilde{C}_{13}$", r"$\tilde{C}_{23}$", r"$\tilde{C}_{33}$"]
        n4_names = [r"$B_{04}$", r"$B_{14}$", r"$B_{24}$", r"$B_{34}$", r"$B_{44}$",
                    r"$\tilde{B}_{14}$", r"$\tilde{B}_{24}$", r"$\tilde{B}_{34}$", r"$\tilde{B}_{44}$",
                    r"$C_{04}$", r"$C_{14}$", r"$C_{24}$", r"$C_{34}$", r"$C_{44}$",
                    r"$\tilde{C}_{14}$", r"$\tilde{C}_{24}$", r"$\tilde{C}_{34}$", r"$\tilde{C}_{44}$"]
        
        n1 = np.empty((N_iter, len(n1_names)))
        n2 = np.empty((N_iter, len(n2_names)))
        n3 = np.empty((N_iter, len(n3_names)))
        n4 = np.empty((N_iter, len(n4_names)))
        for i in range(N_iter):
            n1[i, :] = [B_actions[i, 0], B_actions[i, 4], 
                        B_tilde_actions[i, 0]
            ]
            fill_axis(ax1, n1, ".", n1_names, title=r"$n=1$ weights")

            if max_mode > 1:
                n2[i, :] = [B_actions[i, 1], B_actions[i, 5], B_actions[i, 8], 
                            B_tilde_actions[i, 1], B_tilde_actions[i, 4],
                            C_actions[i, 0], C_actions[i, 3], C_actions[i, 6],
                            C_tilde_actions[i, 0], C_tilde_actions[i, 3]
                ]            
                fill_axis(ax2, n2, ".", n2_names, title=r"$n=2$ weights")
                
            if max_mode > 2:
                n3[i, :] = [B_actions[i, 2], B_actions[i, 6], B_actions[i, 9], B_actions[i, 11],
                            B_tilde_actions[i, 2], B_tilde_actions[i, 5], B_tilde_actions[i, 7], 
                            C_actions[i, 1], C_actions[i, 4], C_actions[i, 7], C_actions[i, 9],
                            C_tilde_actions[i, 1], C_tilde_actions[i, 4], C_tilde_actions[i, 6]
                ]
                fill_axis(ax3, n3, ".", n3_names, title=r"$n=3$ weights")

            if max_mode > 3:
                n4[i, :] = [B_actions[i, 3], B_actions[i, 7], B_actions[i, 10], B_actions[i, 12], B_actions[i, 13],
                            B_tilde_actions[i, 3], B_tilde_actions[i, 6], B_tilde_actions[i, 8], B_tilde_actions[i, 9],
                            C_actions[i, 2], C_actions[i, 5], C_actions[i, 8], C_actions[i, 10], C_actions[i, 11],
                            C_tilde_actions[i, 2], C_tilde_actions[i, 5], C_tilde_actions[i, 7], C_tilde_actions[i, 8]
                ]
                fill_axis(ax4, n4, ".", n4_names, title=r"$n=4$ weights")

            figname = f"mode_weight_seperate_n_noise{sensor_noise}.png"
            
    
    xticks = []
    for reward in rewards:
        xticks.append(f"Reward: {np.round(reward, 2)}")
    
    ax2.set(yticks=[])
    ax3.set(xlabel="Iteration", xticks=(np.arange(N_iter)))
    ax3.set_xticklabels(xticks, rotation=20, size=5)
    ax4.set(xlabel="Iteration", xticks=(np.arange(N_iter)), yticks=[])
    ax4.set_xticklabels(xticks, rotation=20, size=5)
    fig.suptitle(f"Mode weight over iterations, Noise = {sensor_noise}", fontsize=10)
    fig.tight_layout()
    plt.savefig("Reinforcement Learning/Recordings/Images/" + figname)
    plt.show()


    def spherical_harmonic_test(radius_obj2, ml_pair):
        import spherical_harmonics as sh
        # Choose parameters
        eps = 0.1
        viscosity = 1
        N1 = 200
        max_mode = 3
        squirmer_radius = 1.1
        tot_radius = squirmer_radius + radius_obj2
        x1_center = np.array([0, 0, 0])
        x2_center_values = np.arange(tot_radius+0.1, 4*tot_radius, tot_radius/4)
        # Modes
        B = np.zeros((max_mode+1, max_mode+1))
        B_tilde = np.zeros_like(B)
        C = np.zeros_like(B)
        C_tilde = np.zeros_like(B)
        B[1, 1] = 1    
        mode_array = np.array([B, B_tilde, C, C_tilde])
        
        # Calculate force to get the expansion coefficients
        f_ml = np.empty((len(ml_pair), len(x2_center_values)))
        
        # Loop through each target position distance
        for i, x2_center_y in enumerate(x2_center_values):  
            x2_center = np.array([0.2, x2_center_y, 0])
            
            # Force    
            force_with_condition, x1_surface, _ = force_surface_two_objects(N1, max_mode, squirmer_radius, radius_obj2, x1_center, x2_center, mode_array, eps, viscosity, lab_frame=True, return_points=True)
            force = force_with_condition[: 3*N1]        
            force_magnitude = np.sqrt(force[: N1]**2 + force[N1: 2*N1]**2 + force[2*N1: 3*N1]**2)

            # Get angular coordinates
            x = x1_surface[:, 0]
            y = x1_surface[:, 1]
            z = x1_surface[:, 2]
            theta = np.arccos(z / squirmer_radius)  # [0, pi]
            phi = np.arctan2(y, x)  # [0, 2*pi]
            
            # Get expansion coefficients for each ml value at this target position
            for j, ml in enumerate(ml_pair):
                print("")
                print(f"ml = {ml}")
                print(sh.expansion_coefficient_ml(ml[0], ml[1], force_magnitude, phi, theta))
                f_ml[j, i] = sh.expansion_coefficient_ml(ml[0], ml[1], force_magnitude, phi, theta)

        # Plot
        fig, ax = plt.subplots(dpi=150, figsize=(6, 6))
        ax.plot(x2_center_values, f_ml.T, ".")
        ax.set(xlabel="Target y position", ylabel="f_ml coefficient", title="First five expansion coefficients against y-position")
        
        # legend
        ml_str = []
        for ml in ml_pair:
            ml_val_str = f"m={ml[0]}, l={ml[1]}"
            ml_str.append(ml_val_str)
        ax.legend(ml_str)
        plt.show()
        
        
        
# BEM test funktioner
    def test_oseen_given_field():
        N = 250
        r = 1.
        eps = 0.1
        viscosity = 1
        x, y, z, area = canonical_fibonacci_lattice(N, r)
        theta = np.arccos(z / r)
        phi = np.arctan2(y, x)
        # Boundary Conditions
        vx = 1 + 0 * x
        vy = 0 * x
        vz = 0 * x
        
        # Stack
        x = np.stack((x, y, z)).T
        v = np.stack((vx, vy, vz)).T
        v = np.reshape(v, -1, order="F")
        v = np.append(v, np.zeros(6))  # From force=0=Torque

        # Få Oseen matrix på overfladen og løs for kræfterne
        A = oseen_tensor(regularization_offset=eps, dA=area, viscosity=viscosity,
                        evaluation_points=x)
        F = np.linalg.solve(A, v)
        U = F[-6:-3]
        ang_freq = F[-3:]
        # Evaluate i punkter uden for kuglen
        x_e = np.linspace(-3 * r, 3 * r, 25)
        X, Y = np.meshgrid(x_e, x_e)
        x_e = X.ravel()
        y_e = Y.ravel()
        z_e = 0 * X.ravel()
        x_e = np.stack((x_e, y_e, z_e)).T
        
        # Get Oseen and solve for velocities using earlier force
        A_e = oseen_tensor(regularization_offset=eps, dA=area, viscosity=viscosity,
                        evaluation_points=x_e, source_points=x)
        v_e = A_e @ F
        v_e = v_e[:-6]  # Remove Forces
        v_e = np.reshape(v_e, (len(v_e)//3, 3), order="F")

        # Return to squirmer frame by subtracting boundary conditions
        #v_e[:, 0] -= 1
        #v_e[:, 1] -= 1

        # Remove values inside squirmer
        r2 = np.sum(x_e**2, axis=1)
        v_e[r2 < r ** 2, :] = 0
        
        # -- Plot --
        fig, ax = plt.subplots(ncols=1, nrows=1, dpi=150, figsize=(6,6))
        ax_oseen = ax
        ax_oseen.quiver(x_e[:, 0], x_e[:, 1], v_e[:, 0], v_e[:, 1], color="red")
        ax_oseen.set(xlabel="x", ylabel="y", title=r"With Conditions, Artificial field $v_x=1$, lab frame")
        text_min = np.min(x_e)
        text_max = np.max(x_e)
        ax_oseen.text(text_min, text_max, s=f"U={np.round(U, 4)}", fontsize=12)
        ax_oseen.text(text_min, text_max-0.3, s=f"$\omega$={np.round(ang_freq, 4)}", fontsize=12)

        plt.savefig("fluid/images/condition_artificialfield_labframe.png")
        plt.show()
        
        
    def test_oseen_given_force():
        eps = 0.1
        viscosity = 1
        xx = np.linspace(-2, 2, 25)
        X, Y = np.meshgrid(xx, xx)
        X = X.ravel()
        Y = Y.ravel()
        Z = 0 * X
        
        # Source point and force
        xi = np.zeros(3).reshape(1, -1)
        F = np.zeros(9)  # Rotation and U makes bigger
        F[0] = 1.  # Force in x
        #F[-6] = 2.  # Background flow in x
        #F[-1] = -0.5  # Rotation along z-axis
        
        x_e = np.stack((X, Y, Z)).T
        O = oseen_tensor(regularization_offset=eps, dA=0.5, viscosity=viscosity, 
                            evaluation_points=x_e, source_points=xi)
        v = O @ F
        v = v[:-6]  # Remove force and torque
        v = np.reshape(v, (len(v)//3, 3), order='F')
            
        vx = v[:, 0]
        vy = v[:, 1]
        
        fig, ax = plt.subplots(dpi=150, figsize=(6, 6))
        ax.quiver(X, Y, vx, vy, color="red")
        ax.set(xlabel="x", ylabel="y", title="With Conditions, Artificial 'force' $F_x$, lab frame")
        plt.savefig("fluid/images/condition_artificialforce_labframe.png")
        plt.show()  