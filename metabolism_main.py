import numpy as np


#%% -- Functions --
def energy_change(M, F):
    return F - M


def simulation(N, time_steps):
    # Set up initial traits
    mass_0 = np.random.uniform(size=N)
    metabolism_0 = mass_0 ** (3 / 4)  # Average metabolism chosen as initial
    energy_0 = np.ones_like(mass_0)  # For now all are equal. Might want to randomize
    
    # Parameters
    grass_consumption = 0.1  # How much energy is gained from eating grass
    
    # Evolve
    time = np.empty(size=time_steps)
    
    mass_old = mass_0.copy()
    metabolism_old = metabolism_0.copy()
    energy_old = energy_0.copy()
    
    for t in range(len(time)):
        # Random individuals will attack another. Whoever wins, gains energy.
        # All animals with metabolism > 1 consumes grass and gets energy
        # No reproduction
        # Shuffle all values? Right now attacks same person at all times. 
        
        # Fight
        attacks = int(N / 4)  # 25% is in fight, meaning 12.5% make an attack
        fights = np.random.randint(low=0, high=N, size=attacks)
        attacker = fights[:attacks/2]
        attacked = fights[attacks/2:]
        
        win_prob = metabolism_old[attacker] / (metabolism_old[attacker] + metabolism_old[attacked])
        prob_arr = np.random.uniform(size=attacks)
        
        # Update values
        energy_old[fights] = np.where(win_prob > prob_arr, energy_old[fights]+)