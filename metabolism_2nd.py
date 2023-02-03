# -- Imports --
import numpy as np


# -- Functions --
def round_even(x):
    return round(x / 2) * 2


def simulation(N, steps, dt):
    # Data arrays
    mass = np.zeros(shape=(N, steps))
    metabolism = np.zeros_like(mass)
    energy = np.zeros_like(mass)
    
    # Initial values
    mass0 = np.random.uniform(0, 1, size=N)
    metabolism0 = mass0 ** (3 / 4)  # Initial metabolism equal expected metabolism
    energy0 = np.ones_like(mass0) - metabolism0
    
    mass[:, 0] = mass0
    metabolism[:, 0] = metabolism0
    energy[:, 0] = energy0
    
    # Parameters
    frac_fight = 1  # 25% of individuals will end in a fight. 
    
    for i in np.arange(1, steps):
        # Old mass, metabolism and energy data
        m = mass[:, i-1]
        meta = metabolism[:, i-1]
        E = energy[:, i-1]
        
        # -- Fight --
        # Get an even number of fighters and get their indices. Split indices in half into attackers and attacked
        N_fight = round_even(N * frac_fight)
        who_fights = np.random.choice(a=N, size=N_fight, replace=False)
        split = int(N_fight/2)
        attacker = who_fights[:split]
        attacked = who_fights[split:]
        
        # Get probability attacker wins, then compare against random number
        prob_win = meta[attacker] / (meta[attacker] + meta[attacked])
        random_number = np.random.uniform(size=split)
        attacker_win = np.where(prob_win > random_number)
        attacked_win = np.where(prob_win < random_number)
        fights_won = who_fights[attacker_win] 
        fights_lost = who_fights[attacked_win]
        
        print("Win/Lose pos")
        print(attacker_win)
        print(attacked_win)
        
        print("win/lose fights")
        print(fights_won)
        print(fights_lost)

        # Update arrays - PROBLEMET ER + split
        energy[fights_won, i] = E[fights_won] + dt * (E[fights_won + split]- meta[fights_won])  # Euler update 
        energy[fights_lost, i] = E[fights_lost] + dt * (E[fights_lost - split]- meta[fights_lost])
        
        metabolism[:, i] = meta
        mass[:, i] = m
        
    return energy


simulation(N=6, steps=5, dt=1)


        



