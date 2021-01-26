import numpy as np
from scipy.ndimage import convolve
from tqdm import tqdm

def initialize_Latice(m):
    """2D uniform square-lattice"""
    return np.random.choice([-1,1], size=(m,m))

    
def metropolisHastings(n, m, beta, J, B, S, seed=1):
    """Metropolis–Hastings algorithm for the Ising model"""    
    np.random.seed(seed)
    def _H(S):
        """Compute the energy of a given system state of in the Ising model"""
        kernel = np.array([[1,1,1],
                           [1,0,1],
                           [1,1,1]])
        neighbors = convolve(S, kernel, mode="constant")
        return - np.sum(0.5*J*S*neighbors + B*S)

    def _update(S):
        """Randomly pick a spin, flip it, compute contribution and compare w/ old."""
        S_tilde = S.copy()

        cont = _H(S)

        i, j = np.random.randint(low=0, high=m, size=(2,))
        S_tilde[i,j] *= -1 # Flip atom
        new_cont = _H(S_tilde) # Compute new contribution 

        acceptance = np.exp(-beta*(new_cont - cont))

        if new_cont < cont: # Keep spin if new_cont is smaller.
            return S_tilde
        elif np.random.uniform() < acceptance: #Keep with prob exp(-\beta(H_v-H_\mu))
            return S_tilde   
        else: # Stay in same state.
            return S  
    
    # Define Energies and Magnetization    
    Hs = np.zeros(n)
    Ms = np.zeros(n)
    
    for ii in range(n):
        S = _update(S) # New state
        Hs[ii] = _H(S) # Compute & store energy for system
        Ms[ii] = np.sum(S) # Compute & store total magnetic moment of system
    return Hs, Ms, S    