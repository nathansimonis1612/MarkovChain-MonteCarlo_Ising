import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve
from tqdm import tqdm
from itertools import product
import hasting 
plt.style.use("seaborn-white")
np.random.seed(1)

# Define energy computations from ex1.

def energy(S, J=1, B=0):
    """The energy of a given system state of in the Ising model"""
    kernel = np.array([[1,1,1],
                       [1,0,1],
                       [1,1,1]])
    neighbors = convolve(S, kernel, mode="constant") # constant --> fill edges w/ 0
    return - np.sum(0.5*J*S*neighbors + B*S)

# Parameters.
m = 4
beta = 1/3

permutations = list(product([-1,1],repeat=m*m)) # Create all states of system.

n = len(permutations)
Bs = np.zeros(n)
Hs = np.zeros(n)
Ms = np.zeros(n)

for i in range(n):
    S = np.array(permutations[i]).reshape(m,m) # Reshape to m*m 2D latice.
    Hs[i] = energy(S) # Compute & store energy
    Ms[i] = np.sum(S) # Compute & store TMM
    Bs[i] = np.exp(-Hs[i]*beta) # Boltzman
 
Z = np.sum(Bs) # Normalization constant
mbar = 1/Z * np.sum(Ms*Bs) # Exact mbar
print(mbar)

 
N = np.logspace(1,5, 100)  # Multiple runs of N
est_mbar = []
S0 = hasting.initialize_Latice(m) # Same initial state.

for n in N:
    n = int(np.ceil(n))    
    H, M, S = hasting.metropolisHastings(n=n, m=4, beta=1/3, J=1, B=0, S=S0, seed=3)
    est_mbar.append(np.mean(M)) # Compute estimated Mbar

# Plot
plt.plot(N,est_mbar, c="grey", label='Estimated '+r'$\bar{M}(\beta)$')
plt.axhline(mbar, ls="--", c="black", label='Exact '+r'$\bar{M}(\beta)$')
plt.xlabel(r"$n$")
plt.ylabel(r"$\bar{M}(\beta)$")
plt.legend()
plt.savefig("../figures/ex3_mbar_n.pdf")
plt.show()
