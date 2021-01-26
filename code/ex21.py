import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from itertools import product
import hasting 
plt.style.use("seaborn-white")
np.random.seed(1)

# Define parameters.
m = 50
beta = 1
J = 1
B = 0
S0 = hasting.initialize_Latice(m) 

# Run simulations
n = 10000
H, M, S = hasting.metropolisHastings(n, m, beta, J, B, S0)

## Plots
# Energy
plt.plot(H, color="grey")
plt.xlabel(r"$t$")
plt.ylabel(r"$H(S_t)$")
plt.xscale("log")
plt.savefig("../figures/ex21_energy.pdf")
plt.show()


# Magnetic

plt.plot(M, color="grey")
plt.xlabel(r"$t$")
plt.ylabel(r"$M(S_t)$")
#plt.xscale("log")
plt.savefig("../figures/ex21_magnetic.pdf")
plt.show()

# System

fig, axs = plt.subplots(1,2, figsize=(9,9))
axs[0].imshow(S0, aspect="equal", cmap = "Greys")
axs[1].imshow(S, aspect="equal", cmap = "Greys")
plt.tight_layout()

plt.savefig("../figures/ex21_system.pdf")
plt.show()