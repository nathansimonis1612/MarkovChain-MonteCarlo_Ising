import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve
from tqdm import tqdm
from itertools import product
import hasting 
plt.style.use("seaborn-white")
np.random.seed(1)

# Parameters.
np.random.seed(1)
m = 10
S0 = hasting.initialize_Latice(m)

params = {"n":[(10**3)],
         "m":[m],
         "beta":np.linspace(1/3, 1, 10),
         "J":[1],
         "B":[0],
         "S":[S0]}
         
# All possible combinations of parameters. (In this case only \beta changes)
keys, values = zip(*params.items())
experiments = [dict(zip(keys, v)) for v in product(*values)]

M_bar = []
fig = plt.figure(figsize = (10,10))
# Iterate each value of beta.
for comb in experiments:
    H, M, S = hasting.metropolisHastings(**comb) # Run simulation
    plt.plot(M, label=r"$\beta = $"+str(round(comb["beta"],3))) # Plot magnetization for that run.
    plt.xlabel(r"$t$")
    plt.ylabel(r"$M(S_t)$")
    M_bar.append(np.mean(M)) # Compute average total magnetic moment of simulation.
plt.legend()
plt.tight_layout()
plt.savefig("../figures/ex22_magnetic_sims.pdf")
plt.show()

# Plot average total magnetic moment as a function of beta.

plt.plot(params["beta"], M_bar, color="grey")
plt.xlabel(r"$\beta$")
plt.ylabel(r"$\bar{M}(\beta)$")
plt.savefig("../figures/ex22_mbar.pdf")
plt.show()