import numpy as np
import matplotlib.pyplot as plt
from qmater.func import fermi_dirac_derivative
kb = 8.617333262145e-5  # eV K^-1
num = 201

Elist = np.linspace(-0.5, 0.5, num)
flist = np.zeros((num,))
for ind, _E in enumerate(Elist):
    flist[ind] = fermi_dirac_derivative(_E, kb*300)

fig, ax = plt.subplots()
ax.plot(Elist, flist)
plt.show()
