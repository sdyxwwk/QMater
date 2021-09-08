import numpy as np


def fermi_dirac(omega, kbT):
    if omega/kbT > 20.0:
        return 0.0
    elif omega/kbT < -20.0:
        return 1.0
    else:
        return 1.0/(1.0+np.exp(omega/kbT))


def fermi_dirac_derivative(omega, kbT):
    beta = 1/kbT
    if omega*beta > 20.0:
        return 0.0
    elif omega*beta < -20.0:
        return 0.0
    else:
        return -beta*np.exp(beta*omega)/(1.0+np.exp(beta*omega))**2
