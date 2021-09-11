# ===============================================================
# Example: Cubic Band Touching Points in 2D
#
# $$
#     H(k) = \omega_1 k^2 \sigma_0 
#          + [(\alpha k_+^3 + \beta k_-^3) \sigma_+ + H.c.]
#          + (\gamma k_+^3 + \gamma^* k_-^3) \sigma_z
# $$
#
# Ref: W. Wu, et al. arXiv:2105:08424
#
# Author: Weikang Wu (sdyxwwk)
# Email: sdyxwwk@126.com
# Date: 10/09/2021
# ===============================================================
import numpy as np
import qmater as qm
import matplotlib.pyplot as plt

np.set_printoptions(precision=4, suppress=True)

# Pauli matrices
si = np.eye(2)
sx = np.array([[0.0, 1.0], [1.0, 0.0]])
sy = np.array([[0.0, -1.j], [1.j, 0.0]])
sz = np.array([[1.0, 0.0], [0.0, -1.0]])
splus = (sx + 1j*sy)/2
sminus = (sx - 1j*sy)/2

def CBTP_2d(kp, paralist):
    kx, ky, kz = kp
    omega1, alpha, beta, gamma = paralist
    alphac = np.conj(alpha)
    betac = np.conj(beta)
    gammac = np.conj(gamma)
    kp = kx + 1j*ky
    km = kx - 1j*ky

    ham = np.zeros((2, 2), dtype=np.complex128)
    ham += omega1*(kx**2 + ky**2)*si
    ham += (alpha*kp**3 + beta*km**3)*splus
    ham += (alphac*km**3 + betac*kp**3)*sminus
    ham += (gamma*kp**3 + gammac*km**3)*sz
    return ham

def plot_bands(model):
    k = 10.0
    highk = {
        'G': [0., 0., 0.],
        'X': [k, 0., 0.],
        'Y': [0., k, 0.],
        'M': [k/np.sqrt(2), k/np.sqrt(2), 0.],
    }
    kpath_label = ['X', 'G', 'M']
    model.plot_dispersion(
        kpath_label, highk, 
        num_kp_kpath=31, 
        Elim=None
    )

def plot_bands_3d(model):
    kdir1 = [2., 0., 0.]
    kdir2 = [0., 2., 0.]
    kcenter = [0., 0., 0.]
    model.plot_dispersion_3d(
        kdir1, kdir2, kcenter,
        num_kp_dir=31,
        select='All',
    )

def plot_fermisurface(model):
    kdir1 = [6., 0., 0.]
    kdir2 = [0., 6., 0.]
    kcenter = [0., 0., 0.]
    mu = 0.6
    model.plot_fermisurf(
        mu, kdir1, kdir2, kcenter,
        num_kp_dir=301,
    )


def calc_berryphase(model):
    num_kp = 101
    theta = np.linspace(0.0, 2*np.pi, num_kp)
    kpoints = np.zeros((num_kp, 3), dtype=np.float64)
    kpoints[:, 0] = np.cos(theta)
    kpoints[:, 1] = np.sin(theta)
    berry = model.calc_berryphase_1d(
        kpoints,
        num_occupy=1,
        if_eigval=False,
    )
    print(r'$\gamma/2\pi$ = {:2.1f}'.format(berry))

#------------------------------------------------
# Define a model and perform calculations 
#------------------------------------------------
para = (2.0, 1.0, 0.05, 0.0) # omega_1, alpha, beta, gamma
my_kdotp = qm.KdotP(CBTP_2d, num_bands=2, para_list=para)
# plot_bands(my_kdotp)
# plot_bands_3d(my_kdotp)
plot_fermisurface(my_kdotp)
# calc_berryphase(my_kdotp)
