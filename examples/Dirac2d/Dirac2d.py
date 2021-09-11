# ===============================================================
# Example: 2D Dirac point
#
# $$
#     H(k) = v (\tau_z k_x \sigma_x + k_y \sigma_y)
# $$
#
# Reference:
#     [1] S. A. Yang, SPIN 6, 164003 (2016).
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

def Dirac_2d(kp, paralist):
    kx, ky, kz = kp
    v, tau = paralist

    return v*(tau*kx*sx + ky*sy)

def plot_bands(model):
    highk = {
        'G': [0., 0., 0.],
        'X': [1., 0., 0.],
        'Y': [0., 1., 0.],
        'M': [1., 1., 0.],
    }
    kpath_label = ['X', 'G', 'Y']
    model.plot_dispersion(
        kpath_label, highk, 
        num_kp_kpath=31, 
        Elim=None
    )

def plot_bands_3d(model):
    kdir1 = [1., 0., 0.]
    kdir2 = [0., 1., 0.]
    kcenter = [0., 0., 0.]
    model.plot_dispersion_3d(
        kdir1, kdir2, kcenter,
        num_kp_dir=31,
        select='All',
    )

def plot_fermisurface(model):
    kdir1 = [1., 0., 0.]
    kdir2 = [0., 1., 0.]
    kcenter = [0., 0., 0.]
    mu = 0.3
    model.plot_fermisurf(
        mu, kdir1, kdir2, kcenter,
        num_kp_dir=201,
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
my_kdotp = qm.KdotP(Dirac_2d, num_bands=2, para_list=(2.0, 1.0))
# plot_bands(my_kdotp)
# plot_bands_3d(my_kdotp)
# plot_fermisurface(my_kdotp)
calc_berryphase(my_kdotp)
