# ===============================================================
# Example: BBH model
#
# Reference:
# [1] W. A. Benalcazar, B. A. Bernevig and T. L. Hughes, Science 357, 61 (2017) 
#
# Author: Weikang Wu (sdyxwwk)
# Email: sdyxwwk@126.com
# Date: 04/07/2021
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


def Kane_Mele():
    """ Return a Kane-Mele model. """
    lattice = np.array([
        [-np.sqrt(3)/2.0, 3.0/2.0, 0.0],
        [np.sqrt(3)/2.0, 3.0/2.0, 0.0],
        [0.0, 0.0, 10.0],
    ])
    atomsites = {
        'C': np.array([
            [1/3, 1/3, 0.0],
            [2/3, 2/3, 0.0],
        ])
    }
    my_struc = qm.CrystStruct.set_model(lattice, atomsites)

    proj = {'C': ['s']}
    my_model = qm.WannierTB(my_struc, proj, if_soc=True)

    # Nearest hopping
    t = 1.0
    my_model.add_hop([0, 0, 0], [0, 1], t*si)
    my_model.add_hop([-1, 0, 0], [0, 1], t*si)
    my_model.add_hop([0, -1, 0], [0, 1], t*si)

    # SOC term
    lambdaso = 0.8*t
    my_model.add_hop([1, 0, 0], [0, 0], 1j*lambdaso*sz)
    my_model.add_hop([0, -1, 0], [0, 0], 1j*lambdaso*sz)
    my_model.add_hop([-1, 1, 0], [0, 0], 1j*lambdaso*sz)
    my_model.add_hop([0, 1, 0], [1, 1], 1j*lambdaso*sz)
    my_model.add_hop([1, -1, 0], [1, 1], 1j*lambdaso*sz)
    my_model.add_hop([-1, 0, 0], [1, 1], 1j*lambdaso*sz)

    # Rashba term which violates M_{z} symmetry
    lambdaR = 0.1*t
    my_model.add_hop([0, 0, 0], [0, 1], 1j*lambdaR*(sx))
    my_model.add_hop([-1, 0, 0], [0, 1], 1j*lambdaR *
                     (-1./2.*sx-np.sqrt(3)/2.*sy))
    my_model.add_hop([0, -1, 0], [0, 1], 1j*lambdaR *
                     (-1./2.*sx+np.sqrt(3)/2.*sy))

    # staggered sublattice potential which breaks inversion symmetry
    lambdav = 0.1*t
    my_model.add_hop([0, 0, 0], [0, 0], lambdav*si)
    my_model.add_hop([0, 0, 0], [1, 1], -lambdav*si)

    # Magnetic field
    Bz = 0.5
    Bxy = 1.0
    theta = np.pi/5.0
    my_model.add_hop([0, 0, 0], [0, 0], Bz*sz + Bxy *
                     (np.cos(theta)*sx + np.sin(theta)*sy))
    my_model.add_hop([0, 0, 0], [1, 1], Bz*sz + Bxy *
                     (np.cos(theta)*sx + np.sin(theta)*sy))

    return my_model


def plot_bands(model):
    highk = {
        'G': [0.0, 0.0, 0.0],
        'K': [2/3, 1/3, 0.0],
        'M': [0.5, 0.0, 0.0],
    }

    kpath_label = ['G', 'M', 'K', 'G']

    model.plot_band_structure(kpath_label, highk, num_kp_kpath=81)


def plot_wilsonloop(model):
    nkp = 81
    wcc = model.calc_wilsonloop(
        num_kp=nkp,
        origin=[0.0, 0.0, 0.0],
        int_dir=[1.0, 0.0, 0.0],
        var_dir=[0.0, 1.0, 0.0],
        num_occupy=2,
    )
    kp = np.linspace(0.0, 1.0, nkp)

    fig, ax = plt.subplots()
    ax.plot(kp, wcc, '.r')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    ax.set_xlabel(r'$k$')
    ax.set_ylabel(r'$\theta$')
    plt.savefig('wilsonloop.pdf', dpi=300)
    # plt.show()


def plot_berrycurvature(model):
    lat = model.reciprocal_lattice[:2, :2]
    kpos, bc = model.calc_berrycurv_kplane(
        num_kp=101,
        center=[0.5, 0.5, 0.0],
        dir1=[1.0, 0.0, 0.0],
        dir2=[0.0, 1.0, 0.0],
        num_occupy=2,
        index='xy',
    )

    kpos_new = np.dot(kpos, lat)

    fig, ax = plt.subplots()
    ct = ax.contourf(kpos_new[:, :, 0],
                     kpos_new[:, :, 1], bc, 100, cmap="bwr")
    ax.set_xlabel(r'$k_x$')
    ax.set_ylabel(r'$k_y$')
    plt.axis('equal')
    plt.colorbar(ct)
    plt.savefig('berrycurvature.pdf', dpi=300)
    # plt.show()


def plot_bcp(model):
    lat = model.reciprocal_lattice[:2, :2]
    kpos, bc = model.calc_bcp_kplane(
        num_kp=101,
        center=[0.5, 0.5, 0.0],
        dir1=[1.0, 0.0, 0.0],
        dir2=[0.0, 1.0, 0.0],
        num_occupy=2,
        index='xy',
    )

    kpos_new = np.dot(kpos, lat)

    fig, ax = plt.subplots()
    ct = ax.contourf(kpos_new[:, :, 0],
                     kpos_new[:, :, 1], bc, 100,
                     cmap="bwr",
                     #  vmin=-0.1,
                     #  vmax=0.1,
                     )
    ax.set_xlabel(r'$k_x$')
    ax.set_ylabel(r'$k_y$')
    plt.axis('equal')
    plt.colorbar(ct)
    plt.savefig('bcp.pdf', dpi=300)
    # plt.show()


def plot_alpha(model):
    lat = model.reciprocal_lattice[:2, :2]
    kpos, alpha = model.calc_alpha_kplane(
        num_kp=101,
        center=[0.5, 0.5, 0.0],
        dir1=[1.0, 0.0, 0.0],
        dir2=[0.0, 1.0, 0.0],
        num_occupy=2,
        index='xyy',
    )

    kpos_new = np.dot(kpos, lat)

    fig, ax = plt.subplots()
    ct = ax.contourf(kpos_new[:, :, 0],
                     kpos_new[:, :, 1], alpha, 100,
                     cmap="bwr",
                     #  vmin=-0.1,
                     #  vmax=0.1,
                     )
    ax.set_xlabel(r'$k_x$')
    ax.set_ylabel(r'$k_y$')
    plt.axis('equal')
    plt.colorbar(ct)
    plt.savefig('alpha.pdf', dpi=300)
    # plt.show()


if __name__ == '__main__':
    my_model = Kane_Mele()
    # plot_bands(my_model)
    # plot_wilsonloop(my_model)
    # plot_berrycurvature(my_model)
    # plot_bcp(my_model)
    plot_alpha(my_model)
    # print(my_model._calc_alpha_fermisurface([0, 0.4, 0], 0.5, index='xxy'))
