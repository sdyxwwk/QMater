# ===============================================================
# Example: Haldane model
#
# Reference:
#     [1] F. D. M. Haldane, PRL 61, 2015 (1988).
#
# Author: Weikang Wu (sdyxwwk)
# Email: sdyxwwk@126.com
# Date: 11/07/2021
# ===============================================================
import numpy as np
import qmater as qm
import matplotlib.pyplot as plt

# Pauli matrices
si = np.eye(2)
sx = np.array([[0.0, 1.0], [1.0, 0.0]])
sy = np.array([[0.0, -1.j], [1.j, 0.0]])
sz = np.array([[1.0, 0.0], [0.0, -1.0]])


def Haldane():
    """ Return a Haldane model. """
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
    my_model = qm.WannierTB(my_struc, proj, if_soc=False)

    # Nearest hopping
    t1 = 1.0
    my_model.add_hop([0, 0, 0], [0, 1], t1)
    my_model.add_hop([-1, 0, 0], [0, 1], t1)
    my_model.add_hop([0, -1, 0], [0, 1], t1)

    # Next nearest hopping
    phi_a = 0.2
    phi_b = -phi_a
    phi = 2*np.pi/4.0  # (2*phi_a + phi_b)
    t2 = 0.1*t1
    my_model.add_hop([1, 0, 0], [0, 0], t2*np.exp(1.j*phi))
    my_model.add_hop([0, -1, 0], [0, 0], t2*np.exp(1.j*phi))
    my_model.add_hop([-1, 1, 0], [0, 0], t2*np.exp(1.j*phi))
    my_model.add_hop([0, 1, 0], [1, 1], t2*np.exp(-1.j*phi))
    my_model.add_hop([1, -1, 0], [1, 1], t2*np.exp(-1.j*phi))
    my_model.add_hop([-1, 0, 0], [1, 1], t2*np.exp(-1.j*phi))

    # staggered sublattice potential
    M = 0.0*t2
    my_model.add_hop([0, 0, 0], [0, 0], M)
    my_model.add_hop([0, 0, 0], [1, 1], -M)

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
    nkp = 101
    lat = model.reciprocal_lattice[:2, :2]
    kpos, bcx, bcy, bcz = model.calc_berrycurvature(
        num_kp=nkp,
        center=[0.5, 0.5, 0.0],
        dir1=[1.0, 0.0, 0.0],
        dir2=[0.0, 1.0, 0.0],
        num_occupy=2,
    )

    kpos_new = np.dot(kpos, lat)

    fig, ax = plt.subplots()
    ct = ax.contourf(kpos_new[:, :, 0],
                     kpos_new[:, :, 1], bcz, 100, cmap="bwr")
    ax.set_xlabel(r'$k_x$')
    ax.set_ylabel(r'$k_y$')
    plt.axis('equal')
    plt.colorbar(ct)
    plt.savefig('berrycurvature.pdf', dpi=300)
    # plt.show()


if __name__ == '__main__':
    my_model = Haldane()
    plot_bands(my_model)
    # plot_wilsonloop(my_model)
    # plot_berrycurvature(my_model)
