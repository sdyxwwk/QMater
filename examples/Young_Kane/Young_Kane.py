# ===============================================================
# Example: Young-Kane model
#
# Reference:
#     [1] S. M. Young and C. L. Kane, PRL 115, 126803 (2015)
#
# Author: Weikang Wu (sdyxwwk)
# Email: sdyxwwk@126.com
# Date: 22/08/2021
# ===============================================================
import numpy as np
import qmater as qm
import matplotlib.pyplot as plt

# Pauli matrices
si = np.eye(2)
sx = np.array([[0.0, 1.0], [1.0, 0.0]])
sy = np.array([[0.0, -1.j], [1.j, 0.0]])
sz = np.array([[1.0, 0.0], [0.0, -1.0]])


def Young_Kane():
    """ Return a Young-Kane model. """

    lattice = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 10.0],
    ])
    atomsites = {
        'C': np.array([
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.0],
        ]),
    }

    m1 = 0.0; m2 = 0.0 # Displacement of center atom
    atomsites['C'][1] += np.array(
        [-m1*np.sqrt(2)-m2*np.sqrt(2), -m1*np.sqrt(2)+m2*np.sqrt(2), 0.0]
    )

    my_struc = qm.CrystStruct.set_model(lattice, atomsites)

    proj = {'C': ['s']}
    my_model = qm.WannierTB(my_struc, proj, if_soc=True)

    # Spinless hopping
    t = 1.5
    t2 = -0.15
    my_model.add_hop([ 0, 0, 0], [0, 1], t/2.0*si)
    my_model.add_hop([-1, 0, 0], [0, 1], t/2.0*si)
    my_model.add_hop([-1,-1, 0], [0, 1], t/2.0*si)
    my_model.add_hop([ 0,-1, 0], [0, 1], t/2.0*si)

    my_model.add_hop([ 1, 0, 0], [0, 0], t2/2.0*si)
    my_model.add_hop([ 0, 1, 0], [0, 0], t2/2.0*si)
    my_model.add_hop([ 1, 0, 0], [1, 1], t2/2.0*si)
    my_model.add_hop([ 0, 1, 0], [1, 1], t2/2.0*si)

    # Second neighbor SOC
    tso = -0.2
    my_model.add_hop([ 1, 0, 0], [0, 0], tso/2.j*sy)
    my_model.add_hop([ 0, 1, 0], [0, 0],-tso/2.j*sx)
    my_model.add_hop([ 1, 0, 0], [1, 1],-tso/2.j*sy)
    my_model.add_hop([ 0, 1, 0], [1, 1], tso/2.j*sx)

    # Case I: Two symmetry equivalent Dirac points.
    delta1 = 0.5
    my_model.add_hop([ 0, 0, 0], [0, 1],-delta1/4.0*si)
    my_model.add_hop([-1, 0, 0], [0, 1], delta1/4.0*si)
    my_model.add_hop([-1,-1, 0], [0, 1],-delta1/4.0*si)
    my_model.add_hop([ 0,-1, 0], [0, 1], delta1/4.0*si)

    my_model.add_hop([ 0, 0, 0], [0, 1],-m1/2.0*si)
    my_model.add_hop([-1,-1, 0], [0, 1], m1/2.0*si)
    my_model.add_hop([ 0,-1, 0], [0, 1],-m2/2.0*si)
    my_model.add_hop([-1, 0, 0], [0, 1], m2/2.0*si)

    # Case II: Two symmetry equivalent Dirac points.
    delta2 = 0.3
    my_model.add_hop([ 0, 0, 0], [0, 1],-delta2/4.0*si)
    my_model.add_hop([-1, 0, 0], [0, 1],-delta2/4.0*si)
    my_model.add_hop([-1,-1, 0], [0, 1], delta2/4.0*si)
    my_model.add_hop([ 0,-1, 0], [0, 1], delta2/4.0*si)

    # Case III: Three Dirac points.
    tsop = 0.0 # 0.2
    my_model.add_hop([ 1, 0, 0], [0, 0],-tsop/2.j*sx)
    my_model.add_hop([ 0, 1, 0], [0, 0], tsop/2.j*sy)
    my_model.add_hop([ 1, 0, 0], [1, 1], tsop/2.j*sx)
    my_model.add_hop([ 0, 1, 0], [1, 1],-tsop/2.j*sy)

    # Case IV: Line nodes and Weyl points.
    vso = 0.0 # 0.1
    my_model.add_hop([ 0,-1, 0], [0, 1], vso/2.j*sz)
    my_model.add_hop([-1, 0, 0], [0, 1],-vso/2.j*sz)

    # my_model.write_wannier90_hr(
    #     filename='w90_hr.dat',
    #     titleline='Haldane model',
    # )

    return my_model


def plot_bands(model):
    highk = {
        r'$\Gamma$': [0.0, 0.0, 0.0],
        r'$X_1$': [0.5, 0.0, 0.0],
        r'$X_2$': [0.0, 0.5, 0.0],
        r'$M$': [0.5, 0.5, 0.0],
    }

    kpath_label = [r'$\Gamma$', r'$M$', r'$X_1$', r'$\Gamma$', r'$X_2$', r'$M$']

    model.plot_band_structure(kpath_label, highk, num_kp_kpath=81)


def plot_wilsonloop(model):
    nkp = 201
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
        num_occupy=1,
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
    my_model = Young_Kane()
    # my_model.print_info()
    # my_model.structure.print_info()
    plot_bands(my_model)
    # plot_wilsonloop(my_model)
    # plot_berrycurvature(my_model)
