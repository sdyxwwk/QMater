import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.optimize import minimize
from tqdm import tqdm


def get_kpoint_lines(kpath_start, kpath_end, num_kp_kpath=31):
    kpath_start = np.array(kpath_start, dtype=np.float64)
    kpath_end = np.array(kpath_end, dtype=np.float64)
    num_kpath = kpath_start.shape[0]
    mynum = len(kpath_end)
    if mynum != num_kpath:
        raise IndexError(
            "Dimension of starting and ending arrays dismatch!")

    num_kp = num_kp_kpath * num_kpath

    # coordinates of k-points
    kpoints = np.linspace(
        kpath_start, kpath_end, num_kp_kpath, axis=1)
    kpoints = kpoints.reshape(num_kp, 3)

    # positions of k-labels
    klabel_pos = np.zeros((num_kpath + 1, ), dtype=np.float64)
    klabel_pos[1:] = np.cumsum(
        np.linalg.norm(kpath_end - kpath_start, axis=1))

    # positions of k-points
    kpos = np.linspace(
        klabel_pos[:-1], klabel_pos[1:], num_kp_kpath, axis=1)
    kpos = kpos.reshape(num_kp, 1)
    return kpoints, kpos, klabel_pos


def get_kpoint_circle(k_radius, k_center=(0.0, 0.0, 0.0), num_kp=101):
    k_center = np.array(k_center, dtype=np.float64)
    theta = np.linspace(0.0, 2*np.pi, num_kp)
    kpoints = np.zeros((num_kp, 3), dtype=np.float64)
    kpoints[:, 0] = k_radius*np.cos(theta) + k_center[0]
    kpoints[:, 1] = k_radius*np.sin(theta) + k_center[1]
    kpoints[:, 2] = k_center[2]
    return kpoints, theta


def get_kpoint_plane(kplane_center, kplane_dir1, kplane_dir2, num_kp_dir=10):
    kcenter = np.array(kplane_center)
    kdir1 = np.array(kplane_dir1)
    kdir2 = np.array(kplane_dir2)

    korigin = kcenter - kdir1 / 2 - kdir2 / 2
    kpoint1 = np.linspace(korigin, korigin + kdir1, num_kp_dir)
    kpoint2 = np.linspace(korigin, korigin + kdir2, num_kp_dir)

    # coordinates of k-points
    kpoint_array = np.zeros((num_kp_dir, num_kp_dir, 3), dtype=np.float64)
    for ind, kp in enumerate(kpoint1):
        kpoint_array[ind, :, :] = kpoint2 + (kp - korigin)

    _klen1 = np.linalg.norm(kdir1)
    _klen2 = np.linalg.norm(kdir2)

    kpos1 = np.linspace(-_klen1 / 2, _klen1 / 2, num_kp_dir)
    kpos2 = np.linspace(-_klen2 / 2, _klen2 / 2, num_kp_dir)

    # positions of k-points
    kpos_array = np.zeros((num_kp_dir, num_kp_dir, 2), dtype=np.float64)
    kposx, kposy = np.meshgrid(kpos1, kpos2)
    kpos_array[:, :, 0] = np.transpose(kposx)
    kpos_array[:, :, 1] = np.transpose(kposy)

    return kpoint_array, kpos_array


class KdotP(object):
    def __init__(self, func, num_bands=1, para_list=None):
        self.func = func
        self.num_bands = num_bands
        self.parameters = para_list

    @property
    def func(self):
        return self._func

    @func.setter
    def func(self, _func):
        self._func = _func

    @property
    def parameters(self):
        return self._paralist

    @parameters.setter
    def parameters(self, _paralist):
        self._paralist = _paralist

    @property
    def num_bands(self):
        return self._num_bands

    @num_bands.setter
    def num_bands(self, _num_bands):
        self._num_bands = _num_bands

    def _calc_hamk(self, kp):
        return self.func(kp, self.parameters)

    def calc_eigvk(self, kp):
        hamk = self._calc_hamk(kp)
        w, v = np.linalg.eigh(hamk)
        index = np.argsort(w.real)
        eigstat = v[:, index]
        eigval = w[index].real
        return eigval, eigstat

    def calc_berryphase_1d(self, kp_list, num_occupy=1, if_eigval=False):
        num_kp = len(kp_list)
        num_bd = self.num_bands

        # get all eigenstates along the k-path
        eigstat_array = np.zeros(
            (num_kp, num_bd, num_occupy), dtype=np.complex128)
        for ind, kp in enumerate(kp_list):
            eigv, eigs = self.calc_eigvk(kp)
            eigstat_array[ind, :, :] = eigs[:, :num_occupy]

        Lambda = np.eye(num_occupy, dtype=np.complex128)
        for i in range(num_kp-1):
            eig1 = eigstat_array[i, :, :]

            if i == num_kp - 2:
                eigs2 = eigstat_array[0, :, :]
            else:
                eigs2 = eigstat_array[i + 1, :, :]

            Mmnk = np.dot(np.conj(np.transpose(eig1)), eigs2)

            if if_eigval:
                u, s, vh = np.linalg.svd(Mmnk)
                Lambda = np.dot(Lambda, np.dot(u, vh))

                # u = np.conj(np.transpose(u))
                # vh = np.conj(np.transpose(vh))
                # Lambda = np.dot(np.dot(vh, u), Lambda)
            else:
                Lambda = np.dot(Lambda, Mmnk)

        if if_eigval:
            myeigval = np.linalg.eigvals(Lambda)
            berryphase_eigval = (-1.0) * np.angle(myeigval)/2.0/np.pi
            # berryphase_eigval = (-1.0)*np.log(myeigval).real/2.0/np.pi
            berryphase_eigval = np.sort(berryphase_eigval)
            return berryphase_eigval
        else:
            mydet = np.linalg.det(Lambda)
            berryphase = (-1.0) * np.angle(mydet)/2.0/np.pi
            return berryphase

    def plot_dispersion(self, kpath_label, highk, num_kp_kpath=31, Elim=None):
        kpathlist = np.array([highk[a] for a in kpath_label])
        num_kpath = len(kpathlist)-1

        kpath_start = kpathlist[0:num_kpath]
        kpath_end = kpathlist[1:num_kpath+1]
        [kp_list, kpos_list, label_pos] = get_kpoint_lines(
            kpath_start, kpath_end, num_kp_kpath)
        num_kp = num_kp_kpath*num_kpath

        Elist = np.zeros([num_kp, self.num_bands])
        for i in range(num_kp):
            kp = kp_list[i, :]
            _val, _stat = self.calc_eigvk(kp)
            Elist[i, :] = _val

        num_label = len(label_pos)

        Emax = np.amax(Elist)
        Emin = np.amin(Elist)
        if Elim is None:
            ymax = Emax + (Emax - Emin) * 0.05
            ymin = Emin - (Emax - Emin) * 0.05
        else:
            ymin, ymax = Elim

        fig, ax = plt.subplots()
        ax.plot(kpos_list, Elist)

        for i in range(num_label):
            ax.plot([label_pos[i], label_pos[i]], [
                    ymin, ymax], linewidth=0.5, color='k')
        # ax.plot([kpos_list[0], kpos_list[-1]], [0.0, 0.0],
        #        linewidth=0.5, color='k', linestyle="--")
        ax.set_xticks(label_pos)
        ax.set_xticklabels(kpath_label, fontsize=12)
        plt.ylabel('Energy', fontsize=12)
        plt.xlim(kpos_list[0], kpos_list[-1])
        plt.ylim(ymin, ymax)
        plt.savefig('bands.pdf', dpi=300)
        # plt.show()
        # return fig, ax

    def plot_dispersion_3d(self, kdir1, kdir2, kcenter,
                           num_kp_dir=11, select='All'):
        kpoint_array, kpos_array = get_kpoint_plane(
            kcenter, kdir1, kdir2, num_kp_dir)

        if select == 'All':
            select = np.linspace(
                1, self.num_bands, self.num_bands, dtype=np.int)

        Elist = np.zeros([num_kp_dir, num_kp_dir, self.num_bands])

        for i in range(num_kp_dir):
            for j in range(num_kp_dir):
                kp = kpoint_array[i, j, :]
                _val, _stat = self.calc_eigvk(kp)
                Elist[i, j, :] = _val

        Emax = np.amax(Elist)
        Emin = np.amin(Elist)
        zmax = Emax + (Emax - Emin) * 0.05
        zmin = Emin - (Emax - Emin) * 0.05

        try:
            from mayavi import mlab
            for i in select:
                mlab.surf(
                    kpos_array[:, :, 0], kpos_array[:, :, 1], Elist[:, :, i-1]
                )
            mlab.show()
        except:
            fig = plt.figure()
            ax = fig.gca(projection='3d')

            for i in select:
                surf = ax.plot_surface(
                    kpos_array[:, :, 0], kpos_array[:, :, 1], Elist[:, :, i-1],
                    cmap=cm.coolwarm,
                    linewidth=0,
                    antialiased=True,
                    vmin=zmin,
                    vmax=zmax,
                    alpha=0.7,
                )
            # fig.colorbar(surf, shrink=0.5, aspect=5)
            plt.show()
            # return fig, ax

    def plot_fermisurf_kplane(self, mu, kdir1, kdir2, kcenter, num_kp_dir=11):
        kpoint_array, kpos_array = get_kpoint_plane(
            kcenter, kdir1, kdir2, num_kp_dir)

        eta = 5e-3
        ldos = np.zeros([num_kp_dir, num_kp_dir], dtype=np.float64)
        for i in tqdm(range(num_kp_dir)):
            for j in range(num_kp_dir):
                kp = kpoint_array[i, j, :]
                hamk = self._calc_hamk(kp)
                G00 = np.linalg.inv(
                    (mu + 1.j*eta)*np.eye(self.num_bands) - hamk)
                ldos[i, j] = -np.trace(G00.imag)/np.pi

        fig, ax = plt.subplots()
        # cs = ax.contourf(
        #     kpos_array[:, :, 0], kpos_array[:, :, 1], np.log(ldos),
        #     levels=100,
        #     cmap='inferno',
        #     antialiased=False,
        # )
        cs = ax.imshow(
            np.log(ldos),
            cmap='inferno',
        )
        ax.set_aspect('equal', adjustable='box')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel(r'$k_1$')
        ax.set_ylabel(r'$k_2$')
        ax.set_title('E = {:5.3f}'.format(mu))
        fig.colorbar(cs)
        fig.savefig('fermisurf.png', dpi=300)
        # plt.show()

    # NOTE: bands = (kposlist, kx, ky, kz, energy)
    def fit_parameters(self, bands, initial, cutoff=1.0, lplot=True):
        num_kp = bands.shape[0]

        def target_func(paralist):
            self.parameters = paralist

            _sum = 0.0
            for _band in bands:
                _kp = _band[1:4]
                if np.linalg.norm(_kp) < cutoff:
                    _en1 = _band[4:]
                    _en2, _stat = self.calc_eigvk(_kp)
                    _sum += np.linalg.norm(_en2 - _en1)**2
            return _sum

        my_min = minimize(target_func, initial, tol=1e-6)
        print('parameters = ', my_min.x)

        if lplot:
            self.parameters = my_min.x
            kpos_list = bands[:, 0:1]
            Elist = np.zeros((num_kp, self.num_bands), dtype=np.float64)
            for i in range(num_kp):
                Elist[i, :], _stat = self.calc_eigvk(bands[i, 1:4])

            Emax = np.amax(bands[:, 4:])
            Emin = np.amin(bands[:, 4:])
            ymax = Emax + (Emax - Emin) * 0.05
            ymin = Emin - (Emax - Emin) * 0.05

            fig, ax = plt.subplots()
            ax.plot(kpos_list, bands[:, 4:], '.')
            ax.plot(kpos_list, Elist, '--')
            ax.set_ylim([ymin, ymax])
            return fig, ax
            # fig.savefig('fit.pdf', dpt=300)

    def print_info(self):
        num_bands = self.num_bands
        _ham = self.func([0.0, 0.0, 0.0], self.parameters)

        print('========================================')
        print('              k dot p model             ')
        print('----------------------------------------')
        print('Name of bands                 => {:d}'.format(num_bands))
        print('')
        print('On-site term at (0.0, 0.0, 0.0) =>')
        for i in range(num_bands):
            print('{:3d} {:7.3f} +{:7.3f}i'.format(
                i+1, _ham[i, i].real, _ham[i, i].imag))
        print('========================================')
