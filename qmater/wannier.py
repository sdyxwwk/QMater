import os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from qmater.crystal import CrystStruct
from qmater.func import fermi_dirac, fermi_dirac_derivative

# --------------------------------
# Physical Constants
# --------------------------------
hbar = 6.582119569e-16  # eV s
# hbar = 1.054571817e-34 # J s
echarge = 1.602176634e-19  # C
kb = 8.617333262145e-5  # eV K^-1

pauli_0 = np.eye(2, dtype=np.complex128)
pauli_x = np.array([[0., 1.], [1., 0.]], dtype=np.complex128)
pauli_y = np.array([[0., -1j], [1j, 0]], dtype=np.complex128)
pauli_z = np.array([[1., 0.], [0., -1.]], dtype=np.complex128)


class WannierTB(object):
    """ Class WannierTB for a Wannier tight-binding (TB) model. """

    def __init__(self, structure,
                 projectors=None,
                 if_soc=False,
                 fermi=0.0):
        """ Initialization

        Args:
            structure (CrystStruct, optional): a pre-defined crystal structure. Defaults to None.
            projectors (dict, optional): pre-defined projectors. Defaults to None.
            if_soc (bool, optional): whether SOC is included or not. Defaults to False.

        """

        self.structure = structure
        self.if_soc = if_soc
        self.projectors = projectors
        self.fermi = fermi

        # set self.num_wann and self.wannier_centers
        self._set_wannier_parameters()

        self._hoplist = []

    @property
    def structure(self):
        return self._structure

    @structure.setter
    def structure(self, structure):
        if not isinstance(structure, CrystStruct):
            raise ValueError('structure should be a CrystStruct!')
        self._structure = structure

    @property
    def direct_lattice(self):
        return self.structure.direct_lattice

    @property
    def reciprocal_lattice(self):
        return self.structure.reciprocal_lattice

    @property
    def if_soc(self):
        return self._if_soc

    @if_soc.setter
    def if_soc(self, if_soc):
        if not isinstance(if_soc, bool):
            raise ValueError('if_soc should be True or False!')
        self._if_soc = if_soc

    @property
    def projectors(self):
        return self._projectors

    @projectors.setter
    def projectors(self, projectors):
        """ Set projectors from a given simple projector dict.
            - 'p' -> 'pz', 'px', 'py'
            - 'd' -> 'dz2', 'dxz', 'dyz', 'dx2-y2', 'dxy'
        """
        if projectors is None:
            projectors = {}
            for entry in self.structure.elementlist:
                projectors[entry[0]] = ['s']

        if not isinstance(projectors, dict):
            raise ValueError('projectors should be a dict!')

        orbitals = {
            's': ['s'],
            'p': ['pz', 'px', 'py'],
            'd': ['dz2', 'dxz', 'dyz', 'dx2-y2', 'dxy']
        }

        self._projectors = {}
        for _elem, _orb in projectors.items():
            self._projectors[_elem] = []
            for a in _orb:
                if a in orbitals:
                    self._projectors[_elem].extend(orbitals[a])
                else:
                    self._projectors[_elem].append(a)

    @property
    def num_wann(self):
        return self._num_wann

    @property
    def num_rpts(self):
        return len(self._hoplist)

    @property
    def wannier_centers_frac(self):
        return self._wannier_centers[0]

    @property
    def wannier_centers_cart(self):
        return self._wannier_centers[1]

    @property
    def fermi(self):
        return self._fermi

    @fermi.setter
    def fermi(self, fermi):
        self._fermi = fermi

    # ------------------------
    #  internal functions
    # ------------------------
    def _set_wannier_parameters(self):
        """ Set self.num_wann and self.wannier_centers. """
        elementlist = self.structure.elementlist
        atomsites = self.structure.atom_sites
        myprojs = self.projectors
        if_soc = self.if_soc

        num_wann = sum([len(myprojs[_elem]) * _natoms
                        for (_elem, _natoms) in elementlist])

        if if_soc:
            self._num_wann = num_wann * 2
        else:
            self._num_wann = num_wann

        _wcc_frac = np.zeros([self.num_wann, 3], dtype=np.float64)

        nc = 0
        for _elem, _pos in atomsites.items():
            _wcc = np.kron(_pos, np.ones((len(myprojs[_elem]), 1)))
            _wcc_frac[nc:nc + _wcc.shape[0]] = _wcc
            nc += _wcc.shape[0]

        if if_soc:
            nc = self.num_wann // 2
            _wcc_frac[nc:, :] = _wcc_frac[:nc, :]

        _wcc_cart = np.dot(_wcc_frac, self.direct_lattice)
        self._wannier_centers = [_wcc_frac, _wcc_cart]

    # ----------------
    #  set model
    # ----------------
    def _set_hoplist(self, Rvec, rind, hop, deg, mode):
        r""" set or add hopping term in _hoplist
                - hopping direction: (0 + r_{i}) -> (R + r_{j})
                - hopping term: $\langle \phi_{0,i} | H | \phi_{R,j} \rangle$

        Args:
            Rvec (list, int): R-vector
            rind (list, int): Index of orbitals within the home unit cell, i.e. rind = [i, j].
            hop (float or array): hopping term
                    - If lsoc is False, it is a float number;
                    - If lsoc is True, it is a float np.array.
            deg (int): degeneracy
            mode (str): mode of operation
                    - 'set': set a new hopping entry
                    - 'add': add hopping to a current entry

        """

        lsoc = self.if_soc
        num_wann = self.num_wann

        Rlist = [a[0] for a in self._hoplist]

        [rm, rn] = rind

        # get the index of hopping matrix
        # spinless case: (m,n)
        # spinfull case: [[(m, n), (m, n+num_wann/2)],
        #                 [(m+num_wann/2, n+num_wann/2), (m, n+num_wann/2)]]
        if lsoc:
            hop = np.array(hop)
            if hop.shape == (2, 2):
                myid = np.ix_([rm, rm + num_wann // 2],
                              [rn, rn + num_wann // 2])
            else:
                raise Exception("Wrong hopping term: SOC is included!")
        else:
            if type(hop).__name__ not in ['int', 'int64', 'float', 'float64', 'complex', 'complex128']:
                raise Exception("Wrong hopping term: SOC is not included!")
            myid = (rm, rn)

        # set or add hopping term in hopping matrix
        if Rvec in Rlist:
            Rid = Rlist.index(Rvec)
            if mode == 'set':
                # myhop = self._hoplist[Rid][1][myid]
                self._hoplist[Rid][1][myid] = hop
            elif mode == 'add':
                self._hoplist[Rid][1][myid] += hop
        else:
            myH = np.zeros([num_wann, num_wann], dtype=np.complex128)
            myH[myid] = hop
            self._hoplist.append([Rvec, myH, deg])

    def set_hop(self, Rvec, rind, hop, deg=1, lconj=True):
        """ set a new hopping term

        Args:
            Rvec (list, int): R-vector
            rind (list, int): Index of orbitals within the home unit cell.
            hop (float or array): hopping term
            deg (int, optional): degeneracy. Defaults to 1.
            lconj (bool, optional): whether add -R or not. Defaults to True.
        """

        lsoc = self._if_soc

        self._set_hoplist(Rvec, rind, hop, deg, 'set')

        # check if set the conjugate pair at -R
        if lconj and not (Rvec == [0, 0, 0] and rind[0] == rind[1]):
            Rvec1 = [-a for a in Rvec]
            rind1 = rind[::-1]
            if lsoc:
                hop1 = np.transpose(np.conj(hop))
            else:
                hop1 = np.conj(hop)

            self._set_hoplist(Rvec1, rind1, hop1, deg, 'set')

    def add_hop(self, Rvec, rind, hop, deg=1, lconj=True):
        """ add a hopping term to a current term

        Args:
            Rvec (list, int): R-vector
            rind (list, int): Index of orbitals within the home unit cell.
            hop (float or array): hopping term
            deg (int, optional): degeneracy. Defaults to 1.
            lconj (bool, optional): whether add -R or not. Defaults to True.

        """

        lsoc = self.if_soc

        self._set_hoplist(Rvec, rind, hop, deg, 'add')

        # check if add the conjugate pair at -R
        if lconj and not (Rvec == [0, 0, 0] and rind[0] == rind[1]):
            Rvec1 = [-a for a in Rvec]
            rind1 = rind[::-1]
            if lsoc:
                hop1 = np.transpose(np.conj(hop))
            else:
                hop1 = np.conj(hop)
            self._set_hoplist(Rvec1, rind1, hop1, deg, 'add')

    def read_wannier90_hr(self, filename='wannier90_hr.dat', fmt=None):
        """ set _hoplist from Wannier90 Hr file

        Args:
            filename (str, optional): Wannier90 Hr file. Defaults to 'wannier90_hr.dat'.

        Raises:
            Exception: If Hr file is not found, raise Exception.
            Exception: If number of Wannier functions in Hr file mismatch the pre-defined
                       structure and projector, raise Exception.
        """

        if not os.path.exists(filename):
            raise Exception('Does not find the file {:s}'.format(filename))

        with open(filename, 'r') as infile:
            print('========================================')
            print('           Reading Hrfile ...           ')
            print('----------------------------------------')
            infile.readline()
            num_wann = int(infile.readline().split()[0])
            num_rpts = int(infile.readline().split()[0])

            if num_wann != self.num_wann:
                raise Exception(
                    'Number of WFs in {:s} does not match number of orbitals'.format(filename))

            print('Number of wannier functions => {:d}'.format(num_wann))
            print('Number of R-vectors => {:d}'.format(num_rpts))

            deg_rpts = []
            while len(deg_rpts) < num_rpts:
                mylist = [int(a) for a in infile.readline().split()]
                deg_rpts.extend(mylist)

            Rvec = []
            hoplist = []

            for i in tqdm(range(num_rpts)):
                # if num_rpts > 10 and i % int(num_rpts / 10) == 0:
                #     print('R-vector: {:d}'.format(i))

                HmnR = np.zeros((num_wann, num_wann), dtype=np.complex128)
                for j in range(num_wann):
                    for k in range(num_wann):
                        myline = infile.readline().split()
                        if j == 0 and k == 0:
                            myR = [int(a) for a in myline[0:3]]
                            Rvec.append(myR)
                        rm = int(myline[3])-1
                        rn = int(myline[4])-1

                        if fmt == 'spin':
                            num_wann2 = num_wann // 2
                            rm = (rm % 2)*num_wann2 + (rm//2)
                            rn = (rn % 2)*num_wann2 + (rn//2)

                        HmnR[rm, rn] = \
                            np.float64(myline[5]) + 1.j * np.float64(myline[6])

                myhop = [Rvec[i], HmnR, deg_rpts[i]]
                hoplist.append(myhop)

            self._hoplist = hoplist
            print('========================================')
            print('')

    def write_wannier90_hr(self, filename='wannier90_hr.dat', titleline=None):
        """ write Wannier90 Hr file

        Args:
            filename (str, optional): Wannier90 Hr file. Defaults to 'wannier90_hr.dat'.

        """

        num_wann = self.num_wann
        num_rpts = self.num_rpts

        Rvec = [a[0] for a in self._hoplist]
        HmnR = [a[1] for a in self._hoplist]
        deg_rpts = [a[2] for a in self._hoplist]

        if titleline is None:
            titleline = 'written by QMater'

        print('========================================')
        print('          Writing Hrfile ...            ')
        print('----------------------------------------')
        with open(filename, 'w') as outfile:
            outfile.write(titleline + '\n')
            outfile.write('{:12d}\n'.format(num_wann))
            outfile.write('{:12d}\n'.format(num_rpts))
            nc = 0
            for i in range(num_rpts):
                outfile.write('{:5d}'.format(deg_rpts[i]))
                nc += 1
                if nc % 15 == 0:
                    outfile.write('\n')
            outfile.write('\n')

            for i in tqdm(range(num_rpts)):
                # if num_rpts > 10 and i % int(num_rpts / 10) == 0:
                #     print('R-vector: {:d}'.format(i))

                for j in range(num_wann):
                    for k in range(num_wann):
                        outfile.write('{:5d}{:5d}{:5d}{:5d}{:5d}{:12.6f}{:12.6f}\n'.format(
                            # outfile.write('{:12.6f}{:12.6f}{:12.6f}{:5d}{:5d}{:12.6f}{:12.6f}\n'.format(
                            Rvec[i][0], Rvec[i][1], Rvec[i][2], k + 1, j + 1,
                            HmnR[i][k, j].real, HmnR[i][k, j].imag))
        print('========================================')
        print('')

    # ----------------
    #  build model
    # ----------------

    def build_slab(self, num_slab, direction='c'):
        pass

    # ----------------
    #  calculations
    # ----------------

    def _calc_hamk(self, kp):
        """ Calculate Hamiltonian on a k-point (fractional coordinate). """
        kp = np.array(kp, dtype=np.float64)

        num_wann = self.num_wann
        _wcc = self.wannier_centers_frac

        Hamk = np.zeros((num_wann, num_wann), dtype=np.complex128)

        pi2j = 2*np.pi*1.j
        Umatr = np.diag([np.exp(pi2j * np.dot(kp, _wcc[i, :]))
                         for i in range(num_wann)])

        for _hop in self._hoplist:
            myR = np.array(_hop[0])
            deg_rpts = _hop[2]
            HmnR = _hop[1]
            kdotr = np.dot(kp, myR)
            Hamk += np.exp(pi2j * kdotr) * HmnR[:, :]/deg_rpts

        # Convention I
        Hamk = np.dot(np.conj(Umatr), np.dot(Hamk, Umatr))
        return Hamk

    def calc_eigvk(self, kp):
        """ calculate eigenvalues and eigenstates of a k-point (fractional coordinate)

        Args:
            kp (list or array): k-point

        Returns:
            eigstat (array): eigenstates at the k-point
            eigval (array): eigenvalues at the k-point
        """

        hamk = self._calc_hamk(kp)
        w, v = np.linalg.eigh(hamk)
        index = np.argsort(w.real)
        eigstat = v[:, index]
        eigval = w[index].real
        return eigval, eigstat

    def _calc_dhamdk(self, kp):
        r""" calculate the derivative of Hamiltonian in k at a k-point (fractional coordinate)
                - derivative of Hamiltonian: $\nabla_{k} H(k)$
                - $\nabla_{k} H(k) = \sum_{R} iR e^{i k \cdot R} H_{R}$

        Args:
            kp (list): k-point

        Returns:
            [np.array]: derivative of Hamiltonian in k. The shape is (num_wann, num_wann, 3)
        """
        kp = np.array(kp, dtype=np.float64)

        num_wann = self.num_wann
        _wcc = self.wannier_centers_cart
        _struct = self.structure

        # Transform fractional coordinate to Cartesian coordinate for k-point.
        kp = _struct.get_kpoint_cart(kp)

        pi2j = 1.j  # 2*np.pi*1.j
        Umatr = np.diag([np.exp(pi2j * np.dot(kp, _wcc[i, :]))
                         for i in range(num_wann)])

        dUmatr_dk = np.zeros((num_wann, num_wann, 3), dtype=np.complex128)
        for m in range(3):
            dUmatr_dk[:, :, m] = np.diag([1.j * _wcc[i, m] * np.exp(pi2j * np.dot(kp, _wcc[i, :]))
                                          for i in range(num_wann)])

        HamRk = np.zeros((num_wann, num_wann), dtype=np.complex128)
        dHamR_dk = np.zeros((num_wann, num_wann, 3), dtype=np.complex128)

        for _hop in self._hoplist:
            myR = np.dot(np.array(_hop[0]), _struct.direct_lattice)
            HmnR = _hop[1]
            deg_rpts = _hop[2]

            kdotr = np.dot(kp, myR)
            HamRk += deg_rpts * np.exp(pi2j * kdotr) * HmnR[:, :]
            for i in range(3):
                dHamR_dk[:, :, i] += 1.j * myR[i] * \
                    deg_rpts * np.exp(pi2j * kdotr) * HmnR[:, :]

        dHam_dk = np.zeros((num_wann, num_wann, 3), dtype=np.complex128)
        for i in range(3):
            dHam_dk[:, :, i] = np.dot(np.conj(dUmatr_dk[:, :, i]), np.dot(HamRk, Umatr)) + \
                np.dot(np.conj(Umatr), np.dot(dHamR_dk[:, :, i], Umatr)) + \
                np.dot(np.conj(Umatr), np.dot(HamRk, dUmatr_dk[:, :, i]))

        return dHam_dk

    # TODO: Create Singlek class
    def _calc_spinmat(self, kp):
        num_wann = self.num_wann
        # _eigv, _eigs = self.calc_eigvk(kp)

        if self.if_soc:
            spinmat = np.zeros((num_wann, num_wann, 3), dtype=np.complex128)
            # sx = np.kron(pauli_x, np.eye(num_wann//2))
            # sy = np.kron(pauli_y, np.eye(num_wann//2))
            # sz = np.kron(pauli_z, np.eye(num_wann//2))
            # np.dot(np.conj(_eigs.T), np.dot(sx, _eigs))
            spinmat[:, :, 0] = np.kron(pauli_x, np.eye(num_wann//2))
            # np.dot(np.conj(_eigs.T), np.dot(sy, _eigs))
            spinmat[:, :, 1] = np.kron(pauli_y, np.eye(num_wann//2))
            # np.dot(np.conj(_eigs.T), np.dot(sz, _eigs))
            spinmat[:, :, 2] = np.kron(pauli_z, np.eye(num_wann//2))
        else:
            spinmat = np.zeros((num_wann, num_wann, 3), dtype=np.complex128)
            spinmat[:, :, 0] = 2.0*np.eye(num_wann)

        return spinmat

    def calc_fermisurf_kplane(self, mu, kdir1, kdir2, kcenter, num_kp_dir=11):
        kp_array, kpos_array = self.structure.get_kpoint_plane(
            kcenter, kdir1, kdir2, num_kp_dir, frac='F')

        eta = 5e-3
        ldos = np.zeros([num_kp_dir, num_kp_dir], dtype=np.float64)
        for i in tqdm(range(num_kp_dir)):
            for j in range(num_kp_dir):
                kp = kp_array[i, j, :]
                hamk = self._calc_hamk(kp)
                G00 = np.linalg.inv(
                    (mu + 1.j*eta)*np.eye(self.num_wann) - hamk)
                ldos[i, j] = -np.trace(G00.imag)/np.pi

        return kpos_array, ldos

    def _calc_berryphase_1d(self, num_occupy, kp_list, if_eigval=False):
        """ calculate Berry phase along a closed k-path
                - Refer to `Phys. Rev. B 83, 235401 (2011).`

        Args:
            num_occupy (int): number of occupied states
            kp_list (np.array): k-points on the closed k-path
            if_eigval (bool, optional): Defaults to False.
                    - If True, return eigenvalues of unitary rotation matrix.
                    - If False, return the total Berry phase.

        Returns:
            [np.array or float]: eigenvalues or Berry phase based on if_eigval.
        """

        num_wann = self.num_wann
        num_kp = len(kp_list)

        # get all eigenstates along the k-path
        eigstat_array = np.zeros(
            (num_kp, num_wann, num_occupy), dtype=np.complex128)
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
            berryphase_eigval = np.sort(np.mod(berryphase_eigval, 1.0))
            # berryphase_eigval = np.sort(berryphase_eigval)
            return berryphase_eigval
        else:
            mydet = np.linalg.det(Lambda)
            berryphase = (-1.0) * np.angle(mydet)/2.0/np.pi
            return berryphase

    def calc_wilsonloop(self, num_kp, origin=[0, 0, 0],
                        int_dir=[1, 0, 0],
                        var_dir=[0, 1, 0], num_occupy=1):
        """ calculate Wilson loops or hybrid Wannier charge centers (WCCs) on a k-plane

        Args:
            num_kp (int): number of k-points for both the integration or variation direction.
            origin (list, optional): k-point at origin. Defaults to [0, 0, 0].
            int_dir (list, optional): integration direction for WCCs. Defaults to [1, 0, 0].
            var_dir (list, optional): variation direction. Defaults to [0, 1, 0].
            num_occupy (int, optional): number of occupied states. Defaults to 1.

        Returns:
            [np.array]: Wilson loops or hybrid WCCs
        """

        if num_occupy > self.num_wann:
            print('Since num_occupy > num_wann, we set num_occupy = num_wann.')
            num_occupy = self.num_wann

        origin = np.array(origin)
        int_dir = np.array(int_dir)
        var_dir = np.array(var_dir)

        # k-points along integration direction
        int_kp = np.linspace(np.array([0.0, 0.0, 0.0]), int_dir, num_kp)
        # k-points which are variables
        var_kp = np.linspace(origin, origin+var_dir, num_kp)

        wcc = np.zeros((num_kp, num_occupy), dtype=np.float64)
        for ind, kp in enumerate(tqdm(var_kp)):
            # print(ind)
            kp_list = kp + int_kp

            # get Wannier centers of each occupied state
            wcc[ind, :] = self._calc_berryphase_1d(
                num_occupy, kp_list, if_eigval=True)

        return wcc

    def _calc_berrycurv_singlek(self, kp, num_occupy, index='xy'):
        r""" Calculate Berry curvature for a single k-point.

        Refer to `Phys. Rev. B 74, 195118 (2006).`

        Parameters
        ----------
        kp : list or numpy.ndarray
            k-point
        num_occupy : int
            number of occupied states
        index : string
            index of Berry curvature

        Returns
        -------
        tuple
            Berry curvature, i.e. (bcx, bcy, bcz)
        """
        _dhdk = self._calc_dhamdk(kp)
        _eigv, _eigs = self.calc_eigvk(kp)
        d1, d2 = [{'x': 0, 'y': 1, 'z': 2}[i] for i in index]

        # matrix rep of velocity operator
        # v_{mn}^\mu = 1/\hbar \langle u_{m,k}| \partial_\mu H(k) |u_{n,k} \rangle
        velocity = np.zeros((self.num_wann, self.num_wann, 3),
                            dtype=np.complex128)
        for i in range(3):
            velocity[:, :, i] = np.dot(
                np.conj(_eigs.T), np.dot(_dhdk[:, :, i], _eigs))

        omega = np.complex128(0.0)
        eta = 1e-6
        for v in range(num_occupy):
            for c in range(num_occupy, self.num_wann):
                # To avoid the inf value at degenerate points
                w_eta = (_eigv[c] - _eigv[v]) / \
                    ((_eigv[c] - _eigv[v])**2 + eta**2)
                w_eta = w_eta**2
                omega += velocity[v, c, d1]*velocity[c, v, d2]*w_eta

        return -2.0*omega.imag

    def calc_berrycurv_kplane(self, num_kp, center, dir1, dir2,
                              num_occupy=1, index='xy'):
        r""" calculate Berry curvature for a k-plane

        Args:
            num_kp (int): number of k-points on each direction.
            center (list): center of a k-plane. Defaults to [0, 0, 0].
            dir1 (list): direction 1 to define a k-plane. Defaults to [1, 0, 0].
            dir2 (list): direction 2 to define a k-plane. Defaults to [0, 1, 0].
            num_occupy (int, optional): number of occupied states. Defaults to 1.

        Returns:
            [tuple]: (kpos_array, Omegaxy, Omegayz, Omegazx)
                - kpos_array (np.array): position of k-points on k-plane. Shape is (num_kp, num_kp).
                - Omegaxy (np.array): \Omega_{xy}.  Shape is (num_kp, num_kp).
                - Omegayz (np.array): \Omega_{yz}.  Shape is (num_kp, num_kp).
                - Omegazx (np.array): \Omega_{zx}.  Shape is (num_kp, num_kp).
        """

        if num_occupy > self.num_wann:
            print('Since num_occupy > num_wann, we set num_occupy = num_wann.')
            num_occupy = self.num_wann

        mystruct = self.structure

        kp_array, kpos_array = mystruct.get_kpoint_plane(
            center, dir1, dir2, num_kp_dir=num_kp)

        bc = np.zeros((num_kp, num_kp), dtype=np.float64)

        for i in tqdm(range(num_kp)):
            for j in range(num_kp):
                bc[i, j] = self._calc_berrycurv_singlek(
                    kp_array[i, j, :], num_occupy, index=index)

        return kpos_array, bc

    def _calc_bcp_singlek_occupied(self, kp, num_occupy, index='xx'):
        r""" Calculate Berry connection polarizability for occupied bands
        at a single k-point.

        Note: The factor \hbar^2 is eliminated by the velocity operator.

        Parameters
        ----------
        kp : list or numpy.ndarray
            k-point
        num_occupy : int
            number of occupied states
        index : string
            index of Berry connection polarizability

        Returns
        -------
        A float number
            Berry connection polarizability
        """
        _dhdk = self._calc_dhamdk(kp)
        _eigv, _eigs = self.calc_eigvk(kp)
        d1, d2 = [{'x': 0, 'y': 1, 'z': 2}[i] for i in index]

        # matrix rep of velocity operator
        # v_{mn}^\mu = 1/\hbar \langle u_{m,k}| \partial_\mu H(k) |u_{n,k} \rangle
        velocity = np.zeros((self.num_wann, self.num_wann, 3),
                            dtype=np.complex128)
        for i in range(3):
            velocity[:, :, i] = np.dot(
                np.conj(_eigs.T), np.dot(_dhdk[:, :, i], _eigs))

        gk = np.complex128(0.0)
        eta = 1e-6
        for v in range(num_occupy):
            for c in range(num_occupy, self.num_wann):
                # To avoid the inf value at degenerate points
                w_eta = (_eigv[v] - _eigv[c]) / \
                    ((_eigv[v] - _eigv[c])**2 + eta**2)
                w_eta = w_eta**3
                # w_eta = 1.0/(_eigv[v] - _eigv[c])**3
                gk += velocity[v, c, d1]*velocity[c, v, d2]*w_eta

        return 2.0*gk.real

    def _calc_bcp_singlek_fermi(self, kp, mu, index='xx'):
        _dhdk = self._calc_dhamdk(kp)
        _eigv, _eigs = self.calc_eigvk(kp)
        d1, d2 = [{'x': 0, 'y': 1, 'z': 2}[i] for i in index]

        # matrix rep of velocity operator
        # v_{mn}^\mu = 1/\hbar \langle u_{m,k}| \partial_\mu H(k) |u_{n,k} \rangle
        velocity = np.zeros((self.num_wann, self.num_wann, 3),
                            dtype=np.complex128)
        for i in range(3):
            velocity[:, :, i] = np.dot(
                np.conj(_eigs.T), np.dot(_dhdk[:, :, i], _eigs))

        gk = np.complex128(0.0)
        eta = 1e-6
        for n in range(self.num_wann):
            gk0 = np.complex128(0.0)
            for m in range(self.num_wann):
                if m == n:
                    continue
                # To avoid the inf value at degenerate points
                w_eta = (_eigv[n] - _eigv[m]) / \
                    ((_eigv[n] - _eigv[m])**2 + eta**2)
                w_eta = w_eta**3
                # w_eta = 1.0/(_eigv[v] - _eigv[c])**3
                gk0 += velocity[n, m, d1]*velocity[m, n, d2]*w_eta
            gk += gk0*fermi_dirac(_eigv[n]-mu, kbT=kb*300)

        return 2.0*gk.real

    def calc_bcp_kplane(self, num_kp, center, dir1, dir2,
                        num_occupy=1, mu=0.0, index='xx', mode='occupy'):

        if num_occupy > self.num_wann:
            print('Since num_occupy > num_wann, we set num_occupy = num_wann.')
            num_occupy = self.num_wann

        kp_array, kpos_array = self.structure.get_kpoint_plane(
            center, dir1, dir2, num_kp_dir=num_kp, frac='F')

        bcp = np.zeros((num_kp, num_kp), dtype=np.float64)
        for i in tqdm(range(num_kp)):
            for j in range(num_kp):
                kp = kp_array[i, j, :]
                if mode == 'occupy':
                    bcp[i, j] = self._calc_bcp_singlek_occupied(
                        kp, num_occupy, index)
                else:
                    bcp[i, j] = self._calc_bcp_singlek_fermi(kp, mu, index)

        return kpos_array, bcp

    def _calc_alpha_fermisurface(self, kp, mu, index='xxx'):
        _dhdk = self._calc_dhamdk(kp)
        _spin = self._calc_spinmat(kp)
        _eigv, _eigs = self.calc_eigvk(kp)
        d1, d2, d3 = [{'x': 0, 'y': 1, 'z': 2}[i] for i in index]

        # matrix rep of velocity operator
        # v_{mn}^\mu = 1/\hbar \langle u_{m,k}| \partial_\mu H(k) |u_{n,k} \rangle
        velocity = np.zeros((self.num_wann, self.num_wann, 3),
                            dtype=np.complex128)
        for i in range(3):
            velocity[:, :, i] = np.dot(
                np.conj(_eigs.T), np.dot(_dhdk[:, :, i], _eigs))

        # matrix rep of spin operator
        # v_{mn}^\mu = 1/\hbar \langle u_{m,k}| \partial_\mu H(k) |u_{n,k} \rangle
        spin = np.zeros((self.num_wann, self.num_wann, 3),
                        dtype=np.complex128)
        for i in range(3):
            spin[:, :, i] = np.dot(
                np.conj(_eigs.T), np.dot(_spin[:, :, i], _eigs))

        alpha = np.complex128(0.0)
        eta = 1e-6
        for n in range(self.num_wann):
            alpha1 = np.complex128(0.0)
            alpha2 = np.complex128(0.0)
            for m in range(self.num_wann):
                if m == n:
                    continue
                # To avoid the inf value at degenerate points
                w_eta = (_eigv[n] - _eigv[m]) / \
                    ((_eigv[n] - _eigv[m])**2 + eta**2)
                w_eta = w_eta**3
                # w_eta = 1.0/(_eigv[v] - _eigv[c])**3
                alpha1 += velocity[n, m, d2]*velocity[m, n, d3]*w_eta
                alpha2 += spin[n, m, d1]*velocity[m, n, d3]*w_eta
            alpha1 *= -spin[n, n, d1]
            alpha2 *= velocity[n, n, d2]
            alpha += alpha1 * fermi_dirac_derivative(_eigv[n]-mu, kbT=kb*300)
            alpha += alpha2 * fermi_dirac_derivative(_eigv[n]-mu, kbT=kb*300)

        return 2.0*alpha.real

    def _calc_alpha_occupied(self, kp, num_occupy=1, index='xxx'):
        _dhdk = self._calc_dhamdk(kp)
        _spin = self._calc_spinmat(kp)
        _eigv, _eigs = self.calc_eigvk(kp)
        d1, d2, d3 = [{'x': 0, 'y': 1, 'z': 2}[i] for i in index]

        # matrix rep of velocity operator
        # v_{mn}^\mu = 1/\hbar \langle u_{m,k}| \partial_\mu H(k) |u_{n,k} \rangle
        velocity = np.zeros((self.num_wann, self.num_wann, 3),
                            dtype=np.complex128)
        for i in range(3):
            velocity[:, :, i] = np.dot(
                np.conj(_eigs.T), np.dot(_dhdk[:, :, i], _eigs))

        # matrix rep of spin operator
        # v_{mn}^\mu = 1/\hbar \langle u_{m,k}| \partial_\mu H(k) |u_{n,k} \rangle
        spin = np.zeros((self.num_wann, self.num_wann, 3),
                        dtype=np.complex128)
        for i in range(3):
            spin[:, :, i] = np.dot(
                np.conj(_eigs.T), np.dot(_spin[:, :, i], _eigs))

        alpha = np.complex128(0.0)
        eta = 1e-6
        for v in range(num_occupy):
            alpha1 = np.complex128(0.0)
            alpha2 = np.complex128(0.0)
            alpha3 = np.complex128(0.0)
            for c in range(num_occupy, self.num_wann):
                # To avoid the inf value at degenerate points
                w_eta = (_eigv[v] - _eigv[c]) / \
                    ((_eigv[v] - _eigv[c])**2 + eta**2)
                # w_eta = 1.0/(_eigv[v] - _eigv[c])**3

                alpha1 += -3.0*(spin[v, v, d1] - spin[c, c, d1]) * \
                    velocity[v, c, d2]*velocity[c, v, d3]*w_eta**4
                for m in range(self.num_wann):
                    if m != v:
                        # To avoid the inf value at degenerate points
                        w_eta1 = (_eigv[v] - _eigv[m]) / \
                            ((_eigv[v] - _eigv[m])**2 + eta**2)
                        # w_eta1 = 1.0/(_eigv[v] - _eigv[c])**3
                        alpha2 += spin[v, m, d1] * velocity[m, c, d2] * \
                            velocity[c, v, d3]*w_eta1*w_eta**3
                        alpha2 += spin[v, m, d1] * velocity[m, c, d3] * \
                            velocity[c, v, d2]*w_eta1*w_eta**3

                    if m != c:
                        # To avoid the inf value at degenerate points
                        w_eta2 = (_eigv[c] - _eigv[m]) / \
                            ((_eigv[c] - _eigv[m])**2 + eta**2)
                        # w_eta1 = 1.0/(_eigv[v] - _eigv[c])**3
                        alpha3 += spin[c, m, d1] * velocity[m, v, d2] * \
                            velocity[v, c, d3]*w_eta2*w_eta**3
                        alpha3 += spin[c, m, d1] * velocity[m, v, d3] * \
                            velocity[v, c, d2]*w_eta2*w_eta**3

            alpha += alpha1 + alpha2 + alpha3

        return 2.0*alpha.real

    def calc_alpha_kplane(self, num_kp, center, dir1, dir2,
                          mu=0.0, num_occupy=1, index='xxx', mode='occupy'):

        if num_occupy > self.num_wann:
            print('Since num_occupy > num_wann, we set num_occupy = num_wann.')
            num_occupy = self.num_wann

        kp_array, kpos_array = self.structure.get_kpoint_plane(
            center, dir1, dir2, num_kp_dir=num_kp)

        alpha = np.zeros((num_kp, num_kp), dtype=np.float64)
        for i in tqdm(range(num_kp)):
            for j in range(num_kp):
                kp = kp_array[i, j, :]
                if mode == 'occupy':
                    alpha[i, j] = self._calc_alpha_occupied(
                        kp, num_occupy, index)
                else:
                    alpha[i, j] = self._calc_alpha_fermisurface(kp, mu, index)

        return kpos_array, alpha

    def _calc_quanmetric_singlek(self, kp, num_occupy):
        """ Calculate quantum metric tensor for a single k. """
        eta = 1e-6

        num_wann = self.num_wann

        _dhdk = self._calc_dhamdk(kp)
        _eigv, _eigs = self.calc_eigvk(kp)

        # matrix rep of velocity operator
        vx = np.dot(np.conj(_eigs.T), np.dot(_dhdk[:, :, 0], _eigs))
        vy = np.dot(np.conj(_eigs.T), np.dot(_dhdk[:, :, 1], _eigs))
        vz = np.dot(np.conj(_eigs.T), np.dot(_dhdk[:, :, 2], _eigs))

        gxx = np.complex128(0.0)
        gyy = np.complex128(0.0)
        gzz = np.complex128(0.0)

        gxy = np.complex128(0.0)
        gyz = np.complex128(0.0)
        gzx = np.complex128(0.0)

        for m in range(num_occupy):
            for n in range(num_wann):
                if n != m:
                    w_eta = (_eigv[n] - _eigv[m]) / \
                        ((_eigv[n] - _eigv[m])**2 + eta**2)
                    w_eta = w_eta**2
                    gxx += vx[m, n] * vx[n, m] * w_eta
                    gyy += vy[m, n] * vy[n, m] * w_eta
                    gzz += vz[m, n] * vz[n, m] * w_eta

                    gxy += vx[m, n] * vy[n, m] * w_eta
                    gyz += vy[m, n] * vz[n, m] * w_eta
                    gzx += vz[m, n] * vx[n, m] * w_eta

        gxx = gxx.real
        gyy = gyy.real
        gzz = gzz.real

        gxy = gxy.real
        gyz = gyz.real
        gzx = gzx.real

        return gxx, gyy, gzz, gxy, gyz, gzx

    def calc_quantum_metric(self, num_kp,
                            center=[0, 0, 0],
                            dir1=[1, 0, 0],
                            dir2=[0, 1, 0], num_occupy=1):
        r""" calculate quantum metric for a k-plane

        Args:
            num_kp (int): number of k-points on each direction.
            center (list, optional): center of a k-plane. Defaults to [0, 0, 0].
            dir1 (list, optional): direction 1 to define a k-plane. Defaults to [1, 0, 0].
            dir2 (list, optional): direction 2 to define a k-plane. Defaults to [0, 1, 0].
            num_occupy (int, optional): number of occupied states. Defaults to 1.

        Returns:
            [tuple]: (kpos_array, Omegaxy, Omegayz, Omegazx)
                - kpos_array (np.array): position of k-points on k-plane. Shape is (num_kp, num_kp).
                - Omegaxy (np.array): \Omega_{xy}.  Shape is (num_kp, num_kp).
                - Omegayz (np.array): \Omega_{yz}.  Shape is (num_kp, num_kp).
                - Omegazx (np.array): \Omega_{zx}.  Shape is (num_kp, num_kp).
        """

        if num_occupy > self.num_wann:
            print('Since num_occupy > num_wann, we set num_occupy = num_wann.')
            num_occupy = self.num_wann

        mystruct = self.structure

        kp_array, kpos_array = mystruct.get_kpoint_plane(
            center, dir1, dir2, num_kp_dir=num_kp)

        gxx = np.zeros((num_kp, num_kp), dtype=np.float64)
        gyy = np.zeros((num_kp, num_kp), dtype=np.float64)
        gzz = np.zeros((num_kp, num_kp), dtype=np.float64)

        gxy = np.zeros((num_kp, num_kp), dtype=np.float64)
        gyz = np.zeros((num_kp, num_kp), dtype=np.float64)
        gzx = np.zeros((num_kp, num_kp), dtype=np.float64)

        for i in range(num_kp):
            for j in range(num_kp):
                _xx, _yy, _zz, _xy, _yz, _zx = self._calc_quanmetric_singlek(
                    kp_array[i, j, :], num_occupy)
                gxx[i, j] = _xx
                gyy[i, j] = _yy
                gzz[i, j] = _zz

                gxy[i, j] = _xy
                gyz[i, j] = _yz
                gzx[i, j] = _zx

        return kpos_array, gxx, gyy, gzz, gxy, gyz, gzx

    # ----------------
    #  plotting
    # ----------------

    def plot_band_structure(self, kpath_label, highk, num_kp_kpath=31, Elim=None):
        """ plot band structure for a set of given k-paths

        Args:
            kpath_label (list): k-labels
            highk (dict): high-symmetry k-points indicated by k-labels and fractional coordinates.
            num_kp_kpath (int, optional): number of k-points on each k-path. Defaults to 31.
            Elim (list, optional): maximum and minimum energy for plotting band structures. Defaults to None.
        """

        num_wann = self.num_wann
        mystruct = self.structure

        kpathlist = np.array([highk[a] for a in kpath_label])
        num_kpath = len(kpathlist)-1

        kpath_start = kpathlist[0:num_kpath]
        kpath_end = kpathlist[1:num_kpath+1]

        [kp_list, kpos_list, label_pos] = mystruct.get_kpoint_lines(
            kpath_start, kpath_end, num_kp_kpath)

        num_kp = num_kp_kpath*num_kpath
        Elist = np.zeros([num_kp, num_wann])

        for i in range(num_kp):
            kp = kp_list[i, :]
            eigval, eigstat = self.calc_eigvk(kp)
            Elist[i, :] = eigval - self.fermi
            # Elist[i, :] = np.sqrt(np.abs(eigval)) - self.fermi

        # plotting
        num_label = len(label_pos)

        Emax = np.amax(Elist)
        Emin = np.amin(Elist)

        if Elim is None:
            ymax = Emax + (Emax - Emin) * 0.05
            ymin = Emin - (Emax - Emin) * 0.05
        else:
            ymin, ymax = Elim

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(kpos_list, Elist)
        for i in range(num_label):
            ax.plot([label_pos[i], label_pos[i]], [
                    ymin, ymax], linewidth=0.5, color='k')
        ax.plot([kpos_list[0], kpos_list[-1]], [0.0, 0.0],
                linewidth=0.5, color='k', linestyle="--")
        ax.set_xticks(label_pos)
        ax.set_xticklabels(kpath_label, fontsize=18)
        plt.ylabel('Energy', fontsize=18)
        plt.xlim(kpos_list[0], kpos_list[-1])
        plt.ylim(ymin, ymax)
        plt.savefig('bands.pdf', dpi=300)
        # plt.show()

    # ------------------------
    # print information
    # ------------------------

    def print_info(self):
        """ Print information of Wannier TB model. """
        formula = self.structure.formula
        num_wann = self.num_wann
        if_soc = str(self.if_soc)
        num_rpts = self.num_rpts

        h_onsite = None
        for hop in self._hoplist:
            if hop[0] == [0, 0, 0]:
                h_onsite = hop[1]

        print('========================================')
        print('            Wannier TB model            ')
        print('----------------------------------------')
        print('Name of TB model              => {:s}'.format(formula))
        print('Number of Wannier functions   => {:4d}'.format(num_wann))
        print('SOC included                  => {:s}'.format(if_soc))
        print('Number of R-vectors           => {:4d}'.format(num_rpts))
        print('')
        print('On-site term =>')
        if h_onsite is None:
            print(None)
        else:
            for i in range(num_wann):
                print('{:3d} {:7.3f} +{:7.3f}i'.format(
                    i+1, h_onsite[i, i].real, h_onsite[i, i].imag))
        print('========================================')
