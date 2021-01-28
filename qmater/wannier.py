import os
import numpy as np
import matplotlib.pyplot as plt

from qmater.crystal import CrystStruct


class WannierTB(object):
    """ Class WannierTB for a Wannier tight-binding (TB) model

    Attributes:
        structure: A CrystStruct instance representing the crystal lattice.
        if_soc: A boolean indicating whether spin-orbit coupling (SOC) is included or not.
        projectors: A dict consisting of orbitals for each element.
                - e.g. {'Na': ['s'], 'Bi': ['s', 'p']}
        num_wann: An integer indicating the number of Wannier functions.
        wannier_centers: A list consisting the fractional and Cartesian coordinates of Wannier centers.

        _num_rpts: An integer indicating the number of R-vectors.
        _hoplist: A list of lists which include R-vectors, hopping matrix, and degeneracy.
                - [R-vectors, hopping matrix, deg]

    Methods:
        set_hop: set a new hopping term.
        add_hop: add a hopping term to a current term.
        read_wannier90_hr: set _hoplist from Wannier90 Hr file.
        write_wannier90_hr: write Wannier90 Hr file.

        _set_projectors: set projectors from a given simple projector dict.
        _set_num_wannier: set num_wann.
        _set_wannier_centers: set wannier_centers.
        _set_hoplist: set or add hopping term in _hoplist.

        _calc_hamk: calculate Hamiltonian on a k-point (fractional coordinate).
        _calc_dhamdk: calculate the derivative of Hamiltonian in k at a k-point (fractional coordinate).
        _calc_berryphase_1d: calculate Berry phase along a closed k-path.
        _calc_berrycurvature_singlek: calculate Berry curvature for a single k-point.

        calc_eigvk: calculate eigenvalues and eigenstates of a k-point (fractional coordinate).
        calc_wilsonloop: calculate Wilson loops or hybrid Wannier charge centers (WCCs) on a k-plane.
        calc_berrycurvature: calculate Berry curvature for a k-plane.
        calc_quantum_metric: calculate quantum metric for a k-plane.

        plot_band_structure: plot band structure for a set of given k-paths.
        print_info: print information of Wannier TB model.
    """

    def __init__(self, structure=None,
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
        # print(type(structure))
        if not isinstance(structure, CrystStruct):
            raise ValueError('structure should be a CrystStruct!')
        self._structure = structure

    @property
    def projectors(self):
        return self._projectors

    @projectors.setter
    def projectors(self, projectors):
        """ Set projectors from a given simple projector dict.
            - 'p' -> 'pz', 'px', 'py'
            - 'd' -> 'dz2', 'dxz', 'dyz', 'dx2-y2', 'dxy'
        """
        if not isinstance(projectors, dict):
            raise ValueError('projectors should be a dict!')

        orbitals = {
            's': ['s'],
            'p': ['pz', 'px', 'py'],
            'd': ['dz2', 'dxz', 'dyz', 'dx2-y2', 'dxy']
        }

        self._projectors = {}
        for _elem, _orb in projectors.items():
            _orb_temp = [orbitals[a] if a in orbitals else [a] for a in _orb]
            self._projectors[_elem] = [
                a for _entry in _orb_temp for a in _entry]

    @property
    def if_soc(self):
        return self._if_soc

    @if_soc.setter
    def if_soc(self, if_soc):
        if not isinstance(if_soc, bool):
            raise ValueError('if_soc should be True or False!')
        self._if_soc = if_soc

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
        atomlist = self.structure._atomlist
        myprojs = self.projectors
        if_soc = self.if_soc

        num_wann = sum([len(myprojs[_elem]) * _natoms
                        for (_elem, _natoms) in elementlist])

        if if_soc:
            self._num_wann = num_wann * 2
        else:
            self._num_wann = num_wann

        _wcc_frac = np.zeros([self.num_wann, 3], dtype=np.float64)
        _wcc_cart = np.zeros([self.num_wann, 3], dtype=np.float64)

        nc = 0
        for _elem, _pos_frac, _pos_cart in atomlist:
            _wcc = np.kron(_pos_frac, np.ones((len(myprojs[_elem]), 1)))
            _wcc_frac[nc:nc + _wcc.shape[0]] = _wcc
            _wcc = np.kron(_pos_cart, np.ones((len(myprojs[_elem]), 1)))
            _wcc_cart[nc:nc + _wcc.shape[0]] = _wcc
            nc += _wcc.shape[0]

        if if_soc:
            nc = self.num_wann // 2
            _wcc_frac[nc:, :] = _wcc_frac[:nc, :]
            _wcc_cart[nc:, :] = _wcc_cart[:nc, :]

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

            for i in range(num_rpts):
                if num_rpts > 10 and i % int(num_rpts / 10) == 0:
                    print('R-vector: {:d}'.format(i))

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

    # ----------------
    #  write model
    # ----------------
    def write_wannier90_hr(self, filename='wannier90_hr.dat'):
        """ write Wannier90 Hr file

        Args:
            filename (str, optional): Wannier90 Hr file. Defaults to 'wannier90_hr.dat'.

        """

        num_wann = self.num_wann
        num_rpts = self.num_rpts

        Rvec = [a[0] for a in self._hoplist]
        HmnR = [a[1] for a in self._hoplist]
        deg_rpts = [a[2] for a in self._hoplist]

        print('========================================')
        print('          Writing Hrfile ...            ')
        print('----------------------------------------')
        with open(filename, 'w') as outfile:
            outfile.write('written by QMater\n')
            outfile.write('{:12d}\n'.format(num_wann))
            outfile.write('{:12d}\n'.format(num_rpts))
            nc = 0
            for i in range(num_rpts):
                outfile.write('{:5d}'.format(deg_rpts[i]))
                nc += 1
                if nc % 15 == 0:
                    outfile.write('\n')
            outfile.write('\n')

            for i in range(num_rpts):
                if num_rpts > 10 and i % int(num_rpts / 10) == 0:
                    print('R-vector: {:d}'.format(i))

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
            Hamk += deg_rpts * np.exp(pi2j * kdotr) * HmnR[:, :]

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
            berryphase_eigval = np.sort(berryphase_eigval)
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
        int_kp = np.linspace(origin, int_dir, num_kp)
        # k-points which are variables
        var_kp = np.linspace(origin, var_dir, num_kp)

        wcc = np.zeros((num_kp, num_occupy), dtype=np.float64)
        for ind, kp in enumerate(var_kp):
            kp_list = int_kp + (kp - origin)

            # get Wannier centers of each occupied state
            wcc[ind, :] = self._calc_berryphase_1d(
                num_occupy, kp_list, if_eigval=True)

        return wcc

    def _calc_berrycurvature_singlek(self, kp, num_occupy):
        r""" calculate Berry curvature for a single k-point
                - Refer to `Phys. Rev. B 74, 195118 (2006).`

        Args:
            kp (list or numpy.array): k-point
            num_occupy (int): number of occupied states

        Returns:
            [tuple]: Berry curvature, i.e. (Omegaxy, Omegayz, Omegazx).
                - Omegaxy (complex): \Omega_{xy}
                - Omegayz (complex): \Omega_{yz}
                - Omegazx (complex): \Omega_{zx}
        """

        num_wann = self.num_wann

        _dhdk = self._calc_dhamdk(kp)
        _eigv, _eigs = self.calc_eigvk(kp)

        # matrix rep of velocity operator
        vx = np.dot(np.conj(_eigs.T), np.dot(_dhdk[:, :, 0], _eigs))
        vy = np.dot(np.conj(_eigs.T), np.dot(_dhdk[:, :, 1], _eigs))
        vz = np.dot(np.conj(_eigs.T), np.dot(_dhdk[:, :, 2], _eigs))

        omegaxy = np.complex128(0.0)
        omegayz = np.complex128(0.0)
        omegazx = np.complex128(0.0)

        for m in range(num_occupy):
            for n in range(num_occupy, num_wann):
                omegaxy += vx[m, n] * vy[n, m] / ((_eigv[n] - _eigv[m]) ** 2)
                omegayz += vy[m, n] * vz[n, m] / ((_eigv[n] - _eigv[m]) ** 2)
                omegazx += vz[m, n] * vx[n, m] / ((_eigv[n] - _eigv[m]) ** 2)

        omegaxy = -2.0 * omegaxy.imag
        omegayz = -2.0 * omegayz.imag
        omegazx = -2.0 * omegazx.imag

        return omegaxy, omegayz, omegazx

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

    def calc_berrycurvature(self, num_kp,
                            center=[0, 0, 0],
                            dir1=[1, 0, 0],
                            dir2=[0, 1, 0], num_occupy=1):
        r""" calculate Berry curvature for a k-plane

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

        omegaxy = np.zeros((num_kp, num_kp), dtype=np.float64)
        omegayz = np.zeros((num_kp, num_kp), dtype=np.float64)
        omegazx = np.zeros((num_kp, num_kp), dtype=np.float64)

        for i in range(num_kp):
            for j in range(num_kp):
                _xy, _yz, _zx = self._calc_berrycurvature_singlek(
                    kp_array[i, j, :], num_occupy)
                omegaxy[i, j] = _xy
                omegayz[i, j] = _yz
                omegazx[i, j] = _zx

        return kpos_array, omegaxy, omegayz, omegazx

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