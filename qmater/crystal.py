import os
import numpy as np
import spglib as spg
import copy
import itertools


class CrystStruct(object):
    """ Class CrystStruct for a crystal. """

    def __init__(self, lattice=None, atomsites=None):
        self.direct_lattice = lattice
        self.atom_sites = atomsites

    @property
    def direct_lattice(self):
        return self._direct_lattice

    @direct_lattice.setter
    def direct_lattice(self, lattice):
        if lattice is None:
            self._direct_lattice = np.eye(3, dtype=np.float64)
        else:
            self._direct_lattice = np.array(lattice, dtype=np.float64)
        self._reciprocal_lattice = 2.0*np.pi * \
            np.linalg.inv(self._direct_lattice.T)

    @property
    def atom_sites(self):
        return self._atom_sites

    @atom_sites.setter
    def atom_sites(self, atomsites):
        if atomsites is None:
            self._atom_sites = {}
        else:
            self._atom_sites = atomsites

        self._elementlist = [
            (_elem, _pos.shape[0])
            for _elem, _pos in self.atom_sites.items()
        ]

    @property
    def volume(self):
        return abs(np.linalg.det(self._direct_lattice))

    @property
    def reciprocal_lattice(self):
        return self._reciprocal_lattice

    @property
    def elementlist(self):
        return self._elementlist

    @property
    def formula(self):
        return ''.join(
            [a[0] + str(a[1]) for a in self._elementlist])

    @property
    def num_atoms(self):
        return sum([a[1] for a in self._elementlist])

    @property
    def pos_frac(self):
        return np.vstack([a for a in self._atom_sites.values()])

    @property
    def pos_cart(self):
        return np.dot(self.pos_frac, self._direct_lattice)

    @property
    def lattice_constants(self):
        """ Get the lattice constants a, b, c, alpha, beta, gamma. """
        _lattice = self._direct_lattice
        [a, b, c] = [np.linalg.norm(_lattice[i, :]) for i in range(3)]

        adotb = np.dot(_lattice[0], _lattice[1])
        bdotc = np.dot(_lattice[1], _lattice[2])
        cdota = np.dot(_lattice[2], _lattice[0])

        alpha = 180 * np.arccos(bdotc / b / c) / np.pi
        beta = 180 * np.arccos(cdota / a / c) / np.pi
        gamma = 180 * np.arccos(adotb / a / b) / np.pi

        _lattconst = [a, b, c, alpha, beta, gamma]
        return _lattconst

    @property
    def cell(self):
        """ Return cell for spglib. Format refers to "Spglib for Python". """
        numbers = []
        for ind, entry in enumerate(self.elementlist):
            numbers.extend([ind]*entry[1])

        return (self.direct_lattice, self.pos_frac, numbers)

    @property
    def space_group(self):
        """ Get the space group of crystal. """
        return spg.get_spacegroup(self.cell, symprec=1e-3)

    #
    # set model
    #
    @classmethod
    def set_model(cls, lattice, atomsites, frac=True):
        """ set a crystal model from given lattice and atom sites.

        Parameters
        ----------
        lattice : numpy.ndarray
            Direct lattice
        atomsites : dict
            Atom sites.
        frac : bool, optional
            whether positions are fractional or Cartesian. by default True

        Returns
        -------
        CrystStruct
            a CrystStruct instance.

        Raises
        ------
        ValueError
            If `atomsites` is not a dict, raise ValueError.
        """
        if not isinstance(atomsites, dict):
            raise ValueError('atomsites should be a dict!')

        _atomsites = {}
        for _elem, _pos in atomsites.items():
            if frac:
                _atomsites[_elem] = np.mod(_pos, 1.0)
            else:
                _pos_frac = np.dot(_pos, np.linalg.inv(lattice))
                _atomsites[_elem] = np.mod(_pos_frac, 1.0)

        return cls(lattice=lattice, atomsites=_atomsites)

    @classmethod
    def read_poscar(cls, poscar_file):
        """ set a crystal model based on a POSCAR for VASP.

        Parameters
        ----------
        poscar_file : str
            File name of POSCAR.

        Returns
        -------
        CrystStruct
            a CrystStruct instance.

        Raises
        ------
        Exception
            If POSCAR is not found, raise Exception.
        """
        if not os.path.exists(poscar_file):
            raise Exception('Does not find the file {:s}'.format(poscar_file))

        atomsites = {}
        with open(poscar_file, 'r') as fin:
            next(fin)  # skip title line
            scale = float(next(fin))

            lattice = np.genfromtxt(itertools.islice(fin, 3), dtype=np.float64)
            elementlist = fin.readline().split()
            numberlist = [int(a) for a in fin.readline().split()]

            myletter = fin.readline().split()[0][0]
            if myletter in ['S', 's']:
                myletter = fin.readline().split()[0][0]

            for _elem, _num in zip(elementlist, numberlist):
                atomsites[_elem] = np.zeros((_num, 3), dtype=np.float64)
                for i in range(_num):
                    atomsites[_elem][i, :] = np.array(
                        [float(a) for a in fin.readline().split()[:3]])
                # atomsites[_elem] = np.genfromtxt(
                #     itertools.islice(fin, _num), dtype=np.float64)

        if myletter in ['D', 'd']:
            frac = True
        elif myletter in ['C', 'c', 'K', 'k']:
            frac = False
        else:
            frac = True

        return cls.set_model(lattice=lattice, atomsites=atomsites, frac=frac)

    #
    # get information
    #
    def find_neighbors(self, site, func):
        neighbor_list = []
        _pos = self.pos_cart
        _latt = self.direct_lattice
        rc = 6
        r_array = np.mgrid[-rc:rc+1, -rc:rc+1, -rc:rc+1]
        r_array = r_array.reshape(3, (2*rc+1)**3).T

        for _r in r_array:
            _r_cart = np.dot(_r, _latt)
            dr_array = _pos - _pos[site] + _r_cart
            for ind, _dr in enumerate(dr_array):
                if np.linalg.norm(_dr) > 1e-3 and func(_dr):
                    neighbor_list.append([tuple(_r), ind])

        return neighbor_list

    def get_kpoint_cart(self, kpoint):
        """ Get Cartesian coordinate of a given k-point. """
        kp = np.array(kpoint, dtype=np.float64)
        return np.dot(kp, self.reciprocal_lattice)

    def get_kpoint_frac(self, kpoint):
        """ Get fractional coordinate of a given k-point. """
        kp = np.array(kpoint, dtype=np.float64)
        return np.dot(kp, np.linalg.inv(self.reciprocal_lattice))

    def get_kpoint_lines(self, kpath_start, kpath_end, num_kp_kpath=30, frac='N'):
        """ Get k-points of several lines with given start and end k-points.

        Parameters
        ----------
        kpath_start : list or numpy.ndarray
            Starting k-points with fractional coordinates.
        kpath_end : list or numpy.ndarray
            End k-points with fractional coordinates.
        num_kp_kpath : int, optional
            Number of k-points for each k-path. By default 30.
        frac : str, optional
            - 'N' denotes fractional coordinates but Cartesian positions for k-points.
            - 'F' denotes fractional coordinates but fractional positions for k-points.
            - 'C' denotes Cartesian coordinates but Cartesian positions for k-points.
            By default 'N'.

        Returns
        -------
        A tuple of numpy.ndarray
            (kpoints, kpos, klabel_pos)
            - kpoints: coordinates of k-points.
            - kpos: positions of k-points.
            - klabel_pos: positions of k-labels.

        Raises
        ------
        IndexError
            If number of k-points in kpath_start mismatch that in kpath_end, raise IndexError.
        """
        kpath_start = np.array(kpath_start, dtype=np.float64)
        kpath_end = np.array(kpath_end, dtype=np.float64)

        num_kpath = kpath_start.shape[0]
        mynum = len(kpath_end)

        if mynum != num_kpath:
            raise IndexError(
                "The dimension of starting and ending arrays does not match!")

        num_kp = num_kp_kpath * num_kpath

        if frac == 'C':
            kpath_start = self.get_kpoint_cart(kpath_start)
            kpath_end = self.get_kpoint_cart(kpath_end)

        # coordinates of k-points
        kpoints = np.linspace(kpath_start, kpath_end, num=num_kp_kpath, axis=1)
        kpoints = kpoints.reshape(num_kp, 3)

        if frac == 'N':
            kpath_start = self.get_kpoint_cart(kpath_start)
            kpath_end = self.get_kpoint_cart(kpath_end)

        # positions of k-labels
        klabel_pos = np.zeros((num_kpath + 1, ), dtype=np.float64)
        klabel_pos[1:] = np.cumsum(
            np.linalg.norm(kpath_end - kpath_start, axis=1))

        # positions of k-points
        kpos = np.linspace(
            klabel_pos[:-1], klabel_pos[1:], num=num_kp_kpath, axis=1)
        kpos = kpos.reshape(num_kp, 1)

        return kpoints, kpos, klabel_pos

    def get_kpoint_plane(self, kplane_center, kplane_dir1, kplane_dir2, num_kp_dir=10, frac='N'):
        """ Get k-points on a k-plane with given central k-points and two k-directions.

        Parameters
        ----------
        kplane_center : list or numpy.ndarray
            Center of k-plane.
        kplane_dir1 : list or numpy.ndarray
            Direction 1 to define a k-plane.
        kplane_dir2 : list or numpy.ndarray
            Direction 2 to define a k-plane.
        num_kp_dir : int, optional
            Number of k-points along each direction. By default 10.
        frac : str, optional
            - 'N' denotes fractional coordinates but Cartesian positions for k-points.
            - 'F' denotes fractional coordinates but fractional positions for k-points.
            - 'C' denotes Cartesian coordinates but Cartesian positions for k-points.
            By default 'N'.

        Returns
        -------
        A tuple of numpy.ndarray
            (kpoint_array, kpos_array)
            - kpoint_array: coordinates of k-points. Shape is (num_kp_dir, num_kp_dir, 3).
            - kpos_array: positions of k-points. Shape is (num_kp_dir, num_kp_dir, 2).
        """
        kcenter = np.array(kplane_center)
        kdir1 = np.array(kplane_dir1)
        kdir2 = np.array(kplane_dir2)

        if frac == 'C':
            kdir1 = self.get_kpoint_cart(kdir1)
            kdir2 = self.get_kpoint_cart(kdir2)
            kcenter = self.get_kpoint_cart(kcenter)

        korigin = kcenter - kdir1 / 2 - kdir2 / 2
        kpoint1 = np.linspace(korigin, korigin + kdir1, num_kp_dir)
        kpoint2 = np.linspace(korigin, korigin + kdir2, num_kp_dir)

        # coordinates of k-points
        kpoint_array = np.zeros((num_kp_dir, num_kp_dir, 3), dtype=np.float64)
        for ind, kp in enumerate(kpoint1):
            kpoint_array[ind, :, :] = kpoint2 + (kp - korigin)

        if frac == 'N':
            kdir1 = self.get_kpoint_cart(kdir1)
            kdir2 = self.get_kpoint_cart(kdir2)

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

    # TODO: get k-points of a loop defined by func
    def get_kpoint_loop(self, func, num_kp_circle=30, frac='N'):
        pass

    # TODO: get k-points of a closed surface defined by func
    def get_kpoint_sphere(self, func, num_kp_sphere):
        pass

    def get_poscar(self, if_symm=False, to_primitive=True, fname=None):
        """ get POSCAR with or without symmetry considered

        Parameters
        ----------
        if_symm : bool, optional
            whether consider symmetry, by default False
        to_primitive : bool, optional
            whether transform to primitive cell, by default True
        fname : str, optional
            file name of POSCAR, by default None
        """
        elemlist = [entry[0] for entry in self.elementlist]

        if if_symm:
            (_lattice, _positions, _atom_type) = spg.standardize_cell(
                self.cell, to_primitive=to_primitive, no_idealize=False, symprec=1e-3)

            _atomsites_new = {}
            for i in np.unique(_atom_type):
                _elem = elemlist[i]
                _atomsites_new[_elem] = _positions[np.where(_atom_type == i)]
            _elementlist_new = [
                (_elem, _pos.shape[0])
                for _elem, _pos in _atomsites_new.items()
            ]

        else:
            _lattice = self.direct_lattice
            _elementlist_new = self.elementlist
            _atomsites_new = self._atom_sites

        poscar_str = '{:s}\n'.format(self.formula)
        poscar_str += '  {:3.1f}\n'.format(1.0)
        for i in range(3):
            poscar_str += ('{:15.9f}'*3+'\n').format(*_lattice[i][:])
        poscar_str += '  '.join([entry[0] for entry in _elementlist_new])
        poscar_str += '\n'
        poscar_str += '  '.join([str(entry[1]) for entry in _elementlist_new])
        poscar_str += '\n'
        poscar_str += 'Direct\n'
        for entry in _atomsites_new.values():
            for _pos in entry:
                poscar_str += ('{:15.9f}'*3+'\n').format(*_pos)

        if fname is None:
            print(poscar_str)
        else:
            with open(fname, 'w') as fout:
                fout.write(poscar_str)

    #
    # Build Model
    #
    def build_transform(self, rotation=np.eye(3), translation=[0, 0, 0]):
        """ Build a new CrystStruct instance after rotation and translation transformations.

        Parameters
        ----------
        rotation : list or numpy.ndarray, optional
            Rotation matrix. By default numpy.eye(3)
        translation : list or numpy.ndarray, optional
            Translation vector. By default [0, 0, 0]

        Returns
        -------
        CrystStruct
            a new CrystStruct instance
        """
        rot = np.array(rotation)
        trans = np.array(translation)

        lattice_new = np.dot(rot, self.direct_lattice)

        atomsites_new = {}
        for _elem, _pos in self._atom_sites.items():
            atomsites_new[_elem] = np.dot(_pos + trans, np.linalg.inv(rot))

        return CrystStruct().set_model(lattice=lattice_new, atomsites=atomsites_new)

    # TODO: Add index list
    def build_supercell(self, size=[(0, 0), (0, 0), (0, 0)]):
        """ Build a new CrystStruct instance for a supercell.

        Parameters
        ----------
        size : list, optional
            Size of supercell. By default [(0, 0), (0, 0), (0, 0)]

        Returns
        -------
        CrystStruct
            A new CrystStruct instance.

        Raises
        ------
        ValueError
            If size does not contains three tuples, raise ValueError.
        """
        if len(size) != 3:
            raise ValueError('should set three tuples for size!')
        nx, ny, nz = size

        supercell_frac = np.array([
            [i, j, k]
            for i in range(nx[0], nx[1] + 1)
            for j in range(ny[0], ny[1] + 1)
            for k in range(nz[0], nz[1] + 1)
        ], dtype=np.int16)
        supercell_cart = np.dot(supercell_frac, self.direct_lattice)

        num_cell = supercell_frac.shape[0]

        lattice_new = np.dot(
            np.diag([a[1] - a[0] + 1 for a in size]), self.direct_lattice)

        atomsites_new = {}
        for _elem, _pos in self.atom_sites.items():
            _natoms = _pos.shape[0]
            _pos_cart = np.dot(_pos, self.direct_lattice)

            nc = 0
            _pos_cart_new = np.zeros((num_cell * _natoms, 3), dtype=np.float64)
            for shift_cart in supercell_cart:
                _pos_cart_new[nc:nc + _natoms, :] = _pos_cart + shift_cart
                nc += _natoms

            atomsites_new[_elem] = _pos_cart_new

        return CrystStruct().set_model(lattice_new, atomsites_new, frac=False)

    def build_vacuumlayer(self, length=10.0, direction='c'):
        """ Build a new CrystStruct instance after adding a vacuum layer.

        Parameters
        ----------
        length : float, optional
            Thickness of vacuum layer. By default 10.0
        direction : str, optional
            Direction of building vacuum layer. Can be 'a', 'b', or 'c'. By default 'c'

        Returns
        -------
        CrystStruct
            a new CrystStruct instance
        """
        _lattconst = self.lattice_constants
        _lattice = self.direct_lattice

        _dir = {'a': 0, 'b': 1, 'c': 2}[direction]
        _lattice_shift = length * _lattice[_dir, :] / _lattconst[_dir]

        lattice_new = np.zeros((3, 3), dtype=np.float64)
        lattice_new[_dir, :] = _lattice_shift
        lattice_new += _lattice

        atomsites_new = {}
        for _elem, _pos in self.atom_sites.items():
            _pos_cart = np.dot(_pos, _lattice)
            atomsites_new[_elem] = _pos_cart + _lattice_shift/2.0

        return CrystStruct.set_model(lattice_new, atomsites_new, frac=False)

    def build_heterostructure(self, struct, lattice_inplane, direction='c'):
        """ Build a new CrystStruct instance for a heterostructure.

        Parameters
        ----------
        struct : CrystStruct
            Upper layer
        lattice_inplane : numpy.ndarray
            In-plane lattice for heterostructure.
        direction : str, optional
            Direction of building heterostructure. Can be 'a', 'b', or 'c'. By default 'c'

        Returns
        -------
        CrystStruct
            A new CrystStruct instance.
        """
        _nz = {'a': 0, 'b': 1, 'c': 2}[direction]
        _nxy = {'a': (1, 2), 'b': (0, 2), 'c': (0, 1)}[direction]
        _lattice1 = copy.deepcopy(self.direct_lattice)
        _lattice2 = copy.deepcopy(struct.direct_lattice)

        _len1 = np.linalg.norm(_lattice1[_nz, :])
        _len2 = np.linalg.norm(_lattice2[_nz, :])

        _lattice1[_nxy, :] = lattice_inplane
        _lattice2[_nxy, :] = lattice_inplane
        _lattice2[_nz, :] = _len2 * _lattice1[_nz, :] / _len1

        lattice_new = np.zeros((3, 3), dtype=np.float64)
        lattice_new[_nxy, :] = lattice_inplane
        lattice_new[_nz, :] = _lattice1[_nz, :] + _lattice2[_nz, :]

        _atomsites1 = self.atom_sites
        _atomsites2 = struct.atom_sites
        atomsites_new = {}

        for _elem, _pos in _atomsites1.items():
            _pos_cart = np.dot(_pos, _lattice1)
            atomsites_new[_elem] = _pos_cart

        for _elem, _pos in _atomsites2.items():
            _pos_cart = np.dot(_pos, _lattice2) + _lattice1[_nz, :]
            if _elem in atomsites_new.keys():
                atomsites_new[_elem] = np.vstack(
                    (atomsites_new[_elem], _pos_cart))
            else:
                atomsites_new[_elem] = _pos_cart

        return CrystStruct.set_model(lattice_new, atomsites_new, frac=False)

    #
    # print information
    #
    def print_highkpath(self):
        """ Print high-symmetry k-points and recommended k-paths. """
        try:
            import seekpath as skpath
            kinfo = skpath.get_path(self.cell, True, 'hpkot')
            khigh = kinfo['point_coords']
            kpath = kinfo['path']

            print('High-symmetry k-points =>')
            for a in khigh:
                print(('{:8s}->'+'{:11.6f}'*3).format(a, *khigh[a][:]))
            print('')

            print('High-symmetry k-paths =>')
            print('Direct')
            for a0 in kpath:
                for a1 in a0:
                    print(('{:11.6f}'*3 + '  ! {:6s}').format(
                        *khigh[a1][:], a1))
                print('')
        except ImportError:
            print('Seekpath is required.')

    def print_info(self):
        """ Print the basic information of crystal. """
        print('Formula => {:s}'.format(self.formula))

        sg = self.space_group
        if sg:
            print('     SG => {:s}'.format(sg))
            print('')

        _lattconst = self.lattice_constants
        print('      a => {:7.3f} angstrom'.format(_lattconst[0]))
        print('      b => {:7.3f} angstrom'.format(_lattconst[1]))
        print('      c => {:7.3f} angstrom'.format(_lattconst[2]))
        print('  alpha => {:7.3f} degree'.format(_lattconst[3]))
        print('   beta => {:7.3f} degree'.format(_lattconst[4]))
        print('  gamma => {:7.3f} degree'.format(_lattconst[5]))
        print('')

        print('Direct lattice vectors =>')
        _lattice = self.direct_lattice
        for i in range(3):
            print(('{:16.9f}'*3).format(*_lattice[i, :]))
        print('')

        print('Reciprocal lattice vectors =>')
        _reclatt = self.reciprocal_lattice
        for i in range(3):
            print(('{:16.9f}'*3).format(*_reclatt[i, :]))
        print('')

        print('Number of atoms => {:d}'.format(self.num_atoms))
        print('Direct')
        for _elem, _poslist in self._atom_sites.items():
            for ind, _pos in enumerate(_poslist):
                print(('{:6s}'+'{:16.9f}'*3).format(
                    _elem + str(ind), *_pos[:]))
