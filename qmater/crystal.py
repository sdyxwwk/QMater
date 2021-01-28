import os
import numpy as np
import spglib as spg
import copy


class CrystStruct(object):
    """ Class CrystStruct for a crystal

    Attributes:
        direct_lattice: A shape-(3,3) np.array representing the Bravais lattice.
        reciprocal_lattice: A shape-(3,3) np.array representing the reciprocal lattice.
        volume: A float number representing the volume of Bravais lattice.
        elementlist: A list of lists which include the atomic element and the number of atoms.
                - e.g: [['Na', 6], ['Bi', 2]]
        formula: A string indicating the formula of crystal.
        num_atoms: An integer indicating the total number of atoms.
        pos_frac: A shape-(num_atoms,3) np.array representing the fractional position of atoms.
        pos_cart: A shape-(num_atoms,3) np.array representing the Cartesian position of atoms.
        lattice_constants: A list containing lattice constants a, b, c, alpha, beta, gamma.
        cell: A tuple of cell for spglib.
        space_group: A string indicating the space group of crystal

        _atomlist: A list of tuples which include the atomic element and its atomic positions.

    Methods:
        set_model: set a crystal model from given lattice, elementlist, and positions.
        read_poscar: set a crystal model from a POSCAR.

        get_kpoint_cart: get Cartesian coordinate of a given k-point.
        get_kpoint_frac: get fractional coordinate of a given k-point.
        get_kpoint_lines: get k-points of several lines with given start and end k-points.
        get_kpoint_plane: get k-points on a k-plane with given central k-points and two k-directions.

        build_transform: build a new CrystStruct instance after rotation and translation transformations.
        build_supercell: build a new CrystStruct instance for a supercell.
        build_vacuumlayer: build a new CrystStruct instance after adding a vacuum layer.
        build_heterostructure: build a new CrystStruct instance for a heterostructure.

        print_poscar: print POSCAR with or without symmetry considered.
        print_highkpath: print high-symmetry k-points and recommended k-paths.
        print_info: print the basic information of crystal.

        _get_atomlist: get _atomlist.
    """

    def __init__(self):
        self.direct_lattice = np.eye(3, dtype=np.float64)
        self._atomlist = []

    #
    #  Attributes
    #
    @property
    def direct_lattice(self):
        return self._direct_lattice

    @direct_lattice.setter
    def direct_lattice(self, lattice):
        self._direct_lattice = np.array(lattice, dtype=np.float64)
        self._reciprocal_lattice = 2.0*np.pi*np.linalg.inv(
            np.transpose(self._direct_lattice))

    @property
    def volume(self):
        return abs(np.linalg.det(self._direct_lattice))

    @property
    def reciprocal_lattice(self):
        return self._reciprocal_lattice

    @property
    def elementlist(self):
        return [[a[0], a[1].shape[0]] for a in self._atomlist]

    @property
    def formula(self):
        return ''.join(
            [a[0] + str(a[1].shape[0]) for a in self._atomlist])

    @property
    def num_atoms(self):
        return sum([a[1].shape[0] for a in self._atomlist])

    @property
    def pos_frac(self):
        return np.vstack([a[1] for a in self._atomlist])

    @property
    def pos_cart(self):
        return np.vstack([a[2] for a in self._atomlist])

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
        numbers = [
            ind for ind, entry in enumerate(self._atomlist)
            for i in range(entry[1].shape[0])
        ]
        return (self.direct_lattice, self.pos_frac, numbers)

    @property
    def space_group(self):
        """ Get the space group of crystal. """
        return spg.get_spacegroup(self.cell, symprec=1e-1)

    #
    #  Internal Methods
    #
    def _set_atomlist(self, elementlist, positions, frac):
        """ Set self._atomlist and self._position. """
        _dir_latt = self._direct_lattice
        _pos = np.array(positions, dtype=np.float64)

        if frac:
            _pos_frac = np.mod(_pos, 1.0)
            _pos_cart = np.dot(_pos, _dir_latt)
        else:
            _pos_frac = np.dot(_pos, np.linalg.inv(_dir_latt))
            _pos_frac = np.mod(_pos_frac, 1.0)
            _pos_cart = np.dot(_pos_frac, _dir_latt)

        elemlist = [entry[0] for entry in elementlist]
        numlist = np.array([entry[1] for entry in elementlist])

        if sum(numlist) != _pos_frac.shape[0]:
            raise ValueError(
                'Number of atoms in elementlist does not match positions!')

        _start = np.cumsum(numlist) - numlist
        _end = np.cumsum(numlist)

        self._atomlist = [(elem, _pos_frac[n1:n2, :], _pos_cart[n1:n2, :])
                          for elem, n1, n2 in zip(elemlist, _start, _end)]

    #
    # set model
    #
    def set_model(self, lattice, elementlist, positions, frac=True):
        """ set a crystal model from given lattice, elementlist, and positions.

        Args:
            lattice (list): direct_lattice
            elementlist (list): elementlist
            positions (list): pos_frac or pos_cart depending on frac
            frac (bool, optional): whether positions are fractional or Cartesian. Defaults to True.

        Raises:
            ValueError: If number of atoms in pos_frac does not equal num_atoms, raise ValueError.
        """
        self.direct_lattice = lattice
        self._set_atomlist(elementlist, positions, frac)

    def read_poscar(self, poscar_file):
        """ set a crystal model based on a POSCAR for VASP

        Args:
            poscar_file ([str]): POSCAR

        Raises:
            Exception: If POSCAR is not found, raise Exception
        """
        if not os.path.exists(poscar_file):
            raise Exception('Does not find the file {:s}'.format(poscar_file))

        with open(poscar_file, 'r') as infile:
            infile.readline()

            myline = infile.readline().split()
            scale = float(myline[0])

            lattice = np.zeros((3, 3), dtype=np.float64)
            for i in range(3):
                myline = infile.readline().split()
                lattice[i, :] = [scale * float(a) for a in myline[0:3]]

            myline0 = infile.readline().split()
            myline1 = infile.readline().split()
            elementlist = [[_elem, int(_num)]
                           for _elem, _num in zip(myline0, myline1)]
            _num_atoms = sum([entry[1] for entry in elementlist])

            myletter = infile.readline().split()[0][0]
            if myletter in ['S', 's']:
                myletter = infile.readline().split()[0][0]

            positions = np.zeros((_num_atoms, 3), dtype=np.float64)
            for i in range(_num_atoms):
                myline = infile.readline().split()
                positions[i, :] = [float(a) for a in myline[0:3]]

        if myletter in ['D', 'd']:
            frac = True
        elif myletter in ['C', 'c', 'K', 'k']:
            frac = False
        else:
            frac = True

        self.direct_lattice = lattice
        self._set_atomlist(elementlist, positions, frac)

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
        """ get k-points of several lines with given start and end k-points

        Args:
            kpath_start (list): starting k-points with fractional coordinates.
            kpath_end (list): end k-points with fractional coordinates.
            num_kp_kpath (int, optional): number of k-points for each k-path. Defaults to 30.
            frac (str, optional): Defaults to 'N'.
                    - 'N' denotes fractional coordinates but Cartesian positions for k-points.
                    - 'F' denotes fractional coordinates but fractional positions for k-points.
                    - 'C' denotes Cartesian coordinates but Cartesian positions for k-points.

        Raises:
            IndexError: If number of k-points in kpath_start mismatch that in kpath_end, raise IndexError.

        Returns:
            [tuple]: (kpoints, kpos, klabel_pos)
                    - kpoints (np.array): coordinates of k-points
                    - kpos (np.array): positions of k-points
                    - klabel_pos (np.array): positions of k-labels
        """
        _rec_latt = self._reciprocal_lattice
        kpath_start = np.array(kpath_start, dtype=np.float64)
        kpath_end = np.array(kpath_end, dtype=np.float64)

        num_kpath = kpath_start.shape[0]
        mynum = len(kpath_end)

        if mynum != num_kpath:
            raise IndexError(
                "The dimension of starting and ending arrays does not match!")

        num_kp = num_kp_kpath * num_kpath

        if frac == 'C':
            kpath_start = np.dot(kpath_start, _rec_latt)
            kpath_end = np.dot(kpath_end, _rec_latt)

        # coordinates of k-points
        kpoints = np.linspace(kpath_start, kpath_end, num=num_kp_kpath, axis=1)
        kpoints = kpoints.reshape(num_kp, 3)

        if frac == 'N':
            kpath_start = np.dot(kpath_start, _rec_latt)
            kpath_end = np.dot(kpath_end, _rec_latt)

        # positions of k-labels
        klabel_pos = np.zeros((num_kpath + 1, ), dtype=np.float64)
        klabel_pos[1:] = np.cumsum(
            np.linalg.norm(kpath_end - kpath_start, axis=1))

        # positions of k-points
        kpos = np.linspace(
            klabel_pos[:-1], klabel_pos[1:], num=num_kp_kpath, axis=1)
        kpos = kpos.reshape(num_kp, 1)

        return kpoints, kpos, klabel_pos

    # TODO: get k-points of a circle with given central k-points and radius of circle
    def get_kpoint_circle(self, kcircle_center, kcircle_radius, kcircle_normal, num_kp_circle=30, frac='N'):
        pass

    def get_kpoint_plane(self, kplane_center, kplane_dir1, kplane_dir2, num_kp_dir=10, frac='N'):
        """ get k-points on a k-plane with given central k-points and two k-directions

        Args:
            kplane_center (list): center of k-plane
            kplane_dir1 (list): direction 1 to define a k-plane
            kplane_dir2 (list): direction 2 to define a k-plane
            num_kp_dir (int, optional): number of k-points along each direction. Defaults to 10.
            frac (str, optional): Defaults to 'N'.
                    - 'N' denotes fractional coordinates but Cartesian positions for k-points.
                    - 'F' denotes fractional coordinates but fractional positions for k-points.
                    - 'C' denotes Cartesian coordinates but Cartesian positions for k-points.

        Returns:
            [tuple]: (kpoint_array, kpos_array)
                    - kpoint_array (np.array): coordinates of k-points. Shape is (num_kp_dir,num_kp_dir,3).
                    - kpos_array (np.array): positions of k-points. Shape is (num_kp_dir,num_kp_dir,2).

        """

        kcenter = np.array(kplane_center)
        kdir1 = np.array(kplane_dir1)
        kdir2 = np.array(kplane_dir2)

        if frac == 'C':
            kdir1 = np.dot(kdir1, self._reciprocal_lattice)
            kdir2 = np.dot(kdir2, self._reciprocal_lattice)
            kcenter = np.dot(kcenter, self._reciprocal_lattice)

        korigin = np.array(kplane_center) - kdir1 / 2 - kdir2 / 2
        kpoint1 = np.linspace(korigin, korigin + kdir1, num_kp_dir)
        kpoint2 = np.linspace(korigin, korigin + kdir2, num_kp_dir)

        # coordinates of k-points
        kpoint_array = np.zeros((num_kp_dir, num_kp_dir, 3), dtype=np.float64)
        for ind, kp in enumerate(kpoint1):
            kpoint_array[ind, :, :] = kpoint2 + (kp - korigin)

        if frac == 'N':
            kdir1 = np.dot(kdir1, self._reciprocal_lattice)
            kdir2 = np.dot(kdir2, self._reciprocal_lattice)

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

    # TODO: get k-points of a sphere with given central k-points and radius of sphere
    def get_kpoint_sphere(self, ksphere_center, ksphere_radius, num_kp_sphere):
        pass

    #
    # Build Model
    #
    def build_transform(self,
                        rotation=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                        translation=[0, 0, 0]):
        """ build a new CrystStruct instance after rotation and translation transformations.

        Args:
            rotation (list, optional): rotation matrix. Defaults to [[1, 0, 0], [0, 1, 0], [0, 0, 1]].
            translation (list, optional): translation vector. Defaults to [0, 0, 0].

        Returns:
            [CrystStruct]: a new CrystStruct instance
        """
        rot = np.array(rotation)
        trans = np.array(translation)

        lattice_new = np.dot(rot, self.direct_lattice)

        elementlist_new = self.elementlist
        positions_new = np.dot(self.pos_frac + trans, np.linalg.inv(rot))

        struct = CrystStruct()
        struct.set_model(lattice_new,
                         elementlist_new,
                         positions_new,
                         frac=True)
        return struct

    # TODO: Add index list
    def build_supercell(self, size=[(0, 0), (0, 0), (0, 0)]):
        """ build a new CrystStruct instance for a supercell

        Args:
            size (list, optional): size of supercell. Defaults to [(0, 0), (0, 0), (0, 0)].
                    - It consists of three tuples, (xmin, xmax), (ymin, ymax), and (zmin, zmax).
                    - [(0, 0), (0, 0), (0, 0)] indicates the original unit cell (0, 0, 0).

        Raises:
            ValueError: If size does not contains three tuples, raise ValueError.

        Returns:
            [CrystStruct]: a new CrystStruct instance
        """
        if len(size) != 3:
            raise ValueError('should set three tuples for size!')
        nx, ny, nz = size

        _lattice = self.direct_lattice
        _num_atoms = self.num_atoms
        _elementlist = self.elementlist

        supercell_frac = np.array([[i, j, k] for i in range(nx[0], nx[1] + 1)
                                   for j in range(ny[0], ny[1] + 1)
                                   for k in range(nz[0], nz[1] + 1)],
                                  dtype=np.int16)
        supercell_cart = np.dot(supercell_frac, _lattice)

        num_cell = supercell_frac.shape[0]

        lattice_new = np.dot(np.diag([a[1] - a[0] + 1 for a in size]),
                             _lattice)
        elementlist_new = [(entry[0], entry[1] * num_cell)
                           for entry in _elementlist]

        positions_new = np.zeros((num_cell * _num_atoms, 3), dtype=np.float64)
        index_list = []

        nc = 0
        for entry in self._atomlist:
            _elem = entry[0]
            _pos0 = entry[2]
            _natoms = _pos0.shape[0]

            nd = 0
            index_array = np.zeros((num_cell * _natoms, 4), dtype=np.int16)
            for shift_frac, shift_cart in zip(supercell_frac, supercell_cart):
                positions_new[nc:nc + _natoms, :] = _pos0 + shift_cart
                index_array[nd:nd + _natoms, 0] = np.array(range(_natoms))
                index_array[nd:nd + _natoms, 1:4] = shift_frac
                nc += _natoms
                nd += _natoms
            index_list.append(
                [_elem, index_array, positions_new[nc - nd:nc, :]])

        struct = CrystStruct()
        struct.set_model(lattice_new,
                         elementlist_new,
                         positions_new,
                         frac=False)
        return struct, index_list  # (index_array, positions_new)

    def build_vacuumlayer(self, length=10.0, direction='c'):
        """ build a new CrystStruct instance after adding a vacuum layer

        Args:
            length (float, optional): thickness of vacuum layer. Defaults to 10.0.
            direction (str, optional): direction of building vacuum layer.
                                       Can be 'a', 'b', or 'c'. Defaults to 'c'.

        Returns:
            [CrystStruct]: a new CrystStruct instance
        """
        _lattconst = self.lattice_constants
        _lattice = self.direct_lattice
        _elementlist = self.elementlist
        _pos = self.pos_cart

        _dir = {'a': 0, 'b': 1, 'c': 2}[direction]
        _lattice_shift = length * _lattice[_dir, :] / _lattconst[_dir]

        lattice_new = np.zeros((3, 3), dtype=np.float64)
        lattice_new[_dir, :] = _lattice_shift
        lattice_new += _lattice

        elementlist_new = _elementlist
        positions_new = _pos + _lattice_shift / 2.0

        struct = CrystStruct()
        struct.set_model(lattice_new,
                         elementlist_new,
                         positions_new,
                         frac=False)
        return struct

    def build_heterostructure(self, struct, lattice_inplane, direction='c'):
        """ build a new CrystStruct instance for a heterostructure.

        Args:
            struct (CrystStruct): upper layer
            lattice_inplane (np.array): in-plane lattice for heterostructure
            direction (str, optional): direction of building heterostructure.
                                       Can be 'a', 'b', or 'c'. Defaults to 'c'.

        Returns:
            [CrystStruct]: a new CrystStruct instance
        """
        _lattice = self.direct_lattice

        _nz = {'a': 0, 'b': 1, 'c': 2}[direction]
        _nxy = {'a': (1, 2), 'b': (0, 2), 'c': (0, 1)}[direction]
        _lattice1 = copy.deepcopy(_lattice)
        _lattice2 = copy.deepcopy(struct.direct_lattice)

        _len1 = np.linalg.norm(_lattice1[_nz, :])
        _len2 = np.linalg.norm(_lattice2[_nz, :])

        _lattice1[_nxy, :] = lattice_inplane
        _lattice2[_nxy, :] = lattice_inplane
        _lattice2[_nz, :] = _len2 * _lattice1[_nz, :] / _len1

        lattice_new = np.zeros((3, 3), dtype=np.float64)
        lattice_new[_nxy, :] = lattice_inplane
        lattice_new[_nz, :] = _lattice1[_nz, :] + _lattice2[_nz, :]

        _atomlist1 = self._atomlist
        _atomlist2 = struct._atomlist

        _atomlist_new = []
        _elements = []
        for entry in _atomlist1:
            _elem = entry[0]
            _pos_frac = entry[1]
            _pos_cart = np.dot(_pos_frac, _lattice1)
            _atomlist_new.append([_elem, _pos_cart])
            _elements.append(_elem)

        for entry in _atomlist2:
            _elem = entry[0]
            _pos_frac = entry[1]
            _pos_cart = np.dot(_pos_frac, _lattice2)
            _pos_cart += _lattice1[_nz, :]
            if _elem in _elements:
                ind = _elements.index(_elem)
                _temp = np.vstack((_atomlist_new[ind][1], _pos_cart))
                _atomlist_new[ind][1] = _temp
            else:
                _atomlist_new.append([_elem, _pos_cart])
                _elements.append(_elem)

        elementlist_new = [[_elem, _pos.shape[0]]
                           for _elem, _pos in _atomlist_new]
        positions_new = np.vstack([entry[1] for entry in _atomlist_new])

        struct = CrystStruct()
        struct.set_model(lattice_new,
                         elementlist_new,
                         positions_new,
                         frac=False)
        return struct

    #
    # print information
    #
    def print_poscar(self, if_symm=False, to_primitive=True):
        """ print POSCAR with or without symmetry considered

        Args:
            if_symm (bool, optional): whether consider symmetry. Defaults to False.
            to_primitive (bool, optional): whether transform to the primitive cell. Defaults to True.
        """

        elemlist = [entry[0] for entry in self.elementlist]

        if if_symm:
            (_lattice, _positions,
             _atom_type) = spg.standardize_cell(self.cell,
                                                to_primitive=to_primitive,
                                                no_idealize=False,
                                                symprec=1e-2)

            _atomlist_new = [[
                elem, np.array([], dtype=np.float64).reshape(0, 3)
            ] for elem in elemlist]
            for pos, ind in zip(_positions, _atom_type):
                _atomlist_new[ind][1] = np.vstack([_atomlist_new[ind][1], pos])

            _elementlist_new = [[entry[0], entry[1].shape[0]]
                                for entry in _atomlist_new]

        else:
            _lattice = self.direct_lattice
            _elementlist_new = self.elementlist
            _atomlist_new = self._atomlist

        print('{:s}'.format(self.formula))
        print('  {:3.1f}'.format(1.0))
        for i in range(3):
            print('{:15.9f}{:15.9f}{:15.9f}'.format(_lattice[i][0],
                                                    _lattice[i][1],
                                                    _lattice[i][2]))
        print('  '.join([entry[0] for entry in _elementlist_new]))
        print('  '.join([str(entry[1]) for entry in _elementlist_new]))
        print('Direct')
        for entry in _atomlist_new:
            for _pos in entry[1]:
                print('{:15.9f}{:15.9f}{:15.9f}'.format(
                    _pos[0], _pos[1], _pos[2]))

    def print_highkpath(self):
        """ Print high-symmetry k-points and recommended k-paths. """
        try:
            import seekpath as skpath
            kinfo = skpath.get_path(self.cell, True, 'hpkot')
            khigh = kinfo['point_coords']
            kpath = kinfo['path']

            print('High-symmetry k-points =>')
            for a in khigh:
                print('{:8s}->{:11.6f}{:11.6f}{:11.6f}'.format(
                    a, khigh[a][0], khigh[a][1], khigh[a][2]))
            print('')

            print('High-symmetry k-paths =>')
            print('Direct')
            for a0 in kpath:
                for a1 in a0:
                    print('{:11.6f}{:11.6f}{:11.6f}  ! {:6s}'.format(
                        khigh[a1][0], khigh[a1][1], khigh[a1][2], a1))
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
            print('{:16.9f}{:16.9f}{:16.9f}'.format(_lattice[i, 0],
                                                    _lattice[i, 1],
                                                    _lattice[i, 2]))
        print('')

        print('Reciprocal lattice vectors =>')
        _reclatt = self.reciprocal_lattice
        for i in range(3):
            print(' {:16.9f}{:16.9f}{:16.9f}'.format(_reclatt[i, 0],
                                                     _reclatt[i, 1],
                                                     _reclatt[i, 2]))
        print('')

        print('Number of atoms => {:d}'.format(self.num_atoms))
        print('Direct')
        for entry in self._atomlist:
            _elem = entry[0]
            _poslist = entry[1]
            for ind, _pos in enumerate(_poslist):
                _label = _elem + str(ind)
                print('{:6s}{:16.9f}{:16.9f}{:16.9f}'.format(
                    _label, _pos[0], _pos[1], _pos[2]))
