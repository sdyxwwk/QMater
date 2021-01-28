import os
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.collections import LineCollection

from qmater.crystal import CrystStruct


def write_datafile(data_file, xlist, ylist, clist=None):
    """ Write data to file. """
    if len(xlist) != len(ylist):
        raise ValueError('xlist does match ylist!')

    if clist is not None and len(xlist) != len(clist):
        raise ValueError('xlist does match clist!')

    num_bands = ylist[0].shape[1]
    num_path_list = [len(entry) for entry in xlist]
    num_path = len(num_path_list)
    num_points = sum(num_path_list)

    with open(data_file, 'w') as outfile:
        outfile.write('# Written by QMater\n')
        outfile.write(
            '# num_points: {:d}  num_path: {:d}  num_bands: {:d}\n'.format(
                num_points, num_path, num_bands))

        for _num, _entry in zip(num_path_list, xlist):
            outfile.write('# {:6d}{:16.9f}{:16.9f}\n'.format(
                _num, _entry[0], _entry[-1]))

        for j in range(num_bands):
            if clist is None:
                for x_entry, y_entry in zip(xlist, ylist):
                    nx = x_entry.shape[0]
                    for i in range(nx):
                        outfile.write('{:16.9f}{:16.9f}\n'.format(
                            x_entry[i], y_entry[i, j]))
                    outfile.write('\n')
            else:
                for x_entry, y_entry, c_entry in zip(xlist, ylist, clist):
                    nx = x_entry.shape[0]
                    for i in range(nx):
                        outfile.write('{:16.9f}{:16.9f}{:16.9f}\n'.format(
                            x_entry[i], y_entry[i, j], c_entry[i, j]))
                    outfile.write('\n')
            outfile.write('\n')


def plot_multicolor_line(xlist, ylist, clist, xlim, ylim):
    """ Plot multicolored lines. """
    # Parameters for plotting
    xmin, xmax = xlim
    ymin, ymax = ylim

    # Plotting
    fig, ax = plt.subplots(1, 1, sharex=True, sharey=True)
    fig.subplots_adjust(left=0.2, right=0.9, top=0.9, bottom=0.1)
    norm = plt.Normalize(0.0, 1.0)

    for x_entry, y_entry, c_entry in zip(xlist, ylist, clist):
        for _y, _c in zip(y_entry.T, c_entry.T):
            # Create a set of line segments so that we can color them individually
            points = np.array([x_entry, _y]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            # Create a continuous norm to map from data points to colors
            lc = LineCollection(segments, cmap='Reds', norm=norm)
            lc.set_array(_c)
            lc.set_linewidth(1.2)
            ax.add_collection(lc)

    cb = fig.colorbar(lc, ax=ax)
    cb.ax.tick_params(labelsize=10)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    return fig, ax


class BandStruct(object):
    """ Class BandStruct for processing band structures.

    Attributes:
        structure: an object of the class CrystStruct.
        lorbit: LORBIT in VASP
        fermi: fermi energy
        num_kpoints: number of total k-points
        kpath_label: list of k-labels on k-paths and their positions
        num_bands: number of bands
        num_atoms: number of atoms
        orbitals: list of orbitals
        calculation: type of calculations

        _ispin: control std, ncl, or spin
        _num_kp_list: number of k-points for each k-path

        _bandlist: the core list consisting of sublists.
            - Each sublist contains a k-path with [k-points, energy, atom-orbital-spin matrix]

    Examples:

    """

    def __init__(self, structure):
        self.structure = structure
        self._fermi = 0.0
        self._ispin = 1
        self._lorbit = None
        self._num_bands = 0
        self._num_kpoints = 0
        self._kpath_label = []
        self._orbitals = []
        self._bandlist = []

    #
    #  Attributes
    #

    @ property
    def fermi(self):
        return self._fermi

    @ fermi.setter
    def fermi(self, fermi):
        self._fermi = fermi

    @ property
    def calculation(self):
        if self._ispin == 1:
            return 'normal'
        elif self._ispin == 2:
            return 'spin-polarized'
        elif self._ispin == 3:
            return 'spin-orbit'
        else:
            return None

    @ calculation.setter
    def calculation(self, calculation):
        if calculation == 'normal':
            self._ispin = 1
        elif calculation == 'spin-polarized':
            self._ispin = 2
        elif calculation == 'spin-orbit':
            self._ispin = 3
        else:
            raise ValueError(
                'band_type should be \'normal\', \'spin-polarized\', or \'spin-orbit\'!')

    @ property
    def lorbit(self):
        return self._lorbit

    @ lorbit.setter
    def lorbit(self, lorbit):
        self._lorbit = lorbit

    @ property
    def num_atoms(self):
        return self.structure.num_atoms

    @ property
    def num_bands(self):
        return self._num_bands

    @ property
    def num_kpoints(self):
        return self._num_kpoints

    @ property
    def kpath_label(self):
        return self._kpath_label

    @ property
    def orbitals(self):
        return self._orbitals

    @ orbitals.setter
    def orbitals(self, orbitals):
        self._orbitals = orbitals

    #
    # internal functions
    #
    def _read_kpoints_linemode(self, kpoints_file='KPOINTS'):
        """ Read KPOINTS with line mode. """
        if not os.path.exists(kpoints_file):
            raise Exception('Does not find the file {:s}'.format(kpoints_file))

        _struc = self.structure

        with open(kpoints_file, 'r') as infile:
            infile.readline()
            num_kp_kpath = int(infile.readline().split()[0])
            infile.readline()
            if_rec = infile.readline()[0]
            mylist = [line.split()
                      for line in infile.readlines() if line.strip() != ""]

        num_kpath = len(mylist) // 2

        kpath_start = [entry[0:3] for entry in mylist[0::2]]
        kpath_start = np.array(kpath_start, dtype=np.float64)
        kpath_end = [entry[0:3] for entry in mylist[1::2]]
        kpath_end = np.array(kpath_end, dtype=np.float64)

        _label = [entry[-1] for entry in mylist]
        _klabel = _label[0::2] + _label[-1:]

        if if_rec in ['r', 'R', 'd', 'D']:
            frac = 'N'
        else:
            frac = 'F'

        kpoints, kpos, klabel_pos = _struc.get_kpoint_lines(
            kpath_start, kpath_end, num_kp_kpath, frac)

        # set values
        self._num_kp_list = [num_kp_kpath for a in range(num_kpath)]
        self._num_kpoints = num_kp_kpath * num_kpath
        self._kpath_label = _klabel

        nc = 0
        self._bandlist = []
        for _num in self._num_kp_list:
            self._bandlist.append([kpoints[nc:nc + _num, :]])
            nc += _num

    def _get_kpos(self):
        """ Return positions for k-points and k-labels. """
        _klist = [entry[0] for entry in self._bandlist]
        _nkp_kpath = self._num_kp_list
        _num_kp = self._num_kpoints
        _num_kpath = len(_num_kp_kpath)

        kpos = np.zeros((_num_kp,), dtype=np.float64)
        kpos_label = np.zeros((_num_kpath + 1,), dtype=np.float64)

        nc = 0
        for ind, nkp in enumerate(_nkp_kpath):
            _kp = _klist[ind]
            kpos[nc + 1:nc + nkp] = np.linalg.norm(
                _kp[:-1, :] - _kp[1:, :], axis=1)
            kpos_label[ind+1] = np.sum(kpos[nc + 1:nc + nkp])

        kpos = np.cumsum(kpos).reshape(_num_kp, 1)
        kpos_label = np.cumsum(kpos_label).reshape(_num_kpath+1, 1)
        return kpos, kpos_label

    def _get_kpos_v2(self):
        """ Return positions for k-points and k-labels. """
        _klist = [entry[0] for entry in self._bandlist]
        _nkp_kpath = self._num_kp_list
        _num_kp = self._num_kpoints
        _num_kpath = len(_nkp_kpath)
        _rec_latt = self.structure.reciprocal_lattice

        kpos_label = np.zeros((_num_kpath + 1,), dtype=np.float64)

        nc = 0
        kpos_list = []
        kpos_start = 0.0
        for ind, nkp in enumerate(_nkp_kpath):
            kpos = np.zeros((nkp,), dtype=np.float64)

            _kp = np.dot(_klist[ind], _rec_latt)
            kpos[1:nkp] = np.linalg.norm(_kp[:-1, :] - _kp[1:, :], axis=1)
            kpos = np.cumsum(kpos) + kpos_start
            kpos_list.append(kpos)

            kpos_label[ind + 1] = kpos[-1]
            kpos_start = kpos[-1]

        kpos_label = kpos_label
        return kpos_list, kpos_label

    def _get_bands(self):
        """ Return bands for each k-point. """
        pass

    #
    # set model
    #
    def read_procar(self, procar_file, lorbit=10, calculation='normal', fermi=0.0):
        """ Set model from PROCAR. """
        if not os.path.exists(procar_file):
            raise Exception('Does not find the file {:s}'.format(procar_file))

        self._lorbit = lorbit
        self.calculation = calculation
        self.fermi = fermi

        ispin = self._ispin
        mystruc = self.structure

        # read KPOINTS
        self._read_kpoints_linemode()
        _num_kp_kpath = self._num_kp_list

        # read PROCAR
        with open(procar_file, 'r') as infile:
            print('==========================')
            print('    Reading PROCAR ...    ')
            print('--------------------------')
            infile.readline()
            line = infile.readline().split()
            num_kp = int(line[3])
            num_bands = int(line[7])
            num_ions = int(line[11])

            if num_ions != self.structure.num_atoms:
                raise ValueError('Structure does not match PROCAR!')

            if num_kp != self._num_kpoints:
                raise ValueError('KPOINTS does not match PROCAR!')

            print('Number of k-points => {:d}'.format(num_kp))
            print('Number of bands => {:d}'.format(num_bands))
            print('Number of atoms => {:d}'.format(num_ions))

            for ind, nkp in enumerate(_num_kp_kpath):
                _bands = np.zeros((nkp, num_bands), dtype=np.float64)

                if lorbit == 10 and ispin == 3:
                    _charges = np.zeros(
                        (nkp, num_bands, num_ions, 3, 4), dtype=np.float64)
                elif lorbit == 11 and ispin == 3:
                    _charges = np.zeros(
                        (nkp, num_bands, num_ions, 9, 4), dtype=np.float64)

                for nk in range(nkp):
                    if nkp > 10 and nk % int(nkp / 2) == 0:
                        print('k-point {:d} on k-path {:d}'.format(nk, ind))

                    infile.readline()
                    infile.readline()  # k-point
                    infile.readline()

                    for nb in range(num_bands):
                        _ene = float(infile.readline().split()[4])
                        _bands[nk, nb] = _ene - fermi
                        infile.readline()

                        if ind == 0 and nk == 0 and nb == 0:
                            _orb_list = infile.readline().split()[1:-1]
                            if len(_orb_list) != _charges.shape[3]:
                                raise ValueError(
                                    'lorbit does not match PROCAR!')
                        else:
                            infile.readline()

                        if ispin == 3:
                            for ns in range(4):
                                for ni in range(num_ions):
                                    _charges[nk, nb, ni, :, ns] = np.array(
                                        infile.readline().split()[1:-1], dtype=np.float64)
                                infile.readline()  # tot
                        infile.readline()

                # add bands and partial charges to self._bandlist
                self._bandlist[ind].append(_bands)
                self._bandlist[ind].append(_charges)

            print('==========================')
            print('')

        # set values
        self._num_bands = num_bands
        self.orbitals = _orb_list

    #
    # plotting
    #
    def plot_band_structure(self, Elim=[-2.0, 2.0]):
        """ Plot band structure. """
        ymin, ymax = Elim
        kpath_label = self._kpath_label

        kpos_list, label_pos = self._get_kpos_v2()
        num_label = len(label_pos)

        Elist = [entry[1] for entry in self._bandlist]

        # Write data to file.
        write_datafile('bands_qm.dat', kpos_list, Elist)

        # Plot band structure.
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        for i in range(num_label - 1):
            ax.plot(kpos_list[i], Elist[i], linewidth=0.5, color='b')

        for i in range(num_label):
            ax.plot([label_pos[i], label_pos[i]], [ymin, ymax],
                    linewidth=0.5, color='k')
        ax.plot([kpos_list[0][0], kpos_list[-1][-1]], [0.0, 0.0],
                linewidth=0.5, color='k', linestyle="--")

        ax.set_xticks(label_pos)
        ax.set_xticklabels(kpath_label, fontsize=12)
        plt.ylabel('Energy', fontsize=12)
        plt.xlim(kpos_list[0][0], kpos_list[-1][-1])
        plt.ylim(ymin, ymax)
        plt.savefig('bands.pdf', dpi=300)
        # plt.show()

    def plot_fatbands_orbital(self, element, orbital):
        pass

    def plot_fatbands_atom(self, atomlist, Elim=[-2.0, 2.0]):
        """ Plot fatbands for specific atoms. """
        atomlist = np.array(atomlist, dtype=np.integer)
        ispin = self._ispin

        # Get xlist, ylist, and clist
        kpos_list, label_pos = self._get_kpos_v2()

        Elist = [entry[1] for entry in self._bandlist]

        if ispin == 1:
            charge_list = [np.sum(entry[2][:, :, atomlist, :], axis=(0, 3))
                           for entry in self._bandlist]
        elif ispin == 3:
            charge_list = [np.sum(entry[2][:, :, atomlist, :, 0], axis=(0, 3))
                           for entry in self._bandlist]

        # Write data to file.
        write_datafile('fatbands_qm.dat', kpos_list, Elist, charge_list)

        kpath_label = self._kpath_label
        num_label = len(kpath_label)

        # Plot fatbands
        klim = [kpos_list[0][0], kpos_list[-1][-1]]

        fig, ax = plot_multicolor_line(
            kpos_list, Elist, charge_list, klim, Elim)

        for i in range(num_label):
            ax.plot([label_pos[i], label_pos[i]], Elim,
                    linewidth=0.5, color='k')

        # ax.plot([kpos_list[0], kpos_list[-1]], [0.0, 0.0],
        #        linewidth=0.5, color='r', linestyle="--")

        ax.set_xticks(label_pos)
        ax.set_xticklabels(kpath_label, fontsize=18)
        ax.set_ylabel('Energy', fontsize=18)
        fig.savefig('fatbands.pdf', dpi=300)
        # fig.show()
