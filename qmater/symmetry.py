import numpy as np

pauli_0 = np.eye(2, dtype=np.complex128)
pauli_x = np.array([[0., 1.], [1., 0.]], dtype=np.complex128)
pauli_y = np.array([[0., -1j], [1j, 0]], dtype=np.complex128)
pauli_z = np.array([[1., 0.], [0., -1.]], dtype=np.complex128)


class SymmOper(object):
    """ Class SymmOper for a symmetry operation. """

    def __init__(self, rot, trans=None, matrep=None, lconj=False):
        self.operation = (rot, trans)
        self.matrep = matrep
        self.lconj = lconj

    @property
    def operation(self):
        return self._operation

    @operation.setter
    def operation(self, oper):
        rot = np.array(oper[0])
        if oper[1] is None:
            trans = np.array([0, 0, 0])
        else:
            trans = np.array(oper[1])
        self._operation = [rot, trans]

    @property
    def rot(self):
        return self.operation[0]

    @property
    def rot_k(self):
        return np.linalg.inv(self.operation[0].T)

    @property
    def det_rot(self):
        return np.linalg.det(self.operation[0])

    @property
    def trans(self):
        return self.operation[1]

    @property
    def matrep(self):
        return self._matrep

    @matrep.setter
    def matrep(self, matrep):
        if matrep is None:
            self._matrep = None
        else:
            self._matrep = np.array(matrep, dtype=np.complex128)

    @property
    def lconj(self):
        return self._lconj

    @lconj.setter
    def lconj(self, lconj):
        self._lconj = lconj
