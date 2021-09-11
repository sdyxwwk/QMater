#===============================================================
# A script for printing information of POSCAR using QMater
#
# Syntax: python QM_runCrystStruct.py
#
# Read POSCAR and print the structural information,
# the symmetrized POSCAR, and the high-symmetry k-points.
#
# Author: Weikang Wu (sdyxwwk)
# Email: sdyxwwk@126.com
# Date: 27/06/2021
#===============================================================

import qmater as qm

def main():
    poscarfile = 'POSCAR.vasp'

    my_struct = qm.CrystStruct.read_poscar(poscarfile)

    #-----------------------------------------------------------
    # Print structural information.
    #-----------------------------------------------------------
    my_struct.print_info()

    #-----------------------------------------------------------
    # Print POSCAR of the primitive cell.
    #-----------------------------------------------------------
    # my_struct.get_poscar(if_symm=True, to_primitive=True)

    #-----------------------------------------------------------
    # Print high-symmetry k-points and recommended k-paths.
    #-----------------------------------------------------------
    # my_struct.print_highkpath()

    #-----------------------------------------------------------
    # Get Cartesian or fractional coordinate of a k-point.
    #-----------------------------------------------------------
    # kp = [1./3., 1./3., 0.0]
    # print(my_struct.get_kpoint_cart(kp))
    # print(my_struct.get_kpoint_frac(kp))


if __name__ == '__main__':
    main()
