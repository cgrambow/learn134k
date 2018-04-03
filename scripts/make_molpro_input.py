#!/usr/bin/env python
# -*- coding:utf-8 -*-

from __future__ import division

import argparse
import glob
import os
import random
import shutil

from data.structure import Structure


def main():
    data_dir, out_dir, num, method, basis, mem = parse_args()

    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.mkdir(out_dir)

    print('Parsing files...')
    files = glob.iglob(os.path.join(data_dir, '*.xyz'))
    structures = []
    for path in files:
        s = Structure()
        s.parse_structure(path)
        if 'F' not in s.elements:  # Don't use fluorine containing molecules
            structures.append(s)

    print('Making input files...')
    structures = random.sample(structures, num)
    names = []
    for i, s in enumerate(structures):
        name = os.path.splitext(s.file_name)[0]
        molpro_input = os.path.join(out_dir, str(i) + '.in')
        s.make_molpro_input(molpro_input, method=method, basis=basis, mem=mem)
        names.append(name)

    with open(os.path.join(out_dir, '0list_of_names.txt'), 'w') as f:
        for name in names:
            f.write(name + '\n')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Make input files and submission scripts for Molpro jobs.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('data_dir', metavar='DDIR', help='Path to 134k data directory')
    parser.add_argument('out_dir', metavar='ODIR', help='Path to output directory')
    parser.add_argument('-n', '--num', type=int, default=100, metavar='N',
                        help='Number of structures to randomly choose')
    parser.add_argument('--method', default='ccsd(t)-f12a', help='Quantum chemistry method')
    parser.add_argument('--basis', default='cc-pvtz-f12', help='Basis set')
    parser.add_argument('--mem', type=int, default=1900, help='Memory in megawords')
    args = parser.parse_args()

    data_dir = args.data_dir
    out_dir = args.out_dir
    num = args.num
    method = args.method
    basis = args.basis
    mem = args.mem

    return data_dir, out_dir, num, method, basis, mem


if __name__ == '__main__':
    main()
