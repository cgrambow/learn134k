#!/usr/bin/env python
# -*- coding:utf-8 -*-

import argparse
import cPickle as pickle
import glob
import os

from l134k.structure import Structure
from l134k.energy_data import freq_scale_factors
from l134k.util import pickle_dump


def main():
    data_dir, out_file = parse_args()
    structs = []

    print('Parsing files...')
    files = glob.iglob(os.path.join(data_dir, '*.xyz'))
    for path in files:
        s = Structure(path=path)
        if 'F' not in s.elements:  # Don't use fluorine containing molecules
            s.get_enthalpy_of_formation(freq_scale_factor=freq_scale_factors[s.model_chemistry],
                                        apply_bond_corrections=False)
            structs.append(s)

    pickle_dump(out_file, structs, compress=True)
    print('Dumped {} structures to {}'.format(len(structs), out_file))


def parse_args():
    parser = argparse.ArgumentParser(
        description='Parse 134k dataset.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('data_dir', metavar='DIR', help='Path to 134k data directory')
    parser.add_argument('out_file', metavar='FILE', help='Path to output file')
    args = parser.parse_args()

    data_dir = args.data_dir
    out_file = args.out_file

    return data_dir, out_file


if __name__ == '__main__':
    main()
