#!/usr/bin/env python
# -*- coding:utf-8 -*-

import argparse
import glob
import os

from l134k.structure import Structure
from l134k.energy_data import freq_scale_factors
from l134k.util import pickle_dump


def main():
    data_dir, out_file, ignore_file = parse_args()
    structs = []

    ignore = set()
    if ignore_file is not None:
        with open(ignore_file) as f:
            for line in f:
                try:
                    idx = int(line.split()[0])
                except (IndexError, ValueError):
                    continue
                else:
                    ignore.add(idx)

    print('Parsing files...')
    files = glob.iglob(os.path.join(data_dir, '*.xyz'))
    for path in files:
        s = Structure(path=path)
        if s.index in ignore:
            continue
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
    parser.add_argument('--ignore', metavar='FILE', help='Path to file containing list of indices to ignore')
    args = parser.parse_args()

    data_dir = args.data_dir
    out_file = args.out_file
    ignore_file = args.ignore

    return data_dir, out_file, ignore_file


if __name__ == '__main__':
    main()
