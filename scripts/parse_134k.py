#!/usr/bin/env python
# -*- coding:utf-8 -*-

import argparse
import glob
import os

from l134k.structure import Structure
from l134k.energy_data import freq_scale_factors
from l134k.util import pickle_dump


def main():
    data_dir, out_file, ignore_file, names_file = parse_args()
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

    names = None
    if names_file is not None:
        with open(names_file) as f:
            names = [line.strip() for line in f]

    print('Parsing files...')
    if names is None:
        files = glob.iglob(os.path.join(data_dir, '*.xyz'))
    else:
        files = [os.path.join(data_dir, name) for name in names]

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
    parser.add_argument('--names', metavar='FILE', help='Path to file containing list of names to use')
    args = parser.parse_args()

    data_dir = args.data_dir
    out_file = args.out_file
    ignore_file = args.ignore
    names_file = args.names

    return data_dir, out_file, ignore_file, names_file


if __name__ == '__main__':
    main()
