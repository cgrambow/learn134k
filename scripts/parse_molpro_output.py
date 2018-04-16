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
    data_dir, quantum_dir, out_file = parse_args()
    structs = []

    print('Parsing files')
    # We require all matching input and output files from the
    # quantum jobs
    inputs = glob.iglob(os.path.join(quantum_dir, '*.in'))
    for inpath in inputs:
        outpath = os.path.splitext(inpath)[0] + '.out'
        method = basis = energy = None

        with open(inpath, 'r') as f:
            struct_path = os.path.join(data_dir, f.readline().strip().split(',')[1] + '.xyz')

            line = None
            for line in f:
                if 'basis' in line:
                    basis = line.strip().split('=')[-1].lower()
            if line is not None:
                method = line.strip().lower()

        if method is None or basis is None:
            raise Exception('Method and/or basis not found in {}'.format(inpath))

        with open(outpath, 'r') as f:
            for line in reversed(f.readlines()):
                if 'CCSD(T)-F12a total energy' in line:
                    energy = float(line.strip().split()[-1])
                    break
                # Implement other energies here as necessary
            else:
                raise Exception('Energy not found in {}'.format(outpath))

        s = Structure(path=struct_path)
        model_chemistry = method + '/' + basis
        s.set_energy(energy, model_chemistry)
        s.get_enthalpy_of_formation(freq_scale_factor=freq_scale_factors[model_chemistry],
                                    apply_bond_corrections=True)
        structs.append(s)

    pickle_dump(out_file, structs, compress=True)
    print('Dumped {} structures to {}'.format(len(structs), out_file))


def parse_args():
    parser = argparse.ArgumentParser(
        description='Parse Molpro output files.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('data_dir', metavar='DDIR', help='Path to 134k data directory')
    parser.add_argument('quantum_dir', metavar='QDIR', help='Path to quantum calculation files')
    parser.add_argument('out_file', metavar='FILE', help='Path to output file')
    args = parser.parse_args()

    data_dir = args.data_dir
    quantum_dir = args.quantum_dir
    out_file = args.out_file

    return data_dir, quantum_dir, out_file


if __name__ == '__main__':
    main()
