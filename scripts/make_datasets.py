#!/usr/bin/env python
# -*- coding:utf-8 -*-

import argparse
import os
import random

from l134k.util import pickle_load, pickle_dump


def main():
    args = parse_args()
    struct_file = args.data
    out_dir = args.out_dir
    n = args.nsplit

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    print('Loading structures from {}...'.format(struct_file))
    structs = pickle_load(struct_file, compressed=True)
    print('Loaded {} structures'.format(len(structs)))
    random.shuffle(structs)

    bin_size = len(structs) // n
    for i in range(n):
        train = structs[:(i+1)*bin_size]
        test = structs[(i+1)*bin_size:]

        folder = os.path.join(out_dir, 'train_test_{}'.format(i+1))
        if not os.path.exists(folder):
            os.mkdir(folder)
        train_path = os.path.join(folder, 'train_{}.pickle.gz'.format(len(train)))
        test_path = os.path.join(folder, 'test_{}.pickle.gz'.format(len(test)))
        pickle_dump(train_path, train, compress=True)
        pickle_dump(test_path, test, compress=True)


def parse_args():
    parser = argparse.ArgumentParser(description='Make explicit training/testing datasets')
    parser.add_argument('data', metavar='DATA',
                        help='Path to pickled and zipped file containing list of structures')
    parser.add_argument('out_dir', metavar='DIR', help='Output directory')
    parser.add_argument('nsplit', type=int, metavar='N', help='Number of train/test set combinations to make')
    return parser.parse_args()


if __name__ == '__main__':
    main()
