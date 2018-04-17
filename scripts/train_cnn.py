#!/usr/bin/env python
# -*- coding:utf-8 -*-

import argparse
import logging
import os
import random
import shutil
import sys

import numpy as np

from rmgpy.cnn_framework.molecule_tensor import get_attribute_vector_size

from l134k.tensors import struct_to_tensor
from l134k.predictor import Predictor
from l134k.settings import tensor_settings, model_settings, train_settings
from l134k.util import pickle_load


def main():
    # Parse arguments
    args = parse_args()
    struct_file = args.struct_file
    out_dir = args.out_dir
    model_weights_path = args.model
    ndata = args.ndata
    folds = args.folds
    test_split = args.test_split
    save_names = args.save_names

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    level = logging.INFO
    initialize_log(level, os.path.join(out_dir, 'train.log'))

    logging.info('Loading structures from {}...'.format(struct_file))
    structs = pickle_load(struct_file, compressed=True)
    logging.info('Loaded {} structures'.format(len(structs)))

    random.seed(a=0)
    random.shuffle(structs)
    if ndata is not None:
        structs = structs[:ndata]
        logging.info('Randomly selected {} structures'.format(len(structs)))

    # Initialize arrays containing molecule tensors, heats of formation, and smiles
    attribute_vector_size = get_attribute_vector_size(
        add_extra_atom_attribute=tensor_settings['add_extra_atom_attribute'],
        add_extra_bond_attribute=tensor_settings['add_extra_bond_attribute']
    )
    x = np.zeros((len(structs),
                  tensor_settings['padding_final_size'],
                  tensor_settings['padding_final_size'],
                  attribute_vector_size))
    y = np.zeros(len(structs))
    names = []

    logging.info('Converting structures to molecule tensors...')
    i = 0
    for struct in structs:
        mol_tensor = struct_to_tensor(struct, tensor_settings['padding_final_size'],
                                      add_extra_atom_attribute=tensor_settings['add_extra_atom_attribute'],
                                      add_extra_bond_attribute=tensor_settings['add_extra_bond_attribute'])
        if mol_tensor is not None:
            x[i] = mol_tensor
            y[i] = struct.hf298 / 4184.0  # Convert to kcal/mol
            names.append(struct.file_name)
            i += 1

    # Remove empty end of arrays
    x = x[:i]
    y = y[:i]
    logging.info('{} structures converted to tensors'.format(len(x)))

    # Set up class for training, build model, and train
    predictor = Predictor(out_dir)
    predictor.build_model(tensor_settings, **model_settings)
    if model_weights_path is not None:
        predictor.load_weights(model_weights_path)

    # Perform cross-validation if folds was specified
    if folds is None:
        predictor.full_train(x, y, names, test_split, save_names=save_names, **train_settings)
    else:
        predictor.kfcv_train(x, y, names, folds, test_split, save_names=save_names, **train_settings)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train CNN on 134k data to predict Hf298. Requires RMG version with CNN framework.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('struct_file', metavar='FILE',
                        help='Path to pickled and zipped file containing list of structures')
    parser.add_argument('out_dir', metavar='DIR', help='Output directory')
    parser.add_argument('-m', '--model', metavar='PATH', help='Saved model weights to continue training on')
    parser.add_argument('-n', '--ndata', type=int, metavar='N', help='Number of data points to use')
    parser.add_argument('-f', '--folds', type=int, metavar='K',
                        help='If this option is used, perform cross-validation with the given number of folds')
    parser.add_argument('-t', '--test_split', type=float, default=0.1, metavar='S', help='Fraction of data to test on')
    parser.add_argument('-s', '--save_names', action='store_true', help='Store file names')
    return parser.parse_args()


def initialize_log(verbose, log_file_name):
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(verbose)

    # Create console handler and set level to debug; send everything to stdout
    # rather than stderr
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(verbose)

    logging.addLevelName(logging.CRITICAL, 'Critical: ')
    logging.addLevelName(logging.ERROR, 'Error: ')
    logging.addLevelName(logging.WARNING, 'Warning: ')
    logging.addLevelName(logging.INFO, '')
    logging.addLevelName(logging.DEBUG, '')
    logging.addLevelName(1, '')

    # Create formatter and add to console handler
    # formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', '%Y-%m-%d %H:%M:%S')
    # formatter = Formatter('%(message)s', '%Y-%m-%d %H:%M:%S')
    formatter = logging.Formatter('%(levelname)s%(message)s')
    ch.setFormatter(formatter)

    # create file handler
    if os.path.exists(log_file_name):
        directory = os.path.dirname(log_file_name)
        file_name, ext = os.path.splitext(os.path.basename(log_file_name))
        backup = os.path.join(directory, file_name + '_backup' + ext)
        if os.path.exists(backup):
            print('Removing old ' + backup)
            os.remove(backup)
        print('Moving {0} to {1}\n'.format(log_file_name, backup))
        shutil.move(log_file_name, backup)
    fh = logging.FileHandler(filename=log_file_name)
    fh.setLevel(min(logging.DEBUG, verbose))
    fh.setFormatter(formatter)
    # notice that STDERR does not get saved to the log file
    # so errors from underlying libraries (eg. openbabel) etc. that report
    # on stderr will not be logged to disk.

    # remove old handlers!
    while logger.handlers:
        logger.removeHandler(logger.handlers[0])

    # Add console and file handlers to logger
    logger.addHandler(ch)
    logger.addHandler(fh)


if __name__ == '__main__':
    main()
