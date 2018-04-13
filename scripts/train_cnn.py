#!/usr/bin/env python
# -*- coding:utf-8 -*-

import argparse
import logging
import os
import shutil
import sys

from rmgpy.cnn_framework.data import split_test_from_train_and_val, split_inner_val_from_train_data
from rmgpy.cnn_framework.cnn_model import build_model, train_model, save_model
from rmgpy.cnn_framework.molecule_tensor import get_attribute_vector_size

from data.tensors import struct_to_tensor
from data.util import pickle_load

# Tensor settings
tensor_settings = {
    'padding_final_size': 30,
    'add_extra_atom_attribute': True,
    'add_extra_bond_attribute': True,
}

# Model settings
model_settings = {
    'embedding_size': 512,
    'depth': 3,
    'scale_output': 0.05,
    'padding': True,
    'mol_conv_inner_activation': 'tanh',
    'mol_conv_outer_activation': 'softmax',
    'hidden': 50,
    'hidden_activation': 'tanh',
    'output_activation': 'linear',
    'output_size': 1,
    'lr': 0.01,
    'optimizer': 'adam',
    'loss': 'mse',
}

# Training settings
train_settings = {
    'nb_epoch': 150,
    'batch_size': 1,
    'lr_func': "float({0} * np.exp(- epoch / {1}))".format(0.0007, 30.0),
    'patience': 10,
}


def main():
    # Parse arguments
    args = parse_args()
    struct_file = args.struct_file
    out_dir = args.out_dir
    ndata = args.ndata
    test_split = args.test_split
    save_names = args.save_names

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    level = logging.INFO
    initialize_log(level, os.path.join(out_dir, 'train.log'))

    logging.info('Loading structures from {}...'.format(struct_file))
    structs = pickle_load(struct_file, compressed=True)
    logging.info('Loaded {} structures'.format(len(structs)))

    if ndata is not None:
        structs = structs[:ndata]
        logging.info('Randomly selected {} structures'.format(len(structs)))

    # Initialize lists holding molecule tensors, heats of formation, and smiles
    x = []
    y = []
    names = []

    logging.info('Converting structures to molecule tensors...')
    for struct in structs:
        mol_tensor = struct_to_tensor(struct, tensor_settings['padding_final_size'],
                                      add_extra_atom_attribute=tensor_settings['add_extra_atom_attribute'],
                                      add_extra_bond_attribute=tensor_settings['add_extra_bond_attribute'])
        if mol_tensor is not None:
            x.append(mol_tensor)
            y.append(struct.hf298 / 4184.0)  # Convert to kcal/mol
            names.append(struct.file_name)
    logging.info('{} structures converted to tensors'.format(len(x)))

    logging.info('Splitting dataset with a test split of {}'.format(test_split))
    x_test, y_test, x_train, y_train, names_test, names_train = split_test_from_train_and_val(x, y,
                                                                                              extra_data=names,
                                                                                              shuffle_seed=7,
                                                                                              testing_ratio=test_split)
    if save_names:
        names_test_path = os.path.join(out_dir, 'names_test.txt')
        names_train_path = os.path.join(out_dir, 'names_train.txt')
        with open(names_test_path, 'w') as f:
            for name in names_test:
                f.write(name + '\n')
        with open(names_train_path, 'w') as f:
            for name in names_train:
                f.write(name + '\n')

    logging.info('Splitting training data into early-stopping validation and remaining training sets')
    x_train, x_inner_val, y_train, y_inner_val = split_inner_val_from_train_data(x_train, y_train,
                                                                                 shuffle_seed=77,
                                                                                 training_ratio=0.9)

    # Build model
    attribute_vector_size = get_attribute_vector_size(
        add_extra_atom_attribute=tensor_settings['add_extra_atom_attribute'],
        add_extra_bond_attribute=tensor_settings['add_extra_bond_attribute']
    )
    model = build_model(attribute_vector_size=attribute_vector_size, **model_settings)

    logging.info('Training model...')
    logging.info('Training data: {} points'.format(len(x_train)))
    logging.info('Inner validation data: {} points'.format(len(x_inner_val)))
    logging.info('Test data: {} points'.format(len(x_test)))
    model, loss, inner_val_loss, mean_outer_val_loss, mean_test_loss = train_model(model,
                                                                                   x_train, y_train,
                                                                                   x_inner_val, y_inner_val,
                                                                                   x_test, y_test,
                                                                                   X_outer_val=None, y_outer_val=None,
                                                                                   **train_settings)

    logging.info('Saving model')
    model_path = os.path.join(out_dir, 'model')
    save_model(model, loss, inner_val_loss, mean_outer_val_loss, mean_test_loss, model_path)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train CNN on 134k data to predict Hf298. Requires RMG version with CNN framework.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('struct_file', metavar='FILE',
                        help='Path to pickled and zipped file containing list of structures')
    parser.add_argument('out_dir', metavar='DIR', help='Output directory')
    parser.add_argument('-n', '--ndata', type=int, metavar='N', help='Number of data points to use')
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
