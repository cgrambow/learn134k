#!/usr/bin/env python
# -*- coding:utf-8 -*-

import argparse
import logging
import os
import random
import shutil
import sys

from l134k.tensors import structs_to_tensors
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
    test_struct_file = args.test_data
    train_ratio = args.train_ratio
    depth = args.depth
    hidden = args.hidden
    batch_size = args.batch_size
    epochs = args.epochs
    patience = args.patience
    lr = args.learning_rate
    decay = args.lr_decay
    save_names = args.save_names

    model_settings['depth'] = depth
    model_settings['hidden'] = hidden

    train_settings['batch_size'] = batch_size
    train_settings['nb_epoch'] = epochs
    train_settings['patience'] = patience
    train_settings['lr_func'] = "float({0} * np.exp(- epoch / {1}))".format(lr, decay)

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    level = logging.INFO
    initialize_log(level, os.path.join(out_dir, 'train.log'))

    logging.info('Loading structures from {}...'.format(struct_file))
    structs = pickle_load(struct_file, compressed=True)
    logging.info('Loaded {} structures'.format(len(structs)))
    if test_struct_file is not None:
        logging.info('Loading testing structures from {}...'.format(struct_file))
        test_structs = pickle_load(test_struct_file, compressed=True)
        logging.info('Loaded {} testing structures'.format(len(structs)))
    else:
        test_structs = None

    random.seed(a=0)
    random.shuffle(structs)
    if ndata is not None:
        structs = structs[:ndata]
        logging.info('Randomly selected {} structures'.format(len(structs)))

    x, y, names = structs_to_tensors(structs,
                                     tensor_settings['padding_final_size'],
                                     add_extra_atom_attribute=tensor_settings['add_extra_atom_attribute'],
                                     add_extra_bond_attribute=tensor_settings['add_extra_bond_attribute'])
    if test_structs is not None:
        x_test, y_test, _ = structs_to_tensors(test_structs,
                                               tensor_settings['padding_final_size'],
                                               add_extra_atom_attribute=tensor_settings['add_extra_atom_attribute'],
                                               add_extra_bond_attribute=tensor_settings['add_extra_bond_attribute'])
        test_data = (x_test, y_test)
    else:
        test_data = None

    # Set up class for training, build model, and train
    predictor = Predictor(out_dir)
    predictor.build_model(tensor_settings, **model_settings)
    if model_weights_path is not None:
        predictor.load_weights(model_weights_path)

    # Perform cross-validation if folds was specified
    if folds is None:
        predictor.full_train(x, y, names, test_split, train_ratio,
                             test_data=test_data, save_names=save_names, **train_settings)
    else:
        predictor.kfcv_train(x, y, names, folds, test_split, train_ratio, test_data=test_data,
                             save_names=save_names, pretrained_weights=model_weights_path, **train_settings)


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
    parser.add_argument('--test_data', metavar='FILE',
                        help='Pickled file containing list of test data '
                             '(all data in struct_file will be used for training)')
    parser.add_argument('-r', '--train_ratio', type=float, default=0.9, metavar='R',
                        help='Fraction of data to train on (rest is inner validation)')
    parser.add_argument('--depth', type=int, default=3, metavar='D', help='Depth of convolutional layer')
    parser.add_argument('--hidden', type=int, default=50, metavar='H', help='Number of hidden units in dense layer')
    parser.add_argument('-b', '--batch_size', type=int, default=1, metavar='B', help='Batch size')
    parser.add_argument('-e', '--epochs', type=int, default=150, metavar='N', help='Maximum number of epochs')
    parser.add_argument('-p', '--patience', type=int, default=10, metavar='P', help='Patience for early stopping')
    parser.add_argument('--learning_rate', type=float, default=0.0007, metavar='LR', help='Learning rate')
    parser.add_argument('--lr_decay', type=float, default=30.0, metavar='LRD', help='Learning rate decay')
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
