#!/usr/bin/env python
# -*- coding:utf-8 -*-

import argparse

import numpy as np
import pandas as pd

from rmgpy.cnn_framework.molecule_tensor import get_attribute_vector_size

from l134k.tensors import struct_to_tensor
from l134k.predictor import Predictor
from l134k.settings import tensor_settings, model_settings
from l134k.util import pickle_load


def main():
    # Parse arguments
    args = parse_args()
    struct_file = args.struct_file
    out_file = args.out_file
    struct_list = args.struct_list
    model_weights_path = args.model

    # Load saved structures and extract only the ones in struct_list
    print('Loading structures from {}...'.format(struct_file))
    structs = pickle_load(struct_file, compressed=True)
    print('Loaded {} structures'.format(len(structs)))
    with open(struct_list) as f:
        struct_names = set(f.read().split())
    attribute_vector_size = get_attribute_vector_size(
        add_extra_atom_attribute=tensor_settings['add_extra_atom_attribute'],
        add_extra_bond_attribute=tensor_settings['add_extra_bond_attribute']
    )
    x = np.zeros((len(structs),
                  tensor_settings['padding_final_size'],
                  tensor_settings['padding_final_size'],
                  attribute_vector_size))
    y = np.zeros(len(structs))
    smiles = []
    i = 0
    print('Extracting target structures...')
    for struct in structs:
        if struct.file_name in struct_names:
            mol_tensor = struct_to_tensor(struct, tensor_settings['padding_final_size'],
                                          add_extra_atom_attribute=tensor_settings['add_extra_atom_attribute'],
                                          add_extra_bond_attribute=tensor_settings['add_extra_bond_attribute'])
            if mol_tensor is not None:
                x[i] = mol_tensor
                y[i] = struct.hf298 / 4184.0  # Convert to kcal/mol
                smiles.append(struct.smiles)
                i += 1
    # Remove empty end of arrays
    x = x[:i]
    y = y[:i]
    print('{} structures to evaluate CNN on'.format(len(x)))

    # Build and evaluate CNN
    print('Evaluating...')
    predictor = Predictor()
    predictor.build_model(tensor_settings, **model_settings)
    predictor.load_weights(model_weights_path)

    y_pred = predictor.predict(x).flatten()

    df = pd.DataFrame(index=smiles)
    df['Hf298 true (kcal/mol)'] = pd.Series(y, index=df.index)
    df['Hf298 pred (kcal/mol)'] = pd.Series(y_pred, index=df.index)
    diff = abs(df['Hf298 true (kcal/mol)'] - df['Hf298 pred (kcal/mol)'])
    sqe = diff ** 2.0
    df['Hf298 diff (kcal/mol)'] = pd.Series(diff, index=df.index)
    df['Hf298 diff squared (kcal^2/mol^2)'] = pd.Series(sqe, index=df.index)

    descr_ae = df['Hf298 diff (kcal/mol)'].describe()
    count = int(descr_ae.loc['count'])
    mae = descr_ae.loc['mean']
    mae_std = descr_ae.loc['std']

    descr_sqe = df['Hf298 diff squared (kcal^2/mol^2)'].describe()
    rmse = np.sqrt(descr_sqe.loc['mean'])
    rmse_std = np.sqrt(descr_sqe.loc['std'])

    print('Count: {}'.format(count))
    print('MAE: {:.2f}, MAE std: {:.2f}'.format(mae, mae_std))
    print('RMSE: {:.2f}, RMSE std: {:.2f}'.format(rmse, rmse_std))
    df.to_csv(path_or_buf=out_file)
    print('Wrote detailed results to {}'.format(out_file))


def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate trained CNN model. Requires RMG version with CNN framework.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('struct_file', help='Path to pickled and zipped file containing list of structures')
    parser.add_argument('out_file', help='Output file (typically .csv)')
    parser.add_argument('struct_list', help='Path to file containing names of structures to evaluate CNN on')
    parser.add_argument('model', help='Model weights to use for evaluation')
    return parser.parse_args()


if __name__ == '__main__':
    main()
