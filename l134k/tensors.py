#!/usr/bin/env python
# -*- coding:utf-8 -*-

import logging

import numpy as np

from rmgpy.molecule import Molecule
from rmgpy.cnn_framework.molecule_tensor import get_molecule_tensor, pad_molecule_tensor, get_attribute_vector_size
from rmgpy.exceptions import AtomTypeError


def struct_to_tensor(struct, padding_size, add_extra_atom_attribute=True, add_extra_bond_attribute=True):
    try:
        mol = Molecule().fromSMILES(struct.smiles)
    except AtomTypeError:
        try:
            mol = Molecule().fromSMILES(struct.smiles2)
        except AtomTypeError:
            try:
                mol = Molecule().fromInChI(struct.inchi)
            except AtomTypeError:
                logging.warning('Could not convert {}'.format(struct.smiles))
                return None

    mol_tensor = get_molecule_tensor(mol,
                                     add_extra_atom_attribute=add_extra_atom_attribute,
                                     add_extra_bond_attribute=add_extra_bond_attribute)
    return pad_molecule_tensor(mol_tensor, padding_size)


def structs_to_tensors(structs, padding_size, add_extra_atom_attribute=True, add_extra_bond_attribute=True):
    # Initialize arrays containing molecule tensors, heats of formation, and smiles
    attribute_vector_size = get_attribute_vector_size(
        add_extra_atom_attribute=add_extra_atom_attribute,
        add_extra_bond_attribute=add_extra_bond_attribute
    )
    x = np.zeros((len(structs), padding_size, padding_size, attribute_vector_size))
    y = np.zeros(len(structs))
    names = []

    logging.info('Converting structures to molecule tensors...')
    i = 0
    for struct in structs:
        mol_tensor = struct_to_tensor(struct, padding_size,
                                      add_extra_atom_attribute=add_extra_atom_attribute,
                                      add_extra_bond_attribute=add_extra_bond_attribute)
        if mol_tensor is not None:
            x[i] = mol_tensor
            y[i] = struct.hf298 / 4184.0  # Convert to kcal/mol
            names.append(struct.file_name)
            i += 1

    # Remove empty end of arrays
    x = x[:i]
    y = y[:i]
    logging.info('{} structures converted to tensors'.format(len(x)))

    return x, y, names
