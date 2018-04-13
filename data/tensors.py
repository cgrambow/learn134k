#!/usr/bin/env python
# -*- coding:utf-8 -*-

import warnings

from rmgpy.molecule import Molecule
from rmgpy.cnn_framework.molecule_tensor import get_molecule_tensor, pad_molecule_tensor
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
                warnings.warn('Could not convert {}'.format(struct.smiles))
                return None

    mol_tensor = get_molecule_tensor(mol,
                                     add_extra_atom_attribute=add_extra_atom_attribute,
                                     add_extra_bond_attribute=add_extra_bond_attribute)
    return pad_molecule_tensor(mol_tensor, padding_size)
