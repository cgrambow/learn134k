#!/usr/bin/env python
# -*- coding:utf-8 -*-

import os
import warnings

import numpy as np
import pybel

import rmgpy.constants as constants
from rmgpy.statmech import Conformer, IdealGasTranslation, LinearRotor, NonlinearRotor, HarmonicOscillator
from rmgpy.molecule import Molecule
from rmgpy.exceptions import AtomTypeError

import energy_data


class Structure(object):
    # Enthalpy correction terms based on atomic heats of formation in kcal/mol
    enthalpy_corrections = {
        'H': 50.62,
        'C': 169.73,
        'N': 111.49,
        'O': 57.95,
        'F': 17.42,
    }

    atomic_num_dict = {
        1: 'H',
        6: 'C',
        7: 'N',
        8: 'O',
        9: 'F'
    }

    # Symbols used for bond energy corrections based on bond order
    bond_symbols = {
        1: '-',
        2: '=',
        3: '#',
    }

    def __init__(self, path=None):
        # Geometries and energies at B3LYP/6-31G(2df,p) level of
        # theory if parsed directly from dataset

        # Structure, partial charges, and frequencies
        self.natoms = None            # Row 0
        self.elements = None          # Row 2,...,natoms+1
        self.coords = None            # Row 2,...,natoms+1 (Angstrom)
        self.mulliken_charges = None  # Row 2,...,natoms+1 (e)
        self.freqs = None             # Row natoms+2 (cm^-1)
        self.smiles = None            # Row natoms+3  Corresponding to input to B3LYP opt
        self.smiles2 = None           # Row natoms+3  As parsed from the optimized geo
        self.inchi = None             # Row natoms+4

        # Properties line (row 1)
        self.tag = None                       # Col 0
        self.index = None                     # Col 1
        self.rotational_consts = None         # Col 2-4 (GHz)
        self.dipole_mom = None                # Col 5 (Debye)
        self.isotropic_polarizability = None  # Col 6 (Bohr^3)
        self.homo = None                      # Col 7 (Ha)
        self.lumo = None                      # Col 8 (Ha)
        self.gap = None                       # Col 9 (Ha)
        self.electronic_extent = None         # Col 10 (Bohr^2)
        self.zpe = None                       # Col 11 (Ha)
        self.u0 = None                        # Col 12 (Ha)
        self.u298 = None                      # Col 13 (Ha)
        self.h298 = None                      # Col 14 (Ha)
        self.g298 = None                      # Col 15 (Ha)
        self.cv298 = None                     # Col 16 (Ha)

        # Other attributes
        self.file_name = None
        self.model_chemistry = None
        self.e0 = None  # (Ha)
        self.hf298 = None  # J/mol

        # Get structure
        if path is not None:
            self.parse_structure(path)

    def copy(self):
        s = Structure()
        for attr, val in self.__dict__.iteritems():
            setattr(s, attr, val)
        return s

    def __str__(self):
        return '\n'.join('{}: {}'.format(k, v) for k, v in sorted(self.__dict__.iteritems()))

    def parse_structure(self, path):
        """
        Read and parse an extended xyz file from the 134k dataset.
        """
        with open(path, 'r') as f:
            lines = f.readlines()

        self.natoms = int(lines[0].strip())
        props = lines[1].strip().split()
        xyz = np.array([line.split() for line in lines[2:(self.natoms+2)]])
        self.elements = list(xyz[:,0])

        try:
            self.coords = xyz[:,1:4].astype(np.float)
        except ValueError as e:
            if '*^' in str(e):  # Handle powers of 10
                coords_str = xyz[:,1:4]
                coords_flat = np.array([float(s.replace('*^', 'e')) for s in coords_str.flatten()])
                self.coords = coords_flat.reshape(coords_str.shape)
            else:
                raise

        try:
            self.mulliken_charges = xyz[:,4].astype(np.float)
        except ValueError as e:
            if '*^' in str(e):  # Handle powers of 10
                self.mulliken_charges = np.array([float(s.replace('*^', 'e')) for s in xyz[:,4]])
            else:
                raise

        self.freqs = np.array([float(col) for col in lines[self.natoms+2].split()])
        smiles = lines[self.natoms+3].split()
        self.smiles = smiles[0]
        self.smiles2 = smiles[1]
        self.inchi = lines[self.natoms+4].split()[1]

        self.tag = props[0]
        self.index = int(props[1])
        self.rotational_consts = np.array([float(c) for c in props[2:5]])
        self.dipole_mom = float(props[5])
        self.isotropic_polarizability = float(props[6])
        self.homo = float(props[7])
        self.lumo = float(props[8])
        self.gap = float(props[9])
        self.electronic_extent = float(props[10])
        self.zpe = float(props[11])
        self.u0 = float(props[12])
        self.u298 = float(props[13])
        self.h298 = float(props[14])
        self.g298 = float(props[15])
        self.cv298 = float(props[16])

        self.file_name = os.path.basename(path)
        self.model_chemistry = 'b3lyp/6-31g(2df,p)'
        self.e0 = self.u0 - self.zpe

    def set_energy(self, e0_new, model_chemistry):
        """
        Change the electronic energy, but make sure that the other
        thermodynamic quantities remain consistent.
        """
        self.u0 += e0_new - self.e0
        self.u298 += e0_new - self.e0
        self.h298 += e0_new - self.e0
        self.g298 += e0_new - self.e0
        # Also modify Cv?
        self.e0 = e0_new
        self.model_chemistry = model_chemistry.lower()

    def make_molpro_input(self, path, method='ccsd(t)-f12a', basis='cc-pvtz-f12', mem=1000):
        """
        Write a Molpro input file for a single-point energy
        calculation given the method, basis set, and memory
        requirement in megawords. Assumes an RHF orbital calculation.
        """
        with open(path, 'w') as f:
            f.write('***,{}\n'.format(os.path.splitext(self.file_name)[0]))
            f.write('memory,{},m\n'.format(mem))
            f.write('geometry={\n')
            f.write(str(self.natoms) + '\n\n')

            xyz = ['{0}  {1[0]: .8f} {1[1]: .8f} {1[2]: .8f}\n'.format(e, c)
                   for e, c in zip(self.elements, self.coords)]
            f.writelines(xyz)

            f.write('}\n\n')
            f.write('basis={}\n'.format(basis))
            f.write('rhf\n')
            f.write(method + '\n')

    def get_enthalpy_of_formation(self, freq_scale_factor=1.0, apply_bond_corrections=True):
        """
        Calculate the enthalpy of formation at 298.15 K.
        Apply bond energy corrections if desired and if
        model chemistry is compatible.
        """
        temperature = 298.15
        mol = pybel.readstring('smi', self.smiles)  # Use OBMol to extract bond types

        # Use RMG molecule to determine if it's linear
        # Assume it's not linear if we can't parse SMILES with RMG
        rmg_mol = None
        try:
            rmg_mol = Molecule().fromSMILES(self.smiles)
        except AtomTypeError:
            try:
                rmg_mol = Molecule().fromSMILES(self.smiles2)
            except AtomTypeError:
                try:
                    rmg_mol = Molecule().fromInChI(self.inchi)
                except AtomTypeError:
                    warnings.warn('Could not determine linearity from RMG molecule in {}'.format(self.file_name))
        if rmg_mol is not None:
            is_linear = rmg_mol.isLinear()
        else:
            is_linear = False

        # Translation
        translation = IdealGasTranslation()

        # Rotation
        if is_linear:
            rotation = LinearRotor()
        else:
            rotation = NonlinearRotor(rotationalConstant=(self.rotational_consts, 'GHz'))

        # Vibration
        freqs = [f * freq_scale_factor for f in self.freqs]  # Apply scale factor
        vibration = HarmonicOscillator(frequencies=(freqs, 'cm^-1'))

        # Group modes
        modes = [translation, rotation, vibration]
        conformer = Conformer(modes=modes)

        # Energy
        e0 = self.e0 * constants.E_h * constants.Na
        zpe = self.zpe * constants.E_h * constants.Na * freq_scale_factor

        # Bring energy to gas phase reference state at 298.15K
        atom_energies = energy_data.atom_energies[self.model_chemistry]
        for element in self.elements:
            e0 -= atom_energies[element] * constants.E_h * constants.Na
            e0 += self.enthalpy_corrections[element] * 4184.0

        if apply_bond_corrections:
            bond_energies = energy_data.bond_energy_corrections[self.model_chemistry]
            for bond in pybel.ob.OBMolBondIter(mol.OBMol):
                bond_symbol_split = [
                    self.atomic_num_dict[bond.GetBeginAtom().GetAtomicNum()],
                    self.bond_symbols[bond.GetBondOrder()],
                    self.atomic_num_dict[bond.GetEndAtom().GetAtomicNum()]
                ]

                try:
                    bond_energy = bond_energies[''.join(bond_symbol_split)]
                except KeyError:
                    bond_energy = bond_energies[''.join(bond_symbol_split[::-1])]  # Try reverse order

                e0 += bond_energy * 4184.0

        conformer.E0 = (e0 + zpe, 'J/mol')
        self.hf298 = conformer.getEnthalpy(temperature) + conformer.E0.value_si

        return self.hf298
