#!/usr/bin/env python
# -*- coding:utf-8 -*-

import os

import numpy as np

import rmgpy.constants as constants
from rmgpy.statmech import Conformer, IdealGasTranslation, LinearRotor, NonlinearRotor, HarmonicOscillator
from rmgpy.molecule import Molecule


class Structure(object):
    # Atomic reference energies at 0K in Hartree
    atom_energies = {
        'ccsd(t)-f12/cc-pvdz-f12': {  # From Cantherm
            'H': -0.499811124128,
            'N': -54.526406291655,
            'O': -74.995458316117,
            'C': -37.788203485235,
            'S': -397.663040369707
        },
        'ccsd(t)-f12/cc-pvtz-f12': {  # From Cantherm
            'H': -0.499946213243,
            'N': -54.53000909621,
            'O': -75.004127673424,
            'C': -37.789862146471,
            'S': -397.675447487865
        },
        'b3lyp/6-31g(2df,p)': {  # From 134k
            'H': -0.500273,
            'C': -37.846772,
            'N': -54.583861,
            'O': -75.064579,
            'F': -99.718730,
        },
    }

    # Bond energy corrections from Cantherm
    bond_energy_corrections = {
        'ccsd(t)-f12/cc-pvdz-f12': {
            'C-H': -0.46, 'C-C': -0.68, 'C=C': -1.90, 'C#C': -3.13,
            'O-H': -0.51, 'C-O': -0.23, 'C=O': -0.69, 'O-O': -0.02, 'C-N': -0.67,
            'C=N': -1.46, 'C#N': -2.79, 'N-O': 0.74, 'N_O': -0.23, 'N=O': -0.51,
            'N-H': -0.69, 'N-N': -0.47, 'N=N': -1.54, 'N#N': -2.05, 'S-H': 0.87,
            'C-S': 0.42, 'C=S': 0.51, 'S-S': 0.86, 'O-S': 0.23, 'O=S': -0.53,
            'O=S=O': 1.95
        },
        'ccsd(t)-f12/cc-pvtz-f12': {
            'C-H': -0.09, 'C-C': -0.27, 'C=C': -1.03, 'C#C': -1.79,
            'O-H': -0.06, 'C-O': 0.14, 'C=O': -0.19, 'O-O': 0.16, 'C-N': -0.18,
            'C=N': -0.41, 'C#N': -1.41, 'N-O': 0.87, 'N_O': -0.09, 'N=O': -0.23,
            'N-H': -0.01, 'N-N': -0.21, 'N=N': -0.44, 'N#N': -0.76, 'S-H': 0.52,
            'C-S': 0.13, 'C=S': -0.12, 'S-S': 0.30, 'O-S': 0.15, 'O=S': -2.61,
            'O=S=O': 0.27,
        },
        'b3lyp/6-31g(2df,p)': {
            'C-H': 0.25, 'C-C': -1.89, 'C=C': -0.40, 'C#C': -1.50,
            'O-H': -1.09, 'C-O': -1.18, 'C=O': -0.01, 'N-H': 1.36, 'C-N': -0.44,
            'C#N': 0.22, 'C-S': -2.35, 'O=S': -5.19, 'S-H': -0.52,
        }
    }

    # Enthalpy correction terms based on atomic heats of formation in kcal/mol
    enthalpy_corrections = {
        'H': 50.62,
        'C': 169.73,
        'N': 111.49,
        'O': 57.95,
        'F': 17.42,
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
        self.smiles = None            # Row natoms+3
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
        self.smiles = lines[self.natoms+3].split()[1]
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
        mol = Molecule().fromSMILES(self.smiles)

        # Translation
        translation = IdealGasTranslation()

        # Rotation
        if mol.isLinear():
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
        atom_energies = self.atom_energies[self.model_chemistry]
        for element in self.elements:
            e0 -= atom_energies[element] * constants.E_h * constants.Na
            e0 += self.enthalpy_corrections[element] * 4184.0

        if apply_bond_corrections:
            bond_energies = self.bond_energy_corrections[self.model_chemistry]
            for bond in mol.getAllEdges():
                # Sorting in this way makes sure that the order is C, N, O, H
                atom_symbols = sorted([bond.atom1.symbol, bond.atom2.symbol])
                try:
                    h_index = atom_symbols.index('H')
                except ValueError:
                    pass
                else:
                    other_symbol = atom_symbols[1-h_index]
                    atom_symbols = [other_symbol, 'H']
                bond_symbol = atom_symbols[0] + self.bond_symbols[bond.order] + atom_symbols[1]
                e0 += bond_energies[bond_symbol] * 4184.0

        conformer.E0 = (e0 + zpe, 'J/mol')

        return conformer.getEnthalpy(temperature) + conformer.E0.value_si
