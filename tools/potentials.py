import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from rdkit import Chem
from rdkit.Chem import AllChem, rdMolTransforms

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import torch
from torch import Tensor

from torchani.ase import Calculator
from ase.io import read

from hommani.ani_model import CustomAniNet

def set_bond_length(molecule, bond_index, new_length):
    mol_copy = Chem.Mol(molecule)
    conf = mol_copy.GetConformer()
    rdMolTransforms.SetBondLength(conf, bond_index[0], bond_index[1], new_length)
    return mol_copy

def set_bond_angle(molecule, atom_indices, new_angle_deg):
    mol_copy = Chem.Mol(molecule)
    conf = mol_copy.GetConformer()
    rdMolTransforms.SetAngleDeg(conf, atom_indices[0], atom_indices[1], atom_indices[2], new_angle_deg)
    return mol_copy

def plot_potential(t, x1, x2):

    model_dir = '../Step1' #'pretrained_model'
    model_prefix = 'best'
    ensemble = CustomAniNet.load_ensemble(model_dir, model_prefix)

    #import torchani
    #ensemble = torchani.models.ANI2x()

    model = torch.nn.Sequential(ensemble.nn[0], ensemble.nn[1], ensemble.energy_shifter)
    #calculator = Calculator(ensemble.species, model)

    plt.subplot(121)

    mols = read(t+'-bonds.xyz', ':')
    e = []
    for atoms in mols:
        species = ensemble.species_to_tensor(atoms.get_chemical_symbols()).unsqueeze(0)
        coordinates = torch.tensor([atoms.positions], requires_grad=False, device='cpu', dtype=torch.float32)
        energy = ensemble.members_energies(species, coordinates, shift=True)
        #atoms.set_calculator(calculator)
        #energy = atoms.get_potential_energy()
        e.append(energy.detach().numpy())
    
    std = np.std(e, axis=1)
    e = np.mean(e, axis=1)

    plt.plot(x1, e)
    plt.plot(x1, e+std/np.sqrt(8), 'b--', alpha=0.5)
    plt.plot(x1, e-std/np.sqrt(8), 'b--', alpha=0.5)

    df = pd.read_pickle('collectionBOND.pkl')
    df = df.sort_values(('info', 'file_basename'))
    plt.plot(x1, df[('log', 'electronic_energy')].values, 'r-')


    plt.ylabel('Energy (Ha)')
    plt.xlabel('Bond length (Å)')

    plt.subplot(122)
    
    mols = read(t+'-angle.xyz', ':')
    e = []
    for atoms in mols:
        species = ensemble.species_to_tensor(atoms.get_chemical_symbols()).unsqueeze(0)
        coordinates = torch.tensor([atoms.positions], requires_grad=False, device='cpu', dtype=torch.float32)
        energy = ensemble.members_energies(species, coordinates, shift=True)
        #atoms.set_calculator(calculator)
        #energy = atoms.get_potential_energy()
        e.append(energy.detach().numpy())
    
    std = np.std(e, axis=1)
    e = np.mean(e, axis=1)


    plt.plot(x2, e)
    plt.plot(x2, e+std/np.sqrt(8), 'b--', alpha=0.5)
    plt.plot(x2, e-std/np.sqrt(8), 'b--', alpha=0.5)
    plt.xlabel('Bond angle (degree)')

    df = pd.read_pickle('collectionANGLE.pkl')
    df = df.sort_values(('info', 'file_basename'))
    plt.plot(x2, df[('log', 'electronic_energy')].values, 'r-')


    plt.show()
    plt.close()

def create_water(bond_lengths, bond_angles):
    # Example molecule (water)
    mol_smiles = 'O'
    mol = Chem.MolFromSmiles(mol_smiles)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol)

    print("Original Molecule:")
    print(Chem.MolToXYZBlock(mol))

    mol = set_bond_length(mol, (0, 1), 0.96)
    mol = set_bond_length(mol, (0, 2), 0.96)
    mol = set_bond_angle(mol, (1, 0, 2), 104.5)
    
    with open("H2O-bonds.xyz", 'w') as f:
        for bond_length in bond_lengths:
            modified_mol = set_bond_length(mol, (0, 1), bond_length)
            print(Chem.MolToXYZBlock(modified_mol), file=f,end='')

    with open("H2O-angle.xyz", 'w') as f:
        for bond_angle in bond_angles: # Set the new bond angle in degrees
            modified_mol = set_bond_angle(mol, (1, 0, 2), bond_angle)
            print(Chem.MolToXYZBlock(modified_mol), file=f, end='')

def create_ethanol(bond_lengths, bond_angles):
    mol_smiles = 'CCO'
    mol = Chem.MolFromSmiles(mol_smiles)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol)

    print("Original Molecule:")
    print(Chem.MolToXYZBlock(mol))
    
    with open("ethanol-bonds.xyz", 'w') as f:
        for bond_length in bond_lengths:
            modified_mol = set_bond_length(mol, (1, 2), bond_length)
            print(Chem.MolToXYZBlock(modified_mol), file=f,end='')

    with open("ethanol-angle.xyz", 'w') as f:
        for bond_angle in bond_angles: # Set the new bond angle in degrees
            modified_mol = set_bond_angle(mol, (0, 1, 2), bond_angle)
            print(Chem.MolToXYZBlock(modified_mol), file=f, end='')

if __name__ == "__main__":
    
    lengths = np.arange(0.4, 4.0, 0.01)
    angles = np.arange(60.,181.)
    #create_water(lengths, angles)
    create_ethanol(lengths, angles)
    plot_potential('ethanol', lengths, angles)