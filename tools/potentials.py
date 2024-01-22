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

def set_dihedral_angle(molecule, atom_indices, new_angle_deg):
    mol_copy = Chem.Mol(molecule)
    conf = mol_copy.GetConformer()
    rdMolTransforms.SetDihedralDeg(conf, atom_indices[0], atom_indices[1], atom_indices[2], atom_indices[3], new_angle_deg)
    return mol_copy

def plot_potential(t, x1, x2):

    model_dir = '../Step0'
    model_prefix = 'best'
    ensemble = CustomAniNet.load_ensemble(model_dir, model_prefix)

    #model = torch.nn.Sequential(ensemble.nn[0], ensemble.nn[1], ensemble.energy_shifter)
    import torchani
    ani2x = torchani.models.ANI2x()
    calculator = Calculator(ani2x.species, ani2x)

    plt.subplot(121)

    mols = read(t+'-bonds.xyz', ':')
    e = []
    e_ani = []
    for atoms in mols:
        species = ensemble.species_to_tensor(atoms.get_chemical_symbols()).unsqueeze(0)
        coordinates = torch.tensor([atoms.positions], requires_grad=False, device='cpu', dtype=torch.float32)
        energy = ensemble.members_energies(species, coordinates, shift=True).detach().numpy()
        e.append(energy)

        #_, force = ensemble(species, coordinates, return_forces=True)
        #force = force.detach().numpy()
        #print(np.mean(energy), np.sqrt(np.mean(force**2)))

        atoms.set_calculator(calculator)
        energy = atoms.get_potential_energy() * 0.0367502
        e_ani.append(energy)
        
        
    
    """
    e = np.array(e)
    for i in range(8):
        plt.plot(x1, e[:,i], '--', alpha=0.5)
    """
    std = np.std(e, axis=1)
    e = np.mean(e, axis=1)

    plt.plot(x1, e, 'b-', label='NN')
    plt.plot(x1, e+std/np.sqrt(8), 'b--', alpha=0.5)
    plt.plot(x1, e-std/np.sqrt(8), 'b--', alpha=0.5)
    if os.path.exists(t+'BOND.pkl'):
        df = pd.read_pickle(t+'BOND.pkl')
        df = df.sort_values(('info', 'file_basename'))
        plt.plot(x1, df[('log', 'electronic_energy')].values, 'r-', label='B97-3c')

        #for i in range(len(df)):
        #    force = np.array(df[('extra', 'forces')].values[i])
        #    force = np.sqrt(np.mean(force**2))
        #    print(x1[i], df[('log', 'electronic_energy')].values[i], force)

    plt.plot(x1, e_ani, 'g-', label='ANI-2x')

    plt.ylabel('Energy (Ha)')
    plt.xlabel('Bond length (Å)')
    plt.legend()


    plt.subplot(122)
    
    mols = read(t+'-angle.xyz', ':')
    e = []
    e_ani = []
    for atoms in mols:
        species = ensemble.species_to_tensor(atoms.get_chemical_symbols()).unsqueeze(0)
        coordinates = torch.tensor([atoms.positions], requires_grad=False, device='cpu', dtype=torch.float32)
        energy = ensemble.members_energies(species, coordinates, shift=True).detach().numpy()
        e.append(energy)
        
        atoms.set_calculator(calculator)
        energy = atoms.get_potential_energy() * 0.0367502
        e_ani.append(energy)

    """
    e = np.array(e)
    for i in range(8):
        plt.plot(x2, e[:,i], '--', alpha=0.5)

    """
    std = np.std(e, axis=1)
    e = np.mean(e, axis=1)

    plt.plot(x2, e, 'b-')
    plt.plot(x2, e+std/np.sqrt(8), 'b--', alpha=0.5)
    plt.plot(x2, e-std/np.sqrt(8), 'b--', alpha=0.5)

    plt.xlabel('Bond angle (degree)')

    if os.path.exists(t+'ANGLE.pkl'):
        df = pd.read_pickle(t+'ANGLE.pkl')
        df = df.sort_values(('info', 'file_basename'))
        plt.plot(x2, df[('log', 'electronic_energy')].values, 'r-')

    plt.plot(x2, e_ani, 'g-')

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
    AllChem.MMFFOptimizeMolecule(mol)

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

def create_sa(bond_lengths, bond_angles):
    
    mol_smiles = 'OS(=O)(=O)O'
    mol = Chem.MolFromSmiles(mol_smiles)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol)

    conf = mol.GetConformer()
    mol2 = Chem.MolFromXYZFile('sa.xyz')
    conf2 = mol2.GetConformer()
    
    for i in range(mol.GetNumAtoms()):
        conf.SetAtomPosition(i,conf2.GetAtomPosition(i))

    print("Original Molecule:")
    print(Chem.MolToXYZBlock(mol))
    
    with open("sa-bonds.xyz", 'w') as f:
        for bond_length in bond_lengths:
            modified_mol = set_bond_length(mol, (1, 2), bond_length)
            print(Chem.MolToXYZBlock(modified_mol), file=f,end='')

    with open("sa-angle.xyz", 'w') as f:
        for bond_angle in bond_angles: # Set the new bond angle in degrees
            modified_mol = set_bond_angle(mol, (0, 1, 2), bond_angle)
            print(Chem.MolToXYZBlock(modified_mol), file=f, end='')

def ammonia_pes():

    mol_smiles = 'N'
    mol = Chem.MolFromSmiles(mol_smiles)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol)

    conf = mol.GetConformer()
    mol2 = Chem.MolFromXYZFile('am.xyz')
    conf2 = mol2.GetConformer()

def sa_pes():
    mol_smiles = 'OS(=O)(=O)O'
    mol = Chem.MolFromSmiles(mol_smiles)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol)

    conf = mol.GetConformer()
    mol2 = Chem.MolFromXYZFile('sa.xyz')
    conf2 = mol2.GetConformer()


if __name__ == "__main__":
    os.chdir('tools')
    lengths = np.arange(0.4, 4.0, 0.01)
    angles = np.arange(60.,181.)

    #create_sa(lengths, angles)
    #create_water(lengths, angles)
    #create_ethanol(lengths, angles)
    plot_potential('ethanol', lengths, angles)