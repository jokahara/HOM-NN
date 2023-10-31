# -*- coding: utf-8 -*-
from ase.md.langevin import Langevin
from ase.optimize import BFGS
from ase.io.trajectory import Trajectory
from ase.io import write, read
from ase.io.xyz import read_xyz

import torch
from ase import Atoms
import torch.utils.tensorboard

import numpy as np
import pandas as pd

# helper function to convert energy unit from Hartree to kcal/mol
from torchani.ase import Calculator
from torchani.units import hartree2kcalmol

from ensemble import load_ensemble, load_data

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():

    data, energy_shifter = load_data('../ACDB_QM7.pkl')

    ensemble = load_ensemble(model_dir = 'pretrained_model', model_prefix = 'best')
    species_order = ensemble.species
    species_to_tensor = ensemble.species_to_tensor
    print(species_order, species_to_tensor(species_order))

    """
    traj = read('hist1.xyz', ':')
    with torch.no_grad():
        for atoms in traj:
            species = species_to_tensor(atoms.get_chemical_symbols()).unsqueeze(0)
            coordinates =  torch.tensor([atoms.get_positions()],  dtype=torch.float32)
            member_energies = ensemble.members_energies(species, coordinates)
            print(list(np.append([], member_energies.detach().numpy())))
    exit()
    """

    model = torch.nn.Sequential(ensemble.nn[0], ensemble.nn[1])#, energy_shifter)

    # Now let's create a calculator from builtin models:
    calculator = Calculator(species_order, model)

    for i, par in enumerate(data):
        if i < 10:
            continue
        symbols = np.array(species_order)[par['species']]
        positions = [(x,y,z) for x,y,z in par['coordinates']]
        atoms = Atoms(symbols, positions)
        break

    atoms.set_calculator(calculator)
    
    print(len(atoms.positions), "atoms in the cluster")
    print(atoms.get_chemical_symbols())
    print(species_to_tensor(atoms.get_chemical_symbols()))
    print(energy_shifter.self_energies[species_to_tensor(atoms.get_chemical_symbols())].sum())

    species = species_to_tensor(atoms.get_chemical_symbols()).unsqueeze(0)
    coordinates = torch.tensor([atoms.positions], requires_grad=False, device=device, dtype=torch.float32)
    energy = model((species, coordinates)).energies.item()

    from ase.units import Hartree, eV
    print('1 Hartree =', Hartree * eV, 'eV')
    print(Hartree * energy, energy_shifter((species, energy)).energies)


    ###############################################################################
    # Now let's minimize the structure:
    print("Begin minimizing...")
    opt = BFGS(atoms, logfile='-', trajectory='hist.traj')
    opt.run(fmax=0.001, steps=1000)

    trajectory = Trajectory('hist.traj')
    
    write('hist.xyz', trajectory, append=True)

if __name__ == '__main__':
    main()