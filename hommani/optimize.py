# -*- coding: utf-8 -*-
from ase.md.langevin import Langevin
from ase.optimize import BFGS
from ase.io.trajectory import Trajectory
from ase.io import write, read
from ase.io.xyz import read_xyz

import os, sys, torch
from ase import Atoms

import numpy as np
import pandas as pd

# helper function to convert energy unit from Hartree to kcal/mol
from torchani.ase import Calculator
from torchani.units import hartree2kcalmol

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from hommani.ani_model import CustomAniNet
from hommani.datasets import load_data

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def optimize(atoms, bond_lim=0.15, dist_lim=1.7):

    from ase.units import Hartree, eV

    opt = BFGS(atoms, logfile='-')#, trajectory='hist.traj')
    opt.max_steps = 1000
    opt.fmax = 0.001

    # compute initial structure and log the first step
    opt.atoms.get_forces()

    d0 = opt.atoms.get_all_distances()
    internal = (d0 < 1.7)
    external = (d0 > dist_lim)
    has_external = external.sum() > 0
    
    e0 = opt.atoms.get_total_energy()
    # run the algorithm until converged or max_steps reached
    while not opt.converged() and opt.nsteps < opt.max_steps:
        # log the step
        #opt.log()
        #opt.call_observers()
        # compute the next step
        opt.step()
        opt.nsteps += 1

        if opt.fmax > 2:
            break

        d = opt.atoms.get_all_distances()
        if np.abs(d[internal] - d0[internal]).max() > bond_lim:
            break
        if has_external and d[external].min() < dist_lim:
            break
    
    print('Step:', opt.nsteps, 'diff = ', opt.atoms.get_total_energy() - e0)#, np.max(np.abs(d[internal] - d0[internal])), np.min(d[external]))

    return atoms

def opt_sampler(file):
    from ase.units import Hartree, eV
    a = 1 / (Hartree * eV)

    from torchani.neurochem import load_sae
    energy_shifter = load_sae('pretrained_model/sae_linfit.dat')

    ensemble = CustomAniNet.load_ensemble(model_dir = 'pretrained_model', model_prefix = 'best')
    species_order = ensemble.species
    #species_to_tensor = ensemble.species_to_tensor

    model = torch.nn.Sequential(ensemble.nn[0], ensemble.nn[1], energy_shifter)

    # Now let's create a calculator from builtin models:
    calculator = Calculator(species_order, model)

    df = pd.read_pickle(file)
    for i in df.index.values:
        el = df.at[i, ('log', 'electronic_energy')]
        atoms = df.at[i, ('xyz', 'structure')]
        atoms.set_calculator(calculator)
        print(df.at[i, ('info','file_basename')])
        #print(atoms)
        
        atoms = optimize(atoms)

        #opt = BFGS(atoms, logfile='-', trajectory='hist.traj')
        #opt.run(fmax=0.001, steps=500)
        #trajectory = Trajectory('hist.traj')
        #write(df.at[i, ('info','file_basename')]+'.xyz', trajectory)
        
        energy = a*atoms.get_total_energy()
        diff = el - energy
        print('Change:', diff, 'Ha')
        #print(atoms.arrays)
        atoms = Atoms(atoms.symbols, atoms.positions)
        write('test.xyz', atoms)
        df.at[i, ('xyz', 'structure')] = atoms
        df.at[i, ('log', 'electronic_energy')] = energy
        df.at[i, ('info','file_basename')] = df.at[i, ('info','file_basename')]+'_0'
    
    new_file = file.split('/')[-1].split('.')[0]+'_NN.pkl'
    print("Saving to "+new_file)
    df.to_pickle(new_file)

def main():
    _, data, _, energy_shifter = load_data('../Step1/organics.pkl')

    ensemble = CustomAniNet.load_ensemble(model_dir = 'Step1', model_prefix = 'best')
    species_order = ensemble.species
    species_to_tensor = ensemble.species_to_tensor
    print(species_order, species_to_tensor(species_order))

    """
    traj = read('hist.xyz', ':')
    with torch.no_grad():
        d0 = traj[0].get_all_distances()
        internal = (d0 < 1.7)
        external = (d0 > 1.7)
        for atoms in traj:
            
            d = atoms.get_all_distances()
            
            print(np.max(np.abs(d[internal] - d0[internal])), np.min(d[external]))

            species = species_to_tensor(atoms.get_chemical_symbols()).unsqueeze(0)
            coordinates =  torch.tensor([atoms.get_positions()],  dtype=torch.float32)
            member_energies = ensemble.members_energies(species, coordinates)
            #member_energies = energy_shifter((species, member_energies)).energies
            #print(list(np.append([], member_energies.detach().numpy())))
    exit()
    """

    model = torch.nn.Sequential(ensemble.nn[0], ensemble.nn[1], energy_shifter)

    # Now let's create a calculator from builtin models:
    calculator = Calculator(species_order, model)
    
    for i, par in enumerate(data):
        if i < 11:
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
    opt.run(fmax=0.001)

    trajectory = Trajectory('hist.traj')
    write('hist.xyz', trajectory)

if __name__ == '__main__':
    opt_sampler(sys.argv[1])
    #main()