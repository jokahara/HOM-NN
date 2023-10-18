import os
import pickle
import time
#from itertools import chain

import torchani
from torch.utils.data import Dataset

def load_dataset(file,  split=0.8, energy_shifter=None, species_order=['H', 'C', 'N', 'O', 'S']):
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    try:
        path = os.path.dirname(os.path.realpath(__file__))
    except NameError:
        path = os.getcwd()
    dspath = os.path.join(path, file)
    
    pickled_dataset_path = file.split('/')[-1].split('.')[0]+'.pkl'
    # We pickle the dataset after loading to ensure we use the same validation set
    # each time we restart training, otherwise we risk mixing the validation and
    # training sets on each restart.
    for _ in range(5*60):
        if os.path.isfile(pickled_dataset_path):
            print(f'Unpickling preprocessed dataset found in {pickled_dataset_path}')
            with open(pickled_dataset_path, 'rb') as f:
                dataset = pickle.load(f)
            training = dataset['training']
            validation = dataset['validation']
            if energy_shifter == None:
                energy_shifter = torchani.utils.EnergyShifter(None)
            energy_shifter.self_energies = dataset['self_energies']
            break
        elif rank == 0:
            print(f'Processing dataset in {dspath}')
            data = torchani.data.load(dspath, additional_properties=('forces',))
            
            if energy_shifter == None:
                energy_shifter = torchani.utils.EnergyShifter(None)
                data = data.subtract_self_energies(energy_shifter, species_order)\
                            .species_to_indices(species_order)
            else:
                data = list(data.species_to_indices(species_order))
                for i, par in enumerate(data):
                    data[i]['energies'] -= energy_shifter.self_energies[par['species']].sum().item()
                data = torchani.data.TransformableIterable(iter(data))
            training, validation = data.shuffle().split(split, None)
            save_pickled_dataset(pickled_dataset_path, training, validation, energy_shifter)
            
            break
        else:
            time.sleep(1)

    return training, validation, energy_shifter

def save_pickled_dataset(path, training, validation, energy_shifter):
    with open(path, 'wb') as f:
        pickle.dump({'training': training,
                    'validation': validation,
                    'self_energies': energy_shifter.self_energies.cpu()}, f)

class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = list(data.collate(len(data)).cache())[0]
        
    def __len__(self):
        return len(self.data['energies'])

    def __getitem__(self, idx):
        species = self.data['species'][idx]
        coordinates = self.data['coordinates'][idx].float()
        energies = self.data['energies'][idx].float()
        true_forces = self.data['forces'][idx].float()
        
        return species, coordinates, true_forces, energies
    
    def subtract_self_energies(self, energy_shifter):
        self.data
        return

if __name__ == '__main__':
    # device to run the training
    import torch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load pretrained ANI-2x model
    ani2x = torchani.models.ANI2x(periodic_table_index=False, model_index=0).to(device)
    aev_computer = ani2x.aev_computer
    energy_shifter = ani2x.energy_shifter

    species_to_tensor = ani2x.species_to_tensor
    species_order = ani2x.species[:-2]
    nn = ani2x.neural_networks

    dataset, _, e = load_dataset('../data/2sa.h5', split=1.0, energy_shifter=energy_shifter, species_order=species_order)
    print(len(dataset), e)