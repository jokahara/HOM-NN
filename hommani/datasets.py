import os
import pickle
import time
import numpy as np
#from itertools import chain

import torchani
from torch.utils.data import Dataset, DataLoader

def load_data(file, split=1.0, energy_shifter=None, species_order=['H', 'C', 'N', 'O', 'S']):
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    try:
        path = os.path.dirname(os.path.realpath(__file__))
    except NameError:
        path = os.getcwd()
    dspath = os.path.join(path, file)

    if not os.path.exists(dspath):
        print("Error: Dataset file at "+dspath+" not found."); exit()
    
    pickled_dataset_path = file.split('/')[-1].split('.')[0]+'.pkl'
    # We pickle the dataset after loading to ensure we use the same validation set
    # each time we restart training, otherwise we risk mixing the validation and
    # training sets on each restart.
    for _ in range(5*60):
        if os.path.isfile(pickled_dataset_path):
            print(f'Unpickling preprocessed dataset found in {pickled_dataset_path}')
            with open(pickled_dataset_path, 'rb') as f:
                dataset = pickle.load(f)
            data = dataset['data']
            #training = dataset['training']
            #validation = dataset['validation']
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

            data = data.shuffle()
            save_pickled_dataset(pickled_dataset_path, data, energy_shifter)
            break
        else:
            time.sleep(1)

    if split < 1.0:
        training, validation = data.split(split, None)
        return training, validation, energy_shifter
    
    if split > 1:
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=split)
        return data, kf.split(list(range(len(data)))), energy_shifter
    
    return data, energy_shifter

def save_pickled_dataset(path, data, energy_shifter):
    with open(path, 'wb') as f:
        pickle.dump({'data': data, 'self_energies': energy_shifter.self_energies.cpu()}, f)

class CustomDataset(Dataset):
    def __init__(self, data, idx):
        self.data = data
        self.indeces = idx
        
    def __len__(self):
        return len(self.indeces)

    def __getitem__(self, idx):
        i = self.indeces[idx]
        species = self.data['species'][i]
        coordinates = self.data['coordinates'][i].float()
        energies = self.data['energies'][i].float()
        true_forces = self.data['forces'][i].float()
        
        return species, coordinates, true_forces, energies

    def get_train_val_loaders(data, batch_size, kfold, idx):
        n = len(data)
        data = torchani.data.TransformableIterable(iter(data)).collate(n).cache()
        data = list(data)[0]

        for i, (train_idx, val_idx) in enumerate(kfold):
            if i == idx:
                break

        import torch.cuda
        nw = 10 if torch.cuda.is_available() else 8
        train_loader = DataLoader(CustomDataset(data, train_idx), batch_size=batch_size,
                                num_workers=nw, pin_memory=True)
        val_loader = DataLoader(CustomDataset(data, val_idx), batch_size=batch_size,
                                num_workers=nw, pin_memory=True)
        return train_loader, val_loader
    
    def get_test_loader(data, batch_size):
        n = len(data)
        test_idx = list(range(n))
        
        data = torchani.data.TransformableIterable(iter(data)).collate(n).cache()
        data = list(data)[0]

        import torch.cuda
        nw = 10 if torch.cuda.is_available() else 8
        test_loader = DataLoader(CustomDataset(data, test_idx), batch_size=batch_size,
                                num_workers=nw, pin_memory=True)
        return test_loader

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

    dataset, e = load_data('../data/2sa.h5', split=1.0, energy_shifter=energy_shifter, species_order=species_order)
    print(len(dataset), e)