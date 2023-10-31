import os
import pickle
import time
import numpy as np
from itertools import chain

import torchani
from torch.utils.data import Dataset, DataLoader

def load_data(file, kfold=1, train_test_split=0.0, energy_shifter=None, species_order=['H', 'C', 'N', 'O', 'S']):
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
    
    kfold_indices = None
    test = None

    if dspath[-3:] == 'pkl':
        pickled_dataset_path = dspath
    else:
        pickled_dataset_path = dspath.split('/')[-1].split('.')[0]+'.pkl'
    # We pickle the dataset after loading to ensure we use the same validation set
    # each time we restart training, otherwise we risk mixing the validation and
    # training sets on each restart.
    for _ in range(5*60):
        if os.path.isfile(pickled_dataset_path):
            print(f'Unpickling preprocessed dataset found in {pickled_dataset_path}')
            with open(pickled_dataset_path, 'rb') as f:
                dataset = pickle.load(f)
            train = dataset['train']
            if 'kfold' in dataset.keys():
                kfold_indices = dataset['kfold']
            if 'test' in dataset.keys():
                test = dataset['test']
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

            train = data.shuffle()
            if train_test_split > 0.0:
                train, test = train.split(train_test_split, None)
            if kfold > 1:
                from sklearn.model_selection import KFold
                kf = KFold(n_splits=kfold)
                
                kfold_indices = kf.split(list(range(len(train))))
                kfold_indices = list(kfold_indices)
            
            save_pickled_dataset(pickled_dataset_path, train, energy_shifter, test, kfold_indices)
            break
        else:
            time.sleep(1) # other ranks wait for rank 0 to process the dataset

    return train, test, kfold_indices, energy_shifter

def save_pickled_dataset(path, train, energy_shifter, test=None, kfold=None):
    with open(path, 'wb') as f:
        d = {'train': train, 'self_energies': energy_shifter.self_energies.cpu()}
        if test != None:
            d['test'] = test
        if kfold != None:
            d['kfold'] = kfold
        pickle.dump(d, f)

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

    def get_train_val_loaders(data, batch_size, train_val_idx):
        n = len(data)
        data = torchani.data.TransformableIterable(iter(data)).collate(n).cache()
        data = list(data)[0]

        train_idx, val_idx = train_val_idx

        import torch.cuda
        nw = 10 if torch.cuda.is_available() else 8
        train_loader = DataLoader(CustomDataset(data, train_idx), batch_size=batch_size,
                                num_workers=nw, pin_memory=True, shuffle=True)
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

    def merge_datasets(datasets, kfolds=None):
        n = len(datasets)
        data = torchani.data.TransformableIterable(chain(*datasets)).cache()
        
        indeces = None
        if kfolds:
            indeces = kfolds[0]
            lengths = [len(d) for d in datasets]
            for i in range(1,n):
                for k in range(len(indeces)):
                    train, val = kfolds[i][k]
                    train = np.append(indeces[k][0], train + np.sum(lengths[:i], dtype=int))
                    val = np.append(indeces[k][1], val + np.sum(lengths[:i], dtype=int))
                    indeces[k] = (train, val)
        
        return data, indeces