import os, sys
import pickle
import time
import numpy as np
from itertools import chain

import torch.cuda
from torch.utils.data import Dataset, DataLoader
import torchani
from torchani.data import TransformableIterable, IterableAdapter


class DataContainer():

    def __init__(self, train, test=None, kfold_idx=None, energy_shifter=None):
        self.train = train
        self.test = test
        self.kfold = kfold_idx
        self.energy_shifter = energy_shifter

    # adapted from torchani.data.load()
    def _load_pickle(path):

        def pkl_files(path):
            """yield file name of all pkl files in a path"""
            if os.path.isdir(path):
                for f in os.listdir(path):
                    f = os.path.join(path, f)
                    yield from pkl_files(f)
            elif os.path.isfile(path) and path.endswith('.pkl'):
                yield path

        def molecules():
            for file in pkl_files(path):
                with open(file, 'rb') as f:
                    df = pickle.load(f)
                for i in df.index:
                    yield df.loc[i]

        def conformations():
            for m in molecules():
                species = m[('xyz','structure')].symbols
                coordinates = m[('xyz','structure')].positions.astype('float32')
                energy = m[('log', 'electronic_energy')]
                ret = {'species': species, 'coordinates': coordinates, 'energies': energy}
                if 'extra' in m.keys():
                    forces = np.array(m[('extra', 'forces')]).astype('float32')
                    ret['forces'] = forces
                yield ret

        return TransformableIterable(IterableAdapter(lambda: conformations()))

    def load_data(file, kfold=1, train_test_split=0.0, energy_shifter=None, 
                species_order=['H', 'C', 'N', 'O', 'S'], autosave=True):
        
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        
        try:
            path = os.path.dirname(os.path.realpath(__file__))
            if not os.path.exists(dspath):
                raise NameError
        except NameError:
            path = os.getcwd()

        dspath = os.path.join(path, file)
        if not os.path.exists(dspath):
            print("Error: Dataset file at "+dspath+" not found."); exit()
        
        kfold_indices = None
        test = None
        
        pickled_dataset_path = dspath.split('.')[0]+'.nn.pkl'
        
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
                data = DataContainer._load_pickle(dspath)

                if energy_shifter == None:
                    energy_shifter = torchani.utils.EnergyShifter(None)
                    data = data.subtract_self_energies(energy_shifter, species_order)\
                                .species_to_indices(species_order)
                else:
                    data = list(data.species_to_indices(species_order))
                    for i, par in enumerate(data):
                        data[i]['energies'] -= energy_shifter.self_energies[par['species']].sum().item()
                    data = TransformableIterable(iter(data))

                # filter out extreme outliers
                filtered = IterableAdapter(lambda: filter(lambda x: abs(x['energies']) < 1.0, data))
                train = TransformableIterable(filtered).shuffle()
                
                if train_test_split > 0.0:
                    if train_test_split >= 1.0:
                        test = train
                        train = None
                    else:
                        train, test = train.split(train_test_split, None)
                if kfold > 1:
                    from sklearn.model_selection import KFold
                    kf = KFold(n_splits=kfold)
                    
                    kfold_indices = kf.split(list(range(len(train))))
                    kfold_indices = list(kfold_indices)
                
                if autosave:
                    DataContainer._save_pickled_dataset(pickled_dataset_path, train, energy_shifter, test, kfold_indices)
                break
            else:
                time.sleep(1) # other ranks wait for rank 0 to pre-process the dataset
        
        return DataContainer(train, test, kfold_indices, energy_shifter)

    def _save_pickled_dataset(path, train, energy_shifter, test=None, kfold_indices=None):
        with open(path, 'wb') as f:
            d = {'train': train, 'self_energies': energy_shifter.self_energies.cpu()}
            if test != None:
                d['test'] = test
            if kfold_indices != None:
                d['kfold'] = kfold_indices
            pickle.dump(d, f)
        return

    def save(self, path):
        DataContainer._save_pickled_dataset(path, self.train, self.energy_shifter, self.test, self.kfold)
        return
    
    def get_train_val_loaders(self, k=0, batch_size=256):
        if self.train == None:
            print('Error! No training data')
            return False
        if self.kfold == None:
            print('Error! No k-fold indices')
            return False
        
        n = len(self.train)
        data = self.train.collate(n).cache()
        data = list(data)[0]

        train_idx, val_idx = self.kfold[k]

        nw = 10 if torch.cuda.is_available() else 8
        train_loader = DataLoader(CustomDataset(data, train_idx), batch_size=batch_size,
                                num_workers=nw, pin_memory=True, shuffle=True)
        val_loader = DataLoader(CustomDataset(data, val_idx), batch_size=batch_size,
                                num_workers=nw, pin_memory=True)
        
        return train_loader, val_loader
    
    def get_test_loader(self, batch_size=256):
        if self.test == None:
            print('Error! No test data')
            return False
        
        n = len(self.test)
        test_idx = list(range(n))
        
        data = self.test.collate(n).cache()
        data = list(data)[0]

        nw = 10 if torch.cuda.is_available() else 8
        test_loader = DataLoader(CustomDataset(data, test_idx), batch_size=batch_size,
                                num_workers=nw, pin_memory=True)
        return test_loader

    def merge(datasets):
        n = len(datasets)

        trains = []
        tests = []
        kfolds = []
        for d in datasets:
            if d.train:
                trains.append(d.train)
                kfolds.append(d.kfold)
            if d.test:
                tests.append(d.test)
        
        if len(trains) > 0:
            train = torchani.data.TransformableIterable(chain(*trains)).cache()
        if len(tests) > 0:
            test = torchani.data.TransformableIterable(chain(*tests)).cache()
        
        indeces = kfolds[0] if len(kfolds) > 0 else None
        if len(kfolds) > 1:
            indeces = kfolds[0]
            lengths = [len(d) for d in trains]
            for i in range(1,n):
                for k in range(len(indeces)):
                    tra, val = kfolds[i][k]
                    tra = np.append(indeces[k][0], tra + np.sum(lengths[:i], dtype=int))
                    val = np.append(indeces[k][1], val + np.sum(lengths[:i], dtype=int))
                    indeces[k] = (tra, val)

        return DataContainer(train, test, indeces, datasets[0].energy_shifter)


class CustomDataset(Dataset):
    def __init__(self, data, indeces=None):
        self.data = data
        self.indeces = indeces if indeces != None else list(range(len(data)))
        
    def __len__(self):
        return len(self.indeces)

    def __getitem__(self, idx):
        i = self.indeces[idx]
        species = self.data['species'][i]
        coordinates = self.data['coordinates'][i].float()
        energies = self.data['energies'][i].float()
        true_forces = self.data['forces'][i].float()
        
        return species, coordinates, true_forces, energies
    
if __name__ == '__main__':
    if len(sys.argv) > 1:
        DataContainer.load_data(sys.argv[-1])