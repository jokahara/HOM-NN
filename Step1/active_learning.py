
import os, sys
import argparse
import numpy as np

import torch
#from torch import Tensor
import pytorch_lightning as L

import torchani
from torchani.data import TransformableIterable
from torchani.units import hartree2kcalmol

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from sklearn.model_selection import KFold
from hommani.cross_validate import cross_validate, parse_input
from hommani.ani_model import CustomAniNet
from hommani.datasets import CustomDataset, load_data, save_pickled_dataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def sample_qbc_max(model, data_loader, qbc_cutoff=0.23, max_cutoff=1.5):
    
    n = len(data_loader.dataset)
    predicted_energies = []
    true_energies = []
    num_atoms = []

    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            species, coordinates, _, true = batch
            num_atoms = np.append(num_atoms, (species >= 0).sum(dim=1, dtype=true.dtype).detach().numpy())

            # run validation
            true_energies = np.append(true_energies, true)

            # ensemble of energies
            member_energies = model.members_energies(species, coordinates).detach().numpy()

            if i == 0:
                predicted_energies = member_energies
                continue
            predicted_energies = np.append(predicted_energies, member_energies, axis=1)
    
    x = hartree2kcalmol(true_energies)
    y = hartree2kcalmol(predicted_energies)

    max_err = np.max(np.abs(x-y), axis=0)/np.sqrt(num_atoms)
    qbc_factors = np.std(predicted_energies, axis=0) / np.sqrt(num_atoms)

    idx = np.any([qbc_factors > qbc_cutoff, max_err > max_cutoff], axis=0)

    selected = np.arange(n)[idx]
    fail_rate = round(100*len(selected) / n, 2)
    print('Sampled '+str(fail_rate)+'% datapoints above cutoffs')

    #if fail_rate > 5:
    #    queried_idx = np.random.choice(selected, size=size, replace=False)
    #else:
    #    queried_idx = selected
    return selected

def sample_qbc(model, data_loader):
    n = len(data_loader.dataset)
    predicted_energies = []
    true_energies = []
    qbc_factors = []
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            species, coordinates, _, _ = batch

            # mean energy prediction
            energies, qbcs = model.energies_qbcs(species, coordinates)
            predicted_energies = np.append(predicted_energies, energies.detach().numpy())
            # std / sqrt(num_atoms)
            qbc_factors = np.append(qbc_factors, qbcs.detach().numpy())
    
    qbc_factors = hartree2kcalmol(qbc_factors)
    
    idx = np.argsort(qbc_factors)
    ncut = int(len(qbc_factors)*0.95)
    cutoff = 0.23# max(0.23, qbc_factors[idx][ncut:][0])

    idx = qbc_factors > cutoff
    print('Sampled', np.sum(idx), 'clusters with QBC >', cutoff)

    selected = np.arange(n)[idx]
    fail_rate = round(100*len(selected) / n, 2)
    print(str(fail_rate)+'% datapoints failed the test')

    #if fail_rate > 5:
    #    queried_idx = np.random.choice(selected, size=size, replace=False)
    #else:
    #    queried_idx = selected
    return selected


def sample():
    print("Using PyTorch {} and Lightning {}".format(torch.__version__, L.__version__))
    
    model_dir = 'Step1'
    model_prefix = 'best'

    model = CustomAniNet.load_ensemble(model_dir, model_prefix)
    files = sys.argv[1:]

    for f in files:
        train, test, kfold, energy_shifter = load_data('../'+f)
        data_loader = CustomDataset.get_test_loader(list(test), 256)

        # sample data above the error threshold
        sample_idx = sample_qbc_max(model, data_loader)
        if len(kfold) > len(sample_idx):
            print("Too few sampled datapoints. Skipping.")
            continue

        kf = KFold(n_splits=len(kfold))
        test = list(test)
        
        # move sampled data to training set
        sampled = TransformableIterable(iter([test[i] for i in sample_idx])).cache()
        test = np.delete(test, sample_idx, axis=0)
        test = TransformableIterable(iter(test)).cache()

        kfold_indices = list(kf.split(list(range(len(sampled)))))
        train, kfold = CustomDataset.merge_datasets([train, sampled], [kfold, kfold_indices])
        
        print("Saving "+f)
        save_pickled_dataset(f, train, energy_shifter, test, kfold)

    return


if __name__ == '__main__':
    globals()[sys.argv[1]]()
    #sample()
    exit()
    args = parse_input()
    for i in range(8):
        args.i = i
        cross_validate(args)
