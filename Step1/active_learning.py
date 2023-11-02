
import os, sys
import argparse
import numpy as np

import torch
#from torch import Tensor
import pytorch_lightning as L

import torchani
from torchani.data import TransformableIterable, IterableAdapter
#from torchani.units import hartree2kcalmol

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from sklearn.model_selection import KFold
from hommani.cross_validate import cross_validate, parse_input
from hommani.ensemble import load_ensemble, sample_qbc
from hommani.datasets import CustomDataset, load_data, save_pickled_dataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def sample():
    print("Using PyTorch {} and Lightning {}".format(torch.__version__, L.__version__))

    model_dir = 'Step1'
    model_prefix = 'best'

    model = load_ensemble(model_dir, model_prefix)
    files = sys.argv[1:]

    for f in files:
        train, test, kfold, energy_shifter = load_data('../'+f)
        test_loader = CustomDataset.get_test_loader(list(test), 256)

        # sample data above the error threshold
        sample_idx, fail_rate = sample_qbc(model, test_loader)
        
        kf = KFold(n_splits=len(kfold))
        test = list(test)
        
        # move sampled data to training set
        sampled = TransformableIterable(iter([test[i] for i in sample_idx])).cache()
        test = np.delete(test, sample_idx, axis=0)
        test = TransformableIterable(iter(test)).cache()

        kfold_indices = list(kf.split(list(range(len(sampled)))))
        train, kfold = CustomDataset.merge_datasets([train, sampled], [kfold, kfold_indices])
        
        save_pickled_dataset(f, train, energy_shifter, test, kfold)

    return


if __name__ == '__main__':
    args = parse_input()
    sample()
    for i in range(8):
        args.i = i
        cross_validate(args)
