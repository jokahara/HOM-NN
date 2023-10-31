
import os, sys
import argparse

import torch
#from torch import Tensor
import pytorch_lightning as L

import torchani
#from torchani.units import hartree2kcalmol

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from hommani.cross_validate import parse_input, train, CustomAniNet
from hommani.ensemble import sample_qbc, load_ensemble, load_data
from hommani.datasets import CustomDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def sample():
    sample_qbc

def main():
    args = parse_input()

    print("Using PyTorch {} and Lightning {}".format(torch.__version__, L.__version__))

    model_dir = 'pretrained_model'
    model_prefix = 'best'

    model = load_ensemble(model_dir, model_prefix)
    
    batch_size = 256
    energy_shifter, sae_dict = torchani.neurochem.load_sae(model_dir+'/sae_linfit.dat', return_dict=True)
    sample_data, energy_shifter = load_data('../test/sa.h5')

    print('Self atomic energies: ', energy_shifter.self_energies)
    model.energy_shifter = energy_shifter

    test_loader = CustomDataset.get_test_loader(sample_data, 256)
    for i in range(8):
        args.i = i # run each model like this
        train(args)

    return


if __name__ == '__main__':
    main()