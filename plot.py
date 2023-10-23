import os, sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

import seaborn
seaborn.set()

import torch
from torch import Tensor
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as L

import torchani
from torchani.units import hartree2kcalmol

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from hommani.ani_model import CustomAniNet
from hommani.datasets import CustomDataset, load_data

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():
    print("Using PyTorch {} and Lightning {}".format(torch.__version__, L.__version__))
    
    latest_checkpoint = 'latest.ckpt'
    best_model = 'Step0/best.pt'

    ani2x = torchani.models.ANI2x(periodic_table_index=False, model_index=0)
    model = CustomAniNet(ani2x)

    model.nn.load_state_dict(torch.load(best_model, map_location='cpu'))
    model.eval()
    model.freeze()

    batch_size = 256

    energy_shifter, sae_dict = torchani.neurochem.load_sae('sae_linfit.dat', return_dict=True)
    
    training, validation, energy_shifter = load_data('/home/jokahara/PhD/Datasets/ACDB_QM7.h5', 0.8)
    #test, _, _ = load_data('../data/4sa.h5', 0.05, energy_shifter, model.species)

    print('Self atomic energies: ', energy_shifter.self_energies)

    val_loader = DataLoader(CustomDataset(validation), batch_size=batch_size,
                            num_workers=8, pin_memory=True)

    predicted_energies = []
    true_energies = []
    num_atoms = []
    for i, batch in enumerate(val_loader):
        print(i,'/',len(val_loader))
        species, coordinates, _, true = batch
        num_atoms = np.append(num_atoms, (species >= 0).sum(dim=1, dtype=true.dtype).detach().numpy())

        # run validation
        true_energies = np.append(true_energies, true)
        predicted_energies = np.append(predicted_energies, model(species, coordinates).detach().numpy())
    
    x = hartree2kcalmol(true_energies)
    y = hartree2kcalmol(predicted_energies)
    rmse = model.mse(Tensor(x),Tensor(y)).sqrt().item()
    
    print('RMSE = ' + str(rmse) + ' kcal/mol')
    
    loss = np.abs(x-y)/np.sqrt(num_atoms)
    print(str(100*np.sum(loss > 0.4)/len(loss))+'% fail at loss test')
    print(np.sum(loss > 2), 'left outsite range, max = ', np.max(loss))

    plt.subplot(121)
    xy = np.vstack([x,y])
    z = gaussian_kde(xy)(xy)
    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]
    plt.scatter(x, y, c=z, s=2)
    #plt.plot(x, y, '.')
    plt.plot([np.min(x), np.max(x)], [np.min(x), np.max(x)], 'r--', lw=1)
    plt.title('RMSE = ' + str(rmse)[:7] + ' kcal/mol')
    plt.xlabel('Expected (kcal/mol)')
    plt.ylabel('Predicted (kcal/mol)')

    plt.subplot(122)
    plt.hist(loss[loss < 2], bins=50)
    plt.plot([0.4, 0.4], [0, 1000], 'r--')
    plt.title('MAE = ' + str(np.mean(loss))[:7] + ' kcal/mol')
    plt.xlabel('Absolute error / sqrt(N) (kcal/mol)')
    plt.xlim(0,2)
    plt.ylim(0,500)

    plt.show()

if __name__ == '__main__':
    main()