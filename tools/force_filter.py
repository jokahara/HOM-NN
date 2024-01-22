import os, sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

import seaborn as sns
#sns.set()

import torch
from torch import Tensor
from torch.utils.data import DataLoader
import pytorch_lightning as L

import torchani
from torchani.units import hartree2kcalmol

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from hommani.ani_model import CustomAniNet
from hommani.datasets import DataContainer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def plot():
    print("Using PyTorch {} and Lightning {}".format(torch.__version__, L.__version__))
    
    model_dir = 'Step1' #'pretrained_model'
    model_prefix = 'best'

    model = CustomAniNet.load_ensemble(model_dir, model_prefix)
    model.eval()

    energy_shifter, sae_dict = torchani.neurochem.load_sae('pretrained_model/sae_linfit.dat', return_dict=True)
    print('Self atomic energies: ', energy_shifter.self_energies)
    model.energy_shifter = energy_shifter

    files = ['am_sa.nn.pkl', 'acid_base.nn.pkl', 'organics.nn.pkl', 'monomers.nn.pkl']

    for f in files:
        dc = DataContainer.load_data('Step1/'+f)
        #test_loader = dc.get_test_loader()
        test_loader, val_loader = dc.get_train_val_loaders(0)
        predicted_energies = []
        true_energies = []
        num_atoms = []
        qbc_factors = []
        max_force = []

        with torch.no_grad():
            for i, batch in enumerate(test_loader):
                print(i+1,'/',len(test_loader), end='\r', flush=True)
                species, coordinates, forces, true = batch
                num_atoms = np.append(num_atoms, (species >= 0).sum(dim=1, dtype=true.dtype).detach().numpy())

                idx = (true > 0.5)
                if len(true[idx]) > 0:
                    print(true[idx], species[idx], coordinates[idx])
                    shift = energy_shifter((species[idx], true[idx])).energies
                    print(shift)
                    
                true_energies = np.append(true_energies, true)
                for fi in forces:
                    max_force.append(fi.square().sum(1).max().sqrt().item())
                
                # ensemble of energies
                member_energies = model.members_energies(species, coordinates).detach().numpy()
                if len(predicted_energies) == 0:
                    predicted_energies = member_energies
                    continue
                
                predicted_energies = np.append(predicted_energies, member_energies, axis=1)
        
        x = hartree2kcalmol(true_energies)
        y = hartree2kcalmol(predicted_energies)
        for i in range(8):
            rmse = np.sqrt(np.mean((x-y[i])**2))
            print(i,'rmse:',rmse)
        y = np.mean(y, axis=0)
        rmse = np.sqrt(np.mean((x-y)**2))
        
        print('RMSE = ' + str(rmse) + ' kcal/mol')
        
        # absolute errors
        abs_err = np.abs(x-y)
        max_err = np.max(np.abs(predicted_energies-true_energies), axis=0)

        # standard deviations
        std = np.std(predicted_energies, axis=0)
        qbc_factors = hartree2kcalmol(std / np.sqrt(num_atoms))

        print(str(100*np.sum(qbc_factors > 0.23)/len(qbc_factors))+'% of QBCs are >0.23')

        idx = qbc_factors.argsort()
        ncut = int(len(qbc_factors)*0.9)
        cutoff = qbc_factors[idx][ncut:][0]

        # density coloring
        xy = np.vstack([x,y])
        z = gaussian_kde(xy)(xy)
        idx = z.argsort()
        x, y, z = x[idx], y[idx], z[idx]

        plt.subplot(121)
        #plt.errorbar(x, y, yerr=std_err, fmt='.')
        plt.scatter(x, y, alpha=0.5, s=2, c=max_force, label=f.split('.')[0])
        plt.legend()

        #plt.plot(x, y, '.')
        plt.plot([np.min(x), np.max(x)], [np.min(x), np.max(x)], 'r--', lw=1)
        plt.title('RMSE = ' + str(rmse)[:7] + ' kcal/mol')
        plt.xlabel('Expected (kcal/mol)')
        plt.ylabel('Predicted (kcal/mol)')

        plt.subplot(122)
        
        plt.scatter(max_force, hartree2kcalmol(max_err)/np.sqrt(num_atoms), s=2, alpha=0.5)
        #plt.scatter(max_force, qbc_factors, s=2, alpha=0.5)

        print(np.sum(np.array(max_force) > 1.0) / len(max_force))

        plt.title('Max force vs. Energy error (per sqrt(N)) ')
        plt.ylabel('Maximum error (kcal/mol)')
        plt.xlabel('Force (kcal/mol)')

        #plt.plot([0, np.max(abs_err)], [cutoff, cutoff], 'r--')
        plt.ylim(ymin=0)
        plt.xlim(xmin=0)

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    plot()