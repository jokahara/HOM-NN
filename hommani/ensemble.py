import os, sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

#import seaborn
#seaborn.set()

import torch
from torch import Tensor
from torch.utils.data import DataLoader
import pytorch_lightning as L

import torchani
from torchani.units import hartree2kcalmol

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from hommani.ani_model import CustomAniNet
from hommani.datasets import CustomDataset, load_data

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_ensemble(model_dir, model_prefix):

    ani2x = torchani.models.ANI2x(periodic_table_index=False)
    model = CustomAniNet(ani2x)
    
    model_files = list(filter(lambda f: f.split('-')[0]==model_prefix, os.listdir(model_dir)))
    for f in model_files:
        i = int(f.split('-')[-1].split('.')[0])
        print('Loading '+f)
        fpath = model_dir+'/'+f
        state_dict = torch.load(fpath, map_location=device)
        # only take
        dict_filter = filter(lambda kv: kv[0].split('.')[0] == '1', state_dict.items())
        state_dict = {k[2:]: v for k,v in dict_filter}

        model.nn[1][i].load_state_dict(state_dict)
        #model[i].neural_networks.load_state_dict(state_dict)

    return model


def evaluate(model, data_loader, return_qbc=True, max_batches=15):

    predicted_energies = []
    true_energies = []
    num_atoms = []
    qbc_factors = []

    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            print(i+1,'/',len(data_loader), end='\r', flush=True)
            species, coordinates, _, true = batch
            num_atoms = np.append(num_atoms, (species >= 0).sum(dim=1, dtype=true.dtype).detach().numpy())

            true_energies = np.append(true_energies, true)
            
            # run validation
            if return_qbc:
                # mean energy predictions and QBC factors
                energies, qbcs = model.energies_qbcs(species, coordinates)
                predicted_energies = np.append(predicted_energies, energies.detach().numpy())
                qbc_factors = np.append(qbc_factors, qbcs.detach().numpy())
            else:
                # ensemble of energies
                member_energies = model.members_energies(species, coordinates).detach().numpy()
                if len(predicted_energies) == 0:
                    predicted_energies = member_energies
                    continue
                predicted_energies = np.append(predicted_energies, member_energies, axis=1)
            
            if i > max_batches:
                break
    model.train()

    return true_energies, predicted_energies, num_atoms, qbc_factors

def sample_qbc(model, data_loader):
    n = len(data_loader.dataset)
    predicted_energies = []
    true_energies = []
    qbc_factors = []
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            species, coordinates, _, true = batch

            # run validation
            true_energies = np.append(true_energies, true)
            
            # mean energy prediction
            energies, qbcs = model.energies_qbcs(species, coordinates)
            predicted_energies = np.append(predicted_energies, energies.detach().numpy())
            qbc_factors = np.append(qbc_factors, qbcs.detach().numpy())
    
    qbc_factors = hartree2kcalmol(qbc_factors)
    
    idx = np.argsort(qbc_factors)
    ncut = int(len(qbc_factors)*0.9)
    cutoff = max(0.23, qbc_factors[idx][ncut:][0])

    idx = qbc_factors > cutoff
    print('Sampled', np.sum(idx), 'clusters with QBC >', cutoff)

    selected = np.arange(n)[qbc_factors > cutoff]
    fail_rate = round(100*len(selected) / n, 2)
    print(str(fail_rate)+'% datapoints failed the test')
    
    #if fail_rate > 5:
    #    queried_idx = np.random.choice(selected, size=size, replace=False)
    #else:
    #    queried_idx = selected
    return selected, fail_rate


def plot():
    print("Using PyTorch {} and Lightning {}".format(torch.__version__, L.__version__))
    
    model_dir = 'Step1' #'pretrained_model'
    model_prefix = 'best'

    model = load_ensemble(model_dir, model_prefix)
    model.eval()

    batch_size = 256
    energy_shifter, sae_dict = torchani.neurochem.load_sae('pretrained_model/sae_linfit.dat', return_dict=True)
    print('Self atomic energies: ', energy_shifter.self_energies)
    model.energy_shifter = energy_shifter

    #_, test1, _, energy_shifter = load_data('../Step1/ACDB_QM7.pkl')
    files = ['am_sa.pkl', 'acid_base.pkl', 'organics.pkl']


    for f in files:
        _, test, _, energy_shifter = load_data('../Step1/'+f)
        test_loader = CustomDataset.get_test_loader(list(test), batch_size)
        
        predicted_energies = []
        true_energies = []
        num_atoms = []
        qbc_factors = []

        with torch.no_grad():
            for i, batch in enumerate(test_loader):
                print(i+1,'/',len(test_loader), end='\r', flush=True)
                species, coordinates, _, true = batch
                num_atoms = np.append(num_atoms, (species >= 0).sum(dim=1, dtype=true.dtype).detach().numpy())

                #shift = energy_shifter((species, true)).energies
                    
                # run validation
                true_energies = np.append(true_energies, true)
                
                # mean energy prediction
                #energies, qbcs = model.energies_qbcs(species, coordinates)
                #predicted_energies = np.append(predicted_energies, energies.detach().numpy())
                #qbc_factors = np.append(qbc_factors, qbcs.detach().numpy())
                #continue

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
        
        loss = np.abs(x-y)/np.sqrt(num_atoms)
        print(str(100*np.sum(loss > 0.4)/len(loss))+'% fail at loss test')
        print(np.sum(loss > 2), 'left outsite range, max = ', np.max(loss))

        print('Sizes:', np.unique(num_atoms))

        # absolute errors
        abs_err = np.abs(x-y)
        max_err = hartree2kcalmol(np.max(np.abs(true_energies-predicted_energies), axis=0))
    
        # standard deviations
        std = np.std(predicted_energies, axis=0)
        qbc_factors = hartree2kcalmol(std / np.sqrt(num_atoms))

        idx = qbc_factors.argsort()
        ncut = int(len(qbc_factors)*0.9)
        cutoff = qbc_factors[idx][ncut:][0]

        # density coloring
        xy = np.vstack([x,y])
        z = gaussian_kde(xy)(xy)
        idx = z.argsort()
        x, y, z = x[idx], y[idx], z[idx]

        #std_err = std / np.sqrt(8)

        plt.subplot(121)
        #plt.errorbar(x, y, yerr=std_err, fmt='.')
        plt.scatter(x, y, alpha=0.5, s=2, label=f)
        #for n in np.unique(num_atoms):
            #idx = (num_atoms==n)
            #plt.scatter(x[idx], y[idx], s=2, alpha=0.5, label=str(int(n)))

        #idx = qbc_factors >= cutoff
        #ncut = int(len(qbc_factors)*0.1)
        #rand = np.arange(ncut)
        #np.random.shuffle(rand)
        #rand = rand[:ncut//5] # 2% off total data
        #plt.scatter(x[idx][rand], y[idx][rand], s=4, c='black')
        
        plt.legend()

        #plt.plot(x, y, '.')
        plt.plot([np.min(x), np.max(x)], [np.min(x), np.max(x)], 'r--', lw=1)
        plt.title('RMSE = ' + str(rmse)[:7] + ' kcal/mol')
        plt.xlabel('Expected (kcal/mol)')
        plt.ylabel('Predicted (kcal/mol)')

        plt.subplot(122)
        
        #for n in np.unique(num_atoms):
            #idx = (num_atoms==n)
            #plt.scatter(abs[idx], qbc_factors[idx], s=2, alpha=0.5, label=str(int(n)))
            #plt.scatter(qbc_factors[idx], max_err[idx]/np.sqrt(num_atoms[idx]), s=2, alpha=0.5, label=str(int(n)))
        plt.scatter(qbc_factors, max_err, s=2, alpha=0.5)

        print(np.sum(qbc_factors > 1.0) / len(qbc_factors))

        #idx = qbc_factors >= cutoff
        #plt.scatter(abs_err[idx][rand], qbc_factors[idx][rand], s=4, c='black')

        plt.title('QBC vs. Max error (per sqrt(N)) ')
        plt.ylabel('Maximum error (kcal/mol)')
        plt.xlabel('QBC (kcal/mol)')

        #plt.plot([0, np.max(abs_err)], [cutoff, cutoff], 'r--')
        plt.ylim(ymin=0)
        plt.xlim(xmin=0)
        """
        plt.hist(loss[loss < 2], bins=50)
        plt.plot([0.4, 0.4], [0, 1000], 'r--')
        plt.title('MAE = ' + str(np.mean(loss))[:7] + ' kcal/mol')
        plt.xlabel('Absolute error / sqrt(N) (kcal/mol)')
        plt.xlim(0,2)
        plt.ylim(0,500)
        """
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    plot()