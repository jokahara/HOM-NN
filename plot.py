import os
import matplotlib.pyplot as plt
import numpy as np

import torch
from torch import Tensor
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as L
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

import torchani
from torchani.units import hartree2kcalmol

from hommani.ani_model import CustomAniNet
from hommani.datasets import CustomDataset, load_dataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#from sklearn.model_selection import KFold

def main():
    print("Using PyTorch {} and Lightning {}".format(torch.__version__, L.__version__))
    
    latest_checkpoint = 'checkpoint.ckpt'
    ckpt_path = latest_checkpoint if os.path.isfile(latest_checkpoint) else None
    
    best_model = 'best5e-07.pt'
    ani2x = torchani.models.ANI2x(periodic_table_index=False, model_index=0)
    # Load pretrained ANI-2x model
    model = CustomAniNet(ani2x)
    
    model.nn.load_state_dict(torch.load(best_model, map_location='cpu'))
    
    energy_shifter, sae_dict = torchani.neurochem.load_sae('../sae_linfit.dat', return_dict=True)
    print('Self atomic energies: ', energy_shifter.self_energies)

    batch_size = 256
    #training, validation, self_energies = load_dataset('../data/ACDB_forces.h5', 0.8, model.energy_shifter, model.species)
    test, _, _ = load_dataset('../data/4sa.h5', 0.05, energy_shifter, model.species)

    #train_set = CustomDataset(training)
    #train_loader = DataLoader(train_set, batch_size=batch_size,
    #                          num_workers=10, pin_memory=True)
    val_set = CustomDataset(test)
    val_loader = DataLoader(val_set, batch_size=batch_size,
                            num_workers=8, pin_memory=True)
    #for batch in iter(val_loader):
        
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
        if i>10:
            break
    
    x = hartree2kcalmol(true_energies)
    y = hartree2kcalmol(predicted_energies)
    rmse = model.mse(Tensor(x),Tensor(y)).sqrt().item()
    
    print('RMSE = ' + str(rmse) + ' kcal/mol')
    
    loss = np.abs(x-y)/np.sqrt(num_atoms)
    print(str(100*np.sum(loss > 0.4)/len(loss))+'% fail at loss test')
    print(np.sum(loss > 2), 'left outsite range, max = ', np.max(loss))

    plt.subplot(121)
    plt.plot(x, y, '.')
    plt.plot([np.min(x), np.max(x)], [np.min(x), np.max(x)], 'r--')
    plt.title('RMSE = ' + str(rmse) + ' kcal/mol')
    plt.xlabel('Expected (kcal/mol)')
    plt.ylabel('Predicted (kcal/mol)')

    plt.subplot(122)
    plt.hist(loss[loss < 2], bins=50)
    plt.plot([0.4, 0.4], [0, 1000], 'r--')
    plt.title('MAE = ' + str(np.mean(loss)) + ' kcal/mol/atom')
    plt.xlabel('Absolute error / atom (kcal/mol)')
    plt.xlim(0,2)
    plt.ylim(ymin=0)

    plt.show()

if __name__ == '__main__':
    main()