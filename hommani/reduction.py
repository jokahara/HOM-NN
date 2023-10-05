import os
import argparse
from itertools import chain
import numpy as np

import torch
from torch import Tensor
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as L
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

import torchani
from torchani.data import TransformableIterable, IterableAdapter
from torchani.units import hartree2kcalmol

from ani_model import CustomAniNet
from hommani.datasets import CustomDataset, load_dataset, save_pickled_dataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# pick new data from pool to training set
def query(model, pool, n):
    mae = torch.nn.L1Loss(reduction='none')
    batch_size = 1000
    collated_pool = TransformableIterable(
            IterableAdapter(lambda: (x for x in pool))
            ).collate(batch_size).cache()
    n = len(pool)
    error_list = np.array([])

    model.train(False)
    with torch.no_grad():
        for properties in collated_pool:
            species = properties['species'].to(device)
            coordinates = properties['coordinates'].to(device).float()
            true_energies = properties['energies'].to(device).float()
            _, predicted_energies = model((species, coordinates))

            num_atoms = (species >= 0).sum(dim=1, dtype=true_energies.dtype)
            abs_error = (mae(predicted_energies, true_energies) / num_atoms.sqrt())
            error_list = np.append(error_list, abs_error.cpu())
    model.train(True)
    
    selected = np.arange(n)[hartree2kcalmol(error_list) > 0.2]
    fail_rate = round(100*len(selected) / n,2)
    print(str(fail_rate)+'% datapoints failed the test')
    
    if fail_rate > 5:
        queried_idx = np.random.choice(selected, size=n, replace=False)
    else:
        queried_idx = selected
    return queried_idx, fail_rate

def filter_overlaps(data, limit=1e-6):
    # remove datapoints with very close energies
    energies = []
    for i, par in enumerate(data):
        species = par['species']
        energies.append((par['energies']).item())

    order = np.argsort(energies)
    energies = np.array(energies)[order]
    data = np.array(data)[order]

    # find duplicate energies in training data
    duplicates = []
    prev = data[0]['energies']-1
    for i, par in enumerate(data):
        E = par['energies']
        if (E - prev) < limit:
            duplicates.append(i)
        if E > 5.0: # also remove extreme outliers
            duplicates.append(i)
        prev = E

    print('Found', len(duplicates), 'duplicates')
    leftovers = data[duplicates]
    data = np.delete(data, duplicates)
    return data, leftovers

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus', default=1, type=int, metavar='N',
                        help='number of GPUs per node')
    parser.add_argument('--nodes', default=1, type=int, metavar='N',
                        help='number of nodes')
    parser.add_argument('--epochs', default=10, type=int, metavar='N',
                        help='maximum number of epochs to run')
    parser.add_argument('--limit', default=1e-6, type=float, metavar='N',
                        help='energy filter limit')
    args = parser.parse_args()

    print("Using PyTorch {} and Lightning {}".format(torch.__version__, L.__version__))
    
    latest_checkpoint = 'checkpoint'
    ckpt_path = latest_checkpoint if os.path.isfile(latest_checkpoint) else None
    
    ani2x = torchani.models.ANI2x(periodic_table_index=False, model_index=0)
    # Load pretrained ANI-2x model
    model = CustomAniNet(ani2x)
    #model.nn.load_state_dict(torch.load('best.pt', map_location='cpu'))
    
    energy_shifter, sae_dict = torchani.neurochem.load_sae('../sae_linfit.dat', return_dict=True)
    print('Self atomic energies: ', energy_shifter.self_energies)

    training1, validation1, _ = load_dataset('../data/ACDB_forces.h5', 0.8, energy_shifter, model.species)
    pool, _, _ = load_dataset('../data/2sa.h5', 1.0, energy_shifter, model.species)
    test, _, _ = load_dataset('../data/4sa.h5', 0.1, energy_shifter, model.species)

    limit = args.limit
    model.best_model_checkpoint = 'best'+str(limit)+'.pt'
    training2, pool = filter_overlaps(list(pool), limit=limit)

    import random; random.seed(1)
    training2, validation2 = TransformableIterable(iter(training2)).shuffle().split(0.8, None)

    training = TransformableIterable(chain(training1, training2)).cache()
    validation = TransformableIterable(chain(validation1, validation2)).cache()

    print('Using energy limit of', limit)
    print('Training size:', len(training), 'Validation size:', len(validation))

    if os.path.isfile(model.best_model_checkpoint):
        print('Restarting from', model.best_model_checkpoint)
        model.nn.load_state_dict(torch.load(model.best_model_checkpoint, map_location='cpu'))

    n_queries = 10
    for idx in range(n_queries):
        #print('Query:', idx)
        batch_size = 256
        train_set = CustomDataset(training)
        train_loader = DataLoader(train_set, batch_size=batch_size,
                                num_workers=10, pin_memory=True)
        val_set = CustomDataset(validation)
        val_loader = DataLoader(val_set, batch_size=batch_size,
                                num_workers=10, pin_memory=True)
        test_set = CustomDataset(test)
        test_loader = DataLoader(test_set, batch_size=batch_size,
                                num_workers=10, pin_memory=True)
        
        checkpoint_callback = ModelCheckpoint(dirpath="",filename=latest_checkpoint+str(limit))
        trainer = L.Trainer(devices=args.gpus,
                            num_nodes=args.nodes,
                            max_epochs=args.epochs,
                            accumulate_grad_batches=10,
                            check_val_every_n_epoch=1,
                            accelerator=('gpu' if torch.cuda.is_available() else 'cpu'),
                            strategy='ddp_find_unused_parameters_true',
                            callbacks=[checkpoint_callback],
                            log_every_n_steps=1)
        
        print('Initial test set validation')
        trainer.validate(model, test_loader)

        from datetime import datetime
        t0 = datetime.now()
        trainer.fit(model, train_loader, val_loader, ckpt_path=ckpt_path)
        dt = datetime.now() - t0
        print('Training took {}'.format(dt))

        print('Final test set validation')
        trainer.validate(model, val_loader)
        break
        # restart from the best model
        model.nn.load_state_dict(torch.load('best.pt', map_location='cpu'))

        query(model, )
