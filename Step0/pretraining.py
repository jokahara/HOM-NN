
import os, sys
import argparse

import torch
#from torch import Tensor
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as L
from pytorch_lightning.callbacks import ModelCheckpoint

import torchani
#from torchani.units import hartree2kcalmol

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from hommani.ani_model import CustomAniNet
from hommani.datasets import CustomDataset, load_dataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#from sklearn.model_selection import KFold

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus', default=1, type=int, metavar='N',
                        help='number of GPUs per node')
    parser.add_argument('--nodes', default=1, type=int, metavar='N',
                        help='number of nodes')
    parser.add_argument('--epochs', default=10, type=int, metavar='N',
                        help='maximum number of epochs to run')
    parser.add_argument('--atoms', default="H,C,N,O,S", type=str, metavar='N',
                        help='Atoms to train the model on')
    parser.add_argument('--data', default="/home/jokahara/PhD/Datasets/ACDB_QM7.h5", type=str, metavar='N',
                        help='Path to training data')
    args = parser.parse_args()
    
    print("Using PyTorch {} and Lightning {}".format(torch.__version__, L.__version__))
    
    latest_checkpoint = 'latest.ckpt'
    ls = os.listdir()
    for file in ls:
        print(file)
    exit()
    
    ckpt_path = latest_checkpoint if os.path.isfile(latest_checkpoint) else None
    
    ani2x = torchani.models.ANI2x(periodic_table_index=False, model_index=0)
    if ckpt_path:
        # load from latest checkpoint
        model = CustomAniNet.load_from_checkpoint(ckpt_path, pretrained_model=ani2x)
    else:
        # Initialize from pretrained ANI-2x model
        ani2x = torchani.models.ANI2x(periodic_table_index=False, model_index=0)
        model = CustomAniNet(ani2x)
        model.species_to_train = args.atoms.split(',')
    
    print("Training on atoms:", model.species_to_train)
    batch_size = 256
    
    energy_shifter, sae_dict = torchani.neurochem.load_sae('../sae_linfit.dat', return_dict=True)

    training1, validation1, energy_shifter = load_dataset(args.data, 0.8, energy_shifter)
    #model.energy_shifter = energy_shifter

    print('Self atomic energies: ', energy_shifter.self_energies)

    training1 =  torchani.data.TransformableIterable(list(training1))
    train_loader = DataLoader(CustomDataset(training1), batch_size=batch_size,
                              num_workers=10, pin_memory=True)
    val_loader = DataLoader(CustomDataset(validation1), batch_size=batch_size,
                            num_workers=10, pin_memory=True)
    
    checkpoint_callback = ModelCheckpoint(dirpath="",
                                          filename="latest-{epoch:02d}-{validation_rmse:.2f}",
                                          save_top_k=1,
                                          monitor='validation_rmse')
    trainer = L.Trainer(devices=args.gpus,
                        num_nodes=args.nodes,
                        max_epochs=args.epochs,
                        accumulate_grad_batches=10,
                        check_val_every_n_epoch=1,
                        accelerator=('gpu' if torch.cuda.is_available() else 'cpu'),
                        strategy='ddp_find_unused_parameters_true',
                        callbacks=[checkpoint_callback],
                        log_every_n_steps=1)
    
    trainer.validate(model, val_loader)
    
    from datetime import datetime
    t0 = datetime.now()
    trainer.fit(model, train_loader, val_loader, ckpt_path=ckpt_path)
    dt = datetime.now() - t0
    print('Training took {}'.format(dt))

    trainer.validate(model, val_loader)
    trainer.save_checkpoint(latest_checkpoint)

if __name__ == '__main__':
    main()