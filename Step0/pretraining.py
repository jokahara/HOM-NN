import os
import argparse

import torch
#from torch import Tensor
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as L
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

import torchani
#from torchani.units import hartree2kcalmol

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
    args = parser.parse_args()

    print("Using PyTorch {} and Lightning {}".format(torch.__version__, L.__version__))
    
    latest_checkpoint = 'checkpoint.ckpt'
    ckpt_path = latest_checkpoint if os.path.isfile(latest_checkpoint) else None
    
    ani2x = torchani.models.ANI2x(periodic_table_index=False, model_index=0)
    # Load pretrained ANI-2x model
    model = CustomAniNet(ani2x)
    model.nn.load_state_dict(torch.load('best.pt', map_location='cpu'))
    #if ckpt_path:
        #model = model.load_from_checkpoint(ckpt_path)
    
    batch_size = 256
    training1, validation1, self_energies = load_dataset('../data/ACDB_forces.h5', 0.8, model.energy_shifter, model.species)
    #training2, validation2, self_energies = load_dataset('../data/2sa.h5', 0.1, model.energy_shifter, model.species)

    train_set = CustomDataset(training1)
    train_loader = DataLoader(train_set, batch_size=batch_size,
                              num_workers=10, pin_memory=True)
    val_set = CustomDataset(validation1)
    val_loader = DataLoader(val_set, batch_size=batch_size,
                            num_workers=10, pin_memory=True)
    
    checkpoint_callback = ModelCheckpoint(dirpath="",filename=latest_checkpoint)
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