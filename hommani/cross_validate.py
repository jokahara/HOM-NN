
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

from sklearn.model_selection import KFold

def parse_input():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', default=None, type=argparse.FileType('r'), metavar='file.txt',
                        help='input file with arguments')
    parser.add_argument('--gpus', default=1, type=int, metavar='N',
                        help='number of GPUs per node')
    parser.add_argument('--nodes', default=1, type=int, metavar='N',
                        help='number of nodes')
    parser.add_argument('--epochs', default=10, type=int, metavar='N',
                        help='maximum number of epochs to run')
    parser.add_argument('--atoms', default='H,C,N,O,S', type=str, metavar='H,C,O',
                        help='atoms to train the model on')
    parser.add_argument('--model', default='ani-0', type=str, metavar='ani-1',
                        help='model to train')
    parser.add_argument('--data', default='', type=str, metavar='/path/data.h5',
                        help='path to training data')
    parser.add_argument('--ckpt', default='latest', type=str, metavar='latest',
                        help='prefix for checkpoint files')
    args = parser.parse_args()
    if args.f:
        for arg in args.f.readlines():
            arg = list(filter(None, ('--'+arg.strip()).split(' ') ))
            args = parser.parse_args(arg, args)
    
    if args.data == '':
        print('Error: --data input file was not found');exit()

    return args
        
def init():
    
    args = parse_input()
    print("Using PyTorch {} and Lightning {}".format(torch.__version__, L.__version__))

    latest_checkpoint = args.ckpt+'ckpt'
    ls = os.listdir()
    for file in ls:
        if file.split('.')[-1] == 'ckpt':
            if file.split('.')[0].split('-')[0] == 'latest':
                latest_checkpoint = file
    
    ckpt_path = latest_checkpoint if os.path.isfile(latest_checkpoint) else None

    m_type, m_index = args.model.split('-')
    if m_type.lower() != 'ani':
        print("Error: incorrect model type given. (expecting ani)"); exit()

    print('Training '+m_type+'-model index '+m_index)

    ani2x = torchani.models.ANI2x(periodic_table_index=False, model_index=int(m_index))
    if ckpt_path:
        # load from latest checkpoint
        model = CustomAniNet.load_from_checkpoint(ckpt_path, pretrained_model=ani2x)
    else:
        # Initialize from pretrained ANI-2x model
        model = CustomAniNet(ani2x)
        model.species_to_train = args.atoms.split(',')
    
    print("Training on atoms:", model.species_to_train)
    batch_size = 256
    
    energy_shifter, sae_dict = torchani.neurochem.load_sae('../sae_linfit.dat', return_dict=True)

    training, validation, energy_shifter = load_dataset(args.data, 0.8, energy_shifter)
    #model.energy_shifter = energy_shifter
    kf = KFold(n_splits=8)
    kf.get_n_splits()
    print('Self atomic energies: ', energy_shifter.self_energies)

    train_loader = DataLoader(CustomDataset(training), batch_size=batch_size,
                              num_workers=10, pin_memory=True)
    val_loader = DataLoader(CustomDataset(validation), batch_size=batch_size,
                            num_workers=10, pin_memory=True)
    
    checkpoint_callback = ModelCheckpoint(dirpath="",
                                          filename="latest-{epoch:02d}-{validation_rmse:.2f}",
                                          save_top_k=2,
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
    init()