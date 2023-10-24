import os, sys
import argparse
import numpy as np

import torch
#from torch import Tensor
import torch.nn.functional as F
import pytorch_lightning as L
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

import torchani
#from torchani.units import hartree2kcalmol

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from hommani.ani_model import CustomAniNet
from hommani.datasets import CustomDataset, load_data

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def parse_input():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', default=None, type=argparse.FileType('r'), metavar='file.txt',
                        help='input file with arguments')
    parser.add_argument('-i', default=0, type=int, metavar='N',
                        help='model index (0-7)')
    parser.add_argument('--gpus', default=1, type=int, metavar='N',
                        help='number of GPUs per node')
    parser.add_argument('--nodes', default=1, type=int, metavar='N',
                        help='number of nodes')
    parser.add_argument('--epochs', default=10, type=int, metavar='N',
                        help='maximum number of epochs to run')
    parser.add_argument('--batch', default=256, type=int, metavar='N',
                        help='batch size')
    parser.add_argument('--atoms', default='H,C,N,O,S', type=str, metavar='H,C,O',
                        help='atoms to train the model on')
    parser.add_argument('--model', default='ani', type=str, metavar='ANI',
                        help='model to train (index or .pt file)')
    parser.add_argument('--data', default='', type=str, metavar='/path/data.h5',
                        help='path to training data')
    parser.add_argument('--prev', default='', type=str, metavar='/path/data.pkl',
                        help='previous training data')
    parser.add_argument('--init', default='', type=str, metavar='best.pt',
                        help='pretrained model')
    parser.add_argument('--restart', default='latest', type=str, metavar='latest',
                        help='prefix for checkpoint files')
    args = parser.parse_args()
    if args.f:
        for arg in args.f.readlines():
            arg = list(filter(None, ('--'+arg.strip()).split(' ') ))
            args = parser.parse_args(arg, args)
    
    if args.data == '':
        print('Error: --data input file was not found');exit()

    return args
        
def train(args):

    print("Using PyTorch {} and Lightning {}".format(torch.__version__, L.__version__))

    m_index = args.i
    latest_checkpoint = args.restart+"-"+str(m_index)+'.ckpt'
    ckpt_path = latest_checkpoint if os.path.isfile(latest_checkpoint) else None
    
    #checkpoint_files = filter(lambda s: s.split('.')[-1] == 'ckpt', os.listdir())
    #ckpt_path = list(checkpoint_files)[-1]

    m_type = args.model.lower()
    if m_type != 'ani':
        print("Error: incorrect model type given. (expecting ani)"); exit()

    energy_shifter, sae_dict = torchani.neurochem.load_sae('../sae_linfit.dat', return_dict=True)
    data, kfold, energy_shifter = load_data(args.data, split=8, energy_shifter=energy_shifter)
    print('Self atomic energies: ', energy_shifter.self_energies)

    batch_size = args.batch
    train_loader, val_loader = CustomDataset.get_train_val_loaders(data, batch_size, kfold, m_index)

    ani2x = torchani.models.ANI2x(periodic_table_index=False, model_index=m_index)
    if ckpt_path:
        # restarting from latest checkpoint
        print("Restarting training from checkpoint "+ckpt_path)
        model = CustomAniNet.load_from_checkpoint(ckpt_path, pretrained_model=ani2x, energy_shifter=energy_shifter)
    else:
        if args.init != '':
            print('Training '+m_type+'-model initialized from '+args.init)
            model = CustomAniNet(ani2x)
            model.nn.load_state_dict(torch.load(args.init, map_location=device))
        else:
            print('Training '+m_type+'-model initialized from index '+str(m_index))
            # Initialize from pretrained ANI-2x model
            model = CustomAniNet(ani2x)
        
        # best model will be saved at
        model.best_model_checkpoint = 'best-'+str(m_index)+'.pt'

        model.species_to_train = args.atoms.split(',')
        print("Training on atoms:", model.species_to_train)
    
    model.energy_shifter = energy_shifter
    
    # stop training when learning rate is below minimum value
    early_stopping = EarlyStopping(monitor="learning_rate", 
                                   patience=args.epochs, 
                                   stopping_threshold=model.early_stopping_learning_rate*0.9)
    # checkpoint format: latest-2-epoch==10.ckpt
    checkpoint_callback = ModelCheckpoint(dirpath="",
                                        filename=args.restart+"-"+str(m_index)+"-{epoch:02d}",
                                        save_top_k=1,
                                        monitor='validation_rmse')
    trainer = L.Trainer(devices=args.gpus,
                        num_nodes=args.nodes,
                        max_epochs=args.epochs,
                        accumulate_grad_batches=10,
                        check_val_every_n_epoch=1,
                        accelerator=device.type,
                        strategy='ddp_find_unused_parameters_true',
                        callbacks=[early_stopping, checkpoint_callback],
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
    args = parse_input()
    train(args)