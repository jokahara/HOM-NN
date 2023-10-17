import math
from typing import Any, Dict, Tuple, Optional
from pytorch_lightning.utilities.types import STEP_OUTPUT

import torch
from torch import nn, Tensor
import pytorch_lightning as L
from torchmetrics import MeanSquaredError, MeanAbsoluteError

# helper function to convert energy unit from Hartree to kcal/mol
from torchani.units import hartree2kcalmol


class CustomAniNet(L.LightningModule):
    def __init__(self, pretrained_model=None, train_on="H,C,N,O,S".split(',')):
        super().__init__()
        if pretrained_model == None:
            print('Error! Pretrained model was not loaded!'); exit()

        self.energy_shifter = pretrained_model.energy_shifter
        self.species_to_tensor = pretrained_model.species_to_tensor
        self.species = pretrained_model.species
        self.species_to_train = train_on
        self.nn = nn.Sequential(pretrained_model.aev_computer, pretrained_model.neural_networks)

        self.batch_loss = nn.MSELoss(reduction='none')
        self.mse = MeanSquaredError()
        self.mae = MeanAbsoluteError()

        self.best_model_checkpoint = 'best.pt'
        self.early_stopping_learning_rate = 1.0E-5

    def forward(self, species: Tensor, coordinates: Tensor, 
                return_forces: bool = False) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
        if return_forces:
            coordinates.requires_grad_(True)

        energies = self.nn((species, coordinates)).energies

        forces: Optional[Tensor] = None  # noqa: E701
        if return_forces:
            forces = -torch.autograd.grad([energies.sum()], [coordinates], create_graph=True, retain_graph=True)[0]
            return energies, forces
        
        return energies

    def training_step(self, batch, batch_idx):
        species, coordinates, true_forces, true_energies = batch
        AdamW, SGD = self.optimizers()

        #predicted_energies = self(species, coordinates, return_forces=False)
        predicted_energies, forces = self(species, coordinates, return_forces=True)
        num_atoms = (species >= 0).sum(dim=1, dtype=true_energies.dtype)

        # Now the total loss has two parts, energy loss and force loss
        energy_loss = (self.batch_loss(predicted_energies, true_energies) / num_atoms.sqrt()).mean()
        force_loss = (self.batch_loss(true_forces, forces).sum(dim=(1, 2)) / num_atoms).mean()

        force_coefficient = 0.1
        loss = energy_loss + force_coefficient * force_loss

        AdamW.zero_grad()
        SGD.zero_grad()
        self.manual_backward(loss)
        AdamW.step()
        SGD.step()

        self.log("batch_loss", loss)
        return loss    
    
    def validation_step(self, batch, batch_idx):
        species, coordinates, _, true_energies = batch
        # run validation
        predicted_energies = self(species, coordinates)
        self.mse(predicted_energies, true_energies)
        self.mae(predicted_energies, true_energies)
        
        return self.mse 

    def on_validation_epoch_end(self):
        
        rmse = hartree2kcalmol(self.mse.compute().sqrt())
        self.log('validation_rmse', rmse)
        self.log('validation_mae', hartree2kcalmol(self.mae.compute()))

        opt = self.optimizers()
        if len(opt) == 2 and self.trainer.global_rank == 0:
            AdamW, SGD = opt
            AdamW_scheduler, SGD_scheduler = self.lr_schedulers()
            self.trainer.current_epoch

            print('RMSE:', rmse.item(), 'at epoch', AdamW_scheduler.last_epoch + 1)
            
            learning_rate = AdamW.param_groups[0]['lr']
            if learning_rate < self.early_stopping_learning_rate:
                print('Early stopping:')
                self.trainer.should_stop = True
            
            # save the best model
            if AdamW_scheduler.is_better(rmse, AdamW_scheduler.best):
                torch.save(self.nn.state_dict(), self.best_model_checkpoint)

            self.log('best_validation_rmse', AdamW_scheduler.best, rank_zero_only=True)
            self.log('learning_rate', learning_rate, rank_zero_only=True)

            AdamW_scheduler.step(rmse)
            SGD_scheduler.step(rmse)

        self.mse.reset()
        self.mae.reset()
        return 
                            
    def configure_optimizers(self):
        self.automatic_optimization = False
        nn = self.nn[1]
        
        # Setup optimizers for two middle layers. Input and output layers are kept fixed.
        params = []
        for s in self.species_to_train:
            params.append({'params': [nn[s][2].weight], 'weight_decay': 0.00001})
            params.append({'params': [nn[s][4].weight], 'weight_decay': 0.000001})
        AdamW = torch.optim.AdamW(params, lr=1e-3)

        params = []
        for s in self.species_to_train:
            params.append({'params': [nn[s][2].bias]})
            params.append({'params': [nn[s][4].bias]})
        SGD = torch.optim.SGD(params, lr=1e-3)

        # Setting up a learning rate scheduler to do learning rate decay
        AdamW_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(AdamW, factor=0.5, patience=100, threshold=0)
        SGD_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(SGD, factor=0.5, patience=100, threshold=0)
        
        return [AdamW, SGD], [AdamW_scheduler, SGD_scheduler]
    