
from typing import Any, Dict, Tuple, Optional

import torch
from torch import nn, Tensor
import pytorch_lightning as L
from torchmetrics import MeanSquaredError, MeanAbsoluteError

# helper function to convert energy unit from Hartree to kcal/mol
from torchani.units import hartree2kcalmol


class CustomAniNet(L.LightningModule):
    
    def __init__(self, pretrained_model=None, energy_shifter=None, train_on="H,C,N,O,S".split(',')):
        super().__init__()
        if pretrained_model == None:
            print('Error! Pretrained model was not loaded!'); exit()
        
        if energy_shifter == None:
            self.energy_shifter = pretrained_model.energy_shifter
        else:
            self.energy_shifter = energy_shifter

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

    
    def atomic_energies(self, species, coordinates, shift=False, average=False):
        """Calculates predicted atomic energies of all atoms in a molecule

        see `:method:torchani.BuiltinModel.atomic_energies`

        If average is True (the default) it returns the average over all models
        (shape (C, A)), otherwise it returns one atomic energy per model (shape
        (M, C, A))
        """
        species, aevs = self.nn.aev_computer((species, coordinates))
        members_list = []
        for nnp in self.nn[1]:
            members_list.append(nnp._atomic_energies((species, aevs)).unsqueeze(0))
        member_atomic_energies = torch.cat(members_list, dim=0)
        
        if shift:
            self_energies = self.energy_shifter.self_energies.clone().to(species.device)
            self_energies = self_energies[species]
            self_energies[species == torch.tensor(-1, device=species.device)] = torch.tensor(0, device=species.device, dtype=torch.double)
            # shift all atomic energies individually
            assert self_energies.shape == member_atomic_energies.shape[1:]
            member_atomic_energies += self_energies
        if average:
            return member_atomic_energies.mean(dim=0)
        return member_atomic_energies
    
    def members_energies(self, species, coordinates, shift=False):
        """Calculates predicted energies of all member modules

        ..warning::
            Since this function does not call ``__call__`` directly,
            hooks are not registered and profiling is not done correctly by
            pytorch on it. It is meant as a convenience function for analysis
             and active learning.

        .. note:: The coordinates, and cell are in Angstrom, and the energies
            will be in Hartree.

        Args:
            species_coordinates: minibatch of configurations
            cell: the cell used in PBC computation, set to None if PBC is not enabled
            pbc: the bool tensor indicating which direction PBC is enabled, set to None if PBC is not enabled

        Returns:
            species_energies: species and energies for the given configurations
                note that the shape of species is (C, A), where C is
                the number of configurations and A the number of atoms, and
                the shape of energies is (M, C), where M is the number
                of modules in the ensemble

        """
        species, aevs = self.nn[0]((species, coordinates))
        member_outputs = []
        for nnp in self.nn[1]:
            energies = nnp((species, aevs)).energies
            if shift:
                energies = self.energy_shifter((species, energies)).energies
            member_outputs.append(energies.unsqueeze(0))

        return torch.cat(member_outputs, dim=0)
    
    def energies_qbcs(self, species, coordinates, unbiased=True):
        """Calculates predicted predicted energies and qbc factors

        QBC factors are used for query-by-committee (QBC) based active learning
        (as described in the ANI-1x paper `less-is-more`_ ).

        .. _less-is-more:
            https://aip.scitation.org/doi/10.1063/1.5023802

        ..warning::
            Since this function does not call ``__call__`` directly,
            hooks are not registered and profiling is not done correctly by
            pytorch on it. It is meant as a convenience function for analysis
             and active learning.

        .. note:: The coordinates, and cell are in Angstrom, and the energies
            and qbc factors will be in Hartree.

        Args:
            species_coordinates: minibatch of configurations
            cell: the cell used in PBC computation, set to None if PBC is not
                enabled
            pbc: the bool tensor indicating which direction PBC is enabled, set
                to None if PBC is not enabled
            unbiased: if `True` then Bessel's correction is applied to the
                standard deviation over the ensemble member's. If `False` Bessel's
                correction is not applied, True by default.

        Returns:
            species_energies_qbcs: species, energies and qbc factors for the
                given configurations note that the shape of species is (C, A),
                where C is the number of configurations and A the number of
                atoms, the shape of energies is (C,) and the shape of qbc
                factors is also (C,).
        """
        energies = self.members_energies(species, coordinates)

        # standard deviation is taken across ensemble members
        qbc_factors = energies.std(0, unbiased=unbiased)

        # rho's (qbc factors) are weighted by dividing by the square root of
        # the number of atoms in each molecule
        num_atoms = (species >= 0).sum(dim=1, dtype=energies.dtype)
        qbc_factors = qbc_factors / num_atoms.sqrt()
        energies = energies.mean(dim=0)
        assert qbc_factors.shape == energies.shape
        return energies, qbc_factors
    
    def training_step(self, batch, batch_idx):
        species, coordinates, true_forces, true_energies = batch
        AdamW, SGD = self.optimizers()

        #predicted_energies = self(species, coordinates)
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

        # in case configure_optimizers() has not been run yet
        if len(self.optimizers()) == 2 and self.trainer.global_rank == 0:
            AdamW, SGD = self.optimizers()
            AdamW_scheduler, SGD_scheduler = self.lr_schedulers()
            print('RMSE:', rmse.item(), 'at epoch', AdamW_scheduler.last_epoch + 1)
            
            learning_rate = AdamW.param_groups[0]['lr']
            if learning_rate < self.early_stopping_learning_rate:
                print('Early stopping at epoch', self.trainer.current_epoch)
            
            # save the best model
            if AdamW_scheduler.is_better(rmse, AdamW_scheduler.best):
                torch.save(self.nn.state_dict(), self.best_model_checkpoint)

            self.log('best_validation_rmse', AdamW_scheduler.best, rank_zero_only=True)
            self.log('learning_rate', learning_rate, rank_zero_only=True)

            AdamW_scheduler.step(rmse)
            SGD_scheduler.step(rmse)
        else:
            # makes sure 'learning_rate' is logged at least once
            self.log('learning_rate', 1e-3, rank_zero_only=True)

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
    