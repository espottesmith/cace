# the CACE calculator for ASE

from typing import Union

import numpy as np 
import torch

from ase.calculators.calculator import Calculator, all_changes
from ase.stress import full_3x3_to_voigt_6_stress

from ..tools import torch_geometric, torch_tools, to_numpy
from ..data import AtomicData
 
__all__ = ["CACECalculator"]

class CACECalculator(Calculator):
    """CACE ASE Calculator
    args:
        model_path: str or nn.module, path to model
        device: str, device to run on (cuda or cpu)
        compute_stress: bool, whether to compute stress
        compute_charges: bool, whether to compute atomic partial charges
        compute_spins: bool, whether to compute atomic partial spins / magnetic moments
        energy_key: str, key for energy in model output
        forces_key: str, key for forces in model output
        stress_key: str, key for stresses in model output
        charges_key: str, key for atomic partial charges in model output
        spins_key: str, key for atomic partial spins in model output
        energy_units_to_eV: float, conversion factor from model energy units to eV
        length_units_to_A: float, conversion factor from model length units to Angstroms
        charge_units_to_au: float, conversion factor from model charge units to atomic units
        spin_units_to_au: float, conversion factor from model spin units to atomic units
        atomic_energies: dict, dictionary of atomic energies to add to model output
    """

    def __init__(
        self,
        model_path: Union[str, torch.nn.Module],
        device: str,
        energy_units_to_eV: float = 1.0,
        length_units_to_A: float = 1.0,
        charge_units_to_au: float = 1.0,
        spin_units_to_au: float = 1.0,
        compute_stress = False,
        compute_charges = False,
        compute_spins = False,
        energy_key: str = 'energy',
        forces_key: str = 'forces',
        stress_key: str = 'stress',
        charges_key: str = 'charges',
        spins_key: str = 'magmoms',
        atomic_energies: dict = None,
        output_index: int = None, # only used for multi-output models
        **kwargs,
        ):

        Calculator.__init__(self, **kwargs)
        self.implemented_properties = [
            "energy",
            "forces",
            "stress",
            "charges",
            "magmoms",
        ]

        self.results = {}

        if isinstance(model_path, str):
            self.model = torch.load(f=model_path, map_location=device)
        elif isinstance(model_path, torch.nn.Module):
            self.model = model_path
        else:
            raise ValueError("model_path must be a string or nn.Module")
        self.model.to(device)

        self.device = torch_tools.init_device(device)
        self.energy_units_to_eV = energy_units_to_eV
        self.length_units_to_A = length_units_to_A

        try:
            self.cutoff = self.model.representation.cutoff
        except AttributeError:
            self.cutoff = self.model.models[0].representation.cutoff

        self.atomic_energies = atomic_energies

        self.compute_stress = compute_stress
        self.compute_charges = compute_charges
        self.compute_spins = compute_spins

        self.energy_key = energy_key 
        self.forces_key = forces_key
        self.stress_key = stress_key
        self.charges_key = charges_key
        self.spins_key = spins_key

        self.output_index = output_index
        
        for param in self.model.parameters():
            param.requires_grad = False

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        """
        Calculate properties.
        :param atoms: ase.Atoms object
        :param properties: [str], properties to be computed, used by ASE internally
        :param system_changes: [str], system changes since last calculation, used by ASE internally
        :return:
        """
        # call to base-class to set atoms attribute
        Calculator.calculate(self, atoms)

        if not hasattr(self, "output_index"):
            self.output_index = None

        # prepare data
        data_loader = torch_geometric.dataloader.DataLoader(
            dataset=[
                AtomicData.from_atoms(
                    atoms, cutoff=self.cutoff
                )
            ],
            batch_size=1,
            shuffle=False,
            drop_last=False,
        )

        batch_base = next(iter(data_loader)).to(self.device)
        batch = batch_base.clone()
        output = self.model(
            batch.to_dict(),
            training=False,
            compute_stress=self.compute_stress,
            compute_charges=self.compute_charges,
            compute_spins=self.compute_spins,
            output_index=self.output_index
        )
        energy_output = to_numpy(output[self.energy_key])
        forces_output = to_numpy(output[self.forces_key])
        # subtract atomic energies if available
        if self.atomic_energies:
            e0 = sum(self.atomic_energies.get(Z, 0) for Z in atoms.get_atomic_numbers())
        else:
            e0 = 0.0
        
        self.results["energy"] = (energy_output + e0) * self.energy_units_to_eV
        self.results["forces"] = forces_output * self.energy_units_to_eV / self.length_units_to_A

        if self.compute_stress and output[self.stress_key] is not None:
            stress = to_numpy(output[self.stress_key])
            # stress has units eng / len^3:
            self.results["stress"] = (
                stress * (self.energy_units_to_eV / self.length_units_to_A**3)
            )[0]
            self.results["stress"] = full_3x3_to_voigt_6_stress(self.results["stress"])
        
        if self.compute_charges and output[self.charges_key] is not None:
            charges_output = to_numpy(output[self.charges_key])
            self.results["charges"] = charges_output * self.charge_units_to_au
        
        if self.compute_spins and output[self.spins_key] is not None:
            spins_output = to_numpy(output[self.spins_key])
            self.results["magmoms"] = spins_output * self.spin_units_to_au

        return self.results
