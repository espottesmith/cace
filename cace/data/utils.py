###########################################################################################
# Data parsing utilities
# modified from MACE
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import ase
import ase.data
import ase.io
import numpy as np
from ..tools import to_numpy

Vector = np.ndarray  # [3,]
Positions = np.ndarray  # [..., 3]
Forces = np.ndarray  # [..., 3]
Stress = np.ndarray  # [6, ]
Virials = np.ndarray  # [3,3]
Charges = np.ndarray  # [..., 1]
Cell = np.ndarray  # [3,3]
Pbc = tuple  # (3,)

DEFAULT_CONFIG_TYPE = "Default"
DEFAULT_CONFIG_TYPE_WEIGHTS = {DEFAULT_CONFIG_TYPE: 1.0}


@dataclass
class Configuration:
    atomic_numbers: np.ndarray
    positions: Positions  # Angstrom
    energy: Optional[float] = None  # eV
    forces: Optional[Forces] = None  # eV/Angstrom
    stress: Optional[Stress] = None  # eV/Angstrom^3
    virials: Optional[Virials] = None  # eV
    dipole: Optional[Vector] = None  # Debye
    charges: Optional[Charges] = None  # atomic unit
    cell: Optional[Cell] = None
    pbc: Optional[Pbc] = None

    weight: float = 1.0  # weight of config in loss
    energy_weight: float = 1.0  # weight of config energy in loss
    forces_weight: float = 1.0  # weight of config forces in loss
    stress_weight: float = 1.0  # weight of config stress in loss
    virials_weight: float = 1.0  # weight of config virial in loss
    config_type: Optional[str] = DEFAULT_CONFIG_TYPE  # config_type of config


Configurations = List[Configuration]


def random_train_valid_split(
    items: Sequence, valid_fraction: float, seed: int
) -> Tuple[List, List]:
    assert 0.0 < valid_fraction < 1.0

    size = len(items)
    train_size = size - int(valid_fraction * size)

    indices = list(range(size))
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)

    return (
        [items[i] for i in indices[:train_size]],
        [items[i] for i in indices[train_size:]],
    )


def config_from_atoms_list(
    atoms_list: List[ase.Atoms],
    energy_key="energy",
    forces_key="forces",
    stress_key="stress",
    virials_key="virials",
    dipole_key="dipole",
    charges_key="charges",
    config_type_weights: Dict[str, float] = None,
    atomic_energies: Dict[int, float] = None
) -> Configurations:
    """Convert list of ase.Atoms into Configurations"""
    if config_type_weights is None:
        config_type_weights = DEFAULT_CONFIG_TYPE_WEIGHTS

    all_configs = []
    for atoms in atoms_list:
        all_configs.append(
            config_from_atoms(
                atoms,
                energy_key=energy_key,
                forces_key=forces_key,
                stress_key=stress_key,
                virials_key=virials_key,
                dipole_key=dipole_key,
                charges_key=charges_key,
                config_type_weights=config_type_weights,
                atomic_energies=atomic_energies
            )
        )
    return all_configs


def config_from_atoms(
    atoms: ase.Atoms,
    energy_key="energy",
    forces_key="forces",
    stress_key="stress",
    virials_key="virials",
    dipole_key="dipole",
    charges_key="charges",
    config_type_weights: Dict[str, float] = None,
    atomic_energies: Dict[int, float] = None
) -> Configuration:
    """Convert ase.Atoms to Configuration"""
    if config_type_weights is None:
        config_type_weights = DEFAULT_CONFIG_TYPE_WEIGHTS

    atomic_numbers = atoms.get_atomic_numbers()
    energy = atoms.info.get(energy_key, None)  # eV
    # subtract atomic energies if available
    if atomic_energies and energy is not None:
        energy -= sum(atomic_energies.get(Z, 0) for Z in atomic_numbers)
    forces = atoms.arrays.get(forces_key, None)  # eV / Ang
    stress = atoms.info.get(stress_key, None)  # eV / Ang
    virials = atoms.info.get(virials_key, None)
    dipole = atoms.info.get(dipole_key, None)  # Debye
    # Charges default to 0 instead of None if not found
    charges = atoms.arrays.get(charges_key, np.zeros(len(atoms)))  # atomic unit
    pbc = tuple(atoms.get_pbc())
    cell = np.array(atoms.get_cell())
    config_type = atoms.info.get("config_type", "Default")
    weight = atoms.info.get("config_weight", 1.0) * config_type_weights.get(
        config_type, 1.0
    )
    energy_weight = atoms.info.get("config_energy_weight", 1.0)
    forces_weight = atoms.info.get("config_forces_weight", 1.0)
    stress_weight = atoms.info.get("config_stress_weight", 1.0)
    virials_weight = atoms.info.get("config_virials_weight", 1.0)

    # fill in missing quantities but set their weight to 0.0
    if energy is None:
        energy = 0.0
        energy_weight = 0.0
    if forces is None:
        forces = np.zeros(np.shape(atoms.positions))
        forces_weight = 0.0
    if stress is None:
        stress = np.zeros(6)
        stress_weight = 0.0
    if virials is None:
        virials = np.zeros((3, 3))
        virials_weight = 0.0

    return Configuration(
        atomic_numbers=atomic_numbers,
        positions=atoms.get_positions(),
        energy=energy,
        forces=forces,
        stress=stress,
        virials=virials,
        dipole=dipole,
        charges=charges,
        weight=weight,
        energy_weight=energy_weight,
        forces_weight=forces_weight,
        stress_weight=stress_weight,
        virials_weight=virials_weight,
        config_type=config_type,
        pbc=pbc,
        cell=cell,
    )


def load_from_xyz(
    file_path: str,
    config_type_weights: Dict,
    energy_key: str = "energy",
    forces_key: str = "forces",
    stress_key: str = "stress",
    virials_key: str = "virials",
    dipole_key: str = "dipole",
    charges_key: str = "charges",
    atomic_energies: Dict[int, float] = None
) -> Tuple[Dict[int, float], Configurations]:
    atoms_list = ase.io.read(file_path, index=":")

    if not isinstance(atoms_list, list):
        atoms_list = [atoms_list]

    configs = config_from_atoms_list(
        atoms_list,
        config_type_weights=config_type_weights,
        energy_key=energy_key,
        forces_key=forces_key,
        stress_key=stress_key,
        virials_key=virials_key,
        dipole_key=dipole_key,
        charges_key=charges_key,
        atomic_energies=atomic_energies
    )
    return configs

def batch_to_atoms(batched_data: Dict, 
                   pred_data: Optional[Dict] = None,
                   output_file: str = None,
                   energy_key: str = 'energy', 
                   force_key: str = 'forces', 
                   cace_energy_key: str = 'CACE_energy', 
                   cace_force_key: str = 'CACE_forces'):
    """
    Create ASE Atoms objects from batched graph data and write to an XYZ file.

    Parameters:
    - batched_data (Dict): Batched data containing graph information.
    - pred_data (Dict): Predicted data. If not given, the pred_data name is assumed to also be the batched_data.
    - energy_key (str): Key for accessing energy information in batched_data.
    - force_key (str): Key for accessing force information in batched_data.
    - cace_energy_key (str): Key for accessing CACE energy information.
    - cace_force_key (str): Key for accessing CACE force information.
    - output_file (str): Name of the output file to write the Atoms objects.
    """

    if pred_data == None:
        pred_data = batched_data
    atoms_list = []
    batch = batched_data.batch
    num_graphs = batch.max().item() + 1

    for i in range(num_graphs):
        # Mask to extract nodes for each graph
        mask = batch == i

        # Extract node features, edge indices, etc., for each graph
        positions = to_numpy(batched_data['positions'][mask])
        atomic_numbers = to_numpy(batched_data['atomic_numbers'][mask])
        cell = to_numpy(batched_data['cell'][3*i:3*i+3])

        energy = to_numpy(batched_data[energy_key][i])
        forces = to_numpy(batched_data[force_key][mask])
        cace_energy = to_numpy(pred_data[cace_energy_key][i])
        cace_forces = to_numpy(pred_data[cace_force_key][mask])

        # Set periodic boundary conditions if the cell is defined
        pbc = np.all(np.mean(cell, axis=0) > 0)

        # Create the Atoms object
        atoms = ase.Atoms(numbers=atomic_numbers, positions=positions, cell=cell, pbc=pbc)
        atoms.info[energy_key] = energy.item() if np.ndim(energy) == 0 else energy
        atoms.arrays[force_key] = forces
        atoms.info[cace_energy_key] = cace_energy.item() if np.ndim(cace_energy) == 0 else cace_energy
        atoms.arrays[cace_force_key] = cace_forces

        atoms_list.append(atoms)

    # Write all atoms to the output file
    if output_file: 
        ase.io.write(output_file, atoms_list, append=True)
    return atoms_list

