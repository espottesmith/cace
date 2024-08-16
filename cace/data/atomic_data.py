###########################################################################################
# Atomic Data Class for handling molecules as graphs
# modified from MACE
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################

from typing import Dict, Optional, Sequence, Union
from ase import Atoms
import numpy as np
#import torch_geometric
from ..tools import torch_geometric
import torch.nn as nn
import torch.utils.data
from ..tools import voigt_to_matrix

from .neighborhood import get_neighborhood

default_data_key = {
    "energy": "energy",
    "forces": "forces",
    "molecular_index": "molecular_index",
    "stress": "stress",
    "virials": "virials",
    "total_charge": "charge",
    "spin_multiplicity": "spin",
    "partial_charges": "charges",
    "partial_spins": "magmoms",  # TODO: confirm
    "dipole":  None,
    "weights": None,
    "energy_weight": None,
    "force_weight": None,
    "stress_weight": None,
    "virial_weight": None,
    "total_charge_weight": None,
    "spin_multiplicity_weight": None,
    "partial_charges_weight": None,
    "partial_spins_weight": None,
}


class AtomicData(torch_geometric.data.Data):
    atomic_numbers: torch.Tensor
    num_graphs: torch.Tensor
    num_nodes: torch.Tensor
    batch: torch.Tensor
    edge_index: torch.Tensor
    node_attrs: torch.Tensor
    n_atom_basis: torch.Tensor
    positions: torch.Tensor
    shifts: torch.Tensor
    unit_shifts: torch.Tensor
    cell: torch.Tensor
    forces: torch.Tensor
    molecular_index: torch.Tensor
    energy: torch.Tensor
    stress: torch.Tensor
    virials: torch.Tensor
    dipole: torch.Tensor
    total_charge: torch.Tensor
    spin_multiplicity: torch.Tensor
    partial_charges: torch.Tensor
    partial_spins: torch.Tensor
    weight: torch.Tensor
    energy_weight: torch.Tensor
    forces_weight: torch.Tensor
    stress_weight: torch.Tensor
    virials_weight: torch.Tensor
    total_charge_weight: torch.Tensor
    spin_multiplicity_weight: torch.Tensor
    partial_charges_weight: torch.Tensor
    partial_spins_weight: torch.Tensor

    def __init__(
        self,
        edge_index: torch.Tensor,  # [2, n_edges], always sender -> receiver
        atomic_numbers: torch.Tensor,  # [n_nodes]
        positions: torch.Tensor,  # [n_nodes, 3]
        shifts: torch.Tensor,  # [n_edges, 3],
        unit_shifts: torch.Tensor,  # [n_edges, 3]
        num_nodes: Optional[torch.Tensor] = None, #[,]
        cell: Optional[torch.Tensor] = None,  # [3,3]
        forces: Optional[torch.Tensor] = None,  # [n_nodes, 3]
        molecular_index: Optional[torch.Tensor] = None,  # [n_nodes]
        energy: Optional[torch.Tensor] = None,  # [, ]
        stress: Optional[torch.Tensor] = None,  # [1,3,3]
        virials: Optional[torch.Tensor] = None,  # [1,3,3]
        total_charge: Optional[torch.Tensor] = None,  # [, ]
        spin_multiplicity: Optional[torch.Tensor] = None,  # [, ]
        partial_charges: Optional[torch.Tensor] = None,  # [n_nodes, ]
        partial_spins: Optional[torch.Tensor] = None,  # [n_nodes, ]
        additional_info: Optional[Dict] = None, 
        #dipole: Optional[torch.Tensor],  # [, 3]
    ):
        # Check shapes
        if num_nodes is None:
            num_nodes = atomic_numbers.shape[0]
        else:
            assert num_nodes == atomic_numbers.shape[0]
        assert edge_index.shape[0] == 2 and len(edge_index.shape) == 2
        assert positions.shape == (num_nodes, 3)
        assert shifts.shape[1] == 3
        assert unit_shifts.shape[1] == 3
        assert cell is None or cell.shape == (3, 3)
        assert forces is None or forces.shape == (num_nodes, 3)
        assert molecular_index is None or molecular_index.shape == (num_nodes,)
        assert energy is None or len(energy.shape) == 0
        assert stress is None or stress.shape == (1, 3, 3)
        assert virials is None or virials.shape == (1, 3, 3)
        assert total_charge is None or len(total_charge.shape) == 0
        assert spin_multiplicity is None or len(spin_multiplicity.shape) == 0
        assert partial_charges is None or partial_charges.shape == (num_nodes,)
        assert partial_spins is None or partial_spins.shape == (num_nodes,)
        #assert dipole is None or dipole.shape[-1] == 3
        # Aggregate data
        data = {
            "edge_index": edge_index,
            "positions": positions,
            "shifts": shifts,
            "unit_shifts": unit_shifts,
            "cell": cell,
            "atomic_numbers": atomic_numbers,
            "num_nodes": num_nodes,
            "forces": forces,
            "molecular_index": molecular_index,
            "energy": energy,
            "stress": stress,
            "virials": virials,
            "total_charge": total_charge,  # TODO: Since this is a homogeneous graph, should this be a vector, with the same property for each node?
            "spin_multiplicity": spin_multiplicity,  # TODO: see above line
            "charges": charges,
            "spins": spins,
            #"dipole": dipole,
        }
        if additional_info is not None:
            data.update(additional_info)
        super().__init__(**data)

    @classmethod
    def from_atoms(
        cls,
        atoms: Atoms, 
        cutoff: float,
        data_key: Dict[str, str] = None,
        atomic_energies: Optional[Dict[int, float]] = None,
    ) -> "AtomicData":
        if data_key is not None:
            data_key = default_data_key.update(data_key)
        else:
            data_key = default_data_key

        positions = atoms.get_positions()
        pbc = tuple(atoms.get_pbc())
        cell = np.array(atoms.get_cell())
        atomic_numbers = atoms.get_atomic_numbers()

        edge_index, shifts, unit_shifts = get_neighborhood(
            positions=positions,
            cutoff=cutoff,
            pbc=pbc,
            cell=cell
        )

        try:
            energy = atoms.info.get(data_key["energy"], None)  # eV
        except:
            # this ugly bit is for compatibility with newest ASE versions
            if data_key['energy'] == 'energy':
                energy = atoms.get_potential_energy()
            else:
                energy = None

        # subtract atomic energies if available
        if atomic_energies and energy is not None:
            energy -= sum(atomic_energies.get(Z, 0) for Z in atomic_numbers)
        try:
            forces = atoms.arrays.get(data_key["forces"], None)  # eV / Ang
        except:
            if data_key['forces'] == 'forces':
                forces = atoms.get_forces()
            else:
                forces = None
        molecular_index = atoms.arrays.get(data_key["molecular_index"], None) # index of molecules

        stress = atoms.info.get(data_key["stress"], None)  # eV / Ang
        virials = atoms.info.get(data_key["virials"], None)

        total_charge = atoms.info.get(data_key["total_charge"], None)
        spin_multiplicity = atoms.info.get(data_key["spin_multiplicity"], None)

        partial_charges = atoms.arrays.get(data_key["partial_charges"], None)
        partial_spins = atoms.arrays.get(data_key["partial_spins"], None)

        # process these to make tensors
        cell = (
            torch.tensor(cell, dtype=torch.get_default_dtype())
            if cell is not None
            else torch.tensor(
                3 * [0.0, 0.0, 0.0], dtype=torch.get_default_dtype()
            ).view(3, 3)
        )

        forces = (
            torch.tensor(forces, dtype=torch.get_default_dtype())
            if forces is not None
            else None
        )

        molecular_index = (
            torch.tensor(molecular_index, dtype=torch.long)
            if molecular_index is not None
            else None
        )

        energy = (
            torch.tensor(energy, dtype=torch.get_default_dtype())
            if energy is not None
            else None
        )

        stress = (
            voigt_to_matrix(
                torch.tensor(stress, dtype=torch.get_default_dtype())
            ).unsqueeze(0)
            if stress is not None
            else None
        )
        virials = (
            torch.tensor(virials, dtype=torch.get_default_dtype()).unsqueeze(0)
            if virials is not None
            else None
        )

        total_charge = (
            torch.tensor(total_charge, dtype=torch.long)
            if total_charge is not None
            else None
        )
        spin_multiplicity = (
            torch.tensor(spin_multiplicity, dtype=torch.long)
            if spin_multiplicity is not None
            else None
        )

        partial_charges = (
            torch.tensor(partial_charges, dtype=torch.get_default_dtype())
            if partial_charges is not None
            else None
        )
        partial_spins = (
            torch.tensor(partial_spins, dtype=torch.get_default_dtype())
            if partial_spins is not None
            else None
        )

        #  obtain additional info
        # enumerate the data_key and extract data
        additional_info = {}
        for key, kk in data_key.items():
            if kk is None or key in [
                'energy',
                'forces',
                'stress',
                'virial',
                'total_charge',
                'spin_multiplicity',
                'partial_charges',
                'partial_spins',
                'molecular_index'
            ]:
                continue
            else:
                more_info = atoms.info.get(data_key[kk], None)
                if more_info is None:
                    more_info = atoms.arrays.get(data_key[kk], None)
                more_info = (
                    torch.tensor(more_info, dtype=torch.get_default_dtype())
                    if more_info is not None
                    else None
                )
            additional_info[key] = more_info

        return cls(
            edge_index=torch.tensor(edge_index, dtype=torch.long),
            positions=torch.tensor(positions, dtype=torch.get_default_dtype()),
            shifts=torch.tensor(shifts, dtype=torch.get_default_dtype()),
            unit_shifts=torch.tensor(unit_shifts, dtype=torch.get_default_dtype()),
            cell=cell,
            atomic_numbers=torch.tensor(atomic_numbers, dtype=torch.long),
            num_nodes=atomic_numbers.shape[0],
            forces=forces,
            molecular_index=molecular_index,
            energy=energy,
            stress=stress,
            virials=virials,
            total_charge=total_charge,
            spin_multiplicity=spin_multiplicity,
            partial_charges=partial_charges,
            partial_spins=partial_spins,
            additional_info=additional_info,
        )


class AtomicDataWithGlobal(torch_geometric.heterodata.HeteroData):
    num_atom_nodes: torch.Tensor
    atom_edge_index: torch.Tensor
    atomic_numbers: torch.Tensor
    
    positions: torch.Tensor
    shifts: torch.Tensor
    unit_shifts: torch.Tensor
    cell: torch.Tensor
    forces: torch.Tensor
    molecular_index: torch.Tensor
    energy: torch.Tensor
    stress: torch.Tensor
    virials: torch.Tensor
    total_charge: torch.Tensor
    spin_multiplicity: torch.Tensor
    partial_charges: torch.Tensor
    partial_spins: torch.Tensor
    
    weight: torch.Tensor
    energy_weight: torch.Tensor
    forces_weight: torch.Tensor
    stress_weight: torch.Tensor
    virials_weight: torch.Tensor
    total_charge_weight: torch.Tensor
    spin_multiplicity_weight: torch.Tensor
    partial_charges_weight: torch.Tensor
    partial_spins_weight: torch.Tensor

    # dipole: torch.Tensor  # TODO: could add this back in at some point

    def __init__(
        self,
        atom_edge_index: torch.Tensor,  # [2, num_atom_edges], always sender -> receiver
        atomic_numbers: torch.Tensor,  # [num_atom_nodes]
        positions: torch.Tensor,  # [num_atom_nodes, 3]
        shifts: torch.Tensor,  # [num_atom_edges, 3],
        unit_shifts: torch.Tensor,  # [num_atom_edges, 3]
        num_atom_nodes: Optional[torch.Tensor] = None, #[,]
        cell: Optional[torch.Tensor] = None,  # [3,3]
        forces: Optional[torch.Tensor] = None,  # [n_nodes, 3]
        molecular_index: Optional[torch.Tensor] = None,  # [n_nodes]
        energy: Optional[torch.Tensor] = None,  # [, ]
        stress: Optional[torch.Tensor] = None,  # [1,3,3]
        virials: Optional[torch.Tensor] = None,  # [1,3,3]
        total_charge: Optional[torch.Tensor] = None,  # [, ]
        spin_multiplicity: Optional[torch.Tensor] = None,  # [, ]
        partial_charges: Optional[torch.Tensor] = None,  # [n_nodes, ]
        partial_spins: Optional[torch.Tensor] = None,  # [n_nodes, ]
        additional_atom_info: Optional[Dict] = None, 
        additional_global_info: Optional[Dict] = None,
        #dipole: Optional[torch.Tensor],  # [, 3]  # TODO: consider adding this back in
    ):
        # Check shapes
        if num_atom_nodes is None:
            num_atom_nodes = atomic_numbers.shape[0]
        else:
            assert num_atom_nodes == atomic_numbers.shape[0]

        assert atom_edge_index.shape[0] == 2 and len(atom_edge_index.shape) == 2
        assert positions.shape == (num_nodes, 3)
        assert shifts.shape[1] == 3
        assert unit_shifts.shape[1] == 3
        assert cell is None or cell.shape == (3, 3)
        assert forces is None or forces.shape == (num_atom_nodes, 3)
        assert molecular_index is None or molecular_index.shape == (num_atom_nodes,)
        assert energy is None or len(energy.shape) == 0
        assert stress is None or stress.shape == (1, 3, 3)
        assert virials is None or virials.shape == (1, 3, 3)
        assert total_charge is None or len(total_charge.shape) == 0
        assert spin_multiplicity is None or len(spin_multiplicity.shape) == 0
        assert partial_charges is None or partial_charges.shape == (num_atom_nodes,)
        assert partial_spins is None or partial_spins.shape == (num_atom_nodes,)
        #assert dipole is None or dipole.shape[-1] == 3

        atom_data = {
            "positions": positions,
            "shifts": shifts,
            "unit_shifts": unit_shifts,
            "cell": cell,
            "atomic_numbers": atomic_numbers,
            "forces": forces,
            "molecular_index": molecular_index,
            "partial_charges": partial_charges,
            "partial_spins": partial_spins,
            #"dipole": dipole,
        }

        global_data = {
            "total_charge": total_charge,
            "spin_multiplicity": spin_multiplicity,
            "energy": energy,
            "stress": stress,
            "virials": virials,
        }

        if additional_atom_info is not None:
            atom_data.update(additional_atom_info)

        if additional_global_info is not None:
            global_data.update(additional_global_info)

        #TODO: How should you include graph-level properties in a HeteroData?
        # Currently treating them as features of the "global" node

        # Define edges from global node to atoms nodes and vice versa
        g_to_a_edges = torch.tensor(
            [
                [0 for _ in range(num_atom_nodes)],
                [i for i in range(num_atom_nodes)]
            ],
            dtype=torch.long
        )

        a_to_g_edges = torch.tensor(
            [
                [i for i in range(num_atom_nodes)],
                [0 for _ in range(num_atom_nodes)],
            ],
            dtype=torch.long
        )
        
        super().__init__(
            {
                "atom": atom_data,
                "global": global_data,
                ("atom", "a2a", "atom"): {"edge_index": atom_edge_index},
                ("atom", "a2g", "global"): {"edge_index": a_to_g_edges},
                ("global", "g2a", "atom"): {"edge_index": g_to_a_edges}
            }
        )

    @classmethod
    def from_atoms(
        cls,
        atoms: Atoms, 
        cutoff: float,
        data_key: Dict[str, str] = None,
        additional_atom_keys: Dict[str, str] = None,
        additional_global_keys: Dict[str, str] = None,
        atomic_energies: Optional[Dict[int, float]] = None,
    ) -> "AtomicData":

        if data_key is not None:
            data_key = default_data_key.update(data_key)
        else:
            data_key = default_data_key

        positions = atoms.get_positions()
        pbc = tuple(atoms.get_pbc())
        cell = np.array(atoms.get_cell())
        atomic_numbers = atoms.get_atomic_numbers()

        atom_edge_index, shifts, unit_shifts = get_neighborhood(
            positions=positions,
            cutoff=cutoff,
            pbc=pbc,
            cell=cell
        )

        try:
            energy = atoms.info.get(data_key["energy"], None)  # eV
        except:
            # this ugly bit is for compatibility with newest ASE versions
            if data_key['energy'] == 'energy':
                energy = atoms.get_potential_energy()
            else:
                energy = None

        # subtract atomic energies if available
        if atomic_energies and energy is not None:
            energy -= sum(atomic_energies.get(Z, 0) for Z in atomic_numbers)
        try:
            forces = atoms.arrays.get(data_key["forces"], None)  # eV / Ang
        except:
            if data_key['forces'] == 'forces':
                forces = atoms.get_forces()
            else:
                forces = None
        molecular_index = atoms.arrays.get(data_key["molecular_index"], None) # index of molecules

        stress = atoms.info.get(data_key["stress"], None)  # eV / Ang
        virials = atoms.info.get(data_key["virials"], None)

        total_charge = atoms.info.get(data_key["total_charge"], None)
        spin_multiplicity = atoms.info.get(data_key["spin_multiplicity"], None)

        partial_charges = atoms.arrays.get(data_key["partial_charges"], None)
        partial_spins = atoms.arrays.get(data_key["partial_spins"], None)

        # process these to make tensors
        cell = (
            torch.tensor(cell, dtype=torch.get_default_dtype())
            if cell is not None
            else torch.tensor(
                3 * [0.0, 0.0, 0.0], dtype=torch.get_default_dtype()
            ).view(3, 3)
        )

        forces = (
            torch.tensor(forces, dtype=torch.get_default_dtype())
            if forces is not None
            else None
        )

        molecular_index = (
            torch.tensor(molecular_index, dtype=torch.long)
            if molecular_index is not None
            else None
        )

        energy = (
            torch.tensor(energy, dtype=torch.get_default_dtype())
            if energy is not None
            else None
        )

        stress = (
            voigt_to_matrix(
                torch.tensor(stress, dtype=torch.get_default_dtype())
            ).unsqueeze(0)
            if stress is not None
            else None
        )
        virials = (
            torch.tensor(virials, dtype=torch.get_default_dtype()).unsqueeze(0)
            if virials is not None
            else None
        )

        total_charge = (
            torch.tensor(total_charge, dtype=torch.long)
            if total_charge is not None
            else None
        )
        spin_multiplicity = (
            torch.tensor(spin_multiplicity, dtype=torch.long)
            if spin_multiplicity is not None
            else None
        )

        partial_charges = (
            torch.tensor(partial_charges, dtype=torch.get_default_dtype())
            if partial_charges is not None
            else None
        )
        partial_spins = (
            torch.tensor(partial_spins, dtype=torch.get_default_dtype())
            if partial_spins is not None
            else None
        )

        # obtain additional info
        additional_atom_info = {}
        for key, kk in additional_atom_keys.items():
            if key is None or kk is None:
                continue
            else:
                more_info = atoms.info.get(kk, None)
                if more_info is None:
                    more_info = atoms.arrays.get(kk, None)
                more_info = (
                    torch.tensor(more_info, dtype=torch.get_default_dtype())
                    if more_info is not None
                    else None
                )
            additional_atom_info[key] = more_info
        
        additional_global_info = {}
        for key, kk in additional_global_keys.items():
            if key is None or kk is None:
                continue
            else:
                more_info = atoms.info.get(kk, None)
                if more_info is None:
                    more_info = atoms.arrays.get(kk, None)
                more_info = (
                    torch.tensor(more_info, dtype=torch.get_default_dtype())
                    if more_info is not None
                    else None
                )
            additional_global_info[key] = more_info

        return cls(
            atom_edge_index=torch.tensor(atom_edge_index, dtype=torch.long),
            positions=torch.tensor(positions, dtype=torch.get_default_dtype()),
            shifts=torch.tensor(shifts, dtype=torch.get_default_dtype()),
            unit_shifts=torch.tensor(unit_shifts, dtype=torch.get_default_dtype()),
            cell=cell,
            atomic_numbers=torch.tensor(atomic_numbers, dtype=torch.long),
            num_atom_nodes=atomic_numbers.shape[0],
            forces=forces,
            molecular_index=molecular_index,
            energy=energy,
            stress=stress,
            virials=virials,
            total_charge=total_charge,
            spin_multiplicity=spin_multiplicity,
            partial_charges=partial_charges,
            partial_spins=partial_spins,
            additional_atom_info=additional_atom_info,
            additional_global_info=additional_global_info,
        )


def get_data_loader(
    dataset: Union[Sequence[AtomicData], Sequence[AtomicDataWithGlobal]],
    batch_size: int,
    shuffle=True,
    drop_last=False,
) -> torch.utils.data.DataLoader:
    return torch_geometric.dataloader.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
    )
