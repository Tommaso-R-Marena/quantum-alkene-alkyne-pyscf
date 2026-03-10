"""
hamiltonian_utils.py
--------------------
Utilities for generating qubit Hamiltonians from molecular data.
Supports Jordan-Wigner and Bravyi-Kitaev transformations,
active-space selection, and qubit tapering.
"""

import numpy as np
from openfermion import (
    get_fermion_operator,
    jordan_wigner,
    bravyi_kitaev,
    QubitOperator,
)
from openfermion.utils import count_qubits
from openfermion.transforms import freeze_orbitals
from openfermionpyscf import run_pyscf


def get_qubit_hamiltonian(
    mol_data,
    mapping: str = "jordan_wigner",
    n_frozen_core: int = 0,
    n_frozen_virt: int = 0,
):
    """
    Convert MolecularData to a QubitOperator Hamiltonian.

    Parameters
    ----------
    mol_data      : OpenFermion MolecularData (post run_pyscf)
    mapping       : 'jordan_wigner' or 'bravyi_kitaev'
    n_frozen_core : number of core orbitals to freeze
    n_frozen_virt : number of virtual orbitals to freeze

    Returns
    -------
    qubit_hamiltonian : QubitOperator
    n_qubits          : int
    """
    fermion_op = get_fermion_operator(mol_data.get_molecular_hamiltonian())

    if n_frozen_core > 0 or n_frozen_virt > 0:
        n_orb = mol_data.n_orbitals
        occupied = list(range(n_frozen_core))
        virtual = list(range(n_orb - n_frozen_virt, n_orb))
        fermion_op = freeze_orbitals(fermion_op, occupied, virtual)

    if mapping == "jordan_wigner":
        qubit_ham = jordan_wigner(fermion_op)
    elif mapping == "bravyi_kitaev":
        qubit_ham = bravyi_kitaev(fermion_op)
    else:
        raise ValueError(f"Unknown mapping: {mapping}")

    n_qubits = count_qubits(qubit_ham)
    return qubit_ham, n_qubits


def qubit_count_summary(molecules: list, basis: str = "sto-3g"):
    """
    Print qubit count table for a list of molecules under JW and BK mappings.
    Useful for hardware feasibility analysis.
    """
    from src.molecule_builder import run_classical_calcs

    print(f"{'Molecule':<20} {'Basis':<10} {'Electrons':<12} {'JW Qubits':<12} {'BK Qubits':<12}")
    print("-" * 66)
    for name in molecules:
        mol = run_classical_calcs(name, basis=basis, run_ccsd=False, run_fci=False)
        _, jw_q = get_qubit_hamiltonian(mol, mapping="jordan_wigner")
        _, bk_q = get_qubit_hamiltonian(mol, mapping="bravyi_kitaev")
        print(f"{name:<20} {basis:<10} {mol.n_electrons:<12} {jw_q:<12} {bk_q:<12}")
