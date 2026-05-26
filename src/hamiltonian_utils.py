"""
hamiltonian_utils.py
--------------------
Qubit-Hamiltonian construction for the alkene/alkyne project.

Primary path: qiskit_nature.PySCFDriver -> JordanWignerMapper.
Legacy path (kept for unit-test compatibility): openfermion's
get_fermion_operator + jordan_wigner / bravyi_kitaev.
"""

from __future__ import annotations

# ------------------------------------------------------------------
# Legacy OpenFermion symbols (lazy, retained at module scope for tests)
# ------------------------------------------------------------------
try:  # pragma: no cover - exercised via mocks in unit tests
    from openfermion import (
        bravyi_kitaev,
        get_fermion_operator,
        jordan_wigner,
    )
    from openfermion.transforms import freeze_orbitals
    from openfermion.utils import count_qubits
except Exception:  # pragma: no cover
    bravyi_kitaev = jordan_wigner = get_fermion_operator = None  # type: ignore
    freeze_orbitals = None  # type: ignore
    count_qubits = None  # type: ignore


# ==================================================================
# Primary API: qiskit_nature PySCFDriver -> JW qubit Hamiltonian
# ==================================================================

def build_qiskit_nature_hamiltonian(
    molecule_name: str,
    basis: str = "sto-3g",
    active_electrons: int | None = None,
    active_orbitals: int | None = None,
    freeze_core: bool = False,
    charge: int = 0,
    spin: int = 0,
):
    """Build a JW qubit Hamiltonian via qiskit-nature PySCFDriver.

    Returns ``(qubit_op, problem)`` where ``qubit_op`` is a Qiskit
    :class:`SparsePauliOp` and ``problem`` is the
    :class:`ElectronicStructureProblem` (possibly transformed).
    """
    from qiskit_nature.second_q.drivers import PySCFDriver
    from qiskit_nature.second_q.mappers import JordanWignerMapper
    from qiskit_nature.units import DistanceUnit

    from src.molecule_builder import geometry_to_xyz_string

    driver = PySCFDriver(
        atom=geometry_to_xyz_string(molecule_name),
        basis=basis,
        charge=charge,
        spin=spin,
        unit=DistanceUnit.ANGSTROM,
    )
    problem = driver.run()

    if freeze_core:
        from qiskit_nature.second_q.transformers import FreezeCoreTransformer
        problem = FreezeCoreTransformer().transform(problem)

    if active_electrons is not None and active_orbitals is not None:
        from qiskit_nature.second_q.transformers import ActiveSpaceTransformer
        problem = ActiveSpaceTransformer(
            num_electrons=active_electrons,
            num_spatial_orbitals=active_orbitals,
        ).transform(problem)

    fermionic_op = problem.hamiltonian.second_q_op()
    mapper = JordanWignerMapper()
    qubit_op = mapper.map(fermionic_op)
    return qubit_op, problem


def exact_active_space_energy(qubit_op, problem) -> float:
    """Exact diagonalization of the active-space qubit Hamiltonian.

    Returns the FCI energy *within the active space*, i.e. the smallest
    eigenvalue of ``qubit_op`` plus all Hamiltonian energy-shift
    constants (nuclear repulsion + transformer offsets).
    """
    import numpy as np

    consts = getattr(problem.hamiltonian, "constants", None)
    if consts is None:
        shift = float(problem.nuclear_repulsion_energy or 0.0)
    else:
        shift = float(sum(consts.values()))

    mat = qubit_op.to_matrix() if hasattr(qubit_op, "to_matrix") else qubit_op
    eigvals = np.linalg.eigvalsh(mat)
    return float(eigvals[0]) + shift


# ==================================================================
# Legacy API (OpenFermion). Kept for old tests / notebooks.
# ==================================================================

def get_qubit_hamiltonian(
    mol_data,
    mapping: str = "jordan_wigner",
    n_frozen_core: int = 0,
    n_frozen_virt: int = 0,
):
    """Convert OpenFermion MolecularData to a QubitOperator (legacy)."""
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

    return qubit_ham, count_qubits(qubit_ham)


def qubit_count_summary(molecules: list, basis: str = "sto-3g"):
    """Print qubit count table under JW and BK for a list of molecules."""
    from src.molecule_builder import run_classical_calcs

    print(f"{'Molecule':<20} {'Basis':<10} {'Electrons':<12} {'JW Qubits':<12} {'BK Qubits':<12}")
    print("-" * 66)
    for name in molecules:
        mol = run_classical_calcs(name, basis=basis, run_ccsd=False, run_fci=False)
        _, jw_q = get_qubit_hamiltonian(mol, mapping="jordan_wigner")
        _, bk_q = get_qubit_hamiltonian(mol, mapping="bravyi_kitaev")
        print(f"{name:<20} {basis:<10} {mol.n_electrons:<12} {jw_q:<12} {bk_q:<12}")
