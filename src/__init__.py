"""
src — Core helpers for the quantum-alkene-alkyne-pyscf project.

Public modules:
- ``molecule_builder``  — PySCF / OpenFermion molecule builders and classical references.
- ``hamiltonian_utils`` — Jordan-Wigner / Bravyi-Kitaev qubit Hamiltonian construction.
- ``vqe_runner``        — UCCSD-VQE and ADAPT-VQE wrappers (PennyLane).
- ``analysis``          — Energy comparison and chemical-accuracy utilities.

Submodules are not auto-imported here because each pulls in heavy dependencies
(PySCF, OpenFermion, PennyLane). Import what you need explicitly:

    from src.analysis import compute_error_mHa
    from src import hamiltonian_utils

The unit tests in ``tests/unit`` mock these dependencies via
``sys.modules.setdefault`` *before* importing the target submodule, so eager
package-level imports here would break that pattern.
"""

from __future__ import annotations

__version__ = "0.2.0"

__all__ = [
    "analysis",
    "hamiltonian_utils",
    "molecule_builder",
    "vqe_runner",
]
