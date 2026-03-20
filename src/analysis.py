"""
analysis.py
-----------
Energy comparison, error reporting, and chemical accuracy utilities.
Used by notebooks and the test suite.
"""

from __future__ import annotations
import math

CHEM_ACCURACY_mHa: float = 1.6   # 1 kcal/mol ≈ 1.594 mHa
HA_TO_mHA: float = 1000.0


def compute_correlation_energy(
    hf: float,
    fci: float,
    unit: str = "Ha",
) -> float:
    """
    Return the electron correlation energy E_corr = E_FCI - E_HF.

    Parameters
    ----------
    hf   : Hartree-Fock energy (Ha)
    fci  : FCI energy (Ha)
    unit : 'Ha' or 'mHa' — unit of the returned value

    Returns
    -------
    float  (negative for a correlated system, as FCI < HF)
    """
    corr = fci - hf
    if unit == "mHa":
        return corr * HA_TO_mHA
    return corr


def compute_error_mHa(vqe: float, fci: float) -> float:
    """
    Return |E_VQE - E_FCI| in mHa.

    Parameters
    ----------
    vqe : VQE energy (Ha)
    fci : FCI reference energy (Ha)

    Returns
    -------
    float — always non-negative
    """
    return abs(vqe - fci) * HA_TO_mHA


def check_chemical_accuracy(
    vqe: float,
    fci: float,
    threshold_mHa: float = CHEM_ACCURACY_mHa,
) -> bool:
    """
    Return True if |E_VQE - E_FCI| <= threshold_mHa.

    The default threshold is 1.6 mHa (≈ 1 kcal/mol, the conventional
    definition of chemical accuracy in quantum chemistry).
    """
    return compute_error_mHa(vqe, fci) <= threshold_mHa


def format_energy_table(
    energies: dict[str, float],
    fci_energy: float,
    threshold_mHa: float = CHEM_ACCURACY_mHa,
) -> str:
    """
    Return a formatted string table comparing method energies to FCI.

    Parameters
    ----------
    energies      : {method_name: energy_Ha}
    fci_energy    : FCI reference energy (Ha)
    threshold_mHa : chemical accuracy threshold (default 1.6 mHa)

    Returns
    -------
    str — human-readable table suitable for print() or notebook display
    """
    header = f"{'Method':<22} {'Energy (Ha)':>15} {'|ΔE_FCI| (mHa)':>16} {'Chem. acc.':>12}"
    sep = "-" * len(header)
    lines = [header, sep]
    for method, energy in energies.items():
        error = compute_error_mHa(energy, fci_energy)
        acc = "✓" if check_chemical_accuracy(energy, fci_energy, threshold_mHa) else "✗"
        lines.append(f"{method:<22} {energy:>15.8f} {error:>16.4f} {acc:>12}")
    return "\n".join(lines)


def summarise_vqe_result(
    result: dict,
    fci_energy: float,
    molecule_name: str = "",
) -> dict:
    """
    Given a vqe_runner result dict, compute and return a concise
    summary suitable for building benchmark tables.

    Parameters
    ----------
    result        : dict returned by run_vqe_pennylane or run_adapt_vqe
    fci_energy    : FCI reference energy
    molecule_name : optional label for the molecule

    Returns
    -------
    dict with keys: molecule, method, energy, error_mHa,
                    chem_acc, n_params, est_cnots, n_iters
    """
    energy = result.get("energy") or result.get("final_energy")
    if energy is None:
        raise ValueError("Result dict has no 'energy' key.")

    error = compute_error_mHa(energy, fci_energy)
    return {
        "molecule":   molecule_name,
        "method":     result.get("method", "unknown"),
        "energy":     energy,
        "error_mHa":  error,
        "chem_acc":   check_chemical_accuracy(energy, fci_energy),
        "n_params":   result.get("n_params") or result.get("n_operators", 0),
        "est_cnots":  result.get("est_cnot_count", 0),
        "n_iters":    len(result.get("history", [])),
    }
