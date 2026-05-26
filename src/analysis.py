"""
analysis.py
-----------
Energy comparison, unit conversion, and chemical-accuracy utilities.
"""

from __future__ import annotations

# ----------------------------------------------------------------
# Exact unit conversions (1 Hartree = ...)
# ----------------------------------------------------------------
HA_TO_KCAL_MOL: float = 627.5094740631
HA_TO_KJ_MOL:   float = 2625.4996394799
HA_TO_EV:       float = 27.211396
HA_TO_mHA:      float = 1000.0

# Chemical accuracy: 1 kcal/mol -> 1.5936 mHa
CHEM_ACCURACY_KCAL_MOL: float = 1.0
CHEM_ACCURACY_mHa: float = 1.0 / HA_TO_KCAL_MOL * HA_TO_mHA  # ~1.5936


def ha_to_kcal_mol(e_ha: float) -> float:
    return e_ha * HA_TO_KCAL_MOL


def ha_to_kj_mol(e_ha: float) -> float:
    return e_ha * HA_TO_KJ_MOL


def ha_to_ev(e_ha: float) -> float:
    return e_ha * HA_TO_EV


def compute_correlation_energy(hf: float, fci: float, unit: str = "Ha") -> float:
    """E_corr = E_FCI - E_HF (negative for correlated systems)."""
    corr = fci - hf
    if unit == "mHa":
        return corr * HA_TO_mHA
    if unit == "kcal/mol":
        return corr * HA_TO_KCAL_MOL
    return corr


def compute_error_mHa(vqe: float, fci: float) -> float:
    """Absolute error |E_VQE - E_FCI| in mHa (always non-negative)."""
    return abs(vqe - fci) * HA_TO_mHA


def compute_error_kcal_mol(vqe: float, fci: float) -> float:
    """Absolute error |E_VQE - E_FCI| in kcal/mol."""
    return abs(vqe - fci) * HA_TO_KCAL_MOL


def signed_error_mHa(vqe: float, fci: float) -> float:
    """Signed error (E_VQE - E_FCI) in mHa; positive when VQE above FCI."""
    return (vqe - fci) * HA_TO_mHA


def check_chemical_accuracy(
    vqe: float,
    fci: float,
    threshold_mHa: float = CHEM_ACCURACY_mHa,
) -> bool:
    """True iff |E_VQE - E_FCI| ≤ threshold (default ≈ 1.5936 mHa)."""
    return compute_error_mHa(vqe, fci) <= threshold_mHa


def format_energy_table(
    energies: dict[str, float],
    fci_energy: float,
    threshold_mHa: float = CHEM_ACCURACY_mHa,
) -> str:
    """Human-readable energy table comparing methods to FCI."""
    header = f"{'Method':<22} {'Energy (Ha)':>15} {'|ΔE_FCI| (mHa)':>16} {'Chem. acc.':>12}"
    sep = "-" * len(header)
    lines = [header, sep]
    for method, energy in energies.items():
        error = compute_error_mHa(energy, fci_energy)
        acc = "✓" if check_chemical_accuracy(energy, fci_energy, threshold_mHa) else "✗"
        lines.append(f"{method:<22} {energy:>15.8f} {error:>16.4f} {acc:>12}")
    return "\n".join(lines)


def build_comparison_dataframe(
    energies: dict[str, float],
    fci_energy: float,
    threshold_mHa: float = CHEM_ACCURACY_mHa,
):
    """Return a pandas DataFrame with publication columns."""
    import pandas as pd
    rows = []
    for method, e in energies.items():
        err_mha = signed_error_mHa(e, fci_energy)
        err_kcal = err_mha / HA_TO_mHA * HA_TO_KCAL_MOL
        rows.append({
            "Method": method,
            "Energy (Ha)": float(e),
            "Error vs FCI (mHa)": float(err_mha),
            "Error (kcal/mol)": float(err_kcal),
            "Chemical Accuracy?": "✓" if abs(err_mha) <= threshold_mHa else "✗",
        })
    return pd.DataFrame(rows)


def summarise_vqe_result(
    result: dict,
    fci_energy: float,
    molecule_name: str = "",
) -> dict:
    """Concise summary suitable for benchmark tables."""
    energy = (result.get("energy")
              or result.get("final_energy")
              or result.get("final_energy_Ha"))
    if energy is None:
        raise ValueError("Result dict has no energy key.")
    error = compute_error_mHa(energy, fci_energy)
    return {
        "molecule":   molecule_name,
        "method":     result.get("method", "unknown"),
        "energy":     float(energy),
        "error_mHa":  float(error),
        "chem_acc":   check_chemical_accuracy(energy, fci_energy),
        "n_params":   result.get("n_params") or result.get("n_operators", 0),
        "est_cnots":  result.get("est_cnot_count", 0),
        "n_iters":    len(result.get("history") or result.get("energy_history", [])),
    }
