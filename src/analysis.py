"""
analysis.py
-----------
Energy comparison, error analysis, and plotting utilities.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def energy_comparison_table(
    molecule_results: dict,
    reference_method: str = "fci",
) -> pd.DataFrame:
    """
    Build a DataFrame comparing HF, CCSD, VQE, and FCI energies.

    Parameters
    ----------
    molecule_results : dict mapping molecule_name -> dict with keys
                       'hf_energy', 'ccsd_energy', 'vqe_energy', 'fci_energy'
    reference_method : method to use as reference for error calculation

    Returns
    -------
    df : pd.DataFrame
    """
    rows = []
    for name, res in molecule_results.items():
        ref = res.get(f"{reference_method}_energy", np.nan)
        row = {
            "Molecule": name,
            "HF (Ha)": res.get("hf_energy", np.nan),
            "CCSD (Ha)": res.get("ccsd_energy", np.nan),
            "VQE (Ha)": res.get("vqe_energy", np.nan),
            "FCI (Ha)": res.get("fci_energy", np.nan),
            "VQE Error (mHa)": (res.get("vqe_energy", np.nan) - ref) * 1000,
            "CCSD Error (mHa)": (res.get("ccsd_energy", np.nan) - ref) * 1000,
        }
        rows.append(row)
    return pd.DataFrame(rows)


def plot_energy_convergence(history: list, title: str = "VQE Convergence"):
    """Plot VQE energy minimization vs. optimizer iteration."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(history, color="steelblue", linewidth=2)
    ax.axhline(history[-1], color="red", linestyle="--", label=f"Final: {history[-1]:.6f} Ha")
    ax.set_xlabel("Optimizer Iteration", fontsize=12)
    ax.set_ylabel("Energy (Hartree)", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    return fig


def plot_qubit_scaling(molecule_names: list, qubit_counts_jw: list, qubit_counts_bk: list):
    """Bar chart of qubit requirements: JW vs BK for each molecule."""
    x = np.arange(len(molecule_names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width/2, qubit_counts_jw, width, label="Jordan-Wigner", color="steelblue")
    ax.bar(x + width/2, qubit_counts_bk, width, label="Bravyi-Kitaev", color="darkorange")
    ax.set_xticks(x)
    ax.set_xticklabels(molecule_names, rotation=20, ha="right")
    ax.set_ylabel("Number of Qubits", fontsize=12)
    ax.set_title("Qubit Requirements: Alkenes & Alkynes (STO-3G)", fontsize=14)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    return fig
