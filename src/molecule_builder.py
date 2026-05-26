"""
molecule_builder.py
-------------------
PySCF Mole builders for alkenes/alkynes with publication geometries.

Hamiltonian construction is delegated to qiskit_nature.PySCFDriver (see
src.hamiltonian_utils). OpenFermion is only used here for the optional
legacy MolecularData / run_pyscf helpers; those imports are lazy so
the project no longer hard-depends on openfermionpyscf.
"""

from __future__ import annotations

import math

from pyscf import gto

# -------------------------------------------------------------------
# Equilibrium geometries (Angstrom)
# Ethylene  C=C 1.339 Å, C-H 1.086 Å, planar (H-C-H 117.4°)
# Acetylene C≡C 1.203 Å, C-H 1.060 Å, linear
# -------------------------------------------------------------------

_HCH_HALF = math.radians(117.4 / 2.0)
_CH_ETH = 1.086
_DY = _CH_ETH * math.sin(_HCH_HALF)   # ~0.929 Å
_DZ = _CH_ETH * math.cos(_HCH_HALF)   # ~0.547 Å

MOLECULE_GEOMETRIES = {
    "ethylene": [
        ("C", (0.000,  0.000,  0.000)),
        ("C", (0.000,  0.000,  1.339)),
        ("H", (0.000,  _DY,  -_DZ)),
        ("H", (0.000, -_DY,  -_DZ)),
        ("H", (0.000,  _DY,   1.339 + _DZ)),
        ("H", (0.000, -_DY,   1.339 + _DZ)),
    ],
    "acetylene": [
        ("C", (0.000, 0.000,  0.000)),
        ("C", (0.000, 0.000,  1.203)),
        ("H", (0.000, 0.000, -1.060)),
        ("H", (0.000, 0.000,  2.263)),
    ],
    "1-butene": [
        ("C", (0.000,  0.000,  0.000)),
        ("C", (0.000,  0.000,  1.339)),
        ("C", (0.000,  1.284,  2.089)),
        ("C", (0.000,  1.284,  3.571)),
        ("H", (0.000,  0.926,  -0.546)),
        ("H", (0.000, -0.926,  -0.546)),
        ("H", (0.000, -0.926,   1.885)),
        ("H", (0.000,  2.180,  1.480)),
        ("H", (-0.880, 1.284,  4.213)),
        ("H", (0.880,  1.284,  4.213)),
        ("H", (0.000,  2.210,  3.571)),
    ],
    "propyne": [
        ("C", (0.000,  0.000,  0.000)),
        ("C", (0.000,  0.000,  1.206)),
        ("C", (0.000,  0.000,  2.661)),
        ("H", (0.000,  0.000, -1.063)),
        ("H", (1.023,  0.000,  3.060)),
        ("H", (-0.512,  0.887, 3.060)),
        ("H", (-0.512, -0.887, 3.060)),
    ],
    "1,3-butadiene": [
        ("C", (0.000,  0.000,  0.000)),
        ("C", (0.000,  0.000,  1.339)),
        ("C", (0.000,  1.261,  2.076)),
        ("C", (0.000,  1.261,  3.415)),
        ("H", (0.000,  0.926,  -0.546)),
        ("H", (0.000, -0.926,  -0.546)),
        ("H", (0.000, -0.926,   1.885)),
        ("H", (0.000,  2.187,  1.530)),
        ("H", (0.000,  0.335,  3.961)),
        ("H", (0.000,  2.187,  3.961)),
    ],
}

MOLECULE_CHARGES = {k: 0 for k in MOLECULE_GEOMETRIES}
MOLECULE_MULTIPLICITIES = {k: 1 for k in MOLECULE_GEOMETRIES}


def geometry_to_xyz_string(name: str) -> str:
    """Return a ';'-joined XYZ string suitable for qiskit_nature PySCFDriver."""
    geom = MOLECULE_GEOMETRIES[name]
    return "; ".join(f"{sym} {x:.6f} {y:.6f} {z:.6f}" for sym, (x, y, z) in geom)


def build_pyscf_mol(name: str, basis: str = "sto-3g", verbose: int = 0):
    """Return a built PySCF Mole object for the named molecule."""
    geom = MOLECULE_GEOMETRIES[name]
    mol = gto.Mole()
    mol.atom = geom
    mol.basis = basis
    mol.unit = "Angstrom"
    mol.charge = MOLECULE_CHARGES[name]
    mol.spin = MOLECULE_MULTIPLICITIES[name] - 1
    mol.verbose = verbose
    mol.build()
    return mol


# -------------------------------------------------------------------
# Legacy OpenFermion helpers (lazy/optional)
# -------------------------------------------------------------------
# Names retained at module level for backwards-compat with unit tests
# that patch "src.molecule_builder.MolecularData" / ".run_pyscf".

try:  # pragma: no cover - exercised via mock in tests
    from openfermion.chem import MolecularData  # type: ignore
except Exception:  # pragma: no cover
    MolecularData = None  # type: ignore

try:  # pragma: no cover - optional dependency
    from openfermionpyscf import run_pyscf  # type: ignore
except Exception:  # pragma: no cover
    run_pyscf = None  # type: ignore


def build_molecular_data(
    name: str,
    basis: str = "sto-3g",
    multiplicity: int = 1,
    description: str = "",
):
    """Legacy: return an OpenFermion MolecularData (optional dependency)."""
    geom = MOLECULE_GEOMETRIES[name]
    if MolecularData is None:
        raise ImportError("openfermion is required for build_molecular_data")
    return MolecularData(
        geometry=geom,
        basis=basis,
        multiplicity=multiplicity,
        charge=MOLECULE_CHARGES[name],
        description=description or name,
    )


def run_classical_calcs(
    name: str,
    basis: str = "sto-3g",
    run_scf: bool = True,
    run_ccsd: bool = True,
    run_fci: bool = True,
):
    """Legacy HF/CCSD/FCI via OpenFermion-PySCF (optional dependency)."""
    if run_pyscf is None:
        raise ImportError("openfermionpyscf is required for run_classical_calcs")
    mol_data = build_molecular_data(name, basis=basis)
    return run_pyscf(mol_data, run_scf=run_scf, run_ccsd=run_ccsd, run_fci=run_fci)


# -------------------------------------------------------------------
# Classical reference energies via PySCF directly (no openfermionpyscf)
# -------------------------------------------------------------------

def run_pyscf_references(
    name: str,
    basis: str = "sto-3g",
    methods: tuple = ("HF", "MP2", "CISD", "CCSD", "CCSD(T)", "FCI"),
    verbose: int = 0,
) -> dict:
    """Compute classical reference energies (Ha) directly via PySCF.

    Returns {method_name: energy_Ha}. FCI may be skipped silently for
    bases where it would be intractable.
    """
    from pyscf import scf, mp, ci, cc, fci

    mol = build_pyscf_mol(name, basis=basis, verbose=verbose)
    out: dict = {}

    mf = scf.RHF(mol)
    mf.verbose = verbose
    e_hf = float(mf.kernel())
    if "HF" in methods:
        out["HF"] = e_hf

    if "MP2" in methods:
        out["MP2"] = float(mp.MP2(mf).kernel()[0]) + e_hf

    if "CISD" in methods:
        e_corr = float(ci.CISD(mf).kernel()[0])
        out["CISD"] = e_hf + e_corr

    if "CCSD" in methods or "CCSD(T)" in methods:
        mycc = cc.CCSD(mf)
        e_ccsd_corr, _, _ = mycc.kernel()
        if "CCSD" in methods:
            out["CCSD"] = e_hf + float(e_ccsd_corr)
        if "CCSD(T)" in methods:
            try:
                e_t = float(mycc.ccsd_t())
                out["CCSD(T)"] = e_hf + float(e_ccsd_corr) + e_t
            except Exception:
                pass

    if "FCI" in methods:
        try:
            cisolver = fci.FCI(mf)
            e_fci, _ = cisolver.kernel()
            out["FCI"] = float(e_fci)
        except Exception:
            pass

    return out
