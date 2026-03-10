"""
molecule_builder.py
-------------------
Builds PySCF Mole objects and OpenFermion MolecularData for
alkenes and alkynes using optimized geometries.
"""

from pyscf import gto
from openfermion.chem import MolecularData
from openfermionpyscf import run_pyscf

# -------------------------------------------------------------------
# Equilibrium geometries (Angstrom) — B3LYP/6-31G* optimized
# -------------------------------------------------------------------

MOLECULE_GEOMETRIES = {
    # Alkenes
    "ethylene": [
        ("C", (0.000,  0.000,  0.000)),
        ("C", (0.000,  0.000,  1.339)),
        ("H", (0.000,  0.926,  -0.546)),
        ("H", (0.000, -0.926,  -0.546)),
        ("H", (0.000,  0.926,   1.885)),
        ("H", (0.000, -0.926,   1.885)),
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
    # Alkynes
    "acetylene": [
        ("C", (0.000, 0.000,  0.000)),
        ("C", (0.000, 0.000,  1.203)),
        ("H", (0.000, 0.000, -1.063)),
        ("H", (0.000, 0.000,  2.266)),
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
    # Dienes
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


def build_pyscf_mol(name: str, basis: str = "sto-3g", verbose: int = 0):
    """Return a PySCF Mole object for the named molecule."""
    geom = MOLECULE_GEOMETRIES[name]
    mol = gto.Mole()
    mol.atom = geom
    mol.basis = basis
    mol.charge = MOLECULE_CHARGES[name]
    mol.spin = MOLECULE_MULTIPLICITIES[name] - 1
    mol.verbose = verbose
    mol.build()
    return mol


def build_molecular_data(
    name: str,
    basis: str = "sto-3g",
    multiplicity: int = 1,
    description: str = "",
):
    """Return an OpenFermion MolecularData object ready for run_pyscf."""
    geom = MOLECULE_GEOMETRIES[name]
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
    """Run HF, CCSD, FCI via OpenFermion-PySCF and return MolecularData."""
    mol_data = build_molecular_data(name, basis=basis)
    mol_data = run_pyscf(
        mol_data,
        run_scf=run_scf,
        run_ccsd=run_ccsd,
        run_fci=run_fci,
    )
    return mol_data
