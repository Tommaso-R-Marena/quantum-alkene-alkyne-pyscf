"""
tests/unit/test_molecule_builder.py
------------------------------------
Unit tests for src/molecule_builder.py.
All heavy quantum-chemistry calls (run_pyscf) are mocked so
these tests run in <5 s with zero GPU/CPU requirements.
"""

import pytest
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# We mock openfermion, openfermionpyscf, and pyscf before importing src code
# ---------------------------------------------------------------------------
import sys

MOCKED_MODULES = [
    "pyscf", "pyscf.gto", "pyscf.scf", "pyscf.cc",
    "openfermion", "openfermion.chem", "openfermion.utils",
    "openfermion.transforms", "openfermionpyscf",
]
for mod in MOCKED_MODULES:
    sys.modules.setdefault(mod, MagicMock())

from src.molecule_builder import (  # noqa: E402
    MOLECULE_GEOMETRIES,
    MOLECULE_CHARGES,
    MOLECULE_MULTIPLICITIES,
    build_molecular_data,
    run_classical_calcs,
)


class TestMoleculeGeometries:
    """Geometry registry is correct and complete."""

    EXPECTED_MOLECULES = {"ethylene", "1-butene", "acetylene", "propyne", "1,3-butadiene"}

    def test_all_expected_molecules_present(self):
        assert self.EXPECTED_MOLECULES.issubset(set(MOLECULE_GEOMETRIES.keys()))

    @pytest.mark.parametrize("name", ["ethylene", "acetylene", "propyne"])
    def test_geometry_is_list_of_tuples(self, name):
        geom = MOLECULE_GEOMETRIES[name]
        assert isinstance(geom, list)
        assert len(geom) > 0
        for entry in geom:
            assert len(entry) == 2
            symbol, coords = entry
            assert isinstance(symbol, str)
            assert len(coords) == 3

    @pytest.mark.parametrize("name", ["ethylene", "acetylene", "propyne"])
    def test_geometry_coordinates_are_finite_floats(self, name):
        import math
        for _, (x, y, z) in MOLECULE_GEOMETRIES[name]:
            assert math.isfinite(x) and math.isfinite(y) and math.isfinite(z)

    def test_ethylene_atom_count(self):
        geom = MOLECULE_GEOMETRIES["ethylene"]
        symbols = [s for s, _ in geom]
        assert symbols.count("C") == 2
        assert symbols.count("H") == 4

    def test_acetylene_linear_geometry(self):
        """All atoms in acetylene must lie on the z-axis (x=y=0)."""
        for _, (x, y, _) in MOLECULE_GEOMETRIES["acetylene"]:
            assert abs(x) < 1e-9, f"Acetylene x-coord off-axis: {x}"
            assert abs(y) < 1e-9, f"Acetylene y-coord off-axis: {y}"

    def test_acetylene_bond_length_range(self):
        """C≡C bond in acetylene should be 1.15–1.25 Å."""
        carbons = [coords for sym, coords in MOLECULE_GEOMETRIES["acetylene"] if sym == "C"]
        assert len(carbons) == 2
        dz = abs(carbons[1][2] - carbons[0][2])
        assert 1.15 < dz < 1.25, f"C≡C bond length out of range: {dz}"

    def test_ethylene_bond_length_range(self):
        """C=C bond in ethylene should be 1.30–1.36 Å."""
        carbons = [coords for sym, coords in MOLECULE_GEOMETRIES["ethylene"] if sym == "C"]
        assert len(carbons) == 2
        dz = abs(carbons[1][2] - carbons[0][2])
        assert 1.30 < dz < 1.36, f"C=C bond length out of range: {dz}"

    def test_all_charges_are_zero(self):
        for name, charge in MOLECULE_CHARGES.items():
            assert charge == 0, f"{name} has non-zero charge: {charge}"

    def test_all_multiplicities_are_singlet(self):
        for name, mult in MOLECULE_MULTIPLICITIES.items():
            assert mult == 1, f"{name} has non-singlet multiplicity: {mult}"


class TestBuildMolecularData:
    """build_molecular_data returns correct MolecularData args."""

    def test_build_molecular_data_called_with_correct_geometry(self):
        """Verify the geometry passed is the registered one."""
        with patch("src.molecule_builder.MolecularData") as mock_md:
            mock_md.return_value = MagicMock()
            build_molecular_data("ethylene", basis="sto-3g")
            call_kwargs = mock_md.call_args
            geometry_arg = call_kwargs.kwargs.get("geometry") or call_kwargs.args[0]
            assert geometry_arg == MOLECULE_GEOMETRIES["ethylene"]

    def test_build_molecular_data_basis_forwarded(self):
        with patch("src.molecule_builder.MolecularData") as mock_md:
            mock_md.return_value = MagicMock()
            build_molecular_data("acetylene", basis="6-31g")
            call_kwargs = mock_md.call_args
            basis_arg = call_kwargs.kwargs.get("basis") or call_kwargs.args[1]
            assert basis_arg == "6-31g"

    def test_unknown_molecule_raises_key_error(self):
        with pytest.raises(KeyError):
            build_molecular_data("nonexistent_molecule")


class TestRunClassicalCalcs:
    """run_classical_calcs correctly calls run_pyscf with right flags."""

    def test_run_pyscf_called_with_all_flags(self):
        with patch("src.molecule_builder.MolecularData") as mock_md, \
             patch("src.molecule_builder.run_pyscf") as mock_pyscf:
            mock_md.return_value = MagicMock()
            mock_pyscf.return_value = MagicMock()
            run_classical_calcs("ethylene", run_scf=True, run_ccsd=True, run_fci=True)
            mock_pyscf.assert_called_once()
            _, kwargs = mock_pyscf.call_args
            assert kwargs["run_scf"] is True
            assert kwargs["run_ccsd"] is True
            assert kwargs["run_fci"] is True

    def test_run_pyscf_flags_forwarded_correctly(self):
        with patch("src.molecule_builder.MolecularData") as mock_md, \
             patch("src.molecule_builder.run_pyscf") as mock_pyscf:
            mock_md.return_value = MagicMock()
            mock_pyscf.return_value = MagicMock()
            run_classical_calcs("acetylene", run_scf=True, run_ccsd=False, run_fci=False)
            _, kwargs = mock_pyscf.call_args
            assert kwargs["run_ccsd"] is False
            assert kwargs["run_fci"] is False
