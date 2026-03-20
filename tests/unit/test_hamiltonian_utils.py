"""
tests/unit/test_hamiltonian_utils.py
-------------------------------------
Unit tests for src/hamiltonian_utils.py.
All quantum-chemistry libraries are mocked.
"""

import sys
from unittest.mock import MagicMock, patch, call
import pytest
import numpy as np

# Mock all heavy dependencies before import
MOCKED_MODULES = [
    "openfermion", "openfermion.chem", "openfermion.utils",
    "openfermion.transforms", "openfermionpyscf",
    "pyscf", "pyscf.gto", "pyscf.scf",
]
for mod in MOCKED_MODULES:
    sys.modules.setdefault(mod, MagicMock())


class TestGetQubitHamiltonian:
    """Tests for get_qubit_hamiltonian function."""

    def _make_mock_mol(self, n_orbitals=5, n_electrons=10):
        mol = MagicMock()
        mol.n_orbitals = n_orbitals
        mol.n_electrons = n_electrons
        mol.get_molecular_hamiltonian.return_value = MagicMock()
        return mol

    def test_jordan_wigner_path_called(self):
        """Jordan-Wigner branch should call jordan_wigner, not bravyi_kitaev."""
        import src.hamiltonian_utils as hu
        mock_qubit_op = MagicMock()
        mock_jw = MagicMock(return_value=mock_qubit_op)
        mock_bk = MagicMock()
        mock_gfo = MagicMock()
        mock_count = MagicMock(return_value=10)
        with patch.multiple("src.hamiltonian_utils",
                            get_fermion_operator=mock_gfo,
                            jordan_wigner=mock_jw,
                            bravyi_kitaev=mock_bk,
                            count_qubits=mock_count):
            mol = self._make_mock_mol()
            hu.get_qubit_hamiltonian(mol, mapping="jordan_wigner", n_frozen_core=0)
            mock_jw.assert_called_once()
            mock_bk.assert_not_called()

    def test_bravyi_kitaev_path_called(self):
        import src.hamiltonian_utils as hu
        mock_qubit_op = MagicMock()
        mock_jw = MagicMock()
        mock_bk = MagicMock(return_value=mock_qubit_op)
        mock_gfo = MagicMock()
        mock_count = MagicMock(return_value=10)
        with patch.multiple("src.hamiltonian_utils",
                            get_fermion_operator=mock_gfo,
                            jordan_wigner=mock_jw,
                            bravyi_kitaev=mock_bk,
                            count_qubits=mock_count):
            mol = self._make_mock_mol()
            hu.get_qubit_hamiltonian(mol, mapping="bravyi_kitaev")
            mock_bk.assert_called_once()
            mock_jw.assert_not_called()

    def test_invalid_mapping_raises_value_error(self):
        import src.hamiltonian_utils as hu
        with patch.multiple("src.hamiltonian_utils",
                            get_fermion_operator=MagicMock(),
                            count_qubits=MagicMock(return_value=8)):
            mol = self._make_mock_mol()
            with pytest.raises(ValueError, match="Unknown mapping"):
                hu.get_qubit_hamiltonian(mol, mapping="invalid_mapping")

    def test_freeze_orbitals_called_when_frozen_core_nonzero(self):
        import src.hamiltonian_utils as hu
        mock_freeze = MagicMock(return_value=MagicMock())
        mock_jw = MagicMock(return_value=MagicMock())
        with patch.multiple("src.hamiltonian_utils",
                            get_fermion_operator=MagicMock(),
                            freeze_orbitals=mock_freeze,
                            jordan_wigner=mock_jw,
                            count_qubits=MagicMock(return_value=6)):
            mol = self._make_mock_mol(n_orbitals=7)
            hu.get_qubit_hamiltonian(mol, mapping="jordan_wigner", n_frozen_core=2)
            mock_freeze.assert_called_once()
            freeze_args = mock_freeze.call_args
            occupied_arg = freeze_args.args[1] if len(freeze_args.args) > 1 \
                else freeze_args.kwargs.get("occupied")
            assert occupied_arg == [0, 1]

    def test_freeze_orbitals_not_called_when_zero_frozen(self):
        import src.hamiltonian_utils as hu
        mock_freeze = MagicMock()
        with patch.multiple("src.hamiltonian_utils",
                            get_fermion_operator=MagicMock(),
                            freeze_orbitals=mock_freeze,
                            jordan_wigner=MagicMock(return_value=MagicMock()),
                            count_qubits=MagicMock(return_value=10)):
            mol = self._make_mock_mol()
            hu.get_qubit_hamiltonian(mol, mapping="jordan_wigner", n_frozen_core=0)
            mock_freeze.assert_not_called()

    def test_returns_tuple_qubit_op_and_n_qubits(self):
        import src.hamiltonian_utils as hu
        mock_qubit_op = MagicMock()
        with patch.multiple("src.hamiltonian_utils",
                            get_fermion_operator=MagicMock(),
                            jordan_wigner=MagicMock(return_value=mock_qubit_op),
                            count_qubits=MagicMock(return_value=12)):
            mol = self._make_mock_mol()
            result = hu.get_qubit_hamiltonian(mol, mapping="jordan_wigner")
            assert isinstance(result, tuple)
            assert len(result) == 2
            _, n_q = result
            assert n_q == 12
