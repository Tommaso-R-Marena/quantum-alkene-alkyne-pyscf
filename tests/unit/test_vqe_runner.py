"""
tests/unit/test_vqe_runner.py
------------------------------
Unit tests for src/vqe_runner.py.
All PennyLane and quantum-chemistry calls are mocked so tests are fast.
"""

import sys
import math
from unittest.mock import MagicMock, patch, call
import pytest
import numpy as np

# Stub heavy dependencies before import
MOCKED_MODULES = [
    "pennylane", "pennylane.qchem",
    "scipy", "scipy.optimize",
    "openfermion", "openfermion.chem", "openfermion.utils",
    "openfermionpyscf",
]
for mod in MOCKED_MODULES:
    sys.modules.setdefault(mod, MagicMock())


class TestRunVqePennylane:
    """Tests for run_vqe_pennylane (UCCSD-VQE)."""

    def _make_mock_qml(self, energies):
        """Build a minimal pennylane mock that returns given energy sequence."""
        qml = MagicMock()
        energy_iter = iter(energies)
        optimizer = MagicMock()
        optimizer.step_and_cost.side_effect = [
            (np.zeros(4), e) for e in energies
        ]
        qml.GradientDescentOptimizer.return_value = optimizer
        qml.AdamOptimizer.return_value = optimizer
        qml.device.return_value = MagicMock()
        qml.qnode = lambda dev, **kw: lambda fn: fn
        qml.BasisState = MagicMock()
        qml.AllSinglesDoubles = MagicMock()
        qml.expval = MagicMock(return_value=MagicMock())
        qchem = MagicMock()
        qchem.excitations.return_value = ([[0, 2], [1, 3]], [[0, 1, 2, 3]])
        qchem.hf_state.return_value = np.array([1, 1, 0, 0])
        qml.qchem = qchem
        return qml

    def test_result_keys_present(self):
        """Return dict must contain all documented keys."""
        from src.vqe_runner import run_vqe_pennylane
        h_mock = MagicMock()
        energies = [-1.0 + i*0.01 for i in range(10)]
        energies[-1] = energies[-2]  # trigger convergence
        mock_opt = MagicMock()
        mock_opt.step_and_cost.side_effect = [(np.zeros(3), e) for e in energies]
        with patch("src.vqe_runner.qml") as mock_qml:
            mock_qml.AdamOptimizer.return_value = mock_opt
            mock_qml.GradientDescentOptimizer.return_value = mock_opt
            mock_qml.device.return_value = MagicMock()
            mock_qml.qnode = lambda dev, **kw: (lambda fn: fn)
            mock_qml.qchem.excitations.return_value = ([[0, 2]], [[0, 1, 2, 3]])
            mock_qml.qchem.hf_state.return_value = np.array([1, 1, 0, 0])
            result = run_vqe_pennylane(h_mock, n_qubits=4, n_electrons=2,
                                       max_iter=10, verbose=False)
        required_keys = {"method", "energy", "history", "n_params", "est_cnot_count"}
        assert required_keys.issubset(result.keys())

    def test_energy_is_finite(self):
        from src.vqe_runner import run_vqe_pennylane
        h_mock = MagicMock()
        mock_opt = MagicMock()
        mock_opt.step_and_cost.side_effect = [
            (np.zeros(3), -1.0 - i*0.001) for i in range(50)
        ]
        with patch("src.vqe_runner.qml") as mock_qml:
            mock_qml.AdamOptimizer.return_value = mock_opt
            mock_qml.GradientDescentOptimizer.return_value = mock_opt
            mock_qml.device.return_value = MagicMock()
            mock_qml.qnode = lambda dev, **kw: (lambda fn: fn)
            mock_qml.qchem.excitations.return_value = ([[0, 2]], [[0, 1, 2, 3]])
            mock_qml.qchem.hf_state.return_value = np.array([1, 1, 0, 0])
            result = run_vqe_pennylane(h_mock, n_qubits=4, n_electrons=2,
                                       max_iter=50, verbose=False)
        assert math.isfinite(result["energy"])

    def test_n_params_equals_singles_plus_doubles(self):
        from src.vqe_runner import run_vqe_pennylane
        h_mock = MagicMock()
        singles = [[0, 2], [1, 3]]
        doubles = [[0, 1, 2, 3]]
        mock_opt = MagicMock()
        mock_opt.step_and_cost.side_effect = [
            (np.zeros(len(singles)+len(doubles)), -1.0) for _ in range(50)
        ]
        with patch("src.vqe_runner.qml") as mock_qml:
            mock_qml.AdamOptimizer.return_value = mock_opt
            mock_qml.GradientDescentOptimizer.return_value = mock_opt
            mock_qml.device.return_value = MagicMock()
            mock_qml.qnode = lambda dev, **kw: (lambda fn: fn)
            mock_qml.qchem.excitations.return_value = (singles, doubles)
            mock_qml.qchem.hf_state.return_value = np.array([1, 1, 0, 0])
            result = run_vqe_pennylane(h_mock, n_qubits=4, n_electrons=2,
                                       max_iter=50, verbose=False)
        assert result["n_params"] == len(singles) + len(doubles)

    def test_cnot_estimate_formula(self):
        """est_cnot_count = 8 * n_doubles + 2 * n_singles."""
        from src.vqe_runner import run_vqe_pennylane
        singles = [[0, 2], [1, 3]]
        doubles = [[0, 1, 2, 3], [0, 1, 4, 5]]
        expected_cnots = 8 * len(doubles) + 2 * len(singles)
        mock_opt = MagicMock()
        mock_opt.step_and_cost.side_effect = [
            (np.zeros(4), -1.0) for _ in range(50)
        ]
        with patch("src.vqe_runner.qml") as mock_qml:
            mock_qml.AdamOptimizer.return_value = mock_opt
            mock_qml.GradientDescentOptimizer.return_value = mock_opt
            mock_qml.device.return_value = MagicMock()
            mock_qml.qnode = lambda dev, **kw: (lambda fn: fn)
            mock_qml.qchem.excitations.return_value = (singles, doubles)
            mock_qml.qchem.hf_state.return_value = np.array([1, 1, 0, 0])
            result = run_vqe_pennylane(MagicMock(), n_qubits=6, n_electrons=2,
                                       max_iter=50, verbose=False)
        assert result["est_cnot_count"] == expected_cnots

    def test_history_length_positive(self):
        from src.vqe_runner import run_vqe_pennylane
        mock_opt = MagicMock()
        mock_opt.step_and_cost.side_effect = [
            (np.zeros(3), -1.0) for _ in range(10)
        ]
        with patch("src.vqe_runner.qml") as mock_qml:
            mock_qml.AdamOptimizer.return_value = mock_opt
            mock_qml.GradientDescentOptimizer.return_value = mock_opt
            mock_qml.device.return_value = MagicMock()
            mock_qml.qnode = lambda dev, **kw: (lambda fn: fn)
            mock_qml.qchem.excitations.return_value = ([[0, 2]], [[0, 1, 2, 3]])
            mock_qml.qchem.hf_state.return_value = np.array([1, 1, 0, 0])
            result = run_vqe_pennylane(MagicMock(), n_qubits=4, n_electrons=2,
                                       max_iter=10, verbose=False)
        assert len(result["history"]) > 0


class TestBuildOperatorPool:
    """Tests for build_operator_pool."""

    def test_pool_contains_singles_and_doubles(self):
        from src.vqe_runner import build_operator_pool
        singles = [[0, 2], [1, 3]]
        doubles = [[0, 1, 2, 3]]
        with patch("src.vqe_runner.qml") as mock_qml:
            mock_qml.qchem.excitations.return_value = (singles, doubles)
            pool = build_operator_pool(n_qubits=4, n_electrons=2)
        types = [t for _, t, _ in pool]
        assert "single" in types
        assert "double" in types

    def test_pool_labels_are_strings(self):
        from src.vqe_runner import build_operator_pool
        with patch("src.vqe_runner.qml") as mock_qml:
            mock_qml.qchem.excitations.return_value = ([[0, 2]], [[0, 1, 2, 3]])
            pool = build_operator_pool(n_qubits=4, n_electrons=2)
        for label, _, _ in pool:
            assert isinstance(label, str)
            assert len(label) > 0

    def test_pool_size_equals_singles_plus_doubles(self):
        from src.vqe_runner import build_operator_pool
        singles = [[0, 2], [1, 3], [0, 4]]
        doubles = [[0, 1, 2, 3], [0, 1, 4, 5]]
        with patch("src.vqe_runner.qml") as mock_qml:
            mock_qml.qchem.excitations.return_value = (singles, doubles)
            pool = build_operator_pool(n_qubits=6, n_electrons=2)
        assert len(pool) == len(singles) + len(doubles)
