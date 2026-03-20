"""
tests/unit/test_analysis.py
-----------------------------
Unit tests for src/analysis.py — energy comparison and error utilities.
"""

import sys
import math
from unittest.mock import MagicMock
import pytest
import numpy as np

# Stub heavy dependencies
for mod in ["openfermion", "openfermionpyscf", "pyscf", "pennylane"]:
    sys.modules.setdefault(mod, MagicMock())

from src.analysis import (  # noqa: E402
    compute_correlation_energy,
    check_chemical_accuracy,
    format_energy_table,
    compute_error_mHa,
)


CHEM_ACCURACY_mHa = 1.6


class TestComputeCorrelationEnergy:
    def test_correlation_energy_is_negative_for_correlated_system(self):
        # FCI is always lower than HF (variational principle)
        result = compute_correlation_energy(hf=-1.0, fci=-1.1)
        assert result < 0

    def test_correlation_energy_formula(self):
        result = compute_correlation_energy(hf=-1.0, fci=-1.05)
        assert math.isclose(result, -0.05, rel_tol=1e-9)

    def test_correlation_energy_zero_when_equal(self):
        result = compute_correlation_energy(hf=-1.0, fci=-1.0)
        assert result == 0.0

    def test_correlation_energy_in_mHa(self):
        result = compute_correlation_energy(hf=-1.0, fci=-1.001, unit="mHa")
        assert math.isclose(result, -1.0, rel_tol=1e-6)


class TestCheckChemicalAccuracy:
    def test_within_accuracy_returns_true(self):
        # 0.5 mHa < 1.6 mHa threshold
        assert check_chemical_accuracy(vqe=-1.0005, fci=-1.0) is True

    def test_outside_accuracy_returns_false(self):
        # 5 mHa > 1.6 mHa threshold
        assert check_chemical_accuracy(vqe=-1.005, fci=-1.0) is False

    def test_exact_threshold_is_within(self):
        # Exactly 1.6 mHa: should pass (<=)
        fci = -1.0
        vqe = fci - 1.6e-3
        assert check_chemical_accuracy(vqe=vqe, fci=fci) is True

    def test_custom_threshold(self):
        assert check_chemical_accuracy(vqe=-1.001, fci=-1.0, threshold_mHa=0.5) is False
        assert check_chemical_accuracy(vqe=-1.0001, fci=-1.0, threshold_mHa=0.5) is True

    def test_vqe_above_fci_still_checked(self):
        # VQE can be above FCI numerically due to noise; abs() must be used
        assert check_chemical_accuracy(vqe=-0.9995, fci=-1.0) is True


class TestComputeErrorMHa:
    def test_error_is_positive(self):
        error = compute_error_mHa(vqe=-1.001, fci=-1.0)
        assert error > 0

    def test_error_formula(self):
        error = compute_error_mHa(vqe=-1.002, fci=-1.0)
        assert math.isclose(error, 2.0, rel_tol=1e-6)

    def test_error_symmetric_sign(self):
        # |VQE - FCI| should be same regardless of sign
        e1 = compute_error_mHa(vqe=-1.002, fci=-1.0)
        e2 = compute_error_mHa(vqe=-0.998, fci=-1.0)
        assert math.isclose(e1, e2, rel_tol=1e-9)


class TestFormatEnergyTable:
    def test_returns_string(self):
        data = {
            "HF": -1.0,
            "CCSD": -1.01,
            "FCI": -1.02,
            "VQE": -1.019,
        }
        result = format_energy_table(data, fci_energy=-1.02)
        assert isinstance(result, str)

    def test_table_contains_all_methods(self):
        data = {"HF": -1.0, "VQE": -1.01, "FCI": -1.02}
        result = format_energy_table(data, fci_energy=-1.02)
        for method in data:
            assert method in result

    def test_table_contains_chemical_accuracy_symbol(self):
        data = {"VQE": -1.0015, "FCI": -1.0}
        result = format_energy_table(data, fci_energy=-1.0)
        assert "✓" in result or "✗" in result
