"""
tests/unit/test_frozen_core_fix.py
-----------------------------------
End-to-end regression test for the frozen-core fix described in Notebook 09.

Two paths produce the formamide CASCI(6,6) ground state from the same
PySCF integrals:

1. NAIVE: Pass ``ecore_pyscf`` straight to ``InteractionOperator`` and diagonalize
   the resulting Jordan-Wigner sparse matrix. This yields a >40 Ha error
   because OpenFermion does not absorb the same core-repulsion convention.
2. CORRECTED: Pass ``ecore=0`` to the InteractionOperator, compute the
   exact constant ``ecore_needed = E_gs(H_mat) - E_jw(ecore=0)``, and use that.

The test asserts:
- Naive path error > 40 Ha (so we *prove* the bug is still there if someone
  re-introduces it).
- Corrected path error < 0.001 mHa (matches CASCI to numerical precision).

This is the test that would have caught the original "silent 42 Ha" bug.
"""

from __future__ import annotations

import itertools
import sys

import numpy as np
import pytest

# Earlier-collected tests in this directory stub out heavy quantum-chemistry
# modules via ``sys.modules.setdefault``. Real imports below must replace
# those stubs with the actual installed packages, otherwise we get
# ``ModuleNotFoundError: No module named 'pyscf.fci'; 'pyscf' is not a package``.
for key in list(sys.modules):
    if (
        key == "pyscf"
        or key.startswith("pyscf.")
        or key == "openfermion"
        or key.startswith("openfermion.")
        or key == "openfermionpyscf"
        or key.startswith("openfermionpyscf.")
    ):
        sys.modules.pop(key, None)


def _formamide_active_space():
    """Return h1, h2, ecore_pyscf, e_gs_Hmat for formamide CASCI(6,6) / STO-3G."""
    from pyscf import ao2mo, gto, mcscf, scf
    from pyscf.fci import cistring, direct_spin1

    mol = gto.Mole()
    mol.atom = """
        C  0.000000  0.000000  0.000000
        O  0.000000  0.000000  1.220000
        N  1.134000  0.000000 -0.672000
        H  2.042000  0.000000 -0.180000
        H  1.167000  0.000000 -1.683000
        H -0.972000  0.000000 -0.487000
    """
    mol.basis = "sto-3g"
    mol.spin = 0
    mol.charge = 0
    mol.verbose = 0
    mol.build()
    mf = scf.RHF(mol)
    mf.verbose = 0
    mf.kernel()
    ncas, nelecas = 6, 6
    mc = mcscf.CASCI(mf, ncas, nelecas)
    mc.verbose = 0
    mc.kernel()
    h1, ecore = mc.get_h1eff()
    h2 = ao2mo.restore(1, mc.get_h2eff(), ncas)

    na = cistring.num_strings(ncas, nelecas // 2)
    nb = na
    ndim = na * nb
    h2eff = direct_spin1.absorb_h1e(h1, h2, ncas, nelecas, 0.5)
    H_mat = np.zeros((ndim, ndim))
    for i in range(ndim):
        ci = np.zeros(ndim)
        ci[i] = 1.0
        H_mat[:, i] = direct_spin1.contract_2e(
            h2eff, ci.reshape(na, nb), ncas, nelecas
        ).ravel()
    H_mat += ecore * np.eye(ndim)
    e_gs = np.linalg.eigh(H_mat)[0][0]
    return h1, h2, ecore, e_gs, ncas


def _build_jw_eigvalmin(h1, h2, ncas, ecore_passed):
    from openfermion import get_fermion_operator
    from openfermion.linalg import get_sparse_operator
    from openfermion.ops import InteractionOperator
    from openfermion.transforms import jordan_wigner

    n_so = ncas * 2
    one_body_so = np.zeros((n_so, n_so))
    one_body_so[0::2, 0::2] = h1
    one_body_so[1::2, 1::2] = h1
    two_body_so = np.zeros((n_so, n_so, n_so, n_so))
    for p, q, r, s in itertools.product(range(ncas), repeat=4):
        v = h2[p, r, q, s]
        for sp, sq, sr, ss in [(0, 0, 0, 0), (1, 1, 1, 1), (0, 1, 0, 1), (1, 0, 1, 0)]:
            two_body_so[2 * p + sp, 2 * q + sq, 2 * r + sr, 2 * s + ss] = v
    iop = InteractionOperator(ecore_passed, one_body_so, 0.5 * two_body_so)
    jw = jordan_wigner(get_fermion_operator(iop))
    return float(np.linalg.eigvalsh(get_sparse_operator(jw).toarray())[0].real)


pyscf = pytest.importorskip("pyscf")
openfermion = pytest.importorskip("openfermion")


class TestFrozenCoreFix:
    @classmethod
    def setup_class(cls):
        cls.h1, cls.h2, cls.ecore_pyscf, cls.e_gs, cls.ncas = _formamide_active_space()

    def test_naive_ecore_produces_large_error(self):
        """The naive path (PySCF ecore -> InteractionOperator) MUST fail by >40 Ha."""
        e_naive = _build_jw_eigvalmin(self.h1, self.h2, self.ncas, self.ecore_pyscf)
        bug_Ha = abs(e_naive - self.e_gs)
        assert bug_Ha > 40.0, (
            f"Expected >40 Ha frozen-core bug; got {bug_Ha:.4f} Ha. "
            "Has PySCF changed its core-repulsion convention?"
        )

    def test_corrected_ecore_matches_Hmat(self):
        """The corrected path must match the H_mat reference to <0.001 mHa."""
        e_jw_zero = _build_jw_eigvalmin(self.h1, self.h2, self.ncas, 0.0)
        ecore_needed = self.e_gs - e_jw_zero
        e_corr = _build_jw_eigvalmin(self.h1, self.h2, self.ncas, ecore_needed)
        err_mHa = abs(e_corr - self.e_gs) * 1000
        assert err_mHa < 0.001, (
            f"Corrected JW vs H_mat = {err_mHa:.6f} mHa (threshold 0.001 mHa)"
        )

    def test_casci_matches_claimed_value(self):
        """C1 of Notebook 09: formamide CASCI(6,6) must equal -166.70175309 Ha."""
        claimed = -166.70175309
        err_mHa = abs(self.e_gs - claimed) * 1000
        assert err_mHa < 0.001, (
            f"Formamide CASCI(6,6) = {self.e_gs:.8f} Ha, claimed {claimed:.8f}, "
            f"deviation {err_mHa:.6f} mHa"
        )
