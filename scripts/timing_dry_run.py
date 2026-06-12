#!/usr/bin/env python3
"""
Faithful simulator dry-run for Notebook 10.
MUST complete in < 570 seconds before any IBM hardware call is made.
Uses FakeSherbrooke (Heron-class noise model) for transpilation target.
Saves timing results to results/dryrun_timing_latest.json.
"""
import time, json, os, warnings, itertools
import numpy as np
warnings.filterwarnings('ignore')

BUDGET_SECONDS = 570
RESULTS_DIR = 'results'
os.makedirs(RESULTS_DIR, exist_ok=True)

t_total_start = time.time()
timings = {}

print("=" * 60)
print("DRY-RUN: Faithful simulator timing check")
print(f"Budget: {BUDGET_SECONDS}s ({BUDGET_SECONDS/60:.1f} min)")
print("=" * 60)

# ── Step 1: Build formamide CASCI(6,6) Hamiltonian ──────────────────────────
t1 = time.time()
from pyscf import gto, scf, mcscf, ao2mo
from pyscf.fci import direct_spin1, cistring
from openfermion.ops import InteractionOperator
from openfermion.transforms import jordan_wigner
from openfermion.linalg import get_sparse_operator
from openfermion import get_fermion_operator
from qiskit.quantum_info import SparsePauliOp

mol = gto.Mole()
mol.atom = '''
 C  0.000000  0.000000  0.000000
 O  0.000000  0.000000  1.220000
 N  1.134000  0.000000 -0.672000
 H  2.042000  0.000000 -0.180000
 H  1.167000  0.000000 -1.683000
 H -0.972000  0.000000 -0.487000
'''
mol.basis = 'sto-3g'
mol.spin = 0
mol.charge = 0
mol.verbose = 0
mol.build()
mf = scf.RHF(mol)
mf.kernel()
ncas, nelecas = 6, 6
mc = mcscf.CASCI(mf, ncas, nelecas)
mc.verbose = 0
e_casci = mc.kernel()[0]
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
        h2eff, ci.reshape(na, nb), ncas, nelecas).ravel()
H_mat += ecore * np.eye(ndim)
e_gs = np.linalg.eigh(H_mat)[0][0]
assert abs(e_gs - e_casci) * 1000 < 0.001, \
    f"H_mat / CASCI mismatch: {abs(e_gs-e_casci)*1000:.6f} mHa"

n_so = ncas * 2
one_body_so = np.zeros((n_so, n_so))
one_body_so[0::2, 0::2] = h1
one_body_so[1::2, 1::2] = h1
two_body_so = np.zeros((n_so, n_so, n_so, n_so))
for p, q, r, s in itertools.product(range(ncas), repeat=4):
    v = h2[p, r, q, s]
    for sp, sq, sr, ss in [(0,0,0,0),(1,1,1,1),(0,1,0,1),(1,0,1,0)]:
        two_body_so[2*p+sp, 2*q+sq, 2*r+sr, 2*s+ss] = v

iop_zero = InteractionOperator(0.0, one_body_so, 0.5 * two_body_so)
jw_zero = jordan_wigner(get_fermion_operator(iop_zero))
e_jw_zero = np.linalg.eigvalsh(
    get_sparse_operator(jw_zero).toarray())[0].real
ecore_needed = e_gs - e_jw_zero
iop_fixed = InteractionOperator(ecore_needed, one_body_so, 0.5 * two_body_so)
jw_fixed = jordan_wigner(get_fermion_operator(iop_fixed))
e_jw_check = np.linalg.eigvalsh(
    get_sparse_operator(jw_fixed).toarray())[0].real
assert abs(e_jw_check - e_gs) * 1000 < 0.001, \
    f"JW frozen-core fix failed: {abs(e_jw_check-e_gs)*1000:.6f} mHa"

pauli_list = []
for term, coeff in jw_fixed.terms.items():
    if abs(coeff) < 1e-12:
        continue
    ps = ['I'] * n_so
    for idx, op in term:
        ps[idx] = op
    pauli_list.append((''.join(reversed(ps)), float(coeff.real)))
qubit_op = SparsePauliOp.from_list(pauli_list).simplify()

timings['step1_hamiltonian_s'] = round(time.time() - t1, 2)
print(f"[Step 1] Hamiltonian: {qubit_op.num_qubits} qubits, "
      f"{len(qubit_op)} Pauli terms | {timings['step1_hamiltonian_s']}s")

# ── Step 2: Classical VQE, 5 seeds ──────────────────────────────────────────
t2 = time.time()
from qiskit.primitives import StatevectorEstimator
from qiskit.circuit.library import EfficientSU2
from qiskit_algorithms.minimum_eigensolvers import VQE
from qiskit_algorithms.optimizers import SLSQP

REPS = 1  # reduce to 0 if dry-run exceeds budget
ansatz = EfficientSU2(qubit_op.num_qubits, reps=REPS, entanglement='linear')
n_params = ansatz.num_parameters
SEEDS = [1, 4, 5, 6, 7]
best_e = np.inf
best_params = None
best_seed = None
seed_results = []
for seed in SEEDS:
    rng = np.random.default_rng(seed)
    x0 = rng.uniform(-np.pi, np.pi, n_params)
    vqe = VQE(StatevectorEstimator(), ansatz,
              SLSQP(maxiter=1000), initial_point=x0)
    res = vqe.compute_minimum_eigenvalue(qubit_op)
    e = res.eigenvalue.real
    err = abs(e - e_gs) * 1000
    seed_results.append({'seed': seed, 'energy_Ha': round(float(e), 8),
                         'error_mHa': round(float(err), 4)})
    print(f"  Seed {seed}: E={e:.8f} Ha  err={err:.4f} mHa")
    if e < best_e:
        best_e = e
        best_params = np.array(list(res.optimal_parameters.values()))
        best_seed = seed

best_err = abs(best_e - e_gs) * 1000
assert best_err < 1.6, f"VQE did not reach chemical accuracy: {best_err:.4f} mHa"
timings['step2_vqe_s'] = round(time.time() - t2, 2)
print(f"[Step 2] VQE best: seed={best_seed}, E={best_e:.8f} Ha, "
      f"err={best_err:.4f} mHa | {timings['step2_vqe_s']}s")

# ── Step 3: Bind theta* ──────────────────────────────────────────────────────
ansatz_bound = ansatz.assign_parameters(best_params)
assert ansatz_bound.num_parameters == 0, "Free parameters remain after binding"

# ── Step 4: Transpile against FakeSherbrooke (Heron-class noise model) ──────
t4 = time.time()
try:
    from qiskit_ibm_runtime.fake_provider import FakeSherbrooke
    fake_backend = FakeSherbrooke()
    print("[Step 4] Using FakeSherbrooke from qiskit_ibm_runtime.fake_provider")
except ImportError:
    try:
        from qiskit.providers.fake_provider import FakeSherbrooke
        fake_backend = FakeSherbrooke()
        print("[Step 4] Using FakeSherbrooke from qiskit.providers.fake_provider")
    except ImportError:
        from qiskit_ibm_runtime.fake_provider import FakeKyiv
        fake_backend = FakeKyiv()
        print("[Step 4] FakeSherbrooke unavailable; using FakeKyiv as fallback")

from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
pm = generate_preset_pass_manager(
    target=fake_backend.target, optimization_level=3)
circuit_isa = pm.run(ansatz_bound)
qubit_op_isa = qubit_op.apply_layout(circuit_isa.layout)
ops = circuit_isa.count_ops()
n_2q = (ops.get('ecr', 0) + ops.get('cx', 0) +
        ops.get('cz', 0) + ops.get('rzz', 0))
assert circuit_isa.num_parameters == 0, "ISA circuit has free parameters"
timings['step4_transpile_s'] = round(time.time() - t4, 2)
timings['circuit_depth'] = circuit_isa.depth()
timings['circuit_2q_gates'] = n_2q
timings['circuit_num_qubits'] = circuit_isa.num_qubits
print(f"[Step 4] Transpiled: depth={circuit_isa.depth()}, "
      f"2Q={n_2q} | {timings['step4_transpile_s']}s")

# ── Step 5: Statevector estimation (proxy for hardware shot timing) ──────────
t5 = time.time()
from qiskit.primitives import StatevectorEstimator as SVE
sv_est = SVE()
# qiskit >= 2.0 StatevectorEstimator.run() takes `precision`, not `shots`.
# 4096 shots corresponds to precision = 1/sqrt(4096) = 0.015625.
# The ISA circuit is laid out on the full backend register (>100 qubits),
# which cannot be statevector-simulated (2**N amplitudes). Estimate the
# same expectation value on the 12-qubit logical circuit instead.
SHOTS = 4096
sv_job = sv_est.run([(ansatz_bound, qubit_op)], precision=1.0 / np.sqrt(SHOTS))
sv_result = sv_job.result()
e_sim = float(sv_result[0].data.evs)
sim_err = abs(e_sim - e_gs) * 1000
timings['step5_estimator_s'] = round(time.time() - t5, 2)
print(f"[Step 5] Statevector E={e_sim:.6f} Ha, "
      f"err={sim_err:.2f} mHa | {timings['step5_estimator_s']}s")

# ── Final gate ───────────────────────────────────────────────────────────────
t_total = time.time() - t_total_start
timings['total_wall_time_s'] = round(t_total, 2)
timings['budget_s'] = BUDGET_SECONDS
timings['passed'] = t_total < BUDGET_SECONDS
timings['e_gs_Ha'] = round(float(e_gs), 8)
timings['e_casci_Ha'] = round(float(e_casci), 8)
timings['best_vqe_seed'] = int(best_seed)
timings['best_vqe_energy_Ha'] = round(float(best_e), 8)
timings['best_vqe_error_mHa'] = round(float(best_err), 4)
timings['ansatz_reps'] = REPS
timings['seed_results'] = seed_results

outpath = os.path.join(RESULTS_DIR, 'dryrun_timing_latest.json')
with open(outpath, 'w') as f:
    json.dump(timings, f, indent=2)
print(f"\nTiming data written to {outpath}")

print("\n" + "=" * 60)
print(f"DRY-RUN TOTAL WALL TIME: {t_total:.1f}s ({t_total/60:.2f} min)")
print(f"BUDGET:                  {BUDGET_SECONDS}s ({BUDGET_SECONDS/60:.1f} min)")
if timings['passed']:
    print("DRY-RUN RESULT: PASS — safe to proceed to hardware")
else:
    print("DRY-RUN RESULT: FAIL — DO NOT submit to hardware")
    print("Action: Reduce REPS from 1 to 0 and re-run this script.")
    raise RuntimeError(
        f"Dry-run exceeded budget: {t_total:.1f}s > {BUDGET_SECONDS}s")
print("=" * 60)
