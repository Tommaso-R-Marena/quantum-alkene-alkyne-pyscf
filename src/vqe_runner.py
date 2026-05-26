"""
vqe_runner.py
-------------
VQE drivers for the alkene/alkyne project.

Primary path: qiskit-nature 0.7+ UCCSD ansatz, qiskit_aer.AerSimulator
('statevector') with qiskit_aer.primitives.Estimator, SLSQP primary
optimizer (scipy.optimize.minimize) with COBYLA fallback. Returns a
VQEResult-like dict consumed by notebooks and analysis utilities.

Legacy: PennyLane UCCSD-VQE / ADAPT-VQE wrappers, retained because the
existing unit tests patch ``src.vqe_runner.qml``.
"""

from __future__ import annotations

import time

import numpy as np

# Module-level "qml" handle so unit tests can patch it.
try:
    import pennylane as qml  # noqa: F401
except ImportError:  # pragma: no cover
    qml = None  # type: ignore


# ============================================================
# Primary: Qiskit-native UCCSD-VQE on AerSimulator (statevector)
# ============================================================

def run_uccsd_vqe_qiskit(
    qubit_op,
    problem,
    primary: str = "SLSQP",
    fallback: str = "COBYLA",
    ftol: float = 1e-9,
    maxiter: int = 1000,
    initial_point: np.ndarray | None = None,
    verbose: bool = False,
) -> dict:
    """Run UCCSD-VQE for an ElectronicStructureProblem.

    Uses qiskit_aer.AerSimulator(method='statevector') and
    qiskit_aer.primitives.Estimator. SLSQP is tried first; on failure
    or NaN result, COBYLA is used.
    """
    from qiskit_aer import AerSimulator
    from qiskit_aer.primitives import Estimator as AerEstimator
    from qiskit_nature.second_q.circuit.library import UCCSD, HartreeFock
    from qiskit_nature.second_q.mappers import JordanWignerMapper
    from scipy.optimize import minimize

    n_alpha, n_beta = problem.num_particles
    n_spatial = problem.num_spatial_orbitals
    mapper = JordanWignerMapper()

    hf = HartreeFock(
        num_spatial_orbitals=n_spatial,
        num_particles=(n_alpha, n_beta),
        qubit_mapper=mapper,
    )
    ansatz = UCCSD(
        num_spatial_orbitals=n_spatial,
        num_particles=(n_alpha, n_beta),
        qubit_mapper=mapper,
        initial_state=hf,
    )

    backend = AerSimulator(method="statevector")
    estimator = AerEstimator(
        backend_options={"method": "statevector"},
        run_options={"shots": None},
        approximation=True,
    )

    n_params = ansatz.num_parameters
    if initial_point is None:
        initial_point = np.zeros(n_params)

    # Sum of all energy-shift constants (nuclear repulsion + any
    # transformer offsets such as ActiveSpaceTransformer / FreezeCore).
    consts = getattr(problem.hamiltonian, "constants", None)
    if consts is None:
        energy_shift = float(problem.nuclear_repulsion_energy or 0.0)
    else:
        energy_shift = float(sum(consts.values()))

    history: list[float] = []

    def energy_fn(params: np.ndarray) -> float:
        job = estimator.run([ansatz], [qubit_op], [params])
        e_elec = float(job.result().values[0])
        e_total = e_elec + energy_shift
        history.append(e_total)
        if verbose and (len(history) % 25 == 0):
            print(f"  iter {len(history):4d}  E = {e_total:.8f} Ha")
        return e_total

    t0 = time.time()
    used_optimizer = primary
    try:
        res = minimize(
            energy_fn,
            initial_point,
            method=primary,
            options={"ftol": ftol, "maxiter": maxiter},
        )
        final_energy = float(res.fun)
        if not np.isfinite(final_energy):
            raise RuntimeError("Primary optimizer returned non-finite energy")
    except Exception as e:
        if verbose:
            print(f"  Primary {primary} failed ({e}); falling back to {fallback}")
        used_optimizer = fallback
        res = minimize(
            energy_fn,
            initial_point,
            method=fallback,
            options={"maxiter": maxiter, "rhobeg": 0.1, "catol": 1e-8},
        )
        final_energy = float(res.fun)
    wall = time.time() - t0

    try:
        depth = ansatz.decompose().depth()
    except Exception:
        depth = ansatz.depth()

    return {
        "method": "UCCSD-VQE",
        "final_energy_Ha": final_energy,
        "energy": final_energy,
        "n_iterations": len(history),
        "circuit_depth": int(depth),
        "n_parameters": int(n_params),
        "n_params": int(n_params),
        "wall_time_seconds": float(wall),
        "energy_history": history,
        "history": history,
        "optimizer_used": used_optimizer,
        "energy_shift": float(energy_shift),
    }


# ============================================================
# Legacy: UCCSD-VQE (PennyLane)
# ============================================================

def run_vqe_pennylane(
    qubit_hamiltonian,
    n_qubits: int,
    n_electrons: int,
    device: str = "default.qubit",
    stepsize: float = 0.4,
    max_iter: int = 200,
    conv_tol: float = 1e-9,
    verbose: bool = True,
):
    """UCCSD-VQE via PennyLane AllSinglesDoubles (legacy)."""
    singles, doubles = qml.qchem.excitations(n_electrons, n_qubits)
    hf_state = qml.qchem.hf_state(n_electrons, n_qubits)
    dev = qml.device(device, wires=n_qubits)

    @qml.qnode(dev)
    def circuit(params):
        qml.BasisState(hf_state, wires=range(n_qubits))
        qml.AllSinglesDoubles(
            params, wires=range(n_qubits),
            hf_state=hf_state, singles=singles, doubles=doubles
        )
        return qml.expval(qubit_hamiltonian)

    n_params = len(singles) + len(doubles)
    params = np.zeros(n_params)
    opt = qml.GradientDescentOptimizer(stepsize=stepsize)
    energies = []

    for step in range(max_iter):
        params, energy = opt.step_and_cost(circuit, params)
        energies.append(float(energy))
        if verbose and step % 25 == 0:
            print(f"  [UCCSD-VQE] step {step:4d} | E = {energy:.8f} Ha")
        if step > 5 and abs(energies[-1] - energies[-2]) < conv_tol:
            if verbose:
                print(f"  [UCCSD-VQE] Converged at step {step}")
            break

    est_cnots = len(doubles) * 8 + len(singles) * 2

    return {
        "method": "UCCSD-VQE",
        "energy": energies[-1],
        "history": energies,
        "parameters": params,
        "n_params": n_params,
        "n_singles": len(singles),
        "n_doubles": len(doubles),
        "est_cnot_count": est_cnots,
        "n_iterations": len(energies),
    }


# ============================================================
# Legacy: ADAPT-VQE helpers (PennyLane)
# ============================================================

def build_operator_pool(n_qubits: int, n_electrons: int):
    """Return a generalized singles+doubles operator pool (legacy)."""
    singles, doubles = qml.qchem.excitations(n_electrons, n_qubits)
    pool = []
    for s in singles:
        pool.append((f"S_{s[0]}_{s[1]}", "single", s))
    for d in doubles:
        pool.append((f"D_{d[0]}_{d[1]}_{d[2]}_{d[3]}", "double", d))
    return pool


def run_adapt_vqe(
    qubit_hamiltonian,
    n_qubits: int,
    n_electrons: int,
    gradient_threshold: float = 1e-3,
    max_operators: int = 30,
    max_vqe_iter: int = 300,
    stepsize: float = 0.4,
    conv_tol: float = 1e-9,
    device: str = "default.qubit",
    fci_energy: float | None = None,
    verbose: bool = True,
):
    """Legacy PennyLane ADAPT-VQE — kept for backwards compat."""
    import pennylane as qml
    from pennylane import qchem
    from scipy.optimize import minimize

    hf_state = qchem.hf_state(n_electrons, n_qubits)
    pool = build_operator_pool(n_qubits, n_electrons)

    dev = qml.device(device, wires=n_qubits)
    selected_operators: list = []
    params = np.array([])
    energy_history: list = []

    for adapt_iter in range(max_operators):
        gradients = []
        for label, op_type, exc in pool:
            @qml.qnode(dev)
            def probe(theta, exc=exc, op_type=op_type):
                qml.BasisState(hf_state, wires=range(n_qubits))
                for idx2, (_, t2, e2) in enumerate(selected_operators):
                    if t2 == "single":
                        qml.SingleExcitation(params[idx2], wires=e2)
                    else:
                        qml.DoubleExcitation(params[idx2], wires=e2)
                if op_type == "single":
                    qml.SingleExcitation(theta, wires=exc)
                else:
                    qml.DoubleExcitation(theta, wires=exc)
                return qml.expval(qubit_hamiltonian)
            gradients.append(abs((probe(np.pi/2) - probe(-np.pi/2)) / 2.0))

        max_grad = max(gradients)
        best_idx = int(np.argmax(gradients))
        if max_grad < gradient_threshold:
            break
        selected_operators.append(pool[best_idx])
        params = np.append(params, 0.0)

        @qml.qnode(dev)
        def adapt_circuit(p):
            qml.BasisState(hf_state, wires=range(n_qubits))
            for idx, (_, op_type, exc) in enumerate(selected_operators):
                if op_type == "single":
                    qml.SingleExcitation(p[idx], wires=exc)
                else:
                    qml.DoubleExcitation(p[idx], wires=exc)
            return qml.expval(qubit_hamiltonian)

        res = minimize(adapt_circuit, params, method="L-BFGS-B",
                       jac=qml.grad(adapt_circuit),
                       options={"maxiter": max_vqe_iter, "ftol": conv_tol})
        params = res.x
        energy_history.append(float(res.fun))

    n_singles_sel = sum(1 for _, t, _ in selected_operators if t == "single")
    n_doubles_sel = sum(1 for _, t, _ in selected_operators if t == "double")
    est_cnots = n_singles_sel * 2 + n_doubles_sel * 8
    err = (abs(energy_history[-1] - fci_energy) * 1000
           if fci_energy and energy_history else None)
    return {
        "method": "ADAPT-VQE",
        "energy": energy_history[-1] if energy_history else None,
        "history": energy_history,
        "parameters": params,
        "selected_operators": [(l, t) for l, t, _ in selected_operators],
        "n_operators": len(selected_operators),
        "n_singles": n_singles_sel,
        "n_doubles": n_doubles_sel,
        "est_cnot_count": est_cnots,
        "error_mHa": err,
    }


# ============================================================
# Unified comparison (legacy)
# ============================================================

def compare_vqe_methods(
    qubit_hamiltonian,
    n_qubits: int,
    n_electrons: int,
    fci_energy: float | None = None,
    device: str = "default.qubit",
    verbose: bool = True,
):
    uccsd_result = run_vqe_pennylane(qubit_hamiltonian, n_qubits, n_electrons,
                                     device=device, verbose=verbose)
    adapt_result = run_adapt_vqe(qubit_hamiltonian, n_qubits, n_electrons,
                                 fci_energy=fci_energy, device=device, verbose=verbose)
    return {
        "UCCSD-VQE": {
            "energy": uccsd_result["energy"],
            "n_params": uccsd_result["n_params"],
            "est_cnot_count": uccsd_result["est_cnot_count"],
            "error_mHa": (abs(uccsd_result["energy"] - fci_energy) * 1000
                          if fci_energy else None),
            "n_iterations": uccsd_result["n_iterations"],
        },
        "ADAPT-VQE": {
            "energy": adapt_result["energy"],
            "n_params": adapt_result["n_operators"],
            "est_cnot_count": adapt_result["est_cnot_count"],
            "error_mHa": adapt_result["error_mHa"],
            "n_operators_selected": adapt_result["n_operators"],
            "selected": adapt_result["selected_operators"],
        },
    }
