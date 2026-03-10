"""
vqe_runner.py
-------------
UCCSD-VQE and ADAPT-VQE execution wrappers using PennyLane.
Designed for alkene/alkyne active-space Hamiltonians derived from PySCF.
Supports statevector simulator (default.qubit) and IBM Quantum backends.
"""

import numpy as np
from typing import Optional


# ============================================================
# UCCSD-VQE (fixed ansatz)
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
    """
    UCCSD-VQE using PennyLane AllSinglesDoubles ansatz.

    Parameters
    ----------
    qubit_hamiltonian : PennyLane Hamiltonian (from openfermion_to_pennylane)
    n_qubits          : int
    n_electrons       : int (active electrons)
    device            : PennyLane device string
    stepsize          : gradient descent step size
    max_iter          : max optimizer steps
    conv_tol          : convergence threshold on |ΔE|
    verbose           : print progress

    Returns
    -------
    dict : energy, history, parameters, n_params, circuit_depth (estimated)
    """
    import pennylane as qml
    from pennylane import qchem

    singles, doubles = qchem.excitations(n_electrons, n_qubits)
    hf_state = qchem.hf_state(n_electrons, n_qubits)
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
                print(f"  [UCCSD-VQE] Converged at step {step} | E = {energies[-1]:.8f} Ha")
            break

    # Estimate two-qubit gate count from excitation operators
    # Each double ~ 8 CNOTs, each single ~ 2 CNOTs (rough UCCSD estimate)
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
# ADAPT-VQE (adaptive ansatz)
# ============================================================

def _commutator_gradient(hamiltonian, operator, state_fn, wires):
    """
    Estimate the gradient of <H> w.r.t. adding `operator` to the ansatz
    via finite difference on a single-parameter circuit.

    gradient ≈ d/dθ <ψ(θ)|H|ψ(θ)> at θ=0
    """
    import pennylane as qml

    dev = qml.device("default.qubit", wires=wires)

    @qml.qnode(dev)
    def probe(theta):
        state_fn()                    # current ansatz state
        qml.apply(qml.exp(operator, theta))  # probe new operator
        return qml.expval(hamiltonian)

    # Parameter-shift gradient at θ=0
    grad = (probe(np.pi / 2) - probe(-np.pi / 2)) / 2.0
    return abs(float(grad))


def build_operator_pool(n_qubits: int, n_electrons: int):
    """
    Build the generalized singles-and-doubles (GSD) operator pool
    for ADAPT-VQE as a list of PennyLane OrbitalRotation-style operators.

    Returns list of (label, operator_fn) tuples where operator_fn(wires)
    returns the anti-Hermitian generator for that excitation.
    """
    import pennylane as qml
    from pennylane import qchem

    singles, doubles = qchem.excitations(n_electrons, n_qubits)
    pool = []

    for s in singles:
        label = f"S_{s[0]}_{s[1]}"
        pool.append((label, "single", s))

    for d in doubles:
        label = f"D_{d[0]}_{d[1]}_{d[2]}_{d[3]}"
        pool.append((label, "double", d))

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
    fci_energy: Optional[float] = None,
    verbose: bool = True,
):
    """
    ADAPT-VQE: adaptively grow the ansatz by greedily selecting the
    operator from the GSD pool with the largest energy gradient.

    Algorithm (Grimsley et al., Nat. Commun. 2019):
    -----------------------------------------------
    1. Start from Hartree-Fock reference |ψ₀⟩ = |HF⟩
    2. For each operator A_i in the pool, compute |∂E/∂θ_i| at θ=0
    3. Add the operator with the largest gradient to the ansatz
    4. Re-optimize all parameters with VQE
    5. Repeat until max(|∂E/∂θ_i|) < gradient_threshold

    Parameters
    ----------
    qubit_hamiltonian   : PennyLane Hamiltonian
    n_qubits            : int
    n_electrons         : int (active)
    gradient_threshold  : stop when max gradient < this value
    max_operators       : maximum ansatz operators before forced stop
    max_vqe_iter        : VQE iterations per ADAPT cycle
    stepsize            : optimizer step size
    conv_tol            : VQE inner convergence tolerance
    device              : PennyLane device
    fci_energy          : optional FCI reference for error tracking
    verbose             : print progress

    Returns
    -------
    dict : energy, history, selected_operators, parameters,
           n_operators, circuit_depth, error_mHa
    """
    import pennylane as qml
    from pennylane import qchem
    from scipy.optimize import minimize

    hf_state = qchem.hf_state(n_electrons, n_qubits)
    pool = build_operator_pool(n_qubits, n_electrons)
    singles_pool = [(l, d) for l, t, d in pool if t == "single"]
    doubles_pool = [(l, d) for l, t, d in pool if t == "double"]

    dev = qml.device(device, wires=n_qubits)

    # Tracks the growing ansatz
    selected_operators = []  # list of (label, type, excitation_indices)
    params = np.array([])
    energy_history = []      # energy after each ADAPT macro-iteration

    if verbose:
        print(f"ADAPT-VQE | {n_qubits} qubits | {n_electrons} active electrons")
        print(f"Pool size: {len(pool)} operators | threshold: {gradient_threshold}\n")

    for adapt_iter in range(max_operators):

        # ---- Step 1: compute gradients for all pool operators ----
        @qml.qnode(dev)
        def current_state():
            """Current ADAPT ansatz state (no measurement)."""
            qml.BasisState(hf_state, wires=range(n_qubits))
            for idx, (label, op_type, exc) in enumerate(selected_operators):
                if op_type == "single":
                    qml.SingleExcitation(params[idx], wires=exc)
                else:
                    qml.DoubleExcitation(params[idx], wires=exc)
            return qml.state()

        gradients = []
        for label, op_type, exc in pool:
            # Probe circuit: current state + candidate operator
            @qml.qnode(dev)
            def probe_circuit(theta, exc=exc, op_type=op_type):
                qml.BasisState(hf_state, wires=range(n_qubits))
                for idx2, (_, t2, e2) in enumerate(selected_operators):
                    if t2 == "single":
                        qml.SingleExcitation(params[idx2], wires=e2)
                    else:
                        qml.DoubleExcitation(params[idx2], wires=e2)
                # New candidate
                if op_type == "single":
                    qml.SingleExcitation(theta, wires=exc)
                else:
                    qml.DoubleExcitation(theta, wires=exc)
                return qml.expval(qubit_hamiltonian)

            grad = abs((probe_circuit(np.pi/2) - probe_circuit(-np.pi/2)) / 2.0)
            gradients.append(grad)

        max_grad = max(gradients)
        best_idx = int(np.argmax(gradients))

        if verbose:
            print(f"  ADAPT iter {adapt_iter+1:2d} | max|grad| = {max_grad:.6f} "
                  f"| best op: {pool[best_idx][0]}")

        if max_grad < gradient_threshold:
            if verbose:
                print(f"  Converged: max gradient {max_grad:.2e} < {gradient_threshold}")
            break

        # ---- Step 2: add best operator, re-optimize all params ----
        selected_operators.append(pool[best_idx])
        params = np.append(params, 0.0)  # initialize new param at 0

        @qml.qnode(dev)
        def adapt_circuit(p):
            qml.BasisState(hf_state, wires=range(n_qubits))
            for idx, (_, op_type, exc) in enumerate(selected_operators):
                if op_type == "single":
                    qml.SingleExcitation(p[idx], wires=exc)
                else:
                    qml.DoubleExcitation(p[idx], wires=exc)
            return qml.expval(qubit_hamiltonian)

        # Inner VQE optimization (scipy L-BFGS-B for fast convergence)
        res = minimize(
            adapt_circuit, params,
            method="L-BFGS-B",
            jac=qml.grad(adapt_circuit),
            options={"maxiter": max_vqe_iter, "ftol": conv_tol},
        )
        params = res.x
        energy = float(res.fun)
        energy_history.append(energy)

        err_str = ""
        if fci_energy is not None:
            err_str = f" | ΔE={abs(energy-fci_energy)*1000:.4f} mHa"
        if verbose:
            print(f"           Energy = {energy:.8f} Ha{err_str}")

    # Estimate circuit depth: each single ~ 2 CNOTs, each double ~ 8 CNOTs
    n_singles_sel = sum(1 for _, t, _ in selected_operators if t == "single")
    n_doubles_sel = sum(1 for _, t, _ in selected_operators if t == "double")
    est_cnots = n_singles_sel * 2 + n_doubles_sel * 8

    error_mHa = abs(energy_history[-1] - fci_energy) * 1000 if fci_energy else None

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
        "error_mHa": error_mHa,
    }


# ============================================================
# Unified comparison runner
# ============================================================

def compare_vqe_methods(
    qubit_hamiltonian,
    n_qubits: int,
    n_electrons: int,
    fci_energy: Optional[float] = None,
    device: str = "default.qubit",
    verbose: bool = True,
):
    """
    Run both UCCSD-VQE and ADAPT-VQE on the same Hamiltonian and
    return a side-by-side comparison dict. This is the primary
    function for generating publication-ready benchmark tables.
    """
    print("=" * 60)
    print("Running UCCSD-VQE...")
    print("=" * 60)
    uccsd_result = run_vqe_pennylane(
        qubit_hamiltonian, n_qubits, n_electrons,
        device=device, verbose=verbose
    )

    print()
    print("=" * 60)
    print("Running ADAPT-VQE...")
    print("=" * 60)
    adapt_result = run_adapt_vqe(
        qubit_hamiltonian, n_qubits, n_electrons,
        fci_energy=fci_energy, device=device, verbose=verbose
    )

    comparison = {
        "UCCSD-VQE": {
            "energy": uccsd_result["energy"],
            "n_params": uccsd_result["n_params"],
            "est_cnot_count": uccsd_result["est_cnot_count"],
            "error_mHa": abs(uccsd_result["energy"] - fci_energy) * 1000 if fci_energy else None,
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

    if verbose:
        print()
        print("=" * 60)
        print("COMPARISON SUMMARY")
        print("=" * 60)
        print(f"{'Metric':<30} {'UCCSD-VQE':>15} {'ADAPT-VQE':>15}")
        print("-" * 60)
        print(f"{'Energy (Ha)':<30} {comparison['UCCSD-VQE']['energy']:>15.8f} {comparison['ADAPT-VQE']['energy']:>15.8f}")
        if fci_energy:
            print(f"{'Error vs FCI (mHa)':<30} {comparison['UCCSD-VQE']['error_mHa']:>15.4f} {comparison['ADAPT-VQE']['error_mHa']:>15.4f}")
        print(f"{'# Parameters':<30} {comparison['UCCSD-VQE']['n_params']:>15} {comparison['ADAPT-VQE']['n_params']:>15}")
        print(f"{'Est. CNOT count':<30} {comparison['UCCSD-VQE']['est_cnot_count']:>15} {comparison['ADAPT-VQE']['est_cnot_count']:>15}")

    return comparison
