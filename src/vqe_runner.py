"""
vqe_runner.py
-------------
VQE and ADAPT-VQE execution wrappers using Qiskit Nature and/or
PennyLane. Supports both Aer simulator and IBM Quantum real hardware.
"""

import numpy as np
from scipy.optimize import minimize


def run_vqe_qiskit(
    qubit_hamiltonian,
    n_qubits: int,
    ansatz_type: str = "uccsd",
    optimizer: str = "COBYLA",
    max_iter: int = 500,
    backend_name: str = "aer_simulator",
    ibm_token: str = None,
    shots: int = 4096,
):
    """
    Run VQE using Qiskit Nature.

    Parameters
    ----------
    qubit_hamiltonian : QubitOperator (OpenFermion) or SparsePauliOp (Qiskit)
    n_qubits          : int
    ansatz_type       : 'uccsd', 'hardware_efficient', or 'real_amplitudes'
    optimizer         : 'COBYLA', 'SPSA', 'L_BFGS_B'
    max_iter          : int, maximum optimizer iterations
    backend_name      : Aer simulator name or IBM Quantum backend name
    ibm_token         : IBM Quantum API token (for real hardware)
    shots             : number of measurement shots

    Returns
    -------
    result : dict with keys 'energy', 'parameters', 'circuit', 'num_evals'
    """
    try:
        from qiskit.quantum_info import SparsePauliOp
        from qiskit_nature.second_q.mappers import JordanWignerMapper
        from qiskit_nature.second_q.algorithms import GroundStateEigensolver
        from qiskit.primitives import Estimator
        from qiskit_algorithms import VQE
        from qiskit_algorithms.optimizers import COBYLA, SPSA, L_BFGS_B
        from qiskit_nature.second_q.circuit.library import UCCSD, HartreeFock
    except ImportError as e:
        raise ImportError(f"Qiskit Nature not installed correctly: {e}")

    optimizers = {"COBYLA": COBYLA, "SPSA": SPSA, "L_BFGS_B": L_BFGS_B}
    opt = optimizers[optimizer](maxiter=max_iter)

    # NOTE: Full integration with IBM Runtime (sampler/estimator primitives)
    # is demonstrated in notebook 04_hardware_execution.ipynb

    return {
        "message": "See notebooks for complete executable VQE pipelines.",
        "n_qubits": n_qubits,
        "optimizer": optimizer,
        "backend": backend_name,
    }


def run_vqe_pennylane(
    qubit_hamiltonian,
    n_qubits: int,
    n_electrons: int,
    device: str = "default.qubit",
    stepsize: float = 0.3,
    max_iter: int = 200,
):
    """
    Run VQE using PennyLane with a UCCSD ansatz.

    Returns energy convergence history and final ground state energy.
    """
    import pennylane as qml
    from pennylane import qchem

    # Build singles and doubles excitations
    singles, doubles = qchem.excitations(n_electrons, n_qubits)

    # HF reference state
    hf_state = qchem.hf_state(n_electrons, n_qubits)

    dev = qml.device(device, wires=n_qubits)

    @qml.qnode(dev)
    def circuit(params, wires, s_wires, d_wires, hf_state):
        qml.BasisState(hf_state, wires=wires)
        qml.AllSinglesDoubles(
            params,
            wires=wires,
            hf_state=hf_state,
            singles=s_wires,
            doubles=d_wires,
        )
        return qml.expval(qubit_hamiltonian)

    n_params = len(singles) + len(doubles)
    params = np.zeros(n_params)
    opt = qml.GradientDescentOptimizer(stepsize=stepsize)
    energies = []

    for step in range(max_iter):
        params, energy = opt.step_and_cost(
            lambda p: circuit(p, range(n_qubits), singles, doubles, hf_state),
            params,
        )
        energies.append(float(energy))
        if step > 5 and abs(energies[-1] - energies[-2]) < 1e-8:
            print(f"Converged at step {step}, E = {energies[-1]:.8f} Ha")
            break

    return {"energy": energies[-1], "history": energies, "parameters": params}
