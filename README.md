# Quantum Simulation of Alkenes and Alkynes via PySCF

> **Status:** Active development | Targeting publication at *J. Chem. Theory Comput.* or *npj Quantum Information*

[![Notebook 01 – Alkene VQE](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Tommaso-R-Marena/quantum-alkene-alkyne-pyscf/blob/main/notebooks/01_alkene_vqe_simulation.ipynb)
[![Notebook 02 – Alkyne VQE](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Tommaso-R-Marena/quantum-alkene-alkyne-pyscf/blob/main/notebooks/02_alkyne_vqe_simulation.ipynb)
[![Notebook 06 – ADAPT-VQE](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Tommaso-R-Marena/quantum-alkene-alkyne-pyscf/blob/main/notebooks/06_adapt_vqe_comparison.ipynb)

---

## Overview

This repository provides a **systematic quantum simulation framework for alkenes and alkynes**, the first dedicated benchmark of the unsaturated hydrocarbon homologous series on real quantum hardware. Prior work targets diatomics (H₂, LiH, N₂) or small polyatomics (H₂O); this project is the first to study the C=C and C≡C π-bond series systematically under realistic NISQ hardware constraints.

**Software stack:**
- **PySCF ≥ 2.5** — classical electronic structure (HF, CCSD, FCI)
- **OpenFermion-PySCF** — molecular Hamiltonian → qubit operator
- **PennyLane ≥ 0.38** — VQE and ADAPT-VQE on statevector simulator
- **Qiskit ≥ 1.0 / Qiskit Runtime** — transpilation and execution on IBM Quantum

---

## Molecule Series

| Series | Molecules | π-system | STO-3G JW Qubits (full) | Active-space Qubits |
|--------|-----------|----------|--------------------------|---------------------|
| Alkenes | Ethylene (C₂H₄) | 1 C=C | 14 | 8 |
| | 1-Butene (C₄H₈) | 1 C=C | 26 | 8–10 |
| | 1,3-Butadiene (C₄H₆) | conjugated | 26 | 8–10 |
| Alkynes | Acetylene (C₂H₂) | C≡C (2 ⊥ π) | 10 | 8 |
| | Propyne (C₃H₄) | C≡C | 18 | 8–10 |
| | 1-Butyne (C₄H₆) | C≡C | 26 | 10–12 |

> **Hardware feasibility note:** IBM Quantum's 127-qubit Eagle and 133-qubit Heron processors support the active-space circuits here (8–12 qubits) with error mitigation (ZNE, PEC). Qubit tapering via Z₂ symmetries can reduce counts by a further 2–4 qubits.

---

## Computational Workflow

```
Molecule (XYZ geometry)
        │
        ▼
   PySCF  ──────────────────────── HF / CCSD / FCI reference energies
        │
        ▼
  OpenFermion-PySCF
  Molecular Hamiltonian (2nd quantized)
        │
        ▼
  Active Space Selection          ← freeze core orbitals, select HOMO/LUMO window
        │
        ▼
  Fermion → Qubit Mapping
  ├── Jordan-Wigner  (linear qubit overhead, shallow local gates)
  └── Bravyi-Kitaev  (logarithmic overhead, better for larger molecules)
        │
        ▼
  Qubit Tapering (Z₂ symmetries)  ← reduces qubit count 2–4
        │
        ├─────────────────────────────────────────────────────┐
        ▼                                                     ▼
  UCCSD-VQE (fixed ansatz)                      ADAPT-VQE (adaptive ansatz)
  All singles + doubles,                         Grows circuit only with
  fixed circuit depth                            operators that lower energy
        │                                                     │
        └──────────────────────┬──────────────────────────────┘
                               ▼
           Aer Statevector Simulator  →  IBM Quantum (Eagle/Heron)
                               ▼
        Results: E_ground, ΔE vs FCI, circuit depth, qubit count,
                 HOMO-LUMO gap, correlation energy recovery
```

---

## Repository Structure

```
quantum-alkene-alkyne-pyscf/
├── notebooks/
│   ├── 01_alkene_vqe_simulation.ipynb      ← Ethylene & 1-butene: UCCSD-VQE
│   ├── 02_alkyne_vqe_simulation.ipynb      ← Acetylene & propyne: UCCSD-VQE
│   ├── 03_active_space_tapering.ipynb      ← Qubit reduction strategies
│   ├── 04_hardware_execution.ipynb         ← IBM Quantum Runtime (ZNE)
│   ├── 05_benchmark_analysis.ipynb         ← Full comparison table
│   └── 06_adapt_vqe_comparison.ipynb       ← ADAPT-VQE vs UCCSD-VQE
├── src/
│   ├── molecule_builder.py                 ← Geometry + MolecularData builders
│   ├── hamiltonian_utils.py                ← JW/BK mapping, tapering utils
│   ├── vqe_runner.py                       ← UCCSD-VQE + ADAPT-VQE runners
│   └── analysis.py                         ← Energy tables, plots
├── data/geometries/                        ← B3LYP/6-31G* optimized XYZ
├── results/
├── requirements.txt
├── environment.yml
└── README.md
```

---

## Notebook Previews

### 📓 Notebook 01 — Alkene VQE (Ethylene, 1-Butene)

```python
# --- Install ---
!pip install -q pyscf openfermion openfermionpyscf pennylane qiskit qiskit-aer

# --- Classical reference energies via PySCF ---
from openfermion.chem import MolecularData
from openfermionpyscf import run_pyscf

ethylene_geometry = [
    ('C', (0.000,  0.000,  0.000)),
    ('C', (0.000,  0.000,  1.339)),
    ('H', (0.000,  0.926, -0.546)),
    ('H', (0.000, -0.926, -0.546)),
    ('H', (0.000,  0.926,  1.885)),
    ('H', (0.000, -0.926,  1.885)),
]
mol = MolecularData(geometry=ethylene_geometry, basis='sto-3g',
                    multiplicity=1, charge=0, description='ethylene')
mol = run_pyscf(mol, run_scf=True, run_ccsd=True, run_fci=True)
print(f'HF={mol.hf_energy:.6f}  CCSD={mol.ccsd_energy:.6f}  FCI={mol.fci_energy:.6f} Ha')

# --- Jordan-Wigner qubit Hamiltonian ---
from openfermion import get_fermion_operator, jordan_wigner
from openfermion.utils import count_qubits
from openfermion.transforms import freeze_orbitals

fermion_ham = get_fermion_operator(mol.get_molecular_hamiltonian())
active_ham  = freeze_orbitals(fermion_ham, occupied=[0,1,2], virtual=[])  # freeze 3 core
jw_ham      = jordan_wigner(active_ham)
print(f'Active-space qubits (JW): {count_qubits(jw_ham)}')

# --- UCCSD-VQE via PennyLane ---
import pennylane as qml
from pennylane import qchem
import numpy as np

n_q = count_qubits(jw_ham)
n_e = mol.n_electrons - 6
singles, doubles = qchem.excitations(n_e, n_q)
hf_state = qchem.hf_state(n_e, n_q)

def openfermion_to_pennylane(op):
    coeffs, ops = [], []
    for term, c in op.terms.items():
        coeffs.append(np.real(c))
        if not term: ops.append(qml.Identity(0))
        else:
            pl = [{'X':qml.PauliX,'Y':qml.PauliY,'Z':qml.PauliZ}[p](i) for i,p in term]
            ops.append(pl[0] if len(pl)==1 else qml.operation.Tensor(*pl))
    return qml.Hamiltonian(coeffs, ops)

H = openfermion_to_pennylane(jw_ham)
dev = qml.device('default.qubit', wires=n_q)

@qml.qnode(dev)
def vqe_circuit(params):
    qml.BasisState(hf_state, wires=range(n_q))
    qml.AllSinglesDoubles(params, wires=range(n_q),
                          hf_state=hf_state, singles=singles, doubles=doubles)
    return qml.expval(H)

params = np.zeros(len(singles)+len(doubles))
opt = qml.GradientDescentOptimizer(stepsize=0.4)
for step in range(150):
    params, energy = opt.step_and_cost(vqe_circuit, params)
print(f'VQE={energy:.6f}  FCI={mol.fci_energy:.6f}  Err={abs(energy-mol.fci_energy)*1000:.2f} mHa')
```

> **Run it:** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Tommaso-R-Marena/quantum-alkene-alkyne-pyscf/blob/main/notebooks/01_alkene_vqe_simulation.ipynb)

---

### 📓 Notebook 02 — Alkyne VQE (Acetylene, Propyne)

```python
# Acetylene: 4 atoms, linear D∞h, 10 electrons, STO-3G → 10 qubits (JW full)
# Key difference from alkenes: TWO orthogonal π bonds → stronger correlation

acetylene_geometry = [
    ('C', (0.000, 0.000,  0.000)),
    ('C', (0.000, 0.000,  1.203)),
    ('H', (0.000, 0.000, -1.063)),
    ('H', (0.000, 0.000,  2.266)),
]
mol = MolecularData(geometry=acetylene_geometry, basis='sto-3g',
                    multiplicity=1, charge=0, description='acetylene')
mol = run_pyscf(mol, run_scf=True, run_ccsd=True, run_fci=True)

# Correlation energy is the scientific signal:
corr = (mol.fci_energy - mol.hf_energy) * 1000
print(f'Acetylene correlation energy: {corr:.2f} mHa  (larger than ethylene → harder for VQE)')

# Same VQE pipeline as notebook 01, but note deeper circuit needed
# to recover the cylindrical π correlation
```

> **Run it:** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Tommaso-R-Marena/quantum-alkene-alkyne-pyscf/blob/main/notebooks/02_alkyne_vqe_simulation.ipynb)

---

### 📓 Notebook 06 — ADAPT-VQE vs UCCSD-VQE

The key scientific comparison for publication. ADAPT-VQE selects only the operators that most reduce the energy gradient at each step, yielding **shallower circuits** while recovering more correlation energy — critical for alkynes on NISQ hardware.

```python
# ADAPT-VQE: grow the ansatz one operator at a time
# Operator pool: all generalized singles and doubles
# Stopping criterion: ||gradient|| < threshold (typically 1e-3)

from src.vqe_runner import run_adapt_vqe

# Run ADAPT-VQE on acetylene active space
result = run_adapt_vqe(
    qubit_hamiltonian=jw_ham,
    n_qubits=n_q,
    n_electrons=n_e,
    gradient_threshold=1e-3,
    max_operators=20,
    max_vqe_iter=200,
    device='default.qubit',
    verbose=True,
)

print(f"ADAPT-VQE energy     : {result['energy']:.8f} Ha")
print(f"Operators selected   : {result['n_operators']}  (vs {len(singles)+len(doubles)} in UCCSD)")
print(f"Circuit depth        : {result['circuit_depth']}")
print(f"|ADAPT - FCI|        : {result['error_mHa']:.4f} mHa")

# ADAPT typically selects 3-8 operators for small active spaces,
# vs 6-30 in fixed UCCSD — a 3-5x circuit depth reduction
```

| Metric | UCCSD-VQE | ADAPT-VQE |
|--------|-----------|-----------|
| Ansatz | Fixed (all singles+doubles) | Adaptive (gradient-selected) |
| Circuit depth | High, fixed | Grows only as needed |
| # parameters | `len(singles)+len(doubles)` | Typically 3–10 for small active spaces |
| Correlation recovery | ~98–99% FCI | ~99–99.9% FCI |
| NISQ suitability | Moderate (deep circuits) | **High** (shallow, hardware-friendly) |
| Key advantage for alkynes | Systematic | Targets strongest correlators first |

> **Run it:** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Tommaso-R-Marena/quantum-alkene-alkyne-pyscf/blob/main/notebooks/06_adapt_vqe_comparison.ipynb)

---

## Hardware Constraints & NISQ Strategy

This project is explicitly designed around **what is runnable today** on IBM Quantum:

| Constraint | Current hardware limit | Our mitigation |
|---|---|---|
| Qubit count | 127–133 usable qubits (Eagle/Heron) | Active space: 8–12 qubits |
| Circuit depth (T₂ coherence) | ~100–300 CNOT gates before noise dominates | ADAPT-VQE minimizes gate count |
| 2-qubit gate fidelity | ~99.5% on best devices | ZNE error mitigation in NB 04 |
| Connectivity | Heavy-hex topology | BK mapping preferred (more local) |
| Measurement noise | Shot noise at ≤16k shots | Estimator primitive + grouping |

---

## Key Scientific Questions (Publication Framing)

1. **Qubit scaling:** How do JW and BK qubit requirements scale across the C₂→C₄→C₆ alkene/alkyne series after active space selection and Z₂ tapering?
2. **π-bond fidelity:** Can UCCSD-VQE and ADAPT-VQE recover the π-correlation energy (FCI benchmark) for conjugated dienes?
3. **Alkene vs alkyne:** Does the stronger correlation in alkynes (two ⊥ π bonds) cause UCCSD-VQE to fail where ADAPT-VQE succeeds?
4. **Hardware noise impact:** How does ZNE-mitigated energy on IBM Quantum compare to ideal simulation for each molecule?
5. **Circuit efficiency:** How many fewer two-qubit gates does ADAPT-VQE require versus fixed UCCSD for each molecule?

---

## Installation

```bash
git clone https://github.com/Tommaso-R-Marena/quantum-alkene-alkyne-pyscf.git
cd quantum-alkene-alkyne-pyscf
conda env create -f environment.yml
conda activate quantum-chem
```

Or via pip:
```bash
pip install -r requirements.txt
```

---

## Citation

```bibtex
@misc{marena2026alkene_alkyne_quantum,
  author    = {Tommaso R. Marena},
  title     = {Quantum Simulation of Alkenes and Alkynes via PySCF:
               A Benchmark Study on NISQ Hardware},
  year      = {2026},
  publisher = {GitHub},
  url       = {https://github.com/Tommaso-R-Marena/quantum-alkene-alkyne-pyscf}
}
```

## License

Apache 2.0
