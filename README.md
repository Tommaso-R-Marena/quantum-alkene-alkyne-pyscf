# Quantum Simulation of Alkenes and Alkynes via PySCF

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Tommaso-R-Marena/quantum-alkene-alkyne-pyscf/blob/main/notebooks/01_alkene_vqe_simulation.ipynb)

> **Status:** Active development | Targeting publication at a quantum chemistry / quantum computing venue.

## Overview

This repository provides a **systematic quantum simulation framework for alkenes and alkynes** using:
- **PySCF** as the classical electronic structure backend
- **OpenFermion-PySCF** for Hamiltonian generation and fermion-to-qubit mapping
- **Qiskit / Qiskit Nature** for circuit construction and execution on IBM Quantum hardware
- **VQE / ADAPT-VQE** as the variational eigensolver

Whereas prior work has focused on diatomics (H₂, LiH) or small polyatomics (H₂O, NH₃), **no dedicated benchmark study of the alkene/alkyne homologous series on real quantum hardware** exists in the literature. This project fills that gap.

## Molecule Series Studied

| Series | Molecules | Key Feature |
|--------|-----------|-------------|
| Alkenes | Ethylene (C₂H₄), 1-Butene (C₄H₈), 1-Hexene (C₆H₁₂) | C=C π bond, HOMO-LUMO gap |
| Alkynes | Acetylene (C₂H₂), Propyne (C₃H₄), 1-Butyne (C₄H₆) | C≡C triple bond, cylindrical π system |
| Dienes | 1,3-Butadiene (C₄H₆), 1,3-Hexadiene | Conjugation effects |

## Workflow

```
Molecule (XYZ / SMILES)
        |
        v
   PySCF (HF/CCSD/FCI)
        |
        v
  OpenFermion-PySCF
  (Molecular Hamiltonian)
        |
        v
  Fermion → Qubit Mapping
  (Jordan-Wigner / Bravyi-Kitaev)
        |
        v
  Qubit Tapering & Active Space
        |
        v
  VQE / ADAPT-VQE (Qiskit / PennyLane)
        |
        v
  IBM Quantum Hardware / Simulator
        |
        v
  Results: Ground State Energy, Bond Dissociation, Error Analysis
```

## Repository Structure

```
quantum-alkene-alkyne-pyscf/
├── notebooks/
│   ├── 01_alkene_vqe_simulation.ipynb       # Ethylene & butene VQE (starter)
│   ├── 02_alkyne_vqe_simulation.ipynb       # Acetylene & propyne VQE
│   ├── 03_active_space_tapering.ipynb       # Qubit reduction strategies
│   ├── 04_hardware_execution.ipynb          # Running on IBM Quantum real hardware
│   └── 05_benchmark_analysis.ipynb          # VQE vs CCSD vs FCI comparison
├── src/
│   ├── molecule_builder.py                  # Geometry builders for all molecules
│   ├── hamiltonian_utils.py                 # PySCF → OpenFermion pipeline
│   ├── vqe_runner.py                        # VQE/ADAPT-VQE execution wrappers
│   └── analysis.py                          # Energy parsing, plotting, error metrics
├── data/
│   └── geometries/                          # Optimized XYZ files (B3LYP/6-31G*)
├── results/
│   └── .gitkeep
├── requirements.txt
├── environment.yml
└── README.md
```

## Quick Start (Google Colab)

1. Click the **Open in Colab** badge above.
2. Run the first cell to install dependencies (≈3 min).
3. Choose your molecule and basis set.
4. Run VQE on the Aer statevector simulator or connect to IBM Quantum.

## Key Scientific Questions

1. **Qubit scaling**: How does qubit count grow with chain length for alkenes/alkynes after active space selection and tapering?
2. **π-system fidelity**: Can VQE correctly reproduce the π-bond HOMO-LUMO gap for conjugated systems?
3. **Hardware noise impact**: How does decoherence affect energy accuracy for molecules with delocalized electrons?
4. **JW vs BK mapping**: Which fermion-to-qubit mapping yields shallower circuits for unsaturated hydrocarbons?

## Dependencies

```
pyscf>=2.5
openfermion>=1.6
openfermionpyscf>=0.5
qiskit>=1.0
qiskit-nature>=0.7
qiskit-aer>=0.14
pennylane>=0.38
numpy>=1.24
scipy>=1.11
matplotlib>=3.7
pandas>=2.0
```

## Citation

If you use this work, please cite:
```bibtex
@misc{marena2026alkene_alkyne_quantum,
  author       = {Tommaso R. Marena},
  title        = {Quantum Simulation of Alkenes and Alkynes via PySCF},
  year         = {2026},
  publisher    = {GitHub},
  url          = {https://github.com/Tommaso-R-Marena/quantum-alkene-alkyne-pyscf}
}
```

## Contributing

Pull requests welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

Apache 2.0
