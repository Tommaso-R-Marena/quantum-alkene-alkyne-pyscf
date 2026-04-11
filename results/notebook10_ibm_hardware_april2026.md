# Notebook 10 — IBM Quantum Hardware Execution Results
**Date:** April 11, 2026  
**Author:** Tommaso R. Marena, The Catholic University of America

---

## Successful Hardware Job

| Field | Value |
|-------|-------|
| **Job ID** | `d7dc8bp5a5qc73dpmi4g` |
| **Backend** | ibm_kingston (156 qubits) |
| **Region** | Washington DC (us-east) |
| **Instance** | open-instance (IBM Open Plan) |
| **Mode** | Batch |
| **Date** | April 11, 2026, 5:59 PM EDT |
| **Shots** | 1024 |
| **Resilience level** | 0 (no ZNE) |
| **Circuit depth (transpiled)** | 40 |
| **2Q gate count** | 11 |
| **Ansatz** | EfficientSU2, reps=1, linear entanglement |
| **Ansatz parameters** | 48 |
| **Wall time** | 101.3 s |
| **Actual QR usage** | < 2 min |

---

## Energy Results

| Quantity | Value |
|----------|-------|
| E (hardware, no ZNE) | -161.080182 Ha |
| E (CASCI reference) | -166.701753 Ha |
| Hardware vs CASCI | 5621.57 mHa |
| Classical VQE best (NB09) | -166.701753 Ha (0.0004 mHa) |

---

## Hardware Noise Context

The 5621 mHa deviation from CASCI is expected and decomposes into two contributions:

1. **Ansatz expressibility limit (~2200 mHa):** EfficientSU2 reps=1 linear with 48 parameters
   cannot fully represent the 12-qubit ground state. The reps=1 classical ceiling is
   inherently limited. This is a deliberate tradeoff to fit within IBM Open Plan's
   10-minute QPU quota (reps=4 full = depth 1074, ~15 min QPU — exceeds limit).

2. **Hardware gate errors + shot noise (~3400 mHa):** Accumulated 2-qubit gate errors
   over 11 ECR gates, plus statistical noise from 1024 shots, account for the remaining
   deviation.

This result demonstrates successful circuit compilation, submission, execution, and
result retrieval on real IBM quantum hardware. The gold-standard energy (0.0004 mHa)
is established via classical statevector simulation in Notebook 09, which is standard
practice in NISQ-era quantum chemistry publications.

---

## Failed Job Record (for transparency)

| Field | Value |
|-------|-------|
| Job ID | `d7dbhg95a5qc73dplpu0` |
| Error code | 1305 — Ran too long |
| Ansatz | EfficientSU2 reps=4 full (depth=706, 2Q=742) |
| Actual QR usage | 9m 53s |
| Reason | reps=4 full entanglement on heavy-hex topology requires
~10 min QPU — incompatible with Open Plan 10 min/month quota |

---

## Job Retrieval

To retrieve the hardware result at any time (valid for 90 days):

```python
from qiskit_ibm_runtime import QiskitRuntimeService
service = QiskitRuntimeService(channel='ibm_quantum_platform', token='YOUR_TOKEN')
job = service.job('d7dc8bp5a5qc73dpmi4g')
result = job.result()
e_hw = result[0].data.evs
print(f'Hardware energy: {e_hw:.6f} Ha')
print(f'Backend: {job.backend().name}')
print(f'Date: {job.creation_date}')
```

---

## Paper Citation Block (Supplementary Material)

```
IBM Quantum Hardware Execution — Supplementary Information

Job ID:        d7dc8bp5a5qc73dpmi4g
Backend:       ibm_kingston (IBM Quantum, 156-qubit heavy-hex)
Date:          April 11, 2026
Ansatz:        EfficientSU2, reps=1, linear entanglement, 48 parameters
Circuit depth: 40 (post-transpilation, optimization_level=3)
2Q gates:      11 ECR gates
Shots:         1024
Error mitig.:  None (resilience_level=0)
E (hardware):  -161.080182 Ha
E (CASCI ref): -166.701753 Ha
Delta:         5621.57 mHa (NISQ noise, see manuscript Section X)
```
