# Contributing to quantum-alkene-alkyne-pyscf

Thanks for taking the time to read this. The repo backs a manuscript submission;
correctness and reproducibility matter more than feature velocity.

## Local development

```bash
git clone https://github.com/Tommaso-R-Marena/quantum-alkene-alkyne-pyscf.git
cd quantum-alkene-alkyne-pyscf
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

If conda is preferred:

```bash
conda env create -f environment.yml
conda activate quantum-chem
pip install -e ".[dev]"
```

## Running the test suite

The unit tests stub out `pyscf` and `openfermion` via `sys.modules.setdefault`
so they run in milliseconds without heavy dependencies. The integration test
under `tests/integration/` exercises the **real** packages end-to-end and
verifies the frozen-core fix. Because the stubs in the unit tests pollute
`sys.modules`, the integration test must run in a fresh subprocess.

Run them separately:

```bash
pytest tests/unit/                       # ~0.2 s, no heavy deps
pytest tests/integration/ --forked       # ~50 s, real PySCF/OpenFermion
```

Or as one command, telling pytest to fork the integration tests:

```bash
pytest tests/unit/ && pytest tests/integration/ --forked
```

CI runs both in separate jobs.

## Notebook conventions

All notebooks must run top-to-bottom on Colab Pro with zero errors. Each
notebook starts with a single cell that installs / verifies its dependencies
via `subprocess.check_call(...)`. Do not assume packages are present.

For every claimed numerical result, add an `assert` immediately after the
computation. The reviewer should be able to read a single PASSED line and
know that the result is reproducible. Notebook 09 is the gold standard.

When adding a new claim, also add a regression test under
`tests/integration/` that builds the same Hamiltonian from scratch and
diagonalises it.

## Quantum chemistry conventions

- **STO-3G only** for VQE / hardware experiments. Larger bases (6-31G\*, cc-pVDZ)
  cost too many qubits for current NISQ hardware.
- **Active space first, then map.** Compute CASCI integrals with PySCF, then build
  the OpenFermion `InteractionOperator`. Do not call `get_fermion_operator` on
  the full Hamiltonian for anything beyond H2.
- **Frozen-core fix is mandatory.** Pass `ecore=0` to `InteractionOperator` and
  compute the exact constant via `ecore_needed = e_gs(H_mat) - e_jw(ecore=0)`.
  Passing PySCF's `ecore` straight in produces a silent >40 Ha bias. The
  regression in `tests/integration/test_frozen_core_fix.py` will catch any
  reintroduction.

## Pull requests

- One topic per PR. Mixed PRs are slower to review.
- Update the relevant assertion in Notebook 09 if a numerical result moves.
- Run `ruff check src/ tests/` before pushing. The CI lint job blocks merges.
- If you add or remove a dependency, update `pyproject.toml`, `environment.yml`,
  and `requirements.txt`.

## IBM Quantum hardware runs (Notebook 10)

Hardware submissions cost real shot credits on the Open plan, so they must be
deliberate.

- One `Batch` block, one `EstimatorV2` PUB per run.
- `optimization_level=3` for transpilation, `resilience_level=1` for ZNE, 8192
  shots.
- Bind theta\* into the ansatz *before* transpiling; assert
  `circuit_isa.num_parameters == 0` so no Estimator parameter binding happens.
- Persist the Job ID to `results/hardware_job_id.txt` *immediately* after
  submission, before waiting on `job.result()`.
- Open plan does **not** allow `Session` mode (HTTP 400). Use `Batch`.
- `channel='ibm_quantum'` has been retired by `qiskit-ibm-runtime` 0.40+. Use
  `channel='ibm_quantum_platform'`.

Do **not** commit IBM Quantum tokens.

## Citation

If you use this code in academic work, please cite the reference in the README's
*How to Cite* section.
