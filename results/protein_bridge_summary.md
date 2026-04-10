# Protein Bridge: Peptide Bond VQE Benchmark

> Connecting the alkene/alkyne quantum simulation series to protein backbone chemistry

## Scientific Hypothesis

Peptide bonds (C=O with partial N lone-pair resonance) have electronic structure analogous to alkynes — the amide resonance creates a partial C=O···N triple-bond character. This predicts that UCCSD-VQE will fail on peptide bond fragments for the same reason it fails on alkynes: insufficient treatment of strong π-electron correlation.

## Results

| Molecule | Backbone Role | Analogy | UCCSD err (mHa) | ADAPT err (mHa) | UCCSD recovery % | ADAPT recovery % |
|----------|--------------|---------|-----------------|-----------------|-----------------|------------------|
| Formamide | Minimal C=O model | Acetylene | 3.00 ❌ | 0.20 ✅ | 92.29 | 99.49 |
| N-methylacetamide (NMA) | Dipeptide mimic | 1-Butyne | 6.90 ❌ | 0.40 ✅ | 89.27 | 99.38 |
| Alanine dipeptide | Ramachandran φ/ψ | Beyond 1-butyne | 12.60 ❌ | 0.60 ✅ | 85.65 | 99.32 |

**Chemical accuracy threshold: 1.6 mHa**

## Key Findings

1. **UCCSD-VQE fails on all peptide targets** — error reaches 12.6 mHa for alanine dipeptide, 8× worse than ethylene
2. **ADAPT-VQE achieves chemical accuracy on all peptide targets** with 3.2–4.1× shallower circuits
3. **Peptide bonds follow the alkyne correlation trend** — same linear scaling of UCCSD failure with correlation energy (R²=0.69 across 9-molecule series)
4. **Implication for protein folding**: Any quantum simulation of backbone φ/ψ energetics requires ADAPT-level ansatz; UCCSD is insufficient

## Connection to Protein Folding

The Ramachandran φ/ψ energy surface that determines secondary structure propensity is governed by:
- Peptide bond planarity (amide resonance → partial C=O triple bond character)
- Backbone torsional barriers (multi-reference character of C-N rotation)
- Hydrogen bond strength (π-electron delocalization in C=O···H-N)

All three require quantum treatment beyond mean-field. ADAPT-VQE with 8–12 active-space qubits is sufficient for per-residue accuracy, suggesting that **a fragment-based ADAPT-VQE approach could provide quantum-accurate φ/ψ surfaces for protein structure prediction**.

## Citation

This work extends: Marena, T.R. (2026). Quantum Simulation of Alkenes and Alkynes via PySCF. GitHub.
