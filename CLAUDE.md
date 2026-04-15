# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AutoElectrolyte is a high-throughput computational chemistry platform for discovering and analyzing polymer electrolytes for battery applications. It integrates:
- **HiTPoly**: End-to-end pipeline for MD simulations of polymer electrolytes (SMILES → force field parameters → simulation → analysis)
- **ML-based screening**: Bayesian optimization with Gaussian processes over molecular embeddings
- **HTVS integration**: Django backend for submitting jobs to the MIT Supercloud cluster

## Activate

```bash
conda activate hitpoly
```


## Main Entry Points

**Single system setup:**
```bash
python HiTPoly/builder_ligpargen_openmm.py \
  --save_path results/example \
  --smiles_path "[Cu]CCO[Au]" \
  --solvent_count 30 --repeats 50 \
  --salt_type Li.TFSI --concentration 100 \
  --temperature 430 --simu_length 100
```

**Molality-based multi-system setup:**
```bash
python HiTPoly/builder_ligpargen_openmm_from_molality_multi_system.py \
  --smiles_path system_spec.txt \
  --salt_type Li.TFSI --molality_salt 1.0
```

**Screening workflow:**
```bash
python HiTPoly/generate_embedding.py   # Generate MoLFormer + PCA embeddings
python HiTPoly/run_screening.py        # Bayesian optimization loop
```

**Cluster job submission (MIT Supercloud via HTVS Django backend):**
```bash
python run_simulations.py
```

## Architecture

### Pipeline Flow

```
SMILES strings
    → Topology / graph construction  (hitpoly/data/builder.py)
    → ML force field prediction      (hitpoly/models/)
    → 3D conformation + box packing  (hitpoly/writers/box_builder.py)
    → OpenMM / Gromacs scripts       (hitpoly/simulations/)
    → MD simulation (external)
    → Post-simulation analysis       (hitpoly/analysis/)
```

### System Assembly (`hitpoly/writers/`)

- `box_builder.py` — builds long polymer chains by expanding `[Cu]...[Au]` SMILES repeat units; calls Packmol to pack polymer + solvent + ions; generates PDB/topology files

### MD Simulation Setup (`hitpoly/simulations/`)

- `openmm_scripts.py` — multi-phase OpenMM orchestration: NVT equilibration → NPT equilibration → NVT production; writes XML force fields
- `gromacs_writer.py` — generates equivalent Gromacs input files

### Post-Simulation Analysis (`hitpoly/analysis/`)

- `trajectory_analysis.py` — MSD (diffusivity), RDF, trajectory unwrapping, ionic conductivity via correlation functions
- `coordination_analysis.py` — neighbor environments, distance distributions
- `gromacs_edr_reader.py` — parse Gromacs energy trajectories

### Bayesian Optimization Screening (`HiTPoly/screening/`)

- `embedding.py` — IBM MoLFormer-XL embeddings → PCA dimensionality reduction
- `screening_utils.py` — GPyTorch GP fitting (Matern/RBF kernels), K-means clustering of feature space, batch candidate selection per cluster
- Iterative batches stored in `batch0/`, `batch1/`, etc.

### Key Constants and Data (`hitpoly/utils/`, `HiTPoly/data/`)

- `utils/constants.py` — element properties, LJ σ/ε parameters, atom charges
- `utils/geometry_calc.py` — bond lengths, angles, dihedral geometry primitives
- `data/pdb_files/` — ion geometry PDBs (Li, Na, Zn)
- `data/forcefield_files/` — LAMMPS force field reference data

## Polymer SMILES Convention

Polymer chains use Cu/Au as linker atoms to mark repeat-unit ends:
- `[Cu]` — start of repeat unit
- `[Au]` — end of repeat unit
- Example: `[Cu]CCO[Au]` → poly(ethylene oxide) repeat unit

The `create_long_smiles()` function in `box_builder.py` expands these into full-length chains by concatenating repeats and substituting the linkers with proper bonds.

For liquid solvents the start end units are not necessary. 

## HTVS Integration (`run_simulations.py`)

Communicates with a Django REST backend (separate repo) for cluster job management. Key objects: `WorkflowManager`, `Species`, `Geometry`, `RunID`. This file is the bridge between the local HiTPoly pipeline and MIT Supercloud batch submission.
