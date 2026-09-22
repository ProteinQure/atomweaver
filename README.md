# AtomWeaver

**Multi-component flow matching with a structured geometric prior for non-canonical peptide design.**

AtomWeaver designs peptide side chains as *atoms in ℝ³* given a fixed peptide backbone and a target
protein, then reads off residue identity by matching each predicted atom cloud against a reference
library of canonical **and non-canonical** residues. Because identity is a geometric match performed at
read time - not a learned label over a fixed alphabet - a new non-canonical amino acid (NCAA) is
introduced simply by **adding one reference structure to the library, with no retraining.**

This repository provides the trained model weights, inference code to run the model on your own
structures, and the 300-residue reference library, together with tools to extend the residue vocabulary
and refit the read-out. The supported architecture is the released `atomweaver.pt`; historical
research configurations and model-training APIs are not maintained.

> Method: *AtomWeaver: Multi-Component Flow Matching with a Structured Geometric Prior Facilitates
> Non-Canonical Peptide Design* (see [Citation](#citation)).

---

## Installation

Python ≥ 3.11.

```bash
git clone https://github.com/ProteinQure/atomweaver.git
cd atomweaver
pip install -e .        # or: uv sync
```

Dependencies: `torch`, `numpy`, `scipy`, `scikit-learn`, `joblib`, and `typer`.
The design command uses CUDA for sampling; the lower-level `sample.py --device cpu` also supports CPU
inference. The read-out runs on CPU.

### Model weights

The weights are released as an asset on the [Releases page](https://github.com/ProteinQure/atomweaver/releases).
Download `atomweaver.pt` (~192 MB) into `weights/`:

```bash
mkdir -p weights
# download atomweaver.pt from the latest release into weights/
```

The reference libraries and read-out heads are bundled in `data/` and used by default.

---

## Quickstart - design a peptide

Put one or more input PDBs in a directory. Each PDB must contain the **peptide backbone** (the chain to
be designed) and the **target protein**. Then:

```bash
python scripts/joint_diffusion/design.py --pdb-dir examples/ --out OUTDIR
```

This samples side-chain atom clouds for every input (production settings baked in), reads off identity
with the balanced hybrid discretizer over the 300-residue vocabulary, and writes:

| output | contents |
|---|---|
| `OUTDIR/designs.fasta` | one record per (input, sample); canonical residues as 1-letter codes, NCAAs as bracketed CCD codes, e.g. `A[DAL]C[MK8]...` |
| `OUTDIR/designs.csv` | per-position top-3 identities with probabilities |
| `OUTDIR/clouds/` | the raw predicted atom-cloud PDBs |
| `OUTDIR/preds.json` | the full read-out distributions over all 300 types |

Each PDB in `clouds/` has **three chains**: **A** = the target backbone; **B** = the input peptide
(backbone plus its reference side chains, with any NCAA CCD codes preserved); **C** = the generated
side-chain cloud (backbone plus the sampled atoms, named `{element}{slot}`). Chain **C** is the model's
output -- the identities read off it are what land in `designs.csv` / `designs.fasta`.

Useful options:

- `--num-samples N` - designs sampled per input (default 5).
- `--natfreq` - bias composition toward natural (canonical-heavy) frequencies (SwissProt prior).
- `--canon20` - restrict the vocabulary to the 20 canonical amino acids (no NCAA can be called).
- `--dry-run` - print the underlying sampling + read-out commands without running them.

`--natfreq` and `--canon20` are independent and composable.

---

## Refitting the read-out for a custom vocabulary

Identity is read off the geometry at decode time, so **narrowing the vocabulary needs no retraining of the
model** -- only the small CPU read-out head is refit so its classes match your subset. The training clouds
the shipped head was fit on are a Release asset, `readout_train_clouds.npz` (39 MB; 294-dim features of
~274k model-predicted side-chain clouds, labelled by residue type; covers the full shipped 300-type
vocabulary) -- download it into `weights/` alongside the model weights. A refit then takes a few seconds.

Run from the repo root so the package is importable:

```bash
export PYTHONPATH=$PWD

# 1. name the subset -- one CCD code per line, any subset of the shipped vocabulary
printf 'ALA
LEU
LYS
MK8
PTR
DLY
' > residues.txt

# 2. refit the learned read-out head (warm-started from the shipped head; seconds, CPU).
#    --ckpt-fit-cache defaults to weights/readout_train_clouds.npz (the Release asset).
python scripts/joint_diffusion/fit_learned_readout.py \
  --ref-db data/reference_library.pt \
  --phipsi-table data/phipsi_populations.pt \
  --residues residues.txt \
  --warmstart-from data/readout_head_full300.joblib \
  --out my_head.joblib

# 3. design with the custom head -- only the subset's residues can be called
python scripts/joint_diffusion/design.py --pdb-dir examples/ --out OUTDIR --head my_head.joblib
```

A **new** residue outside the shipped 300 needs reference rotamers in the fitting `--ref-db` library
and a refitted head. Pass that library to design with `--eval-db my_library.pt` and the fitted head
with `--head my_head.joblib`. The bundled library already covers the full 300.

---

## Repository layout

```
scripts/joint_diffusion/
  ├─ design.py                  # sampling, read-out, FASTA/CSV export
  ├─ sample.py                  # joint and subset sampling
  ├─ apply_hybrid_readout.py     # hybrid read-out CLI
  ├─ fit_learned_readout.py      # custom-vocabulary read-out fitting
  └─ build_readout_cache.py      # optional reference-only classifier cache utility
atomweaver/joint_diffusion/
  ├─ models.py, diffusion.py    # fixed released architecture and sampling math
  ├─ graph.py                   # radius graphs and edge-type embeddings for SE(3) attention
  ├─ model_loader.py           # strict checkpoint loading
  ├─ sampling.py               # runtime shell scaling and recycle settings
  ├─ datasets.py               # PDB parsing, cropping, and batching
  ├─ reference_library.py      # shared reference-library loading
  └─ matching.py, hybrid_readout.py, readout_features.py
                               # geometric scoring, probability blending, shared features
data/                          # reference libraries, read-out heads, phi/psi table
examples/                      # worked example input
weights/                       # downloaded model weights and read-out training clouds
tests/                         # exact CPU regressions against the original implementation
```

## Development and numerical compatibility

```bash
pip install -e '.[dev]'
ruff check .
ruff format --check .
pytest -q
```

The regression fixtures were captured from the original implementation (`edbaf8c`) with the released
checkpoint. They compare tensors with zero tolerance, including initialization RNG state, joint design,
subset pinning, recycling, parsed datasets, read-out probabilities, and seeded custom-head fitting.
Download the model weights to run sampling tests; those tests skip when the weights are absent.
See [fixture provenance](tests/data/README.md) for exact dependency versions and the optional 250-step
comparison. Exact equality is checked within the same CPU software environment; different PyTorch,
NumPy, or scikit-learn versions can produce different floating-point results.

Registered auxiliary modules and buffers are retained where needed to preserve checkpoint keys and
the original initialization order, including random-number consumption.

---

## Citation

AtomWeaver was introduced in:

```bibtex
@article{kitaygorodsky2026atomweaver,
  author    = {Kitaygorodsky, Alexander and Hostallero, David Earl and
               Broom, Aron and Layne, Elliot and Hwang, Sungwon and
               Kanawaty, Ashlin K. and Babej, Tomáš and
               Butterfoss, Glenn L. and Fingerhuth, Mark},
  title     = {{AtomWeaver}: Multi-Component Flow Matching with a Structured
               Geometric Prior Facilitates Non-Canonical Peptide Design},
  journal   = {bioRxiv},
  year      = {2026},
  publisher = {openRxiv}
}
```

## License

- **Code** - [GNU AGPL-3.0](LICENSE). Derivative works, including networked services built on this code,
  must be released under the same license.
- **Model weights** (`atomweaver.pt`, released as a build asset) - [CC BY 4.0](LICENSE-WEIGHTS).
  Free to use, including commercially, with attribution.
