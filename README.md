# AtomWeaver

**Multi-component flow matching with a structured geometric prior for non-canonical peptide design.**

AtomWeaver designs peptide side chains as *atoms in ℝ³* given a fixed peptide backbone and a target
protein, then reads off residue identity by matching each predicted atom cloud against a reference
library of canonical **and non-canonical** residues. Because identity is a geometric match performed at
read time - not a learned label over a fixed alphabet - a new non-canonical amino acid (NCAA) is
introduced simply by **adding one reference structure to the library, with no retraining.**

This repository provides the trained model weights, inference code to run the model on your own
structures, and the 300-residue reference library, together with tools to extend the residue vocabulary
and refit the read-out.

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

Dependencies: `torch`, `numpy`, `scipy`, `scikit-learn`, `joblib`, `typer`, `matplotlib`,
`biotite`, `pytorch_lightning`. A CUDA GPU is required for sampling; the read-out runs on CPU.

### Model weights

The weights are released as an asset on the [Releases page](https://github.com/ProteinQure/atomweaver/releases).
Download `atomweaver.pt` (~192 MB) into `weights/`:

```bash
mkdir -p weights
# download atomweaver.pt from the latest release into weights/
```

Everything else the model needs: the 300-residue reference library, the read-out head, and the
read-out cache is bundled in `data/` and used by default.

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

Using a genuinely **new** residue (outside the shipped 300) additionally needs its reference rotamers in
the `--ref-db` library and a rebuilt sampling cache (`build_readout_cache.py --eval-db my_library.pt`); the
bundled library already covers the full 300.

---

## Repository layout

```
scripts/joint_diffusion/design.py            # one-command design front door
scripts/joint_diffusion/
  ├─ sample.py                                # sampler (design.py calls this)
  ├─ apply_hybrid_readout.py, hybrid_lib.py   # balanced hybrid discretizer (read-out)
  ├─ fit_learned_readout.py                   # refit the read-out head
  └─ build_readout_cache.py                   # rebuild the sampling cache after a vocab change
atomweaver/joint_diffusion/                     # model: flow matching, SE(3) denoiser, shell prior,
                                              #        model loader, discretization matcher
data/                                         # reference library, read-out heads, sampling cache,
                                              #        phi/psi table
examples/                                     # a worked example input (peptide + target)
weights/                                      # download atomweaver.pt + readout_train_clouds.npz (Release assets) here
```

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
