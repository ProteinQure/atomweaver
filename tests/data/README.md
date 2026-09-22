# Exact numerical regression fixtures

All expected values were captured from the untouched implementation at commit
`edbaf8c`, using the released `weights/atomweaver.pt` and bundled data assets.
They must not be regenerated from the refactored implementation to resolve a
regression failure.

Capture environment: Python 3.14.7, CPU PyTorch 2.14.0+cpu, NumPy 2.5.3,
SciPy 1.18.1, scikit-learn 1.9.1, joblib 1.6.0, and Typer 0.27.2.
Torch and fitting thread pools use two threads. Exact floating-point equality
across other dependency versions or hardware is not assumed.

- `initialization.pt`: ordered checkpoint keys and Torch RNG state after loading
  the model with seed 17. This protects construction order as well as loading.
- `joint.pt`, `subset.pt`, `recycled.pt`: three reverse steps with seed 123,
  using the first three peptide residues, four target residues, and 24 target
  atoms from `examples/9RA5_MK8.pdb`. Cases cover joint design, clean pinning of
  peptide residue 2, and two neighbor recycle passes.
- `full.pt`: three reverse steps with seed 77 on the complete example: 20 peptide
  residues, 45 target residues, and 341 target atoms, including diagnostics.
- `long.pt`: 250 reverse steps with seed 2026 on the cropped joint input,
  including diagnostics and the RNG state after sampling.
- `dataset.pt`: all parsed tensors and metadata for the complete example, both
  before and after collation.
- `features.npy`, `clouds/reference_s0.pdb`, and the four read-out JSON files:
  exact features and predictions for full300/canon20, with and without natfreq.
- `fit_clouds.npz`, `residues.txt`, `fit_expected.npz`: a small ALA/LEU/MK8
  custom-head fit, warm-started from the shipped head with Torch seed 456.
  Comparisons cover class order, prior, scaler statistics, coefficients, and
  intercepts.

Sampling uses shell jitter variance scaling 0.25, five element classes,
reserved-slot-zero exemption, and element temperature 1.0. Each sampling fixture
contains its exact inputs and every returned output, including intermediate
coordinate, element, and mask trajectories. Comparisons use zero tolerance.

The regular suite runs the short checks. Enable the slower complete-example and
250-step comparisons explicitly (the model weights must be present):

```bash
ATOMWEAVER_RUN_SLOW_TESTS=1 pytest -q tests/test_inference.py
```

These fixtures verify CPU behavior in the capture environment, not GPU parity.
