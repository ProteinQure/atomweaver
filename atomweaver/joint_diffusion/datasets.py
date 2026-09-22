"""
Dataset utilities for loading peptide structures from PDB files.

This module provides PyTorch Dataset classes for loading peptide-protein complex
structures, separating backbone and side-chain atoms for the inverse folding
diffusion model.

The key assumption is that the **shorter chain** in each PDB is the peptide ligand,
while the longer chain is the protein target.
"""

from __future__ import annotations

import csv
import logging
import os
import warnings
from collections import OrderedDict
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch.utils.data import Dataset

from .diffusion import NUM_ELEMENT_TYPES  # vocab knob single source of truth (validated in diffusion.py)

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    import numpy as np  # for "np.ndarray" string annotations in the on-the-fly NCAA windowing helpers


def _atomic_torch_save(obj: object, path: "str | Path") -> None:
    """``torch.save`` via a per-process temp file + atomic rename.

    Safe when several Ray DDP workers write the same NFS cache path
    concurrently: every worker computes identical content, writes its own
    pid-suffixed temp file, then atomically renames it into place.
    """
    path = Path(path)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    try:
        torch.save(obj, tmp)
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


# Candidate tuple width: every per-source window candidate is (window_size, ncaa_idx, start).
_CAND_FIELDS = 3


class _NumpyCandidateIndex:
    """Host-RAM-safe store for the per-source NCAA window-candidate index.

    Why this is numpy, not a ``dict[int, list[tuple[int, int, int]]]``
    -----------------------------------------------------------------
    At full monomer scale (~153k source PDBs) the candidate index is the single
    biggest persistent Python structure in the dataset. A plain
    ``dict[int, list[tuple[int, int, int]]]`` is a deep graph of millions of
    tiny Python objects (dict entries, list objects, tuple objects, boxed ints).
    Under a FORKED DataLoader worker, the first read of any of those objects
    bumps its CPython refcount, which writes the object header, which dirties the
    4 KiB page it lives on -- so copy-on-write breaks and the WHOLE index is
    duplicated into that worker's private RSS. Host RAM then scales as
    ``index_size x num_workers`` and OOM-kills the box (the distributed v6 OOM,
    band-aided with ``num_workers=1`` + a 64Gi cap).

    The numpy representation is ONE C buffer per array (CSR-style):

    - ``_data``    : ``int32 [total_candidates, 3]`` -- every source's candidate
      tuples ``(window_size, ncaa_idx, start)`` concatenated in source order.
    - ``_offsets`` : ``int64 [n_sources + 1]`` -- source ``i``'s candidates are
      ``_data[_offsets[i]:_offsets[i + 1]]`` (CSR row pointers).
    - ``_status``  : ``int8  [n_sources]`` -- ``-1`` unknown / not yet parsed,
      ``0`` unusable (negative cache, the old ``None``), ``1`` usable.

    Each array is a single Python object holding a contiguous C buffer, so a
    forked worker reading it bumps ONE refcount (on the ndarray object), leaving
    the multi-megabyte data buffer page-clean -> shared via COW -> host RAM stays
    flat regardless of ``num_workers``. When loaded from the disk cache the arrays
    are ``mmap_mode="r"`` memory-maps, so they're both lower-resident AND shared
    page-cache across workers.

    Lazy phase vs finalized phase
    -----------------------------
    Before the full ``_build_valid_indices`` scan runs, sources may be parsed one
    at a time (``_load_source`` miss). Those land in a small ``_pending`` dict
    overlay (tiny -- a handful of sources, not the whole pool). ``finalize_from``
    (or a cache load) replaces everything with the packed numpy arrays and drops
    the overlay. The public mapping API (``in``, ``[]``, ``[]=``, ``len``,
    ``items``, ``values``) is preserved so existing call sites and tests are
    unchanged; reads after finalize decode a tiny per-call list from the numpy
    slice and never resurrect a persistent per-source Python object graph.
    """

    __slots__ = ("_data", "_n", "_offsets", "_pending", "_status")

    def __init__(self) -> None:
        self._n: int = 0
        # Packed numpy arrays (None until finalize / cache-load).
        self._data = None  # int32 [total_candidates, 3]
        self._offsets = None  # int64 [n_sources + 1]
        self._status = None  # int8 [n_sources]
        # Lazy overlay: idx -> (None | list[tuple[int, int, int]]) for sources parsed
        # before a full finalize. Bounded by however many singletons were touched.
        self._pending: "dict[int, list[tuple[int, int, int]] | None]" = {}

    # ----- packing / loading -------------------------------------------------

    @staticmethod
    def _pack(items: "dict[int, list[tuple[int, int, int]] | None]", n: int):
        """Pack candidates into CSR numpy arrays.

        ``items`` maps ``idx -> candidates-or-None`` (every idx in ``range(n)`` present).
        Returns ``(data, offsets, status)``.
        """
        import numpy as _np

        offsets = _np.zeros(n + 1, dtype=_np.int64)
        status = _np.full(n, -1, dtype=_np.int8)
        rows: list = []
        for i in range(n):
            cands = items.get(i, None)
            if i not in items:
                # Unknown source (never parsed): no row, status stays -1.
                offsets[i + 1] = offsets[i]
                continue
            if cands is None:
                status[i] = 0  # unusable / negative cache
                offsets[i + 1] = offsets[i]
                continue
            status[i] = 1
            rows.extend((int(t[0]), int(t[1]), int(t[2])) for t in cands)
            offsets[i + 1] = offsets[i] + len(cands)
        data = (
            _np.asarray(rows, dtype=_np.int32).reshape(-1, _CAND_FIELDS)
            if rows
            else _np.empty((0, _CAND_FIELDS), dtype=_np.int32)
        )
        return data, offsets, status

    def finalize_from(self, items: "dict[int, list[tuple[int, int, int]] | None]", n: int) -> None:
        """Replace all state with packed numpy arrays built from ``items`` over ``range(n)``."""
        self._n = int(n)
        self._data, self._offsets, self._status = self._pack(items, self._n)
        self._pending = {}

    @property
    def finalized(self) -> bool:
        return self._status is not None

    def to_cache_arrays(self) -> dict:
        """Numpy arrays for ``np.savez`` (only valid once finalized)."""
        import numpy as _np

        return {
            "cand_data": _np.asarray(self._data),
            "cand_offsets": _np.asarray(self._offsets),
            "cand_status": _np.asarray(self._status),
            "cand_n": _np.asarray([self._n], dtype=_np.int64),
        }

    def load_cache_arrays(self, npz) -> None:
        """Adopt mmap'd arrays from a loaded ``np.load(..., mmap_mode='r')`` archive."""
        self._data = npz["cand_data"]
        self._offsets = npz["cand_offsets"]
        self._status = npz["cand_status"]
        self._n = int(npz["cand_n"][0])
        self._pending = {}

    # ----- mapping-like API (preserves the old dict interface) ---------------

    def _status_of(self, idx: int) -> int:
        """-1 unknown, 0 unusable, 1 usable -- checking the overlay first, then numpy."""
        if idx in self._pending:
            return 0 if self._pending[idx] is None else 1
        if self._status is not None and 0 <= idx < self._n:
            return int(self._status[idx])
        return -1

    def __contains__(self, idx: int) -> bool:
        return self._status_of(idx) != -1

    def __getitem__(self, idx: int):
        """Return ``None`` (unusable) or a freshly-decoded ``list[tuple[int, int, int]]``.

        The returned list is a per-call temporary decoded from the shared numpy
        buffer; it is NOT retained, so no persistent per-source Python object
        graph is rebuilt (that would re-introduce the COW blowup).
        """
        if idx in self._pending:
            return self._pending[idx]
        st = self._status_of(idx)
        if st == -1:
            raise KeyError(idx)
        if st == 0:
            return None
        lo = int(self._offsets[idx])
        hi = int(self._offsets[idx + 1])
        block = self._data[lo:hi]
        return [(int(r[0]), int(r[1]), int(r[2])) for r in block]

    def __setitem__(self, idx: int, value) -> None:
        """Lazy single-source insert (overlay), used before a full finalize.

        Once finalized, a source already covered by the numpy arrays (status != -1) is
        authoritative, so re-inserting it (e.g. a heavy-cache-miss re-parse) is a NO-OP --
        this keeps the overlay from growing as sources are re-parsed, preserving the
        host-RAM win (the persistent index stays entirely in the shared numpy buffers).
        """
        if self._status is not None and 0 <= idx < self._n and int(self._status[idx]) != -1:
            return
        self._pending[idx] = None if value is None else [tuple(t) for t in value]

    def get(self, idx: int, default=None):
        if self.__contains__(idx):
            return self.__getitem__(idx)
        return default

    def __len__(self) -> int:
        if self._status is None:
            return len(self._pending)
        # Number of SEEN sources: finalized non-(-1) plus any extra overlay-only ones.
        import numpy as _np

        seen = int((_np.asarray(self._status) != -1).sum())
        extra = sum(1 for i in self._pending if not (0 <= i < self._n and int(self._status[i]) != -1))
        return seen + extra

    def _seen_indices(self):
        import numpy as _np

        idxs = {int(i) for i in _np.nonzero(_np.asarray(self._status) != -1)[0]} if self._status is not None else set()
        idxs.update(self._pending.keys())
        return sorted(idxs)

    def items(self):
        for i in self._seen_indices():
            yield i, self.__getitem__(i)

    def values(self):
        for i in self._seen_indices():
            yield self.__getitem__(i)

    def keys(self):
        return iter(self._seen_indices())

    def __iter__(self):
        return iter(self._seen_indices())


# Backbone atom names (in canonical order)
BACKBONE_ATOMS = ["N", "CA", "C", "O"]

# Standard amino acid 3-letter codes
STANDARD_AA = {
    "ALA",
    "ARG",
    "ASN",
    "ASP",
    "CYS",
    "GLN",
    "GLU",
    "GLY",
    "HIS",
    "ILE",
    "LEU",
    "LYS",
    "MET",
    "PHE",
    "PRO",
    "SER",
    "THR",
    "TRP",
    "TYR",
    "VAL",
}

# 3-letter to 1-letter code mapping
AA_3TO1 = {
    "ALA": "A",
    "CYS": "C",
    "ASP": "D",
    "GLU": "E",
    "PHE": "F",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LYS": "K",
    "LEU": "L",
    "MET": "M",
    "ASN": "N",
    "PRO": "P",
    "GLN": "Q",
    "ARG": "R",
    "SER": "S",
    "THR": "T",
    "VAL": "V",
    "TRP": "W",
    "TYR": "Y",
}

# Canonical sidechain atom ordering for standard amino acids (PDB naming convention)
# This ensures consistent atom ordering between PDB data and reference structures
# Order follows: beta -> gamma -> delta -> epsilon -> zeta (by Greek letter distance from CA)
SIDECHAIN_ATOM_ORDER = {
    "ALA": ["CB"],
    "ARG": ["CB", "CG", "CD", "NE", "CZ", "NH1", "NH2"],
    "ASN": ["CB", "CG", "OD1", "ND2"],
    "ASP": ["CB", "CG", "OD1", "OD2"],
    "CYS": ["CB", "SG"],
    "GLN": ["CB", "CG", "CD", "OE1", "NE2"],
    "GLU": ["CB", "CG", "CD", "OE1", "OE2"],
    "GLY": [],  # No sidechain
    "HIS": ["CB", "CG", "ND1", "CD2", "CE1", "NE2"],
    "ILE": ["CB", "CG1", "CG2", "CD1"],
    "LEU": ["CB", "CG", "CD1", "CD2"],
    "LYS": ["CB", "CG", "CD", "CE", "NZ"],
    "MET": ["CB", "CG", "SD", "CE"],
    "PHE": ["CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ"],
    "PRO": ["CB", "CG", "CD"],  # CD connects back to N
    "SER": ["CB", "OG"],
    "THR": ["CB", "OG1", "CG2"],
    "TRP": ["CB", "CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2"],
    "TYR": ["CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "OH"],
    "VAL": ["CB", "CG1", "CG2"],
}


def load_residue_database(db_path: str | Path) -> dict:
    """
    Load residue database and create name-to-index mapping.

    Parameters
    ----------
    db_path : str or Path
        Path to residue database .pt file.

    Returns
    -------
    dict
        Dictionary containing:
        - coords: (num_residues, max_atoms, 3) tensor
        - masks: (num_residues, max_atoms) tensor
        - one_letter_to_idx: mapping from one-letter code to database index
        - three_letter_to_idx: mapping from three-letter code to database index
        - metadata: original metadata list
    """
    db = torch.load(db_path, weights_only=False)

    # Build one-letter to index mapping
    one_letter_to_idx = {}
    for i, meta in enumerate(db["metadata"]):
        one_letter = meta.get("one_letter", "")
        if one_letter and one_letter not in one_letter_to_idx:
            one_letter_to_idx[one_letter] = i

    # Build three-letter to index mapping (using AA_3TO1)
    three_letter_to_idx = {}
    for three, one in AA_3TO1.items():
        if one in one_letter_to_idx:
            three_letter_to_idx[three] = one_letter_to_idx[one]

    return {
        "coords": db["coords"],
        "masks": db["masks"],
        "one_letter_to_idx": one_letter_to_idx,
        "three_letter_to_idx": three_letter_to_idx,
        "metadata": db["metadata"],
        "num_residues": db["num_residues"],
    }


def res_names_to_indices(res_names: list[str], name_to_idx: dict[str, int]) -> torch.Tensor:
    """
    Convert residue names to database indices.

    Parameters
    ----------
    res_names : list[str]
        List of residue names (3-letter codes like "ALA", "GLY").
    name_to_idx : dict[str, int]
        Mapping from residue name to database index.

    Returns
    -------
    torch.Tensor
        Tensor of indices, shape (len(res_names),). Unknown residues get index 0.
    """
    indices = [name_to_idx.get(name, 0) for name in res_names]
    return torch.tensor(indices, dtype=torch.long)


# Two-character element symbols in the model vocab that are prefixes of PDB atom
# names. Checked before the one-char fallback so CL1/BR2/SE atoms aren't truncated
# to C/B/S. Safe because no canonical backbone/sidechain atom name starts with these
# (carbons are CA/CB/CG/CD/CE/CZ/CH..., never CL/BR/SE).
_TWO_CHAR_ELEMENTS = ("CL", "BR", "SE")


def infer_element_from_atom_name(atom_name: str) -> str:
    """Infer an element symbol from a PDB atom name (fallback when no element column).

    Prefers known two-character symbols (CL, BR, SE) so halogen/selenium atoms are
    not truncated to a single character, then falls back to the first character.

    Parameters
    ----------
    atom_name : str
        The PDB atom name (columns 13-16), e.g. ``CA``, ``CL1``, ``SE``.

    Returns
    -------
    str
        The inferred element symbol (uppercase). Defaults to ``"C"`` for empty input.
    """
    name = atom_name.strip().upper()
    # Strip leading digits (PDB hydrogens are often named 1HG1, 2HD2, ...) so they
    # infer as "H" and get excluded downstream, rather than as a bogus "1" element
    # that survives as a PAD-typed phantom sidechain atom.
    name = name.lstrip("0123456789")
    if not name:
        return "C"
    for sym in _TWO_CHAR_ELEMENTS:
        if name.startswith(sym):
            return sym
    return name[0]


# Expected heavy side-chain atom count per residue, derived from SMILES.
#
# A residue whose deposited composition contradicts its own label is overwhelmingly
# crystallographic disorder: a flexible side chain was not resolved and the model stops at
# CB. Measured on the 11,763-complex training corpus, 2.26% of canonical and 1.15% of
# non-canonical peptide residues deviate, and the rate tracks flexibility exactly as disorder
# predicts (LYS 4.9%, GLU 2.8%, ARG 2.3% vs VAL/THR/PHE ~0.4%).
#
# Such a site must NOT be a redesign target: its ground truth is unknowable, and training on
# it teaches the model that e.g. LYS sometimes has one atom -- biasing generation downward,
# a candidate cause of the persistent ~17% atom undercount. Relabelling it ALA would be
# worse: the backbone was shaped by the real, larger residue, so the model would learn both a
# wrong identity and a wrong geometry-to-identity mapping.
#
# It IS kept in the structure as pocket context. Excluding it from `design_mask` removes the
# corrupted supervision while leaving the backbone -- which is perfectly valid -- in place.
_SIDECHAIN_TOLERANCE = 2  # deviation of >2 heavy atoms excludes the site
_SMILES_TABLE_PATH = Path(__file__).resolve().parents[1] / "data" / "residue_smiles.csv"
_EXPECTED_SIDECHAIN_ATOMS: "dict[str, int] | None" = None


def expected_sidechain_atoms() -> dict[str, int]:
    """``{resname: heavy side-chain atom count}`` from the vendored residue table.

    The count is ``heavy_atoms(unprotected SMILES) - 5``, which removes the amino-acid
    backbone N, CA, C, O and OXT. Verified exact for all 20 canonical residues, and
    :func:`_heavy_sidechain_count` drops OXT to match. It is READ from the table's
    ``n_sidechain_atoms`` column rather than recomputed: the column is generated once, under a
    recorded RDKit version, by ``scripts/data_prep/build_residue_smiles_table.py``, and a test
    asserts every row still equals ``heavy_atoms(smiles) - 5``. Parsing at import time instead
    made two gates silently vanish whenever RDKit was missing, and left them exposed to a
    library upgrade quietly changing a count.

    The table is generated from the company amino-acid database (see the generator's docstring
    for the admission rules and the recorded source commit in
    ``residue_smiles.provenance.json``). A residue absent from it is trusted as-is -- so is
    ``UNK``, which is deliberately excluded because "identity undetermined" is not a
    composition.

    A duplicate resname raises. The table it replaced had 152 of them, one pair being unrelated
    molecules filed under ``E95``, which made the expected count depend on CSV row order.
    """
    global _EXPECTED_SIDECHAIN_ATOMS
    if _EXPECTED_SIDECHAIN_ATOMS is not None:
        return _EXPECTED_SIDECHAIN_ATOMS
    table: dict[str, int] = {}
    try:
        with open(_SMILES_TABLE_PATH, newline="") as fh:
            for row in csv.DictReader(fh):
                if row.get("status") != "ok":
                    continue
                try:
                    n = int(row["n_sidechain_atoms"])
                except (KeyError, TypeError, ValueError):
                    continue
                if n < 0:
                    continue
                resname = row["resname"]
                if resname in table:
                    raise ValueError(
                        f"{_SMILES_TABLE_PATH}: duplicate resname {resname!r}. Expected counts must not "
                        f"depend on row order; regenerate the table with "
                        f"scripts/data_prep/build_residue_smiles_table.py."
                    )
                table[resname] = n
    except OSError as exc:  # table missing -> validate nothing
        _ICODE_LOG.warning("expected_sidechain_atoms: table unavailable (%s); no sites excluded", exc)
    _EXPECTED_SIDECHAIN_ATOMS = table
    return table


def _heavy_sidechain_count(res: dict) -> int:
    """Side-chain heavy atoms, excluding OXT.

    ``BACKBONE_ATOMS`` is ["N", "CA", "C", "O"] and deliberately omits OXT, so
    ``extract_residue_atoms`` files the C-terminal carboxylate oxygen under ``sidechain``.
    Counting it makes every C-terminal residue read as one atom over its label -- which
    inflated the deviation rate from 2.2% to 6.6% before this was caught. OXT is backbone
    chemistry, and the SMILES-derived expectation (heavy - 5) already accounts for both
    terminal oxygens.
    """
    return sum(1 for atom in res["sidechain"] if atom[0] != "OXT")


def exact_composition_mask(residues: "list[dict]") -> "torch.Tensor":
    """False where a residue's composition deviates from its label AT ALL.

    Stricter than :func:`designable_residue_mask`, and used for a different purpose. A residue
    off by one or two atoms is still worth designing -- its backbone is real and the
    ground-truth-free losses (clash, connectivity) apply normally -- but it must not train the
    DISCRETIZATION loss, which matches a generated cloud against reference residue structures.
    An exemplar with the wrong atom count teaches the matcher that e.g. TYR can look like a
    seven-atom side chain, corrupting identity assignment for every later residue of that type.

    Residues absent from the SMILES table are trusted (True), as elsewhere.
    """
    expected = expected_sidechain_atoms()
    keep = torch.ones(len(residues), dtype=torch.bool)
    for i, res in enumerate(residues):
        want = expected.get(res["res_name"])
        if want is not None and _heavy_sidechain_count(res) != want:
            keep[i] = False
    return keep


def designable_residue_mask(residues: "list[dict]", tolerance: int = _SIDECHAIN_TOLERANCE) -> "torch.Tensor":
    """False where a residue's composition contradicts its label by more than ``tolerance``.

    Residues not present in the SMILES table are trusted (True).
    """
    expected = expected_sidechain_atoms()
    keep = torch.ones(len(residues), dtype=torch.bool)
    for i, res in enumerate(residues):
        want = expected.get(res["res_name"])
        if want is None:
            continue
        if abs(_heavy_sidechain_count(res) - want) > tolerance:
            keep[i] = False
    return keep


def apply_designable_gate(design_mask: "torch.Tensor", designable_mask: "torch.Tensor | None") -> "torch.Tensor":
    """Intersect a sampled design mask with the designable-site gate.

    THE ONE place the intersection lives. Call this from every site that produces a
    ``design_mask``; do not open-code ``design_mask & designable_mask`` anywhere else.

    A site whose deposited composition contradicts its own label by more than
    ``_SIDECHAIN_TOLERANCE`` heavy atoms has unknowable ground truth (see
    :func:`designable_residue_mask`). It stays in the structure as pocket context, but it must
    never be a redesign target -- in training it would teach the model that e.g. LYS sometimes
    has one atom, and in evaluation it would score recovery against ground truth we have already
    declared untrustworthy.

    Applying this can leave FEWER than K designed positions. That is deliberate: drawing a
    replacement site would silently change which sites the K-budget covers.

    Why a helper rather than four copies: before this existed the intersection lived only in
    ``BinderDataset.__getitem__``. ``BinderDataset.iter_valid`` and
    ``OnTheFlyNcaaDataset.__getitem__`` (enumeration mode) each synthesised their own design
    mask and skipped it, so the gate protected DataLoader training but not the diagnostic and
    enumeration-eval accessors -- the same one-correct-path-and-N-diverging-copies shape as the
    four private ``peptide_chain_ccds()`` copies this MR removed.

    Parameters
    ----------
    design_mask : torch.Tensor
        ``(L,)`` bool. The sampled design mask. Not mutated.
    designable_mask : torch.Tensor or None
        ``(L,)`` bool from :func:`designable_residue_mask`, or None when the dataset does not
        supply one (then the design mask is returned unchanged -- absence means "no opinion",
        not "nothing is designable").

    Returns
    -------
    torch.Tensor
        The gated mask. The same object as ``design_mask`` when ``designable_mask`` is None,
        otherwise a new tensor.

    Raises
    ------
    ValueError
        If the two masks disagree in length.

    Notes
    -----
    EQUAL LENGTHS ARE REQUIRED, and a mismatch raises rather than truncating to the common
    prefix. Prefix truncation is only meaningful if both masks are index-aligned from position
    0, and the ONLY thing that guarantees that alignment is that both are derived from the same
    residue list -- so a length mismatch is precisely the evidence that they are NOT, and
    truncating would gate the WRONG SITES, silently. Alignment holds by construction at every
    in-repo call site: ``designable_residue_mask`` is computed in ``_load_pdb`` from the
    ``binder_residues`` that ``residues_to_tensors`` turns into ``backbone_coords``, whose
    length is exactly what ``_sample_design_mask`` is given. Windowing happens UPSTREAM of that,
    inside ``OnTheFlyNcaaDataset._parse_pdb_cached``, so a windowed binder gets a
    window-relative mask of the window's own length. ``collate_binders`` normalises its two
    inputs to the item's ``binder_len`` before calling in, keeping its own tolerance for
    hand-built items without weakening this one.

    Idempotent, so calling it twice on the same mask is harmless.
    """
    if designable_mask is None:
        return design_mask
    if design_mask.shape[0] != designable_mask.shape[0]:
        raise ValueError(
            f"design_mask has length {design_mask.shape[0]} but designable_mask has length "
            f"{designable_mask.shape[0]}. The two must be index-aligned over the same residue "
            f"list; a mismatch means they are not, and intersecting the common prefix would "
            f"gate the wrong sites."
        )
    gated = design_mask.clone()
    gated &= designable_mask.to(device=gated.device, dtype=torch.bool)
    return gated


def peptide_chain_ccds(pdb_path: "str | Path", pep_chain: str | None = None) -> "tuple[str, list[str]]":
    """``(peptide_chain_id, [CCD per residue in chain order])``.

    Built on ``parse_pdb_atoms`` / ``extract_residue_atoms`` so the eval path reads residues
    exactly as the dataset does. Four eval scripts previously carried byte-identical private
    copies that keyed on ``(chain, resseq)`` and sorted by ``resseq``. Both are wrong for
    insertion-coded chains:

    * keying on ``resseq`` merges 1 / 1A / 1B, so this returned ``[ALA, GLY]`` where the
      dataset returns ``[ALA, ASP, CYS, GLY]``. Callers then skip the complex on a
      CCD/length mismatch, silently dropping data the dataset handles correctly.
    * sorting by ``resseq`` reorders them. These insertion codes are N-terminal extensions
      numbered backwards -- the chain runs 1B -> 1A -> 1, verified by C->N peptide-bond
      distances of 1.30-1.32 A -- so sorting yields 1, 1A, 1B and reverses the N-terminus.

    Chain order is file order, which is what the PDB format guarantees.
    """
    chains = parse_pdb_atoms(pdb_path)
    if not chains:
        return "", []
    if pep_chain is None or pep_chain not in chains:
        # Shorter chain = peptide, counting DISTINCT residues rather than atoms.
        pep_chain = min(chains, key=lambda c: len({(a["res_id"], a.get("icode", " ")) for a in chains[c]}))
    residues = extract_residue_atoms(chains[pep_chain])
    return pep_chain, [r["res_name"] for r in residues]


# Module-level logger. The module imports logging but never defined one. The `_ICODE_LOG`
# alias is kept because the residue-identity warning below already reads under that name.
_LOG = logging.getLogger(__name__)
_ICODE_LOG = _LOG


def parse_pdb_atoms(pdb_path: str | Path) -> dict[str, list[dict]]:
    """
    Parse atom records from a PDB file.

    Parameters
    ----------
    pdb_path : str or Path
        Path to PDB file.

    Returns
    -------
    chains : dict[str, list[dict]]
        Dictionary mapping chain IDs to lists of atom records.
        Each atom record is a dict with keys:
        - atom_name: str (e.g., 'CA', 'CB')
        - res_name: str (e.g., 'ALA', 'GLY')
        - res_id: int (residue sequence number)
        - icode: str (insertion code, PDB column 27; " " when absent)
        - x, y, z: float (coordinates)
        - element: str (element symbol)
    """
    chains: dict[str, list[dict]] = {}

    with open(pdb_path) as f:
        for line in f:
            if not line.startswith(("ATOM", "HETATM")):
                continue

            # Parse PDB ATOM record format
            atom_name = line[12:16].strip()
            res_name = line[17:20].strip()
            chain_id = line[21].strip() or "A"
            res_id = int(line[22:26].strip())
            # Column 27 is the insertion code. Residues 1 / 1A / 1B are DISTINCT residues
            # that share a sequence number; reading res_id alone merges their atoms into one
            # chimera (e.g. ALA carrying ASP's carboxylate and CYS's sulfur) and silently
            # shortens the chain. Affects 4.6% of the training corpus and 28/100 of test100.
            icode = line[26] if len(line) > 26 else " "
            x = float(line[30:38].strip())
            y = float(line[38:46].strip())
            z = float(line[46:54].strip())
            # Prefer the explicit element column (77-78); fall back to name inference
            # when it's blank or the line is too short (a real NCAA-ingest footgun).
            element_col = line[76:78].strip() if len(line) > 76 else ""
            element = element_col if element_col else infer_element_from_atom_name(atom_name)

            if chain_id not in chains:
                chains[chain_id] = []

            chains[chain_id].append(
                {
                    "atom_name": atom_name,
                    "res_name": res_name,
                    "res_id": res_id,
                    "icode": icode,
                    "x": x,
                    "y": y,
                    "z": z,
                    "element": element,
                }
            )

    return chains


# peptideMPNN-style peptide-complex filename suffixes (4th underscore field). These
# distinguish the designed-peptide manifest format from monomer mutation files such as
# ``1stn_A37_LEU_to_G35.pdb`` (whose 4th field is ``to``).
_CANONICAL_AA3 = frozenset(
    (
        "ALA",
        "ARG",
        "ASN",
        "ASP",
        "CYS",
        "GLN",
        "GLU",
        "GLY",
        "HIS",
        "ILE",
        "LEU",
        "LYS",
        "MET",
        "PHE",
        "PRO",
        "SER",
        "THR",
        "TRP",
        "TYR",
        "VAL",
    )
)

_PEPTIDE_FILENAME_SUFFIXES = (
    "dpbpeptides",
    "pepbdbpeptides",
    "pepgladpeptides",
    "stickystackpeptide",
    # SAbDab antibody CDR pseudo-peptides (scripts/joint_diffusion/prepare_sabdab_cdr_peptides.py):
    # AB_<PDBID>_A<imgtResnum>_sabdabcdr_<srcAbChain>.pdb -- chain A is the designed CDR loop.
    # Without this the parser returns None and the loader falls back to "shortest chain = peptide",
    # which only picks A by luck (loop 8-13 res vs cropped target 54-114) -- exactly the guessing
    # that silently inverted/dropped this dataset before preprocessing.
    "sabdabcdr",
    # Synthetic-dimer + native-gold-dimer NCAA pseudo-peptides
    # (scripts/joint_diffusion/convert_dimer_cifs.py): chain A = the NCAA-bearing peptide (binder),
    # chain B = the merged receptor (target). Same "chain A is the designed peptide" convention.
    "sabdimer",
    # Native-gold dimer (scripts/joint_diffusion/convert_gold_dimer.py): real crystallographic
    # peptide:protein, chain A = peptide (joint-K, no anchor), chain B = merged receptor.
    "golddimer",
)


def peptide_chain_from_filename(pdb_path: "str | Path") -> str | None:
    """
    Recover the designed peptide chain id encoded in a peptideMPNN-style filename.

    peptide-complex PDBs are named ``<chains>_<PDBID>_<chain><resnum>_<suffix>.pdb``,
    e.g. ``BA_1B2B_A30_pepbdbpeptides.pdb`` (peptide chain ``A``),
    ``2M_5XJL_M17_stickystackpeptide.pdb`` (peptide chain ``M``),
    ``1n_6EF3_n15_pepgladpeptides.pdb`` (peptide chain ``n``). The peptide chain id is the
    leading non-digit prefix of the THIRD underscore field -- always a single character
    (a PDB chain id) -- and is the canonical, reliable token (the first field is a
    concatenation of all chains present, e.g. ``BA``/``AB``, so it is NOT used).

    Returns ``None`` for anything that does not match this format (monomer mutation files
    like ``1ads_A15_LEU_to_B42.pdb`` or generic names), so non-peptide datasets fall back
    to the shortest-chain heuristic unchanged.

    Parameters
    ----------
    pdb_path : str or Path
        Path to the PDB file (only the basename stem is inspected).

    Returns
    -------
    str or None
        The single-character peptide chain id, or ``None`` if the filename does not match
        the peptideMPNN peptide-complex format.
    """
    stem = Path(pdb_path).stem
    fields = stem.split("_")
    # Need at least: <chains>_<PDBID>_<chain+resnum>_<suffix...>
    if len(fields) < 4:
        return None
    # The 4th field onward is the dataset suffix; require a known peptide-complex suffix so
    # monomer files (4th field == "to") and generic names are rejected.
    if fields[3] not in _PEPTIDE_FILENAME_SUFFIXES:
        return None
    third = fields[2]
    # Leading non-digit prefix of the 3rd field is the chain id; it must be exactly one
    # character (a real PDB chain id). Anything else (e.g. a 3-letter residue name) is rejected.
    prefix = ""
    for ch in third:
        if ch.isdigit():
            break
        prefix += ch
    if len(prefix) != 1:
        return None
    return prefix


def get_peptide_chain(
    chains: dict[str, list[dict]],
    preferred_chain: str | None = None,
) -> tuple[str, list[dict]]:
    """
    Identify the peptide chain from a PDB.

    Parameters
    ----------
    chains : dict[str, list[dict]]
        Parsed chains from PDB.
    preferred_chain : str or None, optional
        If provided and present in ``chains``, this chain is used as the peptide (e.g. the
        chain id recovered from a peptideMPNN-style filename). Otherwise the peptide falls
        back to the shortest chain.

    Returns
    -------
    chain_id : str
        Chain ID of the peptide.
    atoms : list[dict]
        Atom records for the peptide chain.
    """
    if preferred_chain is not None and preferred_chain in chains:
        return preferred_chain, chains[preferred_chain]

    # Count unique residues per chain
    chain_lengths = {}
    for chain_id, atoms in chains.items():
        residue_ids = {(a["res_id"], a.get("icode", " ")) for a in atoms}
        chain_lengths[chain_id] = len(residue_ids)

    # Peptide is the shorter chain
    peptide_chain = min(chain_lengths, key=chain_lengths.get)
    return peptide_chain, chains[peptide_chain]


def get_peptide_and_target_chains(
    chains: dict[str, list[dict]],
    preferred_chain: str | None = None,
) -> tuple[tuple[str, list[dict]], tuple[str, list[dict]]]:
    """
    Identify peptide and target chains from a PDB.

    Parameters
    ----------
    chains : dict[str, list[dict]]
        Parsed chains from PDB.
    preferred_chain : str or None, optional
        If provided and present in ``chains``, this chain is used as the peptide (e.g. the
        chain id recovered from a peptideMPNN-style filename). Otherwise the peptide falls
        back to the shortest chain. The target is ALWAYS every remaining (non-peptide) chain,
        with all their atoms concatenated into one target.

    Returns
    -------
    peptide : tuple[str, list[dict]]
        (chain_id, atoms) for the peptide chain.
    target : tuple[str, list[dict]]
        (chain_id, atoms) for the COMBINED target -- atoms from all non-peptide chains, each
        atom tagged with its original ``chain_id``. The returned chain id is a combined label
        ("B", or "B+C" for a multi-chain target), used only for display / metadata.

    Raises
    ------
    ValueError
        If fewer than 2 chains are present.
    """
    if len(chains) < 2:
        raise ValueError("Need at least 2 chains for peptide-target complex")

    # Count unique residues per chain
    chain_lengths = {}
    for chain_id, atoms in chains.items():
        residue_ids = {(a["res_id"], a.get("icode", " ")) for a in atoms}
        chain_lengths[chain_id] = len(residue_ids)

    if preferred_chain is not None and preferred_chain in chains:
        peptide_chain = preferred_chain
    else:
        # Sort chains by length; peptide is the shortest.
        sorted_chains = sorted(chain_lengths.keys(), key=lambda c: chain_lengths[c])
        peptide_chain = sorted_chains[0]

    # Target = ALL chains except the peptide, atoms concatenated into a single target.
    # A single-chain complex (one non-peptide chain) reduces to the old behaviour exactly.
    # Overlapping residue numbering across chains (chain B res 1 AND chain C res 1) would
    # otherwise MERGE into one chimeric residue in extract_residue_atoms, which groups on
    # (res_id, icode); tag each atom with its ORIGINAL chain id so the grouping key can keep
    # cross-chain residues distinct (and so any downstream consumer keeps per-atom provenance).
    target_chain_ids = sorted(c for c in chain_lengths if c != peptide_chain)
    target_atoms: list[dict] = []
    for cid in target_chain_ids:
        for atom in chains[cid]:
            target_atoms.append({**atom, "chain_id": cid})
    # combined label only (never used to index ``chains`` downstream -- verified callers treat
    # target_chain_id as a display / metadata string): "B", or "B+C" for a multi-chain target.
    combined_target_id = "+".join(target_chain_ids)

    return (peptide_chain, chains[peptide_chain]), (combined_target_id, target_atoms)


def extract_residue_atoms(
    atoms: list[dict],
    use_canonical_ordering: bool = True,
) -> list[dict]:
    """
    Group atoms by residue and separate backbone from side-chain.

    Parameters
    ----------
    atoms : list[dict]
        Atom records for a chain.
    use_canonical_ordering : bool
        If True, order sidechain atoms according to SIDECHAIN_ATOM_ORDER
        for standard amino acids. This ensures consistent ordering with
        reference structures.

    Returns
    -------
    residues : list[dict]
        List of residue dictionaries, each containing:
        - res_name: str
        - res_id: int
        - backbone: dict[str, tuple[float, float, float]] - backbone atom coords
        - sidechain: list[tuple[str, float, float, float]] - side-chain atoms (canonically ordered)
    """
    # Group atoms by the FULL residue identifier: (res_id, insertion code). Grouping on
    # res_id alone merges 1 / 1A / 1B into a single chimeric residue.

    # Order is FILE ORDER, deliberately not sorted. These insertion codes are N-terminal
    # extensions numbered backwards -- the chain runs 1B -> 1A -> 1 -> 2, verified by
    # C->N peptide-bond distances of 1.30-1.32 A. Sorting by (res_id, icode) would yield
    # 1, 1A, 1B and silently REVERSE the first residues of every affected peptide. PDB files
    # list residues along the chain, so file order is the chain order; for files without
    # insertion codes this is identical to the previous sorted behaviour.
    # Key on (chain_id, res_id, icode). A combined multi-chain target (BUG-2 fix in
    # get_peptide_and_target_chains) concatenates atoms from several chains that can reuse the
    # same residue numbering; without the chain component chain-B res 1 and chain-C res 1 would
    # merge into one chimeric residue. Single-chain inputs (binder, or an untagged legacy target)
    # carry no chain_id -> the component is a constant None, so behaviour is byte-identical.
    residue_atoms: dict[tuple, list[dict]] = {}
    residue_names: dict[tuple, str] = {}
    residue_order: list[tuple] = []

    for atom in atoms:
        key = (atom.get("chain_id"), atom["res_id"], atom.get("icode", " "))
        if key not in residue_atoms:
            residue_atoms[key] = []
            residue_names[key] = atom["res_name"]
            residue_order.append(key)
        residue_atoms[key].append(atom)

    # Loud on collision: a merged residue looks plausible and raises nothing, which is why
    # this went unnoticed. Report once per chain.
    _collided = len(residue_atoms) - len({(k[0], k[1]) for k in residue_atoms})
    if _collided:
        _ICODE_LOG.warning(
            "residues_from_atoms: %d residue(s) share a sequence number and differ only by "
            "insertion code; keeping them distinct (grouping on res_id alone would merge "
            "their atoms into one chimeric residue)",
            _collided,
        )

    # Process each residue
    residues = []
    for key in residue_order:
        res_id = key[1]
        res_name = residue_names[key]
        atoms_list = residue_atoms[key]

        # Separate backbone and side-chain
        backbone = {}
        sidechain_dict = {}  # Use dict for canonical ordering

        for atom in atoms_list:
            name = atom["atom_name"]
            coords = (atom["x"], atom["y"], atom["z"])

            if name in BACKBONE_ATOMS:
                backbone[name] = coords
            elif atom["element"] != "H":  # Exclude hydrogens from side-chain
                # 2026-06-05: carry the real element symbol through so
                # two-char halogens (CL/BR) and Boron survive. Without it the downstream
                # fallback derives element from name[0] (CL1->C, BR1->B), corrupting them.
                sidechain_dict[name] = (*coords, atom["element"])

        # Order sidechain atoms canonically. Each sidechain_dict value is (x, y, z, element).
        if use_canonical_ordering and res_name in SIDECHAIN_ATOM_ORDER:
            canonical_order = SIDECHAIN_ATOM_ORDER[res_name]
            # First add atoms in canonical order
            sidechain = [
                (atom_name, *sidechain_dict[atom_name]) for atom_name in canonical_order if atom_name in sidechain_dict
            ]
            # Then add any remaining atoms not in canonical order (e.g., modified residues)
            sidechain.extend(
                (atom_name, *vals) for atom_name, vals in sidechain_dict.items() if atom_name not in canonical_order
            )
        else:
            # Non-standard residue: keep original order
            sidechain = [(name, *vals) for name, vals in sidechain_dict.items()]

        residues.append(
            {
                "res_name": res_name,
                "res_id": res_id,
                "backbone": backbone,
                "sidechain": sidechain,
            }
        )

    return residues


# Element symbol to type index mapping
# PAD=0 encodes "no atom" (replaces separate mask diffusion track)
# C=1, N=2, O=3, S=4 (shifted +1 vs legacy to make room for PAD)
# 2026-06-05: extended 5->12 to match diffusion.py. Order
# (..., Br, I, Se, B) matches the existing v1 rotamer-DB encoding after the model's
# -1 shift, so no DB rebuild is needed. Keys are uppercase; two-char symbols
# (CL, BR, SE) require uppercased lookup.
# Vocab follows the ATOMWEAVER_ELEMENT_VOCAB knob (see diffusion.py), in lockstep with
# diffusion.NUM_ELEMENT_TYPES. 5-vocab "X" scheme: C/N/O explicit (1/2/3) and EVERY other heavy
# atom (S, P, halogens, Se, B, ...) collapses to the single 4th class id=4 = "X" (= non-CNO), NOT
# dropped to PAD. This keeps train-time element GT consistent with the vocab5-X disc DBs (which
# clamp non-CNO -> X), so NCAA exotic atoms (e.g. SEP phosphate, halogens) are learned as X rather
# than predicted as PAD and then mismatching every X reference. 12-vocab adds explicit P/F/Cl/Br/I/Se/B.
ELEMENT_TO_TYPE = {"C": 1, "N": 2, "O": 3, "S": 4}
if NUM_ELEMENT_TYPES >= 12:  # follows diffusion.NUM_ELEMENT_TYPES (validated 5 or 12) -- no separate env read
    ELEMENT_TO_TYPE.update({"P": 5, "F": 6, "CL": 7, "BR": 8, "I": 9, "SE": 10, "B": 11})
    _NONCNO_ELEMENT_DEFAULT = 0  # vocab12: present atom of an element absent from the table -> PAD (rare)
else:
    _NONCNO_ELEMENT_DEFAULT = 4  # vocab5-X: any non-CNO present heavy atom -> X (id 4), not PAD


# ---------------------------------------------------------------------------
# Reserved-slot atom->slot assignment (14-slot semantics).
#
# The 14 per-residue slots get a fixed semantic meaning so a downstream peptide
# phase can attach chemistry to specific slots:
# slot 0 = "N-connecting" atom (the atom BONDED TO backbone N: proline Cdelta /
# N-methyl / N-acyl carbon).
# slot 1 = Cbeta anchor (the closest side-chain atom to CA).
# slots 2-13 = the rest by ascending radial distance; side-chain body capped at 13.
#
# For the SMALLMOL (crossdocked ligand / pseudo-backbone) pretrain there is no real
# N-connecting atom, so slot 0 is PRIMED GEOMETRICALLY off the PSEUDO-N that
# ``_maybe_add_pseudo_backbone`` synthesizes (placed at CA - 1.47*fwd, behind CA and
# opposite the side-chain centroid). Slot 0 is filled only when the ligand happens to
# have an atom sitting at the N-connecting BOND distance d0 from that pseudo-N -- i.e.
# an atom that genuinely occupies the position where an N-substituent would bond to N.
# Otherwise slot 0 is left ghost/PAD. This mirrors the peptide phase exactly (slot 0 =
# "atom bonded to N") and reuses the pseudo-backbone we already build.
#
# d0 measured from the 6way training corpora (ncaa monomers + peptidemPNN + antibody
# sabdab; ATOMWEAVER_ELEMENT_VOCAB=5, repo parsing) as the distance of each N-connecting
# atom to its OWN backbone N, n = 401,355 (proline Cdelta dominant):
# d0 (N-connecting -> N bond length) = mean 1.469 A, std 0.035 A (pooled).
# d1 (Cbeta radial dist from CA) = mean 1.543 A (informational; slot 1 is always
# the closest atom, so no threshold is applied).
# Slot 0 is primed iff min over ligand atoms of |dist(atom, pseudo-N) - RESERVED_SLOT0_D0|
# <= RESERVED_SLOT0_TOL_N_STD * RESERVED_SLOT0_D0_STD (the nearest-to-d0 atom wins).
# Because the pseudo-N sits AWAY from the side-chain mass, real crossdocked ligands rarely
# have an atom there: with the default 1-std tolerance the fill rate is ~0.2% -- naturally
# occasional and comfortably under the >=5% ceiling set (even a wide +/-0.20 A band
# only reaches ~0.8%, so no tolerance-tightening is needed). The band is exposed as module
# constants so the rate can be retuned without touching the assignment code.
RESERVED_SLOT0_D1_CB = 1.543  # pseudo-Cbeta anchor radial distance from CA (informational)
RESERVED_SLOT0_D0 = 1.469  # N-connecting atom -> backbone-N BOND distance (mean)
RESERVED_SLOT0_D0_STD = 0.035  # its std (pooled proline Cdelta + N-Me/N-acyl carbons)
RESERVED_SLOT0_TOL_N_STD = 1.0  # acceptance half-width in units of RESERVED_SLOT0_D0_STD
# A true N-substituent (N-methyl / N-acyl / proline Cdelta) is bonded to the backbone N but NOT to CA,
# so it lies well beyond a CA covalent bond (~1.5 A) from CA -- proline Cdelta / N-Me carbons sit
# ~2.4 A from CA. A side-chain atom WITHIN this distance of CA is a genuine Cbeta (the anchor) and is
# ineligible for the reserved slot 0. This guard is what keeps a beta-amino-acid Cbeta -- which is
# bonded to BOTH N (~1.47 A) and CA (~1.54 A) -- from being pulled into slot 0.
RESERVED_SLOT0_CA_BOND_MAX = 1.9  # a side-chain atom within this of CA is a Cbeta, not an N-substituent
RESERVED_SLOT0_MAX_BODY = 13  # side-chain body cap: slots 1..13 (slot 0 is reserved)
# Placeholder occupying slot 0 when it is ghost. name=None is the sentinel; residues_to_tensors
# skips it (leaving the slot PAD) and _maybe_add_pseudo_backbone excludes it from frame geometry.
_RESERVED_GHOST_SLOT: tuple = (None, 0.0, 0.0, 0.0, None)


def slot_fill_rate_consistency_check(
    slot_fill_rate: "torch.Tensor | None",
    reserved_slot0: bool,
    *,
    margin: float = 0.2,
) -> bool:
    """Cross-check a model's per-slot fill-rate buffer against a resolved ``reserved_slot0``.

    The shell buffers are the AUTHORITATIVE record of the slot layout a checkpoint was trained
    under -- the stored hparam can be wrong (e.g. the Ray name-bridge may record a value that does
    not match what actually ran). This guard reads the distinguishing signature off the model's
    ``_slot_fill_rate`` buffer (shape ``(max_sc,)``):

      * ``reserved_slot0=True`` -> slot 0 is the N-connecting atom (rare) so it is MOSTLY GHOST,
        while slot 1 is the Cbeta anchor (near-always filled): ``fill_rate[0] << fill_rate[1]``.
      * ``reserved_slot0=False`` -> slot 0 IS the Cbeta anchor (near-always filled), so
        ``fill_rate[0]`` is high.

    Best-effort: if the buffer is missing or unpopulated (``sum <= 0``) the check is SKIPPED with a
    warning (older checkpoints may never have populated it). When populated, a buffer signature
    that CLEARLY contradicts ``reserved_slot0`` raises ``ValueError``.

    Parameters
    ----------
    slot_fill_rate : torch.Tensor or None
        The model's ``_slot_fill_rate`` buffer (or ``None`` if absent).
    reserved_slot0 : bool
        The resolved slot layout to validate against.
    margin : float, optional
        Minimum ``fill_rate[1] - fill_rate[0]`` gap required to declare a ``reserved_slot0=False``
        contradiction (avoids false errors on near-equal / noisy buffers). Default 0.2.

    Returns
    -------
    bool
        ``True`` if the buffer was populated and the check ran (consistent); ``False`` if skipped.

    Raises
    ------
    ValueError
        If the populated buffer's signature clearly contradicts ``reserved_slot0``.
    """
    if slot_fill_rate is None:
        warnings.warn(
            "reserved_slot0 buffer cross-check skipped: model has no _slot_fill_rate buffer.",
            stacklevel=2,
        )
        return False
    fr = slot_fill_rate.detach().float().reshape(-1).cpu()
    if fr.numel() < 2 or float(fr.sum()) <= 0.0:
        warnings.warn(
            "reserved_slot0 buffer cross-check skipped: _slot_fill_rate is unpopulated "
            f"(numel={fr.numel()}, sum={float(fr.sum()):.4g}).",
            stacklevel=2,
        )
        return False
    s0, s1 = float(fr[0]), float(fr[1])
    if reserved_slot0:
        # Expect slot0 low, slot1 high. Contradiction = slot0 filled like a Cbeta (high AND >= slot1).
        if s0 > 0.5 and s0 >= s1:
            raise ValueError(
                "reserved_slot0=True contradicts the checkpoint's _slot_fill_rate buffer: "
                f"slot0 fill={s0:.3f} >= slot1 fill={s1:.3f} (slot0 is filled like the Cbeta anchor, "
                "i.e. a reserved_slot0=False layout). Load with --no-reserved-slot0, or verify the "
                "checkpoint's slot layout."
            )
    else:
        # Expect slot0 high (Cbeta). Contradiction = slot0 clearly low while slot1 is high (reserved sig).
        if s0 < 0.5 and (s1 - s0) > margin:
            raise ValueError(
                "reserved_slot0=False contradicts the checkpoint's _slot_fill_rate buffer: "
                f"slot0 fill={s0:.3f} << slot1 fill={s1:.3f} (slot0 is mostly ghost, i.e. a "
                "reserved_slot0=True layout). Load with --reserved-slot0, or verify the checkpoint's "
                "slot layout."
            )
    return True


def resolve_reserved_slot0(
    stored: bool | None,
    cli_value: bool | None,
    *,
    slot_fill_rate: "torch.Tensor | None" = None,
) -> bool:
    """Resolve the effective ``reserved_slot0`` for an eval/resume load.

    The value STORED in the checkpoint hparams is authoritative-by-default so the slot layout is
    auto-detected instead of relying on a manually passed ``--reserved-slot0`` flag:

      * stored present, explicit ``cli_value`` given and DISAGREES -> raise (catch the mismatch).
      * stored present -> use stored.
      * stored absent (older ckpt) -> warn + fall back to the CLI
        flag (``cli_value``; ``None`` -> ``False``, the historical default).

    Whatever value is resolved is then cross-checked against ``slot_fill_rate`` (the authoritative
    shell buffer) via :func:`slot_fill_rate_consistency_check`, which raises on a clear contradiction.

    Parameters
    ----------
    stored : bool or None
        ``ckpt["hyper_parameters"].get("reserved_slot0")`` (``None`` if the ckpt does not store it).
    cli_value : bool or None
        The explicit CLI flag (``--reserved-slot0`` / ``--no-reserved-slot0``); ``None`` = not passed.
    slot_fill_rate : torch.Tensor or None, optional
        The model's ``_slot_fill_rate`` buffer for the authoritative cross-check.

    Returns
    -------
    bool
        The resolved ``reserved_slot0``.

    Raises
    ------
    ValueError
        If ``cli_value`` disagrees with a stored value, or the buffer cross-check contradicts.
    """
    if stored is not None:
        if cli_value is not None and bool(cli_value) != bool(stored):
            raise ValueError(
                "reserved_slot0 mismatch: checkpoint stored reserved_slot0="
                f"{bool(stored)} but the command line explicitly passed reserved_slot0={bool(cli_value)}. "
                "Drop the explicit --reserved-slot0/--no-reserved-slot0 flag (the checkpoint value is used "
                "automatically) or correct it to match the checkpoint."
            )
        resolved = bool(stored)
    else:
        resolved = bool(cli_value) if cli_value is not None else False
        warnings.warn(
            "checkpoint has no stored reserved_slot0 hparam (older checkpoint); falling back to the CLI "
            f"flag -> reserved_slot0={resolved}. Pass --reserved-slot0/--no-reserved-slot0 to match the "
            "checkpoint's training-time slot layout.",
            stacklevel=2,
        )
    slot_fill_rate_consistency_check(slot_fill_rate, resolved)
    return resolved


def _apply_reserved_slot0(res: dict) -> None:
    """Reorder a residue's side chain into the reserved-slot layout (both data paths).

    Shared by the smallmol/pseudo_backbone path and the real-residue peptide/NCAA path -- the
    slot semantics are identical on both, so both callers use this one function. It reads the
    backbone-N from ``res["backbone"]["N"]`` to pick slot 0: on the smallmol path that is the
    pseudo-N synthesized by ``_maybe_add_pseudo_backbone`` (so this MUST run after it there); on
    the real-residue path that is the REAL backbone-N already parsed onto the residue. When no N
    is present (pseudo-backbone not added, or a backbone missing N), slot 0 is left ghost/PAD.

    Mutates ``res["sidechain"]`` in place so that, after ``residues_to_tensors`` enumerates it
    slot-by-slot:
      * slot 1 = the closest atom to pseudo-CA (the pseudo-Cbeta anchor). PAD only when the sole
        side-chain atom is itself the N-substituent promoted to slot 0 (e.g. sarcosine).
      * slot 0 = the atom bonded to the pseudo-N -- the non-anchor atom whose distance to the pseudo-N
        is within ``RESERVED_SLOT0_TOL_N_STD * RESERVED_SLOT0_D0_STD`` of the N-connecting bond length
        ``RESERVED_SLOT0_D0`` and nearest to it; otherwise a ghost placeholder (leaving slot 0 PAD).
        No pseudo-N (pseudo-backbone not added) => ghost. Special case for a Cbeta-LESS N-substituent
        (sarcosine, N-substituted glycines): if the closest-to-CA atom picked as the anchor is itself a
        valid N-substituent (bonded to N, NOT bonded to CA) and slot 0 is still empty, it is promoted
        into slot 0 and the anchor is re-picked from the remaining atoms. A genuine Cbeta (bonded to CA)
        is never promoted, so beta-amino acids stay byte-identical.
      * slots 2..13 = the remaining atoms by ascending radial distance from pseudo-CA,
      * the side-chain body (slots 1..13) is capped at ``RESERVED_SLOT0_MAX_BODY`` atoms,
        dropping the most-distal on overflow (mirrors the ``[:max_sidechain_atoms]`` cap).

    No-op when the residue has no CA or no side-chain atoms.
    """
    import numpy as np

    bb = res["backbone"]
    if "CA" not in bb:
        return
    sc = res["sidechain"]
    if len(sc) == 0:
        return

    ca = np.array(bb["CA"], dtype=np.float64)
    xyz = np.array([atom[1:4] for atom in sc], dtype=np.float64)  # (N, 3); atom = (name, x, y, z[, elem])
    radii = np.linalg.norm(xyz - ca, axis=1)  # radial distance from pseudo-CA
    order = [int(i) for i in np.argsort(radii, kind="stable")]  # ascending radius, stable
    anchor_idx = order[0]  # provisional slot 1 = pseudo-Cbeta anchor (closest to CA)

    # slot 0 = the non-anchor atom bonded to the pseudo-N (nearest to the N-connecting bond length).
    slot0_idx = None
    d_to_n = None
    tol = RESERVED_SLOT0_TOL_N_STD * RESERVED_SLOT0_D0_STD
    if "N" in bb:
        n_xyz = np.array(bb["N"], dtype=np.float64)
        d_to_n = np.linalg.norm(xyz - n_xyz, axis=1)
        best_score = None
        for k in range(len(sc)):
            if k == anchor_idx:
                continue
            score = abs(float(d_to_n[k]) - RESERVED_SLOT0_D0)
            if score <= tol and (best_score is None or score < best_score):
                best_score = score
                slot0_idx = k

    # Cbeta-LESS N-substituent fix (e.g. sarcosine = N-methyl-glycine; N-(cyclopropylmethyl)-glycine):
    # with no real Cbeta, the sole N-substituent is the closest atom to CA, so it was grabbed as the
    # anchor above and slot 0 left empty -- making the residue read like its unsubstituted parent
    # (SAR ~ ALA). If the provisional anchor is ITSELF a valid N-substituent -- bonded to the backbone
    # N (within the RESERVED_SLOT0_D0 window) and NOT bonded to CA (radial distance beyond a CA covalent
    # bond) -- and slot 0 is still empty, promote the anchor into slot 0 and re-pick the anchor from the
    # remaining atoms. The ``radii >= RESERVED_SLOT0_CA_BOND_MAX`` guard means a genuine Cbeta (bonded to
    # CA) is never promoted, so beta-amino acids -- whose Cbeta is bonded to BOTH N and CA -- are left
    # byte-identical; only true Cbeta-less N-substituted residues change.
    if (
        slot0_idx is None
        and d_to_n is not None
        and float(radii[anchor_idx]) >= RESERVED_SLOT0_CA_BOND_MAX
        and abs(float(d_to_n[anchor_idx]) - RESERVED_SLOT0_D0) <= tol
    ):
        slot0_idx = anchor_idx
        anchor_idx = next((i for i in order if i != slot0_idx), None)

    used = {i for i in (slot0_idx, anchor_idx) if i is not None}
    # slots 1..13: anchor first, then remaining atoms (excl slot 0) by ascending radius; capped at 13.
    body_idx = ([anchor_idx] if anchor_idx is not None else []) + [i for i in order if i not in used]
    body_atoms = [sc[i] for i in body_idx][:RESERVED_SLOT0_MAX_BODY]
    slot0_atom = sc[slot0_idx] if slot0_idx is not None else _RESERVED_GHOST_SLOT
    res["sidechain"] = [slot0_atom, *body_atoms]


def _maybe_add_pseudo_backbone(res: dict, reserved_slot0: bool = False) -> bool:
    """If a residue has only CA in backbone, add deterministic pseudo-N/C/O.

    Constructs a local coordinate frame from pseudo-CA + sidechain atom geometry,
    then places synthetic N, C, O at standard peptide bond lengths. This gives
    BackboneEncoder a meaningful 4-point frame instead of a degenerate CA-only input.
    Used for smallmol (crossdocked) pretrain where each molecule is a single pseudo-CA
    residue with no real backbone. Frame is determined by the molecule's own geometry
    (equivariant). Returns True if a pseudo-backbone was added.

    Ported from mainline atomweaver (smallmol pretrain support).

    When ``reserved_slot0`` is set (smallmol reserved-slot layout), this runs BEFORE
    ``_apply_reserved_slot0`` on the ORIGINAL (unordered) side chain, and the secondary
    axis is built from the pseudo-Cbeta anchor (the closest atom to CA -- the eventual
    slot-1 atom) rather than the arbitrary first atom. The synthesized pseudo-N is then
    what ``_apply_reserved_slot0`` uses to pick slot 0. Ghost placeholders (``name is
    None``), if ever present, are excluded from the frame geometry.
    """
    import numpy as np

    bb = res["backbone"]
    sc = res["sidechain"]
    # reserved_slot0: exclude any ghost placeholder from the atom set used to build the
    # frame; the geometry is defined by the REAL side-chain atoms only.
    real_sc = [s for s in sc if s[0] is not None] if reserved_slot0 else sc
    if "N" in bb or "C" in bb or len(real_sc) < 2:
        return False
    if "CA" not in bb:
        return False

    ca = np.array(bb["CA"], dtype=np.float64)
    sc_coords = np.array([s[1:4] for s in real_sc], dtype=np.float64)  # (N, 3)

    # Primary axis: CA -> sidechain centroid
    centroid = sc_coords.mean(axis=0)
    fwd = centroid - ca
    fwd_norm = np.linalg.norm(fwd)
    if fwd_norm < 1e-6:
        return False
    fwd = fwd / fwd_norm

    # Secondary axis: orthogonal component of CA -> anchor atom.
    # reserved_slot0: the anchor is the pseudo-Cbeta = the atom CLOSEST to CA (the eventual
    # slot-1 atom), NOT the arbitrary first atom; otherwise (default) it is the first
    # side-chain atom, as before (flag-off byte-identical).
    if reserved_slot0:
        v0 = sc_coords[int(np.argmin(np.linalg.norm(sc_coords - ca, axis=1)))] - ca
    else:
        v0 = sc_coords[0] - ca
    v0_norm = np.linalg.norm(v0)
    if v0_norm < 1e-6:
        v0 = np.array([0.0, 1.0, 0.0])
    else:
        v0 = v0 / v0_norm

    perp = v0 - np.dot(v0, fwd) * fwd
    perp_norm = np.linalg.norm(perp)
    if perp_norm < 1e-6:
        # fwd and v0 parallel -- pick arbitrary orthogonal
        ref = np.array([1.0, 0.0, 0.0]) if abs(fwd[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        perp = np.cross(fwd, ref)
        perp = perp / np.linalg.norm(perp)
    else:
        perp = perp / perp_norm

    third = np.cross(fwd, perp)

    # Place pseudo-backbone at standard bond lengths (N behind CA, C perpendicular, O off C).
    bb["N"] = tuple((ca - 1.47 * fwd).tolist())
    bb["C"] = tuple((ca + 1.52 * perp).tolist())
    bb["O"] = tuple((ca + 1.52 * perp + 1.24 * third).tolist())
    return True


def residues_to_tensors(
    residues: list[dict],
    max_sidechain_atoms: int = 14,
) -> dict[str, torch.Tensor]:
    """
    Convert residue data to tensors for the model.

    Parameters
    ----------
    residues : list[dict]
        Residue data from extract_residue_atoms.
    max_sidechain_atoms : int
        Maximum number of side-chain atoms to pad to.

    Returns
    -------
    data : dict[str, torch.Tensor]
        Dictionary containing:
        - backbone_coords: (L, 4, 3) - N, CA, C, O coordinates
        - backbone_mask: (L, 4) - which backbone atoms are present
        - sidechain_coords: (L, max_sc, 3) - side-chain coordinates
        - sidechain_mask: (L, max_sc) - which side-chain atoms are valid
        - sidechain_element_types: (L, max_sc) - element type indices for sidechain atoms
        - res_names: list[str] - residue names
    """
    num_residues = len(residues)

    backbone_coords = torch.zeros(num_residues, 4, 3)
    backbone_mask = torch.zeros(num_residues, 4, dtype=torch.bool)
    sidechain_coords = torch.zeros(num_residues, max_sidechain_atoms, 3)
    sidechain_mask = torch.zeros(num_residues, max_sidechain_atoms, dtype=torch.bool)
    sidechain_element_types = torch.full((num_residues, max_sidechain_atoms), 0, dtype=torch.long)  # 0 = PAD
    res_names = []

    for i, res in enumerate(residues):
        res_names.append(res["res_name"])

        # Backbone
        for j, atom_name in enumerate(BACKBONE_ATOMS):
            if atom_name in res["backbone"]:
                coords = res["backbone"][atom_name]
                backbone_coords[i, j] = torch.tensor(coords)
                backbone_mask[i, j] = True

        # Side-chain (up to max atoms)
        for j, atom_data in enumerate(res["sidechain"][:max_sidechain_atoms]):
            # Reserved-slot ghost placeholder (name is None): leave this slot PAD (mask False,
            # element 0) but keep its index so downstream slots stay aligned. Only ever present
            # on the smallmol reserved-slot0 path; the default path emits no such entries, so
            # this branch is inert (byte-identical output) when the flag is off.
            if atom_data[0] is None:
                continue
            # atom_data is (name, x, y, z) or (name, x, y, z, element)
            if len(atom_data) == 4:
                name, x, y, z = atom_data
                # Infer element from atom name (CL/BR/SE-aware, not just first char)
                element = infer_element_from_atom_name(name)
            else:
                name, x, y, z, element = atom_data

            sidechain_coords[i, j] = torch.tensor([x, y, z])
            sidechain_mask[i, j] = True
            # vocab5: non-CNO present heavy atom -> X(4); vocab12: unknown element -> PAD(0)
            sidechain_element_types[i, j] = ELEMENT_TO_TYPE.get(element.upper(), _NONCNO_ELEMENT_DEFAULT)

    return {
        "backbone_coords": backbone_coords,
        "backbone_mask": backbone_mask,
        "sidechain_coords": sidechain_coords,
        "sidechain_mask": sidechain_mask,
        "sidechain_element_types": sidechain_element_types,
        "res_names": res_names,
    }


def flatten_all_atoms(
    backbone_coords: torch.Tensor,
    backbone_mask: torch.Tensor,
    sidechain_coords: torch.Tensor,
    sidechain_mask: torch.Tensor,
    sidechain_element_types: torch.Tensor | None = None,
    residue_types: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """
    Flatten backbone and sidechain atoms into a single tensor with per-atom features.

    Parameters
    ----------
    backbone_coords : torch.Tensor
        Backbone coordinates of shape (L, 4, 3).
    backbone_mask : torch.Tensor
        Backbone mask of shape (L, 4).
    sidechain_coords : torch.Tensor
        Sidechain coordinates of shape (L, max_sc, 3).
    sidechain_mask : torch.Tensor
        Sidechain mask of shape (L, max_sc).
    sidechain_element_types : torch.Tensor, optional
        Element types for sidechain atoms of shape (L, max_sc). Values in [0,3] for C,N,O,S.
    residue_types : torch.Tensor, optional
        Residue type indices of shape (L,).

    Returns
    -------
    dict with:
        coords : torch.Tensor
            All atom coordinates of shape (N_atoms, 3).
        mask : torch.Tensor
            Mask of shape (N_atoms,) - all True (only valid atoms returned).
        residue_idx : torch.Tensor
            Residue index for each atom of shape (N_atoms,).
        atom_type : torch.Tensor
            Atom type: 0-3 for backbone (N,CA,C,O), 4-17 for sidechain slots 0-13.
        element_type : torch.Tensor
            Element type: 0=C, 1=N, 2=O, 3=S, -1=unknown. Shape (N_atoms,).
        residue_type : torch.Tensor
            Residue type index for each atom. Shape (N_atoms,). -1 if not provided.
        is_backbone : torch.Tensor
            Boolean mask for backbone atoms. Shape (N_atoms,).
        ca_coords : torch.Tensor
            CA coordinates of shape (L, 3) for prefiltering.
    """
    seq_len = backbone_coords.shape[0]
    max_sc = sidechain_coords.shape[1]
    device = backbone_coords.device

    # PAD-aware (v1) target-backbone element codes: real ELEMENT_TO_TYPE values so the
    # backbone shares the 1-based sidechain element vocabulary (0=PAD), instead of the
    # legacy [1,0,0,2] which put CA/C on PAD=0 and collided with sidechain codes.
    # v1 is the ONLY path on this branch (no toggle).
    bb_element_types = torch.tensor(
        [ELEMENT_TO_TYPE["N"], ELEMENT_TO_TYPE["C"], ELEMENT_TO_TYPE["C"], ELEMENT_TO_TYPE["O"]],
        dtype=torch.long,
        device=device,
    )  # N, CA, C, O

    # Collect per-atom data
    coords_list = []
    residue_idx_list = []
    atom_type_list = []
    element_type_list = []
    is_backbone_list = []

    # Backbone atoms
    bb_flat = backbone_coords.view(-1, 3)  # (L*4, 3)
    bb_mask_flat = backbone_mask.view(-1)  # (L*4,)
    bb_valid_idx = torch.where(bb_mask_flat)[0]

    if len(bb_valid_idx) > 0:
        bb_valid_coords = bb_flat[bb_valid_idx]
        coords_list.append(bb_valid_coords)

        # Residue index for each backbone atom
        bb_residue_idx = torch.arange(seq_len, device=device).unsqueeze(1).expand(-1, 4).reshape(-1)
        residue_idx_list.append(bb_residue_idx[bb_valid_idx])

        # Atom type: 0-3 for N, CA, C, O
        bb_atom_type = torch.arange(4, device=device).unsqueeze(0).expand(seq_len, -1).reshape(-1)
        atom_type_list.append(bb_atom_type[bb_valid_idx])

        # Element type for backbone
        bb_element_type = bb_element_types.unsqueeze(0).expand(seq_len, -1).reshape(-1)
        element_type_list.append(bb_element_type[bb_valid_idx])

        # Is backbone flag
        is_backbone_list.append(torch.ones(len(bb_valid_idx), dtype=torch.bool, device=device))

    # Sidechain atoms
    sc_flat = sidechain_coords.view(-1, 3)  # (L*max_sc, 3)
    sc_mask_flat = sidechain_mask.view(-1)  # (L*max_sc,)
    sc_valid_idx = torch.where(sc_mask_flat)[0]

    if len(sc_valid_idx) > 0:
        sc_valid_coords = sc_flat[sc_valid_idx]
        coords_list.append(sc_valid_coords)

        # Residue index for each sidechain atom
        sc_residue_idx = torch.arange(seq_len, device=device).unsqueeze(1).expand(-1, max_sc).reshape(-1)
        residue_idx_list.append(sc_residue_idx[sc_valid_idx])

        # Atom type: 4-17 for sidechain slots 0-13 (offset by 4 to distinguish from backbone)
        sc_atom_type = torch.arange(max_sc, device=device).unsqueeze(0).expand(seq_len, -1).reshape(-1) + 4
        atom_type_list.append(sc_atom_type[sc_valid_idx])

        # Element type for sidechains
        if sidechain_element_types is not None:
            sc_element_flat = sidechain_element_types.view(-1)
            element_type_list.append(sc_element_flat[sc_valid_idx])
        else:
            element_type_list.append(torch.full((len(sc_valid_idx),), -1, dtype=torch.long, device=device))

        # Is backbone flag
        is_backbone_list.append(torch.zeros(len(sc_valid_idx), dtype=torch.bool, device=device))

    # Concatenate all
    if coords_list:
        all_coords = torch.cat(coords_list, dim=0)
        all_residue_idx = torch.cat(residue_idx_list, dim=0)
        all_atom_type = torch.cat(atom_type_list, dim=0)
        all_element_type = torch.cat(element_type_list, dim=0)
        all_is_backbone = torch.cat(is_backbone_list, dim=0)
    else:
        all_coords = torch.zeros(0, 3, device=device)
        all_residue_idx = torch.zeros(0, dtype=torch.long, device=device)
        all_atom_type = torch.zeros(0, dtype=torch.long, device=device)
        all_element_type = torch.zeros(0, dtype=torch.long, device=device)
        all_is_backbone = torch.zeros(0, dtype=torch.bool, device=device)

    # Residue type for each atom
    if residue_types is not None:
        if len(all_residue_idx) > 0:
            all_residue_type = residue_types[all_residue_idx]
        else:
            all_residue_type = torch.zeros(0, dtype=torch.long, device=device)
    else:
        all_residue_type = torch.full((len(all_coords),), -1, dtype=torch.long, device=device)

    # Extract CA coordinates (index 1 in backbone: N, CA, C, O)
    ca_coords = backbone_coords[:, 1, :]  # (L, 3)

    return {
        "coords": all_coords,
        "mask": torch.ones(len(all_coords), dtype=torch.bool, device=device),
        "residue_idx": all_residue_idx,
        "atom_type": all_atom_type,
        "element_type": all_element_type,
        "residue_type": all_residue_type,
        "is_backbone": all_is_backbone,
        "ca_coords": ca_coords,
    }


def _geometric_k_report(r: float, lengths=(8, 12, 16)) -> str:
    """Human-readable 'what you actually get' table for a geometric K law, for error messages."""
    rows = []
    for length in lengths:
        w = [r**i for i in range(length)]
        z = sum(w)
        p1, p2, pl = w[0] / z, (w[1] / z if length > 1 else 0.0), w[-1] / z
        mean = sum((i + 1) * wi / z for i, wi in enumerate(w))
        rows.append(f"      L={length:<3d} P(K=1)={p1:.3f}  P(K=2)={p2:.3f}  P(K=L)={pl:.4f}  meanK={mean:.2f}")
    return "\n".join(rows)


def _validate_geometric_k(r: float, kl, bridge, k1) -> None:
    """Reject r together with any explicit 3-mode weight, and say what r actually gives.

    Why they are mutually exclusive rather than merely redundant: the geometric is A*r^(K-1), and A is
    NOT free -- sum-to-1 pins it. So r alone determines BOTH ends,
        P(K=1) = (1-r)/(1-r^L)   and   P(K=L) = P(K=1) * r^(L-1),
    and there is no leftover degree of freedom to hit a separately-specified K=L weight. Honouring both
    would require rescaling, which changes the very numbers that were asked for (only their RATIO
    survives). Worse, there is a bound no parameterisation escapes: for any NON-INCREASING law on
    {1..L}, P(K=L) is the minimum, and a minimum cannot exceed the mean, so P(K=L) <= 1/L (0.083 at
    L=12). A requested kl above that is unreachable, and solving for it silently yields r>1 -- a law
    that RISES toward K=L, the opposite of the intent.
    """
    if not (0.0 < r <= 1.0):
        raise ValueError(
            f"random_binder_k_geometric_r={r} must be in (0, 1]. r>1 would make P(K) INCREASE toward "
            "K=L (the opposite of a left-shifted law); r=1 is the uniform special case."
        )
    conflicting = [
        name
        for name, v in (
            ("random_binder_k_final_kl", kl),
            ("random_binder_k_final_bridge", bridge),
            ("random_binder_k_final_k1", k1),
        )
        if v is not None
    ]
    if conflicting:
        raise ValueError(
            "random_binder_k_geometric_r=%.3f cannot be combined with %s.\n"
            "  The geometric is A*r^(K-1) and A is pinned by sum-to-1, so r is the ONLY free knob:\n"
            "  it sets BOTH ends at once (P(K=1)=(1-r)/(1-r^L), P(K=L)=P(K=1)*r^(L-1)). There is no\n"
            "  degree of freedom left to also honour an explicit K=L / bridge / K=1 weight -- and any\n"
            "  non-increasing law obeys P(K=L) <= 1/L regardless (0.083 at L=12), so a larger kl is\n"
            "  unreachable at ANY r <= 1.\n"
            "  Leave those unset. With r=%.3f you actually get:\n%s\n"
            "  If you need a bigger K=L share than that, drop r and use the 3-mode weights instead --\n"
            "  accepting that the K=L bin is then a deliberate MODE, not a monotone tail."
            % (r, " + ".join(conflicting), r, _geometric_k_report(r))
        )


class BinderDataset(Dataset):
    """
    Dataset for loading binder-target complexes from PDB files.

    Extracts the binder chain (shorter chain) and target chain (longer chain)
    from each PDB. Returns binder backbone/sidechain coordinates (to be diffused)
    and target structure/sequence for conditioning.

    Parameters
    ----------
    pdb_dir : str or Path
        Directory containing PDB files.
    max_sidechain_atoms : int
        Maximum number of side-chain atoms per residue.
    max_binder_length : int, optional
        Maximum binder length. Longer binders are skipped.
    max_target_length : int, optional
        Maximum target length. Longer targets are truncated.
    file_list : list[str], optional
        Specific list of PDB filenames to use. If None, uses all .pdb files.
    max_files : int, optional
        Maximum number of PDB files to consider. Limits the initial file list
        before validation, useful for faster loading when only a subset is needed.
    min_sidechain_atoms : int, optional
        Minimum total sidechain atoms required. Binders with fewer are skipped.
        Default 1 (skip all-glycine peptides with no sidechains).
    """

    def __init__(
        self,
        pdb_dir: str | Path,
        max_sidechain_atoms: int = 14,
        max_binder_length: int | None = 32,
        max_target_length: int | None = 256,
        file_list: Sequence[str] | None = None,
        max_files: int | None = None,
        min_sidechain_atoms: int = 1,
        holdout_resnames: frozenset[str] | None = None,
        pseudo_backbone: bool = False,
        reserved_slot0: bool = False,
        random_binder_k_min: int = 1,
        random_binder_k_mix_prob: float | None = None,
        random_binder_k_ramp: bool = False,
        random_binder_k_ramp_total_epochs: int = 0,
        random_binder_k_ramp_start_frac: float = 1.0 / 3.0,
        random_binder_k_ramp_end_frac: float = 2.0 / 3.0,
        random_binder_k_mix_prob_ramp: bool = False,
        random_binder_k_mix_prob_ramp_start_frac: float = 0.5,
        random_binder_k_mix_prob_ramp_end_frac: float = 0.75,
        random_binder_k_final_mix: str | None = None,
        random_binder_k_reverse_start_frac: float = 0.4,
        random_binder_k_reverse_end_frac: float = 0.9,
        random_binder_k_reverse_intro_frac: float = 0.6,
        random_binder_k_final_kl: float | None = None,
        random_binder_k_final_bridge: float | None = None,
        random_binder_k_final_k1: float | None = None,
        random_binder_k_geometric_r: float | None = None,
        ncaa_anchor_weight: float = 1.0,
        interface_anchor_weight: float = 1.0,
        interface_only: bool = False,
        fcc_per_env: bool = False,
        proximity_target_crop: bool = True,
    ):
        self.pdb_dir = Path(pdb_dir)
        self.max_sidechain_atoms = max_sidechain_atoms
        self.max_binder_length = max_binder_length
        self.max_target_length = max_target_length
        self.min_sidechain_atoms = min_sidechain_atoms
        # NCAA holdout exclusion: any PDB whose binder (shorter chain) contains one of
        # these residue names is dropped at indexing time, so the model never sees it --
        # keeps the manuscript holdouts as genuine zero-shot (evalD), not train-seen (evalC).
        self.holdout_resnames = holdout_resnames or frozenset()
        self.pseudo_backbone = pseudo_backbone  # smallmol pretrain: synth pseudo-N/C/O for CA-only molecules
        # Reserved-slot layout: reorder the binder side chain so slot 0 is the N-connecting atom,
        # slot 1 the Cbeta anchor, slots 2..13 the rest by radius. Applies on BOTH data paths:
        # the smallmol/pseudo_backbone path (slot 0 primed off the synthesized pseudo-N) and the
        # real-residue peptide/NCAA path (slot 0 picked off the REAL backbone-N). This must match
        # the reserved-slot base checkpoint a resume runs off; default off = current behavior.
        self.reserved_slot0 = reserved_slot0
        # Binder-preserving K-masking (proper inpainting): each access designs K of the L binder
        # residues; the other L-K are GT-pinned context that STAYS binder-classed (never wrapped as
        # target). mix_prob=None disables it (design all = native all-site). When on:
        # P(K=K_max)=mix_prob (full design -- preserves 50% all-site training), else K~Uniform[k_min,K_max].
        # K_max=L per peptide, unless the optional ramp is on (bridges single-site->multi-site by
        # ramping K_max k_min->L over training; off for peptides).
        self.random_binder_k_min = random_binder_k_min
        self.random_binder_k_mix_prob = random_binder_k_mix_prob
        self.random_binder_k_ramp = random_binder_k_ramp
        self.random_binder_k_ramp_total_epochs = random_binder_k_ramp_total_epochs
        self.random_binder_k_ramp_start_frac = random_binder_k_ramp_start_frac
        self.random_binder_k_ramp_end_frac = random_binder_k_ramp_end_frac
        # mix_prob CURRICULUM: ramp the all-site probability from 1.0 (full-joint K=L only) down to
        # random_binder_k_mix_prob (the base) over [start_frac, end_frac] of training, so inpainting is
        # phased in only after the model first learns the easier full-joint task. Off => static base.
        self.random_binder_k_mix_prob_ramp = random_binder_k_mix_prob_ramp
        self.random_binder_k_mix_prob_ramp_start_frac = random_binder_k_mix_prob_ramp_start_frac
        self.random_binder_k_mix_prob_ramp_end_frac = random_binder_k_mix_prob_ramp_end_frac
        # THREE-WAY FINAL-MIX K schedule (alternative ramp TARGET; independent of the mix_prob curriculum
        # above). When "three_way", the per-access K is drawn so the RAMP TARGET is a 3-way mixture over
        # the design count K: 1/3 K=1 (single-site), 1/3 K~Uniform{2..L-1} (multi-site interior), 1/3 K=L
        # (full joint). A probability p_mix ramps 0->1 over the existing K-ramp window
        # ([random_binder_k_ramp_start_frac, random_binder_k_ramp_end_frac], horizon
        # random_binder_k_ramp_total_epochs): at the start every draw is K=1 (single-site-always); by the
        # ramp end p_mix=1 so every draw samples the full 3-way. None / "uniform" => legacy behaviour.
        # REVERSED (joint-first) FINAL-MIX K schedule (alternative ramp DIRECTION; independent of the
        # three_way schedule above). When "reversed", training STARTS at 100% K=L (full joint), holds until
        # random_binder_k_reverse_start_frac, then relaxes DOWN to a 3-mode final mix
        # (K=L / bridge K~Uniform{2..L-1} / K=1). Phase A [start, intro) grows the bridge from the top of the
        # interior (K=L-1) downward while K=L recedes; phase B [intro, end) introduces K=1 LAST as its own
        # bin -- seeded at the bridge per-member level (NOT 0) so it enters on equal footing with K=2/K=3 and
        # then climbs to its final share. The three finals (kl/bridge/k1) must be >= 0 and sum to 1.
        self.random_binder_k_reverse_start_frac = random_binder_k_reverse_start_frac
        self.random_binder_k_reverse_end_frac = random_binder_k_reverse_end_frac
        self.random_binder_k_reverse_intro_frac = random_binder_k_reverse_intro_frac
        self.random_binder_k_geometric_r = random_binder_k_geometric_r
        # Soft anchor toward the binder's NCAA site (the sole non-canonical residue). 1.0 == unbiased
        # (base behaviour, no anchor); >1 up-weights the NCAA position in the K-subset draw WITHOUT
        # changing the K budget -- same mechanism OnTheFlyNcaaDataset uses for the monomer window,
        # grafted onto the defined-pair path for the synthetic NCAA dimer (single-site focus). Gold
        # dimer / peptideMPNN / antibody leave it at 1.0 (joint design, no site bias).
        if not (ncaa_anchor_weight >= 1.0):
            raise ValueError(f"ncaa_anchor_weight must be >= 1.0 (1.0 == unbiased; got {ncaa_anchor_weight})")
        self.ncaa_anchor_weight = float(ncaa_anchor_weight)
        # Interface-anchored design masking. 1.0 == unbiased (base behaviour, no interface bias, and
        # NO per-PDB SASA cost incurred). >1 up-weights the binder's INTERFACE residues in the without-
        # replacement K-subset draw, so the design mask leans interface-heavy while the K budget is
        # preserved EXACTLY. Interface == the new two-axis classifier's is_contacting axis (any CA-inclusive
        # sidechain-role heavy atom within 4 Å of a target heavy atom AND the pseudo-Cβ ray points at the
        # target) -- surfaced as label "Interface" by the interface-first collapse of classify_bei_batch
        # (bei_def="aron_overlap", the default). Computed ONCE per PDB in the cached-load path (_load_pdb),
        # the same burial classifier.
        if not (interface_anchor_weight >= 1.0):
            raise ValueError(f"interface_anchor_weight must be >= 1.0 (1.0 == unbiased; got {interface_anchor_weight})")
        self.interface_anchor_weight = float(interface_anchor_weight)
        if self.interface_anchor_weight > 1.0:
            # Fail loud rather than silently degrade to uniform masking: interface anchoring needs the
            # SASA-based B/E/I classifier, so a missing biotite would otherwise make the flag a no-op.
            from atomweaver.joint_diffusion.bei import _HAS_BIOTITE

            if not _HAS_BIOTITE:
                raise RuntimeError(
                    "interface_anchor_weight > 1.0 requires biotite for interface (B/E/I) classification, "
                    "but biotite is unavailable -- refusing to silently fall back to uniform masking."
                )
        # Interface-ONLY design masking (hard restriction, distinct from the soft interface_anchor_weight
        # up-weighting above). When True the designable K-mask draws its sites EXCLUSIVELY from the binder's
        # INTERFACE (is_contacting) residues (same collapsed classify_bei_batch Interface set as
        # interface_anchor_weight); every non-interface residue stays GT-pinned context.
        # K is capped at the interface count.
        # The interface set is computed & cached ONCE per PDB in _load_pdb (same path as the anchor). Off
        # (default) => the normal weighted/uniform K-subset draw, byte-identical to base behaviour.
        self.interface_only = bool(interface_only)
        # Per-environment FCC slopes: cache a per-residue B/E/I env code (data["bei_env"]) so the model's
        # fill-corrected-coord loss can subtract an env-specific slope (Interface/Buried/Exposed) instead
        # of the single scalar. Only computes/caches bei_env when set; off (default) => no bei_env key,
        # model falls back to the scalar fcc_slope (byte-identical).
        self.fcc_per_env = bool(fcc_per_env)
        # Target hard-cap policy when a target exceeds max_target_length. True (default, the bug
        # fix): keep the max_target_length target residues CLOSEST (min Cα-distance) to any binder
        # Cα, so a far-numbered interface (e.g. peptide contacting target residues 300-400 of a
        # 500-residue target) always survives the cap. False (legacy / ablation): keep the first
        # max_target_length by INDEX, which silently drops any high-index interface.
        self.proximity_target_crop = bool(proximity_target_crop)
        if (self.interface_only or self.fcc_per_env) and self.interface_anchor_weight <= 1.0:
            # Fail loud on a missing biotite for the SAME reason the anchor does: both features need the
            # SASA-based B/E/I classifier, so a missing biotite would silently degrade them to no-ops
            # (interface_only -> uniform draw; fcc_per_env -> all-Exposed slopes). The anchor already
            # ran this check above when its weight > 1, so only re-check when it did not.
            from atomweaver.joint_diffusion.bei import _HAS_BIOTITE

            if not _HAS_BIOTITE:
                raise RuntimeError(
                    "interface_only / fcc_per_env require biotite for B/E/I classification, but biotite "
                    "is unavailable -- refusing to silently fall back (uniform masking / all-Exposed slopes)."
                )
        if random_binder_k_geometric_r is not None:
            _validate_geometric_k(
                random_binder_k_geometric_r,
                random_binder_k_final_kl,
                random_binder_k_final_bridge,
                random_binder_k_final_k1,
            )
        _kl = 1.0 / 3.0 if random_binder_k_final_kl is None else random_binder_k_final_kl
        self.random_binder_k_final_kl = _kl
        # None bridge/k1 split the remaining (1 - kl) mass evenly across the bridge and K=1 bins.
        self.random_binder_k_final_bridge = (
            random_binder_k_final_bridge if random_binder_k_final_bridge is not None else (1.0 - _kl) / 2.0
        )
        self.random_binder_k_final_k1 = (
            random_binder_k_final_k1 if random_binder_k_final_k1 is not None else (1.0 - _kl) / 2.0
        )
        self.random_binder_k_final_mix = random_binder_k_final_mix
        self._current_epoch = 0
        if random_binder_k_final_mix is not None and random_binder_k_final_mix not in (
            "uniform",
            "three_way",
            "reversed",
        ):
            raise ValueError(
                "random_binder_k_final_mix must be None, 'uniform', 'three_way', or 'reversed', got "
                f"{random_binder_k_final_mix!r}"
            )
        if random_binder_k_final_mix == "reversed":
            kl_f = self.random_binder_k_final_kl
            br_f = self.random_binder_k_final_bridge
            k1_f = self.random_binder_k_final_k1
            if not (kl_f >= 0.0 and br_f >= 0.0 and k1_f >= 0.0) or abs((kl_f + br_f + k1_f) - 1.0) > 1e-6:
                raise ValueError(
                    "random_binder_k_final_mix='reversed' requires non-negative finals summing to 1, got "
                    f"kl={kl_f}, bridge={br_f}, k1={k1_f} (sum={kl_f + br_f + k1_f})"
                )
            s_f = self.random_binder_k_reverse_start_frac
            e_f = self.random_binder_k_reverse_end_frac
            i_f = self.random_binder_k_reverse_intro_frac
            if not (0.0 <= s_f < e_f <= 1.0):
                raise ValueError(
                    "random_binder_k_final_mix='reversed' requires reverse fracs to satisfy "
                    f"0 <= start_frac < end_frac <= 1, got start={s_f}, end={e_f}"
                )
            if not (0.0 <= i_f <= 1.0):
                raise ValueError(
                    f"random_binder_k_final_mix='reversed' requires 0 <= intro_frac <= 1, got intro_frac={i_f}"
                )
        if random_binder_k_mix_prob is not None and not 0.0 <= random_binder_k_mix_prob <= 1.0:
            raise ValueError(f"random_binder_k_mix_prob must be in [0, 1], got {random_binder_k_mix_prob}")
        if random_binder_k_ramp and not (0.0 <= random_binder_k_ramp_start_frac < random_binder_k_ramp_end_frac <= 1.0):
            raise ValueError(
                "random_binder_k_ramp fracs must satisfy 0 <= start_frac < end_frac <= 1 "
                f"(got start={random_binder_k_ramp_start_frac}, end={random_binder_k_ramp_end_frac})"
            )
        if random_binder_k_mix_prob_ramp and not (
            0.0 <= random_binder_k_mix_prob_ramp_start_frac < random_binder_k_mix_prob_ramp_end_frac <= 1.0
        ):
            raise ValueError(
                "random_binder_k_mix_prob_ramp fracs must satisfy 0 <= start_frac < end_frac <= 1 "
                f"(got start={random_binder_k_mix_prob_ramp_start_frac}, end={random_binder_k_mix_prob_ramp_end_frac})"
            )
        if random_binder_k_mix_prob_ramp and random_binder_k_mix_prob is None:
            # The curriculum ramps the all-site prob from 1.0 down to this base; with no base, K-masking
            # is off entirely and the ramp is a silent no-op (design-all forever). Fail loud instead.
            raise ValueError(
                "random_binder_k_mix_prob_ramp=True requires random_binder_k_mix_prob (the ramp's base "
                "all-site probability) to be set, but got None -- the curriculum would be a silent no-op."
            )

        # Collect PDB files
        if file_list is not None:
            self.pdb_files = [self.pdb_dir / f for f in file_list]
        else:
            self.pdb_files = sorted(self.pdb_dir.glob("*.pdb"))

        # Structurally exclude holdout-containing PDBs (before the max_files cap so the
        # cap yields that many holdout-free samples).
        if self.holdout_resnames:
            n_before = len(self.pdb_files)
            self.pdb_files = [p for p in self.pdb_files if not self._binder_has_holdout(p)]
            n_dropped = n_before - len(self.pdb_files)
            print(
                f"[BinderDataset] holdout filter ({sorted(self.holdout_resnames)}): "
                f"dropped {n_dropped}/{n_before} PDBs containing a holdout binder residue"
            )

        # Limit file list if max_files specified (before validation scan)
        if max_files is not None and len(self.pdb_files) > max_files:
            self.pdb_files = self.pdb_files[:max_files]

        # Filter by length if needed (lazy, done on access)
        self._cached_data: dict[int, dict] = {}
        self._valid_indices: list[int] | None = None
        # Samples whose design mask the designable-site gate emptied outright (see
        # _gated_design_mask). Per instance; dataloader workers each hold their own copy.
        self._emptied_design_mask_count = 0

    def _binder_has_holdout(self, pdb_path: Path) -> bool:
        """True if the binder (shorter) chain contains any holdout residue name.

        SOURCE OF TRUTH for the holdout set:
          the residue-metadata CSV used to build the reference library
        (next to the frozen clustering files). It lists every residue with BOTH code schemes
        (``residue code`` / ``ccd_code``) and a ``holdout`` flag. The holdout NCAAs are all ``no code``
        (empty residue code), so structures name them by their ``ccd_code`` -- which is why the
        ``holdout_resnames`` passed here (and the sampler D_SET) are CCD codes, matched against
        raw residue names. If the holdout set ever changes, re-derive both lists from that CSV
        rather than editing the hardcoded string, or training-exclusion and evalD silently drift
        apart. (Verified 2026-07: the holdout NCAAs do not occur in the monomer set at all, so
        exclusion is a correct no-op there; they DO occur in gold-dimer -> routed to evalD.)
        """
        try:
            chains = parse_pdb_atoms(pdb_path)
            if not chains:
                return False
            preferred = peptide_chain_from_filename(pdb_path)
            _, binder_atoms = get_peptide_chain(chains, preferred_chain=preferred)
            return any(atom["res_name"] in self.holdout_resnames for atom in binder_atoms)
        except (ValueError, KeyError, IndexError):
            # Unparseable here -> keep it; _load_pdb will reject it at access time if invalid.
            return False

    def _parse_pdb_cached(self, idx: int) -> tuple | None:
        """Parse a PDB into ``(binder_chain_id, target_chain_id, binder_residues,
        target_residues)`` -- the INVARIANT part of ``_load_pdb`` (independent of
        K-mask / pseudo_backbone / holdout / per-access design_mask).

        Exists as an override seam: ``OnTheFlyNcaaDataset`` replaces ONLY this
        method to slice a fresh (binder window, radial-cropped target) pair from a
        single-chain source PDB, while inheriting the entire downstream tensorization
        + design_mask path unchanged. The base implementation parses both chains from
        a pre-built 2-chain PDB exactly as the legacy inline code did.

        Returns None if the PDB is unparseable or has fewer than 2 chains.
        """
        pdb_path = self.pdb_files[idx]
        try:
            chains = parse_pdb_atoms(pdb_path)
            if not chains or len(chains) < 2:
                return None
            # Get both binder (peptide) and target chains. For peptideMPNN-style filenames the
            # designed peptide chain is encoded in the filename (and may NOT be the shortest
            # chain -- e.g. the insulin A-chain leak); prefer it when present.
            preferred = peptide_chain_from_filename(pdb_path)
            (binder_chain_id, binder_atoms), (target_chain_id, target_atoms) = get_peptide_and_target_chains(
                chains, preferred_chain=preferred
            )
            binder_residues = extract_residue_atoms(binder_atoms)
            target_residues = extract_residue_atoms(target_atoms)
        except (OSError, ValueError, KeyError):
            return None
        return (binder_chain_id, target_chain_id, binder_residues, target_residues)

    def _load_pdb(self, idx: int) -> dict | None:
        """Load and process a single PDB file with both binder and target chains."""
        if idx in self._cached_data:
            return self._cached_data[idx]

        pdb_path = self.pdb_files[idx]

        try:
            parsed = self._parse_pdb_cached(idx)
            if parsed is None:
                return None
            binder_chain_id, target_chain_id, binder_residues, target_residues = parsed

            # Smallmol pretrain: crossdocked molecules are a single pseudo-CA residue with no real
            # N/C/O backbone -> synthesize a deterministic pseudo-backbone so BackboneEncoder gets a frame.
            if self.pseudo_backbone:
                for _res in binder_residues:
                    # Build the pseudo-backbone (incl. the pseudo-N) FIRST; the reserved-slot
                    # reorder then picks slot 0 as the ligand atom bonded to that pseudo-N.
                    _maybe_add_pseudo_backbone(_res, reserved_slot0=self.reserved_slot0)
                    if self.reserved_slot0:
                        _apply_reserved_slot0(_res)
            elif self.reserved_slot0:
                # Real-residue (peptide / NCAA monomer) path: apply the SAME reserved-slot layout
                # as the smallmol base, but slot 0 is picked off the REAL backbone-N already on
                # res["backbone"]["N"] -- no pseudo-backbone is synthesized. Keeps binder slot
                # semantics (slot 0 = N-connecting, slot 1 = Cbeta, radial, body cap 13) identical
                # to the reserved-slot base the checkpoint was trained under, so a resume
                # off it feeds matching slots instead of the default Cbeta-first-radial order.
                for _res in binder_residues:
                    _apply_reserved_slot0(_res)

            # Filter by binder length
            if self.max_binder_length and len(binder_residues) > self.max_binder_length:
                return None

            if len(binder_residues) == 0:
                return None

            # Cap target dimensionality if too long. Proximity crop (default) keeps the residues
            # nearest the binder so the interface always survives; the legacy index crop keeps the
            # first N and can silently drop a high-index interface (BUG-1 fix).
            if self.max_target_length and len(target_residues) > self.max_target_length:
                if self.proximity_target_crop:
                    keep_idx = _closest_target_indices(
                        _residue_ca_array(binder_residues),
                        _residue_ca_array(target_residues),
                        self.max_target_length,
                    )
                    target_residues = [target_residues[i] for i in keep_idx]
                else:
                    target_residues = target_residues[: self.max_target_length]

            # Sites whose deposited composition contradicts their label are kept as pocket
            # context but excluded from design (see designable_residue_mask).
            designable = designable_residue_mask(binder_residues)
            # Stricter gate for the discretization loss only: any deviation disqualifies the
            # residue as a matching exemplar, even the small ones design tolerates.
            exact_composition = exact_composition_mask(binder_residues)

            # Convert to tensors
            binder_data = residues_to_tensors(binder_residues, self.max_sidechain_atoms)
            target_data = residues_to_tensors(target_residues, self.max_sidechain_atoms)

            # Filter by minimum sidechain atoms (skip all-glycine peptides)
            if self.min_sidechain_atoms > 0:
                n_sidechain_atoms = binder_data["sidechain_mask"].sum().item()
                if n_sidechain_atoms < self.min_sidechain_atoms:
                    return None

            # Carried through so __getitem__ can intersect it with the sampled design mask.
            binder_data["designable_mask"] = designable
            binder_data["exact_composition_mask"] = exact_composition

            # Compute target residue type indices for detailed atom features
            from atomweaver.joint_diffusion.models import RESIDUE_TO_IDX

            target_residue_types = torch.tensor(
                [RESIDUE_TO_IDX.get(name, RESIDUE_TO_IDX["UNK"]) for name in target_data["res_names"]], dtype=torch.long
            )

            # Flatten all target atoms for EGNN graph with detailed per-atom features
            target_flat = flatten_all_atoms(
                target_data["backbone_coords"],
                target_data["backbone_mask"],
                target_data["sidechain_coords"],
                target_data["sidechain_mask"],
                sidechain_element_types=target_data["sidechain_element_types"],
                residue_types=target_residue_types,
            )

            data = {
                # Binder (peptide) data - what we're denoising
                "backbone_coords": binder_data["backbone_coords"],
                "backbone_mask": binder_data["backbone_mask"],
                "sidechain_coords": binder_data["sidechain_coords"],
                "sidechain_mask": binder_data["sidechain_mask"],
                "sidechain_element_types": binder_data["sidechain_element_types"],
                "res_names": binder_data["res_names"],
                # The two composition gates. They MUST be copied out here: every consumer
                # (__getitem__, iter_valid, the enumeration eval, collate_binders) reads them
                # off the ITEM, and `binder_data` is a local that dies with this call. They were
                # stranded on it when the gate landed, which made apply_designable_gate a
                # no-op everywhere -- `data.get("designable_mask")` was always None -- and
                # left `residue_in_disc_db` ungated. Index-aligned with `backbone_coords` by
                # construction: both are computed from `binder_residues`, the SAME list that
                # `residues_to_tensors` tensorizes, AFTER any windowing (OnTheFlyNcaaDataset
                # windows inside `_parse_pdb_cached`, upstream of this method). So they are
                # window-relative whenever the binder is, and always length L.
                "designable_mask": binder_data["designable_mask"],
                "exact_composition_mask": binder_data["exact_composition_mask"],
                # Target data - conditioning context (for cross-attention)
                "target_backbone_coords": target_data["backbone_coords"],
                "target_backbone_mask": target_data["backbone_mask"],
                "target_sidechain_coords": target_data["sidechain_coords"],
                "target_sidechain_mask": target_data["sidechain_mask"],
                "target_res_names": target_data["res_names"],
                # Target data - for EGNN graph (all atoms flattened with per-atom features)
                "target_coords": target_flat["coords"],
                "target_mask": target_flat["mask"],
                "target_ca_coords": target_flat["ca_coords"],
                "target_atom_residue_idx": target_flat["residue_idx"],
                "target_atom_type": target_flat["atom_type"],
                "target_atom_element_type": target_flat["element_type"],
                "target_atom_residue_type": target_flat["residue_type"],
                "target_atom_is_backbone": target_flat["is_backbone"],
                # Metadata
                "pdb_name": pdb_path.stem,
                "binder_chain_id": binder_chain_id,
                "target_chain_id": target_chain_id,
            }

            # NCAA anchor (opt-in via ncaa_anchor_weight > 1): stash the binder-relative index of the
            # sole non-canonical residue so the design mask biases toward it. Deterministic per
            # structure, so caching it is safe; None when the binder is all-canonical (no anchor).
            if self.ncaa_anchor_weight > 1.0:
                ncaa_pos = next(
                    (i for i, nm in enumerate(binder_data["res_names"]) if nm not in _CANONICAL_AA3),
                    None,
                )
                if ncaa_pos is not None:
                    data["_ncaa_window_pos"] = ncaa_pos

            # Interface anchor (opt-in via interface_anchor_weight > 1): compute the binder's INTERFACE
            # residues ONCE per PDB here (SASA needs the whole peptide+target system) and cache the
            # window-relative indices. These index the SAME array whose length drives _sample_design_mask
            # (data["backbone_coords"].shape[0] == binder residue count L), so the design mask and the
            # interface set share one indexing. classify_bei_batch is the exact burial classifier
            # uses (Interface == sidechain ΔSASA >= 16 Å² vs target). Deterministic per structure -> safe
            # to cache. Only paid when the anchor is on; biotite-missing / errors -> empty list (no bias).
            # interface_positions is needed for BOTH the soft anchor (weight > 1) and the hard
            # interface_only restriction. bei_env is needed for the per-env FCC slopes. All three
            # derive from the SAME classify_bei_batch pass, so run it once and split the outputs
            # (avoids a second SASA pass when e.g. interface_only + fcc_per_env are both on).
            need_iface = self.interface_anchor_weight > 1.0 or self.interface_only
            if need_iface or self.fcc_per_env:
                labels = self._classify_bei_labels(data)
                if need_iface:
                    data["interface_positions"] = [p for p, lab in labels.items() if lab == "Interface"]
                if self.fcc_per_env:
                    data["bei_env"] = self._bei_env_from_labels(labels, data["backbone_coords"].shape[0])

            self._cached_data[idx] = data
            return data

        except (OSError, ValueError, KeyError):
            return None

    def _classify_bei_labels(self, data: dict) -> dict:
        """Shared B/E/I (Buried/Exposed/Interface) classification for this windowed complex.

        Runs the SAME ``classify_bei_batch`` (default ``bei_def="aron_overlap"``: the
        two-axis burial + is_contacting classifier, collapsed interface-first so Interface == is_contacting)
        on THIS PDB's windowed binder coords + target. Returns ``{p: label}`` with p relative to
        ``data["backbone_coords"]`` (== the ``length`` passed to ``_sample_design_mask``).

        On classification error the behaviour depends on which feature needs the labels:
          * a HARD feature (``interface_only`` silently disables the restriction; ``fcc_per_env``
            silently mislabels every env) => RAISE, so a broken SASA pass can never masquerade as a
            benign all-Exposed / no-interface batch;
          * otherwise (soft ``interface_anchor_weight`` only, or no BEI feature) => log a warning and
            return ``{}`` (callers treat that as "no interface set" -- graceful degradation).
        The ``_HAS_BIOTITE`` short-circuit stays graceful here because dataset ``__init__`` already
        fails loud at construction when a BEI feature is enabled without biotite.
        """
        from atomweaver.joint_diffusion.bei import _HAS_BIOTITE, classify_bei_batch

        if not _HAS_BIOTITE:
            return {}
        try:
            return classify_bei_batch(
                data["sidechain_coords"],
                data["sidechain_mask"],
                data.get("sidechain_element_types"),
                data["backbone_coords"],
                data["backbone_mask"],
                target_coords=data.get("target_coords"),
                target_mask=data.get("target_mask"),
                target_el=data.get("target_atom_element_type"),
            )
        except Exception as exc:
            if getattr(self, "interface_only", False) or getattr(self, "fcc_per_env", False):
                raise RuntimeError(
                    "BEI classification failed but a HARD feature needs the labels "
                    f"(interface_only={getattr(self, 'interface_only', False)}, "
                    f"fcc_per_env={getattr(self, 'fcc_per_env', False)}); refusing to silently degrade "
                    f"(interface_only would drop the restriction / fcc_per_env would mislabel envs). "
                    f"Original error: {exc!r}"
                ) from exc
            logging.warning("BEI classification failed; returning no interface set (soft degrade): %r", exc)
            return {}

    def _compute_interface_positions(self, data: dict) -> list[int]:
        """Window-relative binder indices whose sidechain is INTERFACE (is_contacting: within 4 Å of the
        target AND pseudo-Cβ-facing it, under the default aron_overlap def).

        Thin wrapper over ``_classify_bei_labels`` kept for the interface-anchor call sites / tests.
        Graceful: biotite-missing / classification error => ``[]`` (no anchoring, plain uniform draw).
        """
        return [p for p, lab in self._classify_bei_labels(data).items() if lab == "Interface"]

    @staticmethod
    def _bei_env_from_labels(labels: dict, length: int) -> torch.Tensor:
        """Per-residue B/E/I env code (int tensor length ``length``): 0=Interface, 1=Buried, 2=Exposed.

        Window-relative (same L as ``backbone_coords``, matching ``interface_positions``). Residues with
        no label -- biotite missing / classification failed (``labels == {}``), or glycine/no-sidechain --
        default to Exposed (2), the neutral env whose FCC slope is the mildest correction.
        """
        env = torch.full((length,), 2, dtype=torch.long)  # default Exposed=2
        code = {"Interface": 0, "Buried": 1, "Exposed": 2}
        for p, lab in labels.items():
            if 0 <= p < length:
                env[p] = code.get(lab, 2)
        return env

    def _compute_bei_env(self, data: dict) -> torch.Tensor:
        """Per-residue B/E/I env code for THIS PDB's window (see ``_bei_env_from_labels``)."""
        return self._bei_env_from_labels(self._classify_bei_labels(data), data["backbone_coords"].shape[0])

    def _build_valid_indices(self) -> None:
        """Build list of valid indices (PDBs that load successfully)."""
        if self._valid_indices is not None:
            return

        self._valid_indices = []
        for i in range(len(self.pdb_files)):
            if self._load_pdb(i) is not None:
                self._valid_indices.append(i)

    def __len__(self) -> int:
        """Return number of valid complexes.

        NOTE: This returns an ESTIMATE (90% of file count) to avoid expensive
        validation scan. The actual count is determined lazily during iteration.
        """
        if self._valid_indices is not None:
            return len(self._valid_indices)
        # Estimate: assume ~90% of PDB files are valid (avoids scanning all files)
        return int(len(self.pdb_files) * 0.9)

    def set_k_ramp_epoch(self, epoch: int) -> None:
        """Set current epoch for the K_max ramp (name matches collect_k_ramp_datasets). No-op unless ramp on."""
        self._current_epoch = int(epoch)

    def _design_k_max(self, length: int) -> int:
        """K_max for the design draw: L, unless the optional ramp is on (k_min -> L over epochs)."""
        if not self.random_binder_k_ramp or self.random_binder_k_ramp_total_epochs <= 0:
            return length
        prog = self._current_epoch / max(self.random_binder_k_ramp_total_epochs, 1)
        s, e = self.random_binder_k_ramp_start_frac, self.random_binder_k_ramp_end_frac
        frac = 0.0 if prog <= s else (1.0 if prog >= e else (prog - s) / (e - s))
        k_min = max(1, self.random_binder_k_min or 1)
        return int(min(max(round(k_min + frac * (length - k_min)), k_min), length))

    def _effective_mix_prob(self) -> "float | None":
        """Curriculum-adjusted all-site probability for the current epoch.

        Ramps from 1.0 (full-joint K=L ONLY) down to the base ``random_binder_k_mix_prob`` over
        [start_frac, end_frac] of training (horizon = ``random_binder_k_ramp_total_epochs``). Returns
        the static base when the curriculum is off so existing K-mask runs are unchanged.
        """
        base = self.random_binder_k_mix_prob
        if base is None or not self.random_binder_k_mix_prob_ramp or self.random_binder_k_ramp_total_epochs <= 0:
            return base
        prog = self._current_epoch / max(self.random_binder_k_ramp_total_epochs, 1)
        s, e = self.random_binder_k_mix_prob_ramp_start_frac, self.random_binder_k_mix_prob_ramp_end_frac
        frac = 0.0 if prog <= s else (1.0 if prog >= e else (prog - s) / (e - s))
        return 1.0 + frac * (base - 1.0)  # 1.0 (full-joint) at start -> base by end_frac

    def _three_way_p_mix(self) -> float:
        """Probability that the per-access K is drawn from the full 3-way mixture (vs. forced K=1).

        Only meaningful when ``random_binder_k_final_mix == "three_way"``. ``p_mix`` ramps 0 -> 1 over the
        existing K-ramp window ([``random_binder_k_ramp_start_frac``, ``random_binder_k_ramp_end_frac``],
        horizon ``random_binder_k_ramp_total_epochs``): single-site-always (K=1) at the start, full 3-way
        by the ramp end. With no ramp horizon set, ``p_mix`` is pinned at 1.0 (mixture active immediately).
        """
        if self.random_binder_k_ramp_total_epochs <= 0:
            return 1.0
        prog = self._current_epoch / max(self.random_binder_k_ramp_total_epochs, 1)
        s, e = self.random_binder_k_ramp_start_frac, self.random_binder_k_ramp_end_frac
        return 0.0 if prog <= s else (1.0 if prog >= e else (prog - s) / (e - s))

    def _sample_three_way_k(self, length: int) -> int:
        """Draw K for the 3-way final-mix schedule: ramp from K=1-always to the full 3-way mixture.

        With probability ``p_mix`` (ramping 0 -> 1 over the K-ramp window) sample the 3-way mixture over
        the design count K -- 1/3 K=1, 1/3 K~Uniform{2..L-1} (multi-site interior), 1/3 K=L -- otherwise
        K=1 (single-site). Falls back gracefully for short binders: L<=1 -> K=1; L==2 -> the interior
        branch is empty so it collapses onto the K=1/K=L endpoints (each 1/2 of the mixture draws).
        """
        if length <= 1:
            return 1
        if torch.rand(()).item() >= self._three_way_p_mix():
            return 1  # mixture not active yet -> single-site
        u = torch.rand(()).item()
        if u < 1.0 / 3.0:
            return 1  # single-site
        if u < 2.0 / 3.0:
            if length < 3:
                # No interior {2..L-1} for L<3: re-route this third onto the K=1 / K=L endpoints (1/2 each).
                return 1 if torch.rand(()).item() < 0.5 else length
            return int(torch.randint(2, length, (1,)).item())  # Uniform{2..L-1} (high exclusive)
        return length  # full joint

    def _reversed_k_state(self, length: int) -> "tuple[float, float, float, int]":
        """Schedule state ``(w_kl, w_bridge, w_k1, floor)`` for the reversed (joint-first) K schedule.

        Only meaningful when ``random_binder_k_final_mix == "reversed"``. Starts at 100% K=L (full joint)
        and holds until ``random_binder_k_reverse_start_frac`` (``s``); over [``s``, intro) (phase A) it grows
        the bridge from the top of the interior (``K=L-1``, via the descending ``floor``) downward while K=L
        recedes; over [intro, ``random_binder_k_reverse_end_frac``] (phase B) it introduces K=1 LAST as its own
        bin -- seeded at the bridge per-member level ``w_b / (L-1)`` (NOT 0) so it enters on equal footing with
        K=2/K=3 -- and relaxes (w_kl, w_bridge, w_k1) to the validated finals (kl_f, br_f, k1_f). ``floor`` is
        the inclusive low end of the bridge's ``Uniform{floor..L-1}`` draw: ``L-1`` at phase-A entry, ramping
        to 2 by intro, then pinned at 2. ``intro = s + intro_frac * (e - s)``.
        """
        kl_f = self.random_binder_k_final_kl
        br_f = self.random_binder_k_final_bridge
        k1_f = self.random_binder_k_final_k1
        s, e = self.random_binder_k_reverse_start_frac, self.random_binder_k_reverse_end_frac
        if self.random_binder_k_ramp_total_epochs <= 0:
            # Missing/zero horizon -> pin 100% joint (K=L), the SAFE pre-ramp default: with no horizon the
            # schedule progress is undefined, so a missing horizon is documented as "100% K=L".
            # With horizon auto-fill working (the training launcher derives it) this branch
            # should not be hit in practice; it's a defensive default that never silently jumps to the
            # final inpainting-heavy mix.
            return 1.0, 0.0, 0.0, length
        prog = self._current_epoch / max(self.random_binder_k_ramp_total_epochs, 1)
        if prog <= s:
            return 1.0, 0.0, 0.0, length  # pure joint (K=L only)
        if prog >= e:
            return kl_f, br_f, k1_f, 2
        intro = s + self.random_binder_k_reverse_intro_frac * (e - s)
        # End-of-phase-A targets: half the eventual K=1 mass is "pre-loaded" into K=L and the bridge so the
        # phase-A endpoint and phase-B start are continuous (K=1 then peels its share off the bridge).
        w_a = kl_f + k1_f / 2.0  # K=L weight at the A->B handoff
        w_b = br_f + k1_f / 2.0  # bridge weight at the A->B handoff (K=1 still 0 here)
        top = length - 1
        if prog < intro:
            # PHASE A [s, intro): K=L recedes 1.0 -> w_a, bridge grows 0 -> w_b (bridge member K=1 not yet active).
            a = (prog - s) / (intro - s)
            w_kl = 1.0 - a * (1.0 - w_a)
            w_bridge = a * w_b
            w_k1 = 0.0
            floor = max(2, round(top - a * (top - 2)))  # L-1 -> 2
            return w_kl, w_bridge, w_k1, floor
        # PHASE B [intro, e): introduce K=1 seeded at the bridge per-member level, then relax all three to finals.
        b = (prog - intro) / (e - intro)
        w_k1_init = w_b / max(length - 1, 1)  # per-member bridge level at entry (NOT 0)
        w_kl = w_a + b * (kl_f - w_a)
        w_k1 = w_k1_init + b * (k1_f - w_k1_init)
        w_bridge = (w_b - w_k1_init) + b * (br_f - (w_b - w_k1_init))
        return w_kl, w_bridge, w_k1, 2

    def _geometric_k_probs(self, length: int) -> "torch.Tensor":
        """P(K) ∝ r^(K-1) over K=1..L, normalized. Left-shifted, MONOTONE non-increasing for r<=1.

        The alternative to the 3-mode {K=L, uniform bridge, K=1} mixture, for sources where full-joint
        design is not the point (the NCAA monomer/dimer sets: peptideMPNN + antibody already supply
        joint-mode training, and those sets are scaffold-poor so joint design overfits them).

        r is the ONLY knob, and it is the only one there is room for: A in ``A*r^(K-1)`` is pinned by
        sum-to-1, so r alone fixes BOTH ends -- P(1) = (1-r)/(1-r^L) and P(L) = P(1)*r^(L-1). You choose
        one end; the other follows. See ``_validate_geometric_k`` for why a separate K=L weight cannot
        be honoured on top of this.
        """
        r = float(self.random_binder_k_geometric_r)
        k = torch.arange(length, dtype=torch.float64)  # exponents 0..L-1 for K=1..L
        w = r**k
        return (w / w.sum()).to(torch.float32)

    def _sample_reversed_k(self, length: int) -> int:
        """Draw K for the reversed (joint-first) final-mix schedule. See ``_reversed_k_state`` for the schedule.

        Samples the 3-mode mixture {K=L, bridge K~Uniform{floor..L-1}, K=1} at the current schedule weights,
        OR -- when ``random_binder_k_geometric_r`` is set -- a pure geometric P(K) ∝ r^(K-1) over 1..L.
        Falls back gracefully for short binders: L<=1 -> K=1; L<3 has no interior bridge so the bridge mass is
        renormalized onto the K=L / K=1 endpoints.
        """
        if length <= 1:
            return 1
        # getattr (not direct attr): _sample_reversed_k is exercised by lightweight BinderDataset
        # shims that bypass __init__ (tests, K-schedule probes) and never set this optional field.
        if getattr(self, "random_binder_k_geometric_r", None) is not None:
            # Joint-first ramp is preserved: blend from the K=L point mass to the geometric across
            # [s, e], so prog<=s is still 100% joint and prog>=e is the pure geometric. With the NCAA
            # per-source override (s=0.0, e=0.0001) this is the geometric from ~ep0, as intended.
            s, e = self.random_binder_k_reverse_start_frac, self.random_binder_k_reverse_end_frac
            total = max(self.random_binder_k_ramp_total_epochs, 1)
            prog = self._current_epoch / total
            if self.random_binder_k_ramp_total_epochs <= 0 or prog <= s:
                return length
            alpha = 1.0 if prog >= e else (prog - s) / max(e - s, 1e-9)
            if torch.rand(()).item() >= alpha:
                return length
            probs = self._geometric_k_probs(length)
            return int(torch.multinomial(probs, 1).item()) + 1  # index 0 -> K=1
        w_kl, w_bridge, w_k1, floor = self._reversed_k_state(length)
        if length < 3:
            # No interior {floor..L-1} bridge for L<3: renormalize onto the K=1 / K=L endpoints.
            z = w_kl + w_k1
            return length if (z <= 0 or torch.rand(()).item() < w_kl / z) else 1
        u = torch.rand(()).item()
        if u < w_kl:
            return length  # full joint
        if u < w_kl + w_bridge:
            return int(torch.randint(floor, length, (1,)).item())  # Uniform{floor..L-1} (high exclusive)
        return 1  # single-site

    def _sample_design_mask(
        self,
        length: int,
        forced_include: int | None = None,
        weighted_include: "int | Sequence[int] | None" = None,
        weighted_include_weight: float = 1.0,
        restrict_include: "Sequence[int] | None" = None,
    ) -> torch.Tensor:
        """Per-access binder design mask: which residues to design.

        All-True when K-masking is off. Otherwise P(K=K_max)=mix_prob (full design, preserves all-site
        training), else K~Uniform[k_min, K_max]; the other L-K residues stay GT-pinned binder context.

        The drawn budget K is the SOLE driver of HOW MANY positions are designed. Exactly K sites are
        always returned; the optional ``forced_include`` / ``weighted_include`` knobs only bias WHICH K
        sites are chosen -- they never change the count.

        Parameters
        ----------
        length : int
            Binder (window) length L.
        forced_include : int, optional
            A binder-window-relative residue index that is HARD-FORCED into the designed positions
            (off by default, None). Used by ``OnTheFlyNcaaDataset``'s enumeration eval path to
            guarantee the systematic per-NCAA single-site eval always designs the enumerated NCAA:
            its slot is reserved, then the remaining K-1 sites are drawn from the others. Out-of-range
            / negative indices are ignored. Distinct from ``weighted_include`` (a soft bias).
        weighted_include : int or sequence of int, optional
            Binder-window-relative residue index(es) that are UP-WEIGHTED (not forced) in the without-
            replacement site draw (off by default, None). Accepts EITHER a single int (the NCAA single-
            site anchor used by ``OnTheFlyNcaaDataset``'s TRAINING path) OR a sequence of ints
            (the INTERFACE positions, ΔSASA >= 16 Å², used by the interface-anchored design mask). ALL
            valid in-range indices in the (possibly multi-element) set get the SAME
            ``weighted_include_weight``. The K budget is preserved exactly; a single anchor's inclusion
            probability is ``w / (w + L - 1)`` at K=1 and rises to 1.0 at K=L (a multi-element set is
            enriched jointly). With ``weighted_include_weight == 1.0`` (the default) -- or an empty /
            None set -- the draw is statistically identical to uniform site selection. Out-of-range /
            negative indices are ignored. Ignored when ``forced_include`` is supplied.
        weighted_include_weight : float
            Multinomial weight applied to ``weighted_include`` (default 1.0 == unbiased/uniform). Only
            consulted when ``weighted_include`` is a valid in-range index. Values >= 1 bias toward the
            site; values in [0, 1) would de-bias it (callers reject < 1 upstream).
        restrict_include : sequence of int, optional
            HARD restriction (interface-only design): when provided and non-empty, the designed sites are
            drawn EXCLUSIVELY from these window-relative indices and the drawn budget K is capped at
            ``len(restrict_include)`` (design all of them if K would exceed the count). Every index outside
            the set stays GT-pinned context. Takes precedence over ``forced_include`` / ``weighted_include``
            (the interface-only anchor kwargs never combine them). None / empty => no restriction, so the
            normal weighted/uniform draw runs unchanged (byte-identical when interface_only is off).
        """
        three_way = self.random_binder_k_final_mix == "three_way"
        reversed_mix = self.random_binder_k_final_mix == "reversed"
        # K-mask is OFF (design-all) only when NONE of the schedules is engaged. three_way / reversed each
        # drive their own K draw independently of random_binder_k_mix_prob (which can stay None under them).
        if (self.random_binder_k_mix_prob is None and not three_way and not reversed_mix) or length <= 1:
            return torch.ones(length, dtype=torch.bool)
        if three_way:
            # 3-way final-mix schedule: ramp single-site-always -> {1/3 K=1, 1/3 K~Uniform{2..L-1}, 1/3 K=L}.
            k = self._sample_three_way_k(length)
        elif reversed_mix:
            # reversed (joint-first) final-mix schedule: 100% K=L -> {K=L, bridge K~U{2..L-1}, K=1}.
            k = self._sample_reversed_k(length)
        else:
            k_max = self._design_k_max(length)
            k_min = min(max(1, self.random_binder_k_min or 1), k_max)  # k_min can exceed a short L -> clamp to k_max
            mix = self._effective_mix_prob()  # curriculum-adjusted all-site prob (== base when ramp off)
            if torch.rand(()).item() < mix:
                k = k_max  # all-site (K=L) -- exactly mix_prob of draws
            else:
                # uniform over [k_min, k_max-1]: EXCLUDE k_max so K=L occurs ONLY in the all-site branch above
                # => exactly mix_prob of draws are all-site; the rest are genuine partial-design tasks (K<L).
                # max(k_min+1, k_max) guards the degenerate k_min==k_max case (randint high is exclusive).
                k = int(torch.randint(k_min, max(k_min + 1, k_max), (1,)).item())
        k = min(k, length)  # guard: never request more sites than exist
        mask = torch.zeros(length, dtype=torch.bool)
        # HARD interface-only restriction: draw the K designed sites EXCLUSIVELY from restrict_include,
        # capping K at the interface count. Empty / None => skip (the anchor kwargs only set this when the
        # interface set is non-empty), so the weighted/forced paths below run byte-identically when off.
        if restrict_include:
            restrict_idxs = [int(i) for i in restrict_include if 0 <= int(i) < length]
            if restrict_idxs:
                k = min(k, len(restrict_idxs))
                pool = torch.tensor(restrict_idxs, dtype=torch.long)
                pick = pool[torch.randperm(pool.numel())[:k]]
                mask[pick] = True
                return mask
        forced_valid = forced_include is not None and 0 <= forced_include < length
        # Normalize weighted_include to a list of in-range indices. Accepts a single int (NCAA anchor --
        # identical to the historical scalar path) OR a sequence of ints (interface positions). None /
        # empty / all-out-of-range => no up-weighting (statistically uniform, byte-identical to the old
        # randperm path when the weight is 1.0).
        if weighted_include is None:
            weighted_idxs: list[int] = []
        elif isinstance(weighted_include, int):
            weighted_idxs = [weighted_include] if 0 <= weighted_include < length else []
        else:
            weighted_idxs = [int(i) for i in weighted_include if 0 <= int(i) < length]
        if forced_valid:
            # HARD FORCE (enumeration eval): guarantee the anchor is designed. Reserve its slot, then
            # fill the remaining K-1 uniformly from the other positions. Kept distinct from the
            # weighted path so the systematic per-NCAA single-site eval is unaffected by the training
            # anchor weight.
            mask[forced_include] = True
            if k > 1:
                others = torch.tensor([i for i in range(length) if i != forced_include], dtype=torch.long)
                pick = others[torch.randperm(others.numel())[: k - 1]]
                mask[pick] = True
        else:
            # WEIGHTED SUBSET WITHOUT REPLACEMENT: K is preserved exactly; only the per-site weight of
            # `weighted_include` (NCAA) biases WHICH K sites are chosen. With weight 1.0 (canonical
            # default) this reduces to a uniform K-subset, statistically identical to the old randperm
            # path. multinomial(replacement=False) requires num_samples <= #non-zero-weight entries;
            # all weights here are strictly positive (>=1 on the bias site, 1.0 elsewhere) and k<=length,
            # so the edge case is already guarded by the k=min(k,length) clamp above.
            weights = torch.ones(length, dtype=torch.float)
            for wi in weighted_idxs:
                weights[wi] = float(weighted_include_weight)
            pick = torch.multinomial(weights, num_samples=k, replacement=False)
            mask[pick] = True
        return mask

    def _design_mask_anchor_kwargs(
        self, ncaa_window_pos: int | None, interface_positions: "list[int] | None" = None
    ) -> dict:
        """Per-access kwargs threading the (optional) NCAA / INTERFACE anchors into ``_sample_design_mask``.

        Base behaviour honours two soft, opt-in anchors that bias WHICH K sites are designed (never how
        many -- the K budget is preserved):

        * ``interface_anchor_weight`` > 1: up-weight the binder's INTERFACE residues
          (``interface_positions``, ΔSASA >= 16 Å² vs target, computed & cached in ``_load_pdb``).
        * ``ncaa_anchor_weight`` > 1: up-weight the stashed NCAA position (synthetic NCAA dimer).

        When BOTH are active the NCAA position is UNIONed into the interface set and both are biased
        under a SINGLE shared ``weighted_include_weight`` (the interface weight). This is deliberate:
        ``_sample_design_mask`` applies one multinomial weight to every up-weighted index, so rather
        than juggling two per-site weights we let the interface weight govern the whole anchored set
        (the NCAA site is generally already interface, so this is usually a no-op union).

        When neither anchor is engaged (canonical / joint sources, or the weights at 1.0) returns an
        empty dict and the design mask draws a plain uniform K-subset. ``OnTheFlyNcaaDataset``
        overrides this for the monomer-window path.

        The HARD ``interface_only`` restriction takes precedence over both soft anchors: when it is on and
        an interface set exists, the design mask draws its sites EXCLUSIVELY from that set (``restrict_include``).
        When ``interface_only`` is on but the interface set is empty, it falls through to the normal draw.
        """
        if getattr(self, "interface_only", False) and interface_positions:
            return {"restrict_include": list(interface_positions)}
        iface_on = getattr(self, "interface_anchor_weight", 1.0) > 1.0 and bool(interface_positions)
        ncaa_on = ncaa_window_pos is not None and getattr(self, "ncaa_anchor_weight", 1.0) > 1.0
        if iface_on:
            positions = list(interface_positions)
            if ncaa_on and ncaa_window_pos not in positions:
                positions.append(ncaa_window_pos)  # UNION NCAA into interface set (shared iface weight)
            return {"weighted_include": positions, "weighted_include_weight": self.interface_anchor_weight}
        if ncaa_on:
            return {"weighted_include": ncaa_window_pos, "weighted_include_weight": self.ncaa_anchor_weight}
        return {}

    def _gated_design_mask(self, design_mask: "torch.Tensor", data: dict) -> "torch.Tensor":
        """Apply the designable-site gate and make an EMPTIED mask visible rather than silent.

        The gate can leave fewer than K designed positions, and at K=1 that is zero. A sample
        with no designed position carries NO SUPERVISION for any design-restricted loss: the
        training step masks every such term to the designed set, and each is a masked
        ``sum() / (mask.sum() + eps)``, so the item contributes exactly zero and still consumes
        its slot in the batch. That is the bad outcome -- not a crash, not a skip, just a wasted
        sample -- and it was previously indistinguishable from a normal one.

        NOT redrawn and NOT skipped, deliberately. Redrawing would silently change which sites
        the K budget covers (see :func:`apply_designable_gate`), and skipping would make dataset
        length depend on a per-access random draw. Measured rates on the real NCAA sources make
        that the right trade: 0.000% of synthetic/modelled NCAA sites are gated (they match their
        SMILES exactly, as expected), against 0.51-0.63% on the native/crystal gold sources,
        where the gated sites are genuinely disorder-truncated (DLY, MLY, CCS resolved to 1-3
        side-chain atoms). No source chain in any of those sets has zero designable sites, so
        only a K=1 draw that lands on a disordered site can empty the mask.

        Logged with exponential backoff (occurrences 1, 2, 4, 8, ...) so a ~0.5% rate reports
        itself without flooding a training log. The count is per dataset instance, and dataloader
        workers each hold their own copy.
        """
        gated = apply_designable_gate(design_mask, data.get("designable_mask"))
        if design_mask.any() and not gated.any():
            self._emptied_design_mask_count += 1
            n = self._emptied_design_mask_count
            if n & (n - 1) == 0:  # powers of two
                _LOG.warning(
                    "Designable-site gate emptied the design mask for %s: all %d sampled "
                    "position(s) have a deposited composition that contradicts their label, so "
                    "this sample carries no design supervision and still consumes a batch slot "
                    "(%d such sample(s) so far in this dataset instance).",
                    data.get("pdb_name", "<unknown>"),
                    int(design_mask.sum()),
                    n,
                )
        return gated

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Get binder-target complex data by index (with a fresh per-access binder K-mask)."""
        self._build_valid_indices()
        real_idx = self._valid_indices[idx]
        data = self._load_pdb(real_idx)
        if data is None:
            return data
        # Shallow-copy so the per-access design_mask never mutates _load_pdb's cached dict.
        data = dict(data)
        # OnTheFlyNcaaDataset stashes the NCAA's window-relative index here so the design mask
        # can bias toward the NCAA site (soft anchor). None / absent for canonical datasets (no anchor).
        anchor_kwargs = self._design_mask_anchor_kwargs(
            data.pop("_ncaa_window_pos", None), data.get("interface_positions")
        )
        design_mask = self._sample_design_mask(data["backbone_coords"].shape[0], **anchor_kwargs)
        # Never designate a site whose ground truth contradicts its own label. See
        # apply_designable_gate for why this can leave FEWER than K designed positions, and
        # _gated_design_mask for what happens (and gets logged) when that is zero.
        data["design_mask"] = self._gated_design_mask(design_mask, data)
        return data

    def iter_valid(self, max_samples: int | None = None) -> "Iterator[dict[str, torch.Tensor]]":
        """
        Lazily iterate over valid samples without pre-scanning all files.

        This is more efficient than __len__() + __getitem__() when you don't need
        to know the total count upfront, as it avoids loading all PDBs to build
        the valid indices list.

        Parameters
        ----------
        max_samples : int, optional
            Maximum number of valid samples to yield. If None, yields all valid samples.

        Yields
        ------
        dict[str, torch.Tensor]
            Sample data dictionary for each valid PDB.
        """
        count = 0
        for idx in range(len(self.pdb_files)):
            if max_samples is not None and count >= max_samples:
                break
            data = self._load_pdb(idx)
            if data is not None:
                count += 1
                data = dict(data)  # add design_mask here too so diagnostics via iter_valid see K-masks
                anchor_kwargs = self._design_mask_anchor_kwargs(
                    data.pop("_ncaa_window_pos", None), data.get("interface_positions")
                )
                # Same designable-site gate as __getitem__. Without it this accessor -- used by
                # diagnostics and eval, never by the DataLoader -- could hand out design masks
                # covering sites whose ground truth we have declared untrustworthy.
                data["design_mask"] = self._gated_design_mask(
                    self._sample_design_mask(data["backbone_coords"].shape[0], **anchor_kwargs),
                    data,
                )
                yield data


# Backwards compatibility alias
PeptideDataset = BinderDataset


def _residue_ca_array(residues: "list[dict]") -> "np.ndarray":
    """``(N, 3)`` array of each residue's Cα coordinate; NaN where the residue has no CA.

    Residue dicts store backbone atoms under ``res["backbone"]`` keyed by atom name (see
    ``extract_residue_atoms``); the Cα is ``res["backbone"]["CA"]`` when present.
    """
    import numpy as _np

    out = _np.full((len(residues), 3), _np.nan, dtype=float)
    for i, res in enumerate(residues):
        ca = res.get("backbone", {}).get("CA")
        if ca is not None:
            out[i] = ca[:3]
    return out


def _closest_target_indices(
    binder_ca: "np.ndarray",
    target_ca: "np.ndarray",
    max_keep: int,
) -> "np.ndarray":
    """Indices into ``target_ca`` of the ``max_keep`` residues closest to any binder Cα.

    Distance is the min over binder Cα (interface proximity). Residues with a non-finite Cα
    (missing CA / NaN) are pushed to the far end so they are dropped FIRST. Ties and the
    degenerate no-finite-binder-Cα case fall back to original residue order (stable), so the
    result then matches the legacy first-N index crop. Returned indices are sorted ascending,
    preserving original target order among the kept residues.
    """
    import numpy as _np

    n_t = len(target_ca)
    if n_t <= max_keep:
        return _np.arange(n_t, dtype=int)

    finite_b = _np.isfinite(binder_ca).all(axis=-1)
    finite_t = _np.isfinite(target_ca).all(axis=-1)
    min_d = _np.full(n_t, _np.inf, dtype=float)
    if finite_b.any():
        bca = binder_ca[finite_b]
        diffs = bca[:, None, :] - target_ca[None, :, :]
        d = _np.sqrt((diffs**2).sum(axis=-1)).min(axis=0)  # (T,) min over binder
        min_d = _np.where(finite_t, d, _np.inf)
    # lexsort by (index, distance): distance is primary, index breaks ties -> stable order,
    # and an all-inf min_d (no finite binder Cα, or all-NaN target) degrades to first-N.
    order = _np.lexsort((_np.arange(n_t), min_d))
    return _np.sort(order[:max_keep])


def _radial_crop_target_indices(
    binder_ca: "np.ndarray",
    target_ca: "np.ndarray",
    radius: float,
) -> "np.ndarray":
    """Return indices into ``target_ca`` whose Cα lies within ``radius`` Å of any
    binder Cα. Vectorized; cheap for B <= ~20, T up to a few hundred.

    NaN Cα coords (residues missing a CA atom) are excluded.

    Ported verbatim from mainline atomweaver (monomer/NCAA on-the-fly windowing).
    """
    import numpy as _np

    if len(binder_ca) == 0 or len(target_ca) == 0:
        return _np.array([], dtype=int)
    finite_t = _np.isfinite(target_ca).all(axis=-1)
    finite_b = _np.isfinite(binder_ca).all(axis=-1)
    if not finite_b.any():
        return _np.array([], dtype=int)
    bca = binder_ca[finite_b]
    diffs = bca[:, None, :] - target_ca[None, :, :]
    d2 = (diffs**2).sum(axis=-1)
    r2 = float(radius) ** 2
    mask = (d2 <= r2).any(axis=0) & finite_t
    return _np.where(mask)[0]


class _IndexBuildShim:
    """Tiny picklable stand-in for the spawn-worker parse.

    Exposes exactly the attributes that ``OnTheFlyNcaaDataset._parse_source`` /
    ``_enumerate_candidates`` read.

    Spawn-context worker processes re-import this module fresh (no inherited parent
    state / no CUDA fork), so the parse worker must be a module-level function with
    picklable arguments. Rather than duplicate the (carefully validated) enumeration
    logic, we reconstruct a minimal object carrying the parse-relevant config + the
    chunk's PDB paths, then call the UNBOUND ``_parse_source`` / ``_enumerate_candidates``
    methods on it. Result is bit-identical to the in-process serial path by construction
    (same code path), while the shim is cheap to pickle and ships no heavy state.
    """

    __slots__ = (
        "holdout_resnames",
        "max_binder_length",
        "max_sidechain_atoms",
        "min_target_size",
        "pdb_files",
        "target_radius",
        "window_size_max",
        "window_size_min",
    )

    def __init__(self, cfg: dict, pdb_files: "list[Path]") -> None:
        self.pdb_files = pdb_files
        self.holdout_resnames = cfg["holdout_resnames"]
        self.window_size_min = cfg["window_size_min"]
        self.window_size_max = cfg["window_size_max"]
        self.min_target_size = cfg["min_target_size"]
        self.max_binder_length = cfg["max_binder_length"]
        self.max_sidechain_atoms = cfg["max_sidechain_atoms"]
        self.target_radius = cfg["target_radius"]

    # `_parse_source` / `_enumerate_candidates` are bound to the dataset's UNBOUND methods
    # just after the class definition (forward reference), so the shim runs the exact same
    # enumeration code as the in-process serial path -- no logic duplication, no drift.


def _index_build_worker(args: "tuple[dict, list[Path]]") -> "list[tuple[int, list | None]]":
    """Module-level (picklable) parse worker for the spawn-context parallel index build.

    Takes ``(cfg, pdb_paths)`` -- a picklable parse config dict and the chunk's PDB paths --
    and returns ``[(local_idx, candidates)]`` (``candidates is None`` for unusable sources).
    Spawn workers re-import this module and do NOT share the parent's host RAM, so they avoid
    both the CUDA fork deadlock and the copy-on-write host-RAM blowup of a forked pool.
    """
    cfg, pdb_paths = args
    shim = _IndexBuildShim(cfg, list(pdb_paths))
    out: list[tuple[int, list | None]] = []
    for local_i in range(len(pdb_paths)):
        entry = shim._parse_source(local_i)
        out.append((local_i, None if entry is None else entry["candidates"]))
    return out


class OnTheFlyNcaaDataset(BinderDataset):
    """``BinderDataset`` variant that loads single-chain NCAA source PDBs and
    slices a fresh (binder window, radial-cropped target) pair at every
    ``__getitem__`` call instead of consuming pre-built 2-chain PDBs.

    Why this exists
    ---------------
    The pre-built monomer pipeline emits a fixed N windows per source, so any given
    source's NCAA is permanently placed at one offset inside its window. At large
    source-pool scale (~18K source PDBs) the budget forces ~1 window
    per source, and a fixed NCAA offset could be memorized in late training. Picking
    the window on the fly removes that risk: every epoch every source is windowed
    differently.

    Adaptation to this branch (kmask-inpainting / vocab5-X)
    -------------------------------------------------------
    Mainline's on-the-fly dataset is coupled to mainline's ANCHOR-based K-mask
    (``random_binder_k_max`` + ``_pre_drawn_K`` + ``ncaa_position_bias_prob`` +
    ``filter_oversized`` + ``deterministic_binder_k``). THIS branch uses the
    design-mask K-mask (``random_binder_k_mix_prob`` + ``_sample_design_mask``,
    applied per-access over the binder window). We express mainline's NCAA-anchor
    REQUIREMENT natively in the design-mask style rather than porting that
    machinery verbatim:

    1. The window merely CONTAINS the NCAA -- ``_enumerate_candidates`` keeps every
       containing window (any NCAA offset, including the edges) and ``_pick_candidate``
       samples one uniformly. No central-band preference: edge placements are valid
       training signal and dropping them used to silently de-enumerate a second NCAA.
    2. The NCAA is UP-WEIGHTED (not forced) in the design-mask site draw during training via
       ``ncaa_design_anchor_weight`` (default 4.0). The window-relative NCAA index is threaded
       through ``_ncaa_window_pos`` into the parent's ``_sample_design_mask`` as
       ``weighted_include`` with that weight. The K budget is sampled as usual and is the SOLE
       driver of HOW MANY positions are designed; the weight only biases WHICH K sites are
       chosen (weighted subset sampling without replacement). The NCAA's inclusion probability is
       ``w / (w + L - 1)`` at K=1, rising to 1.0 at K=L. When the NCAA is NOT picked, it is left
       as clean pinned conditioning context so neighbours learn to design around an undesigned
       NCAA (bidirectional objective). The enumeration eval path uses a SEPARATE hard-force
       (``forced_include``, prob 1.0) so systematic per-NCAA eval is unaffected by the training
       weight. No ``_pre_drawn_K`` / ``ncaa_position_bias_prob`` coupling is needed.

    Everything else downstream -- vocab5-X element encoding, max_sidechain_atoms=16
    slots, pseudo_backbone, holdout filtering, the per-access ``design_mask`` -- is
    inherited from the parent unchanged.

    Enumeration mode (``enumerate_ncaa_positions``, eval only) iterates one sample
    per (PDB, NCAA position), pinning the anchor at each NCAA in turn for the
    systematic per-NCAA discretization eval. Unused by training.

    Parameters
    ----------
    source_pdb_dir : str or Path
        Directory of single-chain source PDBs. Each PDB must contain at least
        one residue whose 3-letter code lies outside ``STANDARD_AA`` (the NCAA).
        Multi-chain source PDBs are accepted; the longest chain is used.
    window_size_min, window_size_max : int
        Binder window size range (inclusive). Default 8..14 matches the pre-built
        pipeline.
    target_radius : float
        Radial crop radius around binder Cα atoms. Set 0 to keep the whole
        rest-of-chain as target. Default 15.0 Å matches the pre-built pipeline.
    min_target_size : int
        Skip windows whose radial-cropped target has fewer residues. Default 20.
    deterministic_binder_k : bool
        If True, the window for each source is chosen by a stable hash of the PDB
        stem so val/test runs see a fixed window across epochs (epoch-stable
        val_loss). Random otherwise. Name kept for parity with the dataloader
        plumbing; here it gates ONLY window determinism (no K coupling on this
        branch).
    ncaa_design_anchor_weight : float
        Multinomial up-weight (default 4.0) applied to the NCAA site in the TRAINING design-mask
        site draw (weighted subset sampling without replacement). The K budget is preserved
        EXACTLY -- this only biases WHICH K sites are chosen, never how many. The NCAA's inclusion
        probability is ``w / (w + L - 1)`` at K=1 and 1.0 at K=L; when the NCAA is not picked it
        stays as clean pinned conditioning context (bidirectional objective). ``weight == 1.0`` is
        unbiased/uniform; values < 1 would de-bias the NCAA and are rejected (must be >= 1.0). Has
        NO effect in enumeration mode, which HARD-FORCES the anchor (prob 1.0) via a separate path.
    Remaining keyword arguments are forwarded to ``BinderDataset.__init__``.
    """

    def __init__(
        self,
        source_pdb_dir: "str | Path",
        *,
        window_size_min: int = 8,
        window_size_max: int = 14,
        target_radius: float = 15.0,
        min_target_size: int = 20,
        deterministic_binder_k: bool = False,
        ncaa_design_anchor_weight: float = 4.0,
        enumerate_ncaa_positions: bool = False,
        enumeration_exclude_resnames: "frozenset[str] | None" = None,
        source_cache_maxsize: int = 4096,
        **binder_dataset_kwargs,
    ) -> None:
        if window_size_min < 1 or window_size_max < window_size_min:
            raise ValueError(f"Invalid window size range: [{window_size_min}, {window_size_max}]")
        if target_radius < 0:
            raise ValueError(f"target_radius must be >= 0 (got {target_radius})")
        if min_target_size < 0:
            raise ValueError(f"min_target_size must be >= 0 (got {min_target_size})")
        if source_cache_maxsize < 1:
            raise ValueError(f"source_cache_maxsize must be >= 1 (got {source_cache_maxsize})")
        # Weight must be >= 1.0: 1.0 == unbiased uniform draw; > 1.0 biases the NCAA toward the
        # designed set. A weight < 1 would actively DE-bias the NCAA (push it out of the designed
        # set), which is never the intent -- reject it loudly rather than silently de-anchoring.
        if not (ncaa_design_anchor_weight >= 1.0):
            raise ValueError(
                f"ncaa_design_anchor_weight must be >= 1.0 (1.0 == unbiased; got {ncaa_design_anchor_weight})"
            )

        # The parent's `__init__` holdout filter (`_binder_has_holdout`) parses EVERY source
        # PDB up front to inspect binder-chain residue names. For the on-the-fly NCAA dataset
        # that filter is BOTH redundant and wrong:
        # - redundant: `_parse_source` already drops any source containing a holdout residue
        # (it walks the whole chain, not just a 2-chain "binder"), so holdout sources never
        # produce a candidate and are pruned during `_build_valid_indices`.
        # - wrong/wasteful: `_binder_has_holdout` uses the 2-chain `get_peptide_chain` split,
        # which is meaningless for these single-chain source PDBs, and at 153k sources its
        # serial parse pass cost ~minutes per dataset instance for zero benefit.
        # So we withhold `holdout_resnames` from the parent (skipping that parse pass) and set
        # `self.holdout_resnames` ourselves afterward, preserving the `_parse_source` holdout drop.
        _holdout = binder_dataset_kwargs.pop("holdout_resnames", None)
        super().__init__(pdb_dir=source_pdb_dir, holdout_resnames=None, **binder_dataset_kwargs)
        self.holdout_resnames = frozenset(_holdout) if _holdout else frozenset()
        self.window_size_min = int(window_size_min)
        self.window_size_max = int(window_size_max)
        self.target_radius = float(target_radius)
        self.min_target_size = int(min_target_size)
        self.deterministic_binder_k = bool(deterministic_binder_k)
        self.ncaa_design_anchor_weight = float(ncaa_design_anchor_weight)
        # Two-tier per-worker cache (memory-bounded -- 153k+ sources can't all fit):

        # _source_candidates: idx -> candidate list (or None for unusable sources).
        # LIGHTWEIGHT (a few ints per `(window_size, ncaa_idx, start)` tuple) and
        # PERSISTENT -- every enumerated source keeps its candidates forever so the
        # index built by `_build_valid_indices` / `_build_ncaa_position_index`
        # survives without holding the heavy parse. `None` marks an unusable source
        # (unparseable / no NCAA / too short / holdout / no valid window).

        # _source_cache: idx -> heavy parse dict (residues, ncaa_indices, ca_coords,
        # candidates) -- bounded LRU (OrderedDict, `move_to_end` on hit, pop oldest
        # when over `source_cache_maxsize`). On a miss `_load_source` re-parses from
        # disk (deterministic from file content; the candidates are already known
        # from `_source_candidates`, so re-parse is cheap and correct).
        self.source_cache_maxsize = int(source_cache_maxsize)
        # COW-safe candidate index. Backed by a numpy CSR store (one shared C buffer per array)
        # once `_build_valid_indices` finalizes -- so forked DataLoader workers share it via
        # copy-on-write instead of each duplicating a 153k-entry Python dict/list/tuple graph.
        # Mapping API (`in`, `[]`, `len`, `.items()`, `.values()`) is preserved for call sites/tests.
        self._source_candidates = _NumpyCandidateIndex()
        self._source_cache: "OrderedDict[int, dict]" = OrderedDict()
        # Anchor plumbing. `_forced_anchor` (enumeration eval) pins one source-chain NCAA
        # position into the window + design mask for the next `__getitem__`; single-use,
        # consumed in `_parse_pdb_cached`. `_last_ncaa_window_pos` is the window-relative
        # NCAA index of the most recently sliced window, injected into the data dict so the
        # parent's design-mask sampler always includes the NCAA site.
        self._forced_anchor: int | None = None
        self._last_ncaa_window_pos: int | None = None
        # Enumeration mode (eval): walk every NCAA position, force each designed once.
        self.enumerate_ncaa_positions = bool(enumerate_ncaa_positions)
        self.enumeration_exclude_resnames = enumeration_exclude_resnames or frozenset()
        # Flat (pdb_idx, ncaa_source_position) list built lazily for enumeration mode.
        self._ncaa_position_index: list[tuple[int, int]] | None = None

    # ------------------------------------------------------------------
    # Disk cache of the valid-index build (the ~15 min cold-build is the #1
    # iteration killer; cache it keyed on every enumeration-relevant input).
    # ------------------------------------------------------------------

    def _index_build_config(self) -> dict:
        """Picklable parse config shipped to spawn workers AND used in the cache key.

        Holds EXACTLY the attributes that ``_parse_source`` / ``_enumerate_candidates`` read
        (so a worker shim reproduces the in-process parse bit-for-bit). NOT included here: the
        pdb file list (keyed separately) -- this dict is per-worker config, replicated identically.
        """
        return {
            "holdout_resnames": frozenset(self.holdout_resnames),
            "window_size_min": int(self.window_size_min),
            "window_size_max": int(self.window_size_max),
            "min_target_size": int(self.min_target_size),
            "max_binder_length": (None if self.max_binder_length is None else int(self.max_binder_length)),
            "max_sidechain_atoms": int(self.max_sidechain_atoms),
            "target_radius": float(self.target_radius),
        }

    def _source_stat_fingerprint(self) -> list:
        """Cheap stat-based content fingerprint of the source PDBs (no content hash).

        The paths alone don't detect in-place REGENERATION of the 153k monomer PDBs (same paths,
        new bytes -> stale cached indices). Hashing 153k file contents would cost minutes and defeat
        the cache; instead we fold cheap ``os.stat`` metadata that changes when files are rewritten:

          * per SOURCE DIRECTORY (the parent dirs of ``pdb_files``): the dir's own ``st_mtime_ns``
            (bumps when files are added/removed/renamed in it) and the count of our files under it;
          * a bounded, deterministic SAMPLE of up to 64 files spread across the sorted list, each
            contributing ``(st_size, st_mtime_ns)`` -- a regenerated file changes mtime (and usually
            size), so any in-place rewrite that touches a sampled file misses the cache.

        Stat-based (not content-hash) is the deliberate cost/safety balance: O(#dirs + 64) stat
        calls vs O(153k) reads. It catches the common regenerate-in-place case (mtimes move) at
        negligible cost; it does NOT catch a byte-identical rewrite that preserves size+mtime on an
        UNSAMPLED file (vanishingly rare, and a schema bump or cache-disable env is the escape hatch).
        Missing files (a path that vanished) contribute a sentinel so they still perturb the key.
        """
        paths = list(self.pdb_files)
        # Per-directory metadata: dir mtime + how many of our files live under it.
        dir_counts: dict[str, int] = {}
        for p in paths:
            d = str(Path(p).parent)
            dir_counts[d] = dir_counts.get(d, 0) + 1
        dir_fp = []
        for d in sorted(dir_counts):
            try:
                dir_fp.append([d, int(Path(d).stat().st_mtime_ns), dir_counts[d]])
            except OSError:
                dir_fp.append([d, -1, dir_counts[d]])
        # Deterministic sample of <=64 files spread evenly across the (already sorted) list.
        n = len(paths)
        if n == 0:
            sample_idx: list[int] = []
        else:
            step = max(1, n // 64)
            sample_idx = list(range(0, n, step))[:64]
        file_fp = []
        for i in sample_idx:
            try:
                st = Path(paths[i]).stat()
                file_fp.append([str(paths[i]), int(st.st_size), int(st.st_mtime_ns)])
            except OSError:
                file_fp.append([str(paths[i]), -1, -1])
        return [dir_fp, file_fp]

    def _index_cache_key(self) -> str:
        """SHA-256 hex digest over EVERY input that determines which candidates are valid.

        Any change to a keyed input MUST miss the cache (correctness is paramount). Keyed inputs:
        the sorted source PDB path list (the exact files enumerated), the holdout resname set, the
        window min/max, max_binder_length, max_sidechain_atoms, target_radius, and min_target_size
        (the full enumeration-filter set in ``_enumerate_candidates`` / ``_parse_source``), PLUS a
        cheap stat-based source fingerprint (``_source_stat_fingerprint``) so an in-place regeneration
        of the PDBs under the same paths misses the cache. A schema version prefix invalidates every
        cache if this code's enumeration semantics change.
        """
        import hashlib
        import json

        cfg = self._index_build_config()
        payload = {
            "schema": 3,  # bump if enumeration semantics change so old caches miss
            "pdb_files": [str(p) for p in self.pdb_files],
            "source_stat": self._source_stat_fingerprint(),
            "holdout_resnames": sorted(cfg["holdout_resnames"]),
            "window_size_min": cfg["window_size_min"],
            "window_size_max": cfg["window_size_max"],
            "min_target_size": cfg["min_target_size"],
            "max_binder_length": cfg["max_binder_length"],
            "max_sidechain_atoms": cfg["max_sidechain_atoms"],
            "target_radius": cfg["target_radius"],
        }
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(blob).hexdigest()

    def _index_cache_dir(self) -> "Path | None":
        """Writable cache directory, or None if caching is disabled / no dir is writable.

        ``ATOMWEAVER_INDEX_CACHE_DISABLE=1`` disables caching entirely. Otherwise use
        ``ATOMWEAVER_INDEX_CACHE_DIR`` (default ``~/.cache/atomweaver_ncaa_index``), falling back to
        ``/tmp/atomweaver_ncaa_index`` if the chosen dir can't be created/written.
        """
        import os as _os

        if _os.environ.get("ATOMWEAVER_INDEX_CACHE_DISABLE", "") not in ("", "0", "false", "False"):
            return None
        candidates = []
        env_dir = _os.environ.get("ATOMWEAVER_INDEX_CACHE_DIR")
        if env_dir:
            candidates.append(Path(env_dir))
        else:
            candidates.append(Path.home() / ".cache" / "atomweaver_ncaa_index")
        candidates.append(Path("/tmp") / "atomweaver_ncaa_index")
        for d in candidates:
            try:
                d.mkdir(parents=True, exist_ok=True)
                probe = d / ".write_probe"
                probe.write_text("ok")
                probe.unlink()
                return d
            except OSError:
                continue
        return None

    def _index_cache_path(self) -> "Path | None":
        d = self._index_cache_dir()
        if d is None:
            return None
        # A DIRECTORY of separate `.npy` files (not an `.npz` zip): `np.load(*.npy, mmap_mode="r")`
        # genuinely memory-maps each array. (`mmap_mode` is silently ignored for `.npz` because the
        # arrays live inside a zip, defeating the whole point.) The loaded candidate index then stays
        # on disk, paged in on demand and shared read-only across forked DataLoader workers -- the
        # proper host-RAM fix for the COW blowup.
        return d / f"valid_index_{self._index_cache_key()}"

    # Per-array `.npy` filenames inside the cache directory.
    _CACHE_ARRAY_NAMES = ("cand_data", "cand_offsets", "cand_status", "cand_n", "valid_indices", "key")

    def _quarantine_cache_dir(self, path: "Path") -> None:
        """Move a bad/corrupt cache dir aside so the next save can write cleanly.

        ``Path.replace`` (the atomic rename in ``_save_index_cache``) FAILS onto a non-empty dir, so
        a corrupt/partial cache at ``path`` would otherwise persist and force a rebuild every launch.
        Rename it to ``<path>.corrupt.<pid>.<ns>`` (best-effort; fall back to ``rmtree``) to clear the
        target. Never raises -- a failure here just means we retry quarantine on the next launch.
        """
        import os as _os
        import shutil as _shutil
        import time as _time

        try:
            if not path.exists():
                return
            dest = path.with_name(f"{path.name}.corrupt.{_os.getpid()}.{_time.time_ns()}")
            try:
                path.replace(dest)
            except OSError:
                # Couldn't rename (e.g. dest exists / cross-device): remove outright instead.
                _shutil.rmtree(path, ignore_errors=True)
        except OSError:
            pass

    def _load_index_cache(self) -> bool:
        """Try to load the valid index from disk (mmap'd). Returns True on a verified hit.

        The directory name embeds the cache key, but we ALSO re-verify the stored ``key`` array
        inside the directory (defence against hash collision / stale dir) before trusting it. On any
        mismatch we return False; on any DECODE/VERIFY ERROR (corrupt or partial cache) we ALSO
        quarantine the bad dir so the next save can write cleanly, then return False
        (the caller rebuilds).

        The candidate arrays are loaded with ``mmap_mode="r"`` so the big ``cand_data`` /
        ``cand_offsets`` buffers stay on disk and are paged in on demand -- shared read-only across
        forked workers (no per-worker copy), the proper host-RAM fix for the COW blowup.
        """
        import numpy as _np

        path = self._index_cache_path()
        if path is None or not path.is_dir():
            return False
        files = {name: path / f"{name}.npy" for name in self._CACHE_ARRAY_NAMES}
        # A partial/missing-file dir is corrupt: quarantine it so a later save isn't blocked.
        if any(not f.exists() for f in files.values()):
            self._quarantine_cache_dir(path)
            return False
        try:
            # key is tiny -- load fully (no mmap) and verify before touching the big arrays.
            key = bytes(_np.load(files["key"]).tolist()).decode()
            if key != self._index_cache_key():
                # Key mismatch is a STALE (not corrupt) dir for a different config; the directory
                # name embeds our key, so a mismatching key here means a hash collision / dir reuse.
                # Quarantine it so our save can claim this path.
                self._quarantine_cache_dir(path)
                return False
            arrays = {
                "cand_data": _np.load(files["cand_data"], mmap_mode="r"),
                "cand_offsets": _np.load(files["cand_offsets"], mmap_mode="r"),
                "cand_status": _np.load(files["cand_status"], mmap_mode="r"),
                "cand_n": _np.load(files["cand_n"]),  # 1 element, no mmap needed
            }
            self._source_candidates.load_cache_arrays(arrays)
            # valid_indices is tiny -- materialize as a plain Python list (its own small buffer).
            self._valid_indices = [int(i) for i in _np.load(files["valid_indices"]).tolist()]
        except Exception:
            # Any decode/verify failure => treat as a miss AND quarantine the bad dir.
            self._quarantine_cache_dir(path)
            return False
        return True

    def _save_index_cache(self) -> None:
        """Atomically persist the valid index as a directory of `.npy` arrays (tmp dir + rename).

        Serializes the CSR candidate arrays (``cand_data``/``cand_offsets``/``cand_status``/
        ``cand_n``) plus the small ``valid_indices`` and the cache ``key`` as separate uncompressed
        `.npy` files (so they're mmap'able on load). The whole directory is built under a per-pid
        temp name then atomically renamed into place. Best-effort: never raises. Requires the
        candidate store to be finalized.
        """
        import os as _os
        import shutil as _shutil

        import numpy as _np

        path = self._index_cache_path()
        if path is None:
            return
        if not self._source_candidates.finalized:
            return
        arrays = self._source_candidates.to_cache_arrays()
        arrays["valid_indices"] = _np.asarray(list(self._valid_indices or []), dtype=_np.int64)
        # Store the key as a uint8 byte buffer (allow_pickle=False safe; verified on load).
        arrays["key"] = _np.frombuffer(self._index_cache_key().encode(), dtype=_np.uint8)

        tmp = path.with_name(path.name + f".tmp.{_os.getpid()}")
        try:
            if tmp.exists():
                _shutil.rmtree(tmp, ignore_errors=True)
            tmp.mkdir(parents=True, exist_ok=True)
            for name, arr in arrays.items():
                _np.save(tmp / f"{name}.npy", _np.ascontiguousarray(arr))
            # Path.replace -> os.replace: atomic for a same-filesystem rename, BUT it fails onto a
            # non-empty dir. Two cases when `path` already exists:
            # (a) a concurrent writer already wrote VALID identical content -> our save is
            # redundant; tolerate the race and drop our tmp.
            # (b) a CORRUPT/partial dir squats at `path` -> it would block this and
            # every future save forever; quarantine it aside, then retry the rename once so a
            # good cache lands.
            try:
                tmp.replace(path)
            except OSError:
                if path.is_dir() and self._cache_dir_is_valid(path):
                    _shutil.rmtree(tmp, ignore_errors=True)  # (a) good cache already present
                else:
                    self._quarantine_cache_dir(path)  # (b) clear the squatter
                    try:
                        tmp.replace(path)
                    except OSError:
                        _shutil.rmtree(tmp, ignore_errors=True)
        except OSError:
            _shutil.rmtree(tmp, ignore_errors=True)

    def _cache_dir_is_valid(self, path: "Path") -> bool:
        """True iff ``path`` holds a complete, key-matching cache.

        Used to tell a winning concurrent writer (a) from a corrupt squatter (b) in
        ``_save_index_cache``. Cheap: existence of all array files + the tiny ``key`` array matches
        our key. Never raises.
        """
        import numpy as _np

        try:
            files = {name: path / f"{name}.npy" for name in self._CACHE_ARRAY_NAMES}
            if any(not f.exists() for f in files.values()):
                return False
            key = bytes(_np.load(files["key"]).tolist()).decode()
            return key == self._index_cache_key()
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Source-level cache: parse + enumerate all valid windows once.
    # ------------------------------------------------------------------

    def _parse_source(self, idx: int) -> dict | None:
        """Parse source PDB ``idx`` from disk and precompute every valid window candidate.

        Pure function of file content + config: no caching, no side effects (so a cache
        miss can safely re-parse and get the IDENTICAL result).

        A candidate is ``(window_size, ncaa_idx, start)`` such that the binder
        window contains the NCAA AND the radial-cropped target has at least
        ``min_target_size`` residues AND the window fits ``max_binder_length`` AND
        (when applicable) contains no residue whose sidechain heavy-atom count
        exceeds ``max_sidechain_atoms`` (would be silently truncated by
        ``residues_to_tensors``). Returns None for sources that are unparseable,
        have no NCAA, are too short, contain a holdout residue, or produce no
        valid candidate.
        """
        pdb_path = self.pdb_files[idx]
        try:
            chains = parse_pdb_atoms(pdb_path)
        except (OSError, ValueError, KeyError):
            return None
        if not chains:
            return None
        # source PDBs are single-chain; if multi-chain, use the longest chain.
        if len(chains) == 1:
            atoms = next(iter(chains.values()))
        else:
            atoms = max(chains.values(), key=lambda a: len({(x["res_id"], x.get("icode", " ")) for x in a}))

        try:
            residues = extract_residue_atoms(atoms)
        except (ValueError, KeyError):
            return None
        if not residues:
            return None

        ncaa_indices = [i for i, r in enumerate(residues) if r.get("res_name") not in STANDARD_AA]
        if not ncaa_indices:
            return None

        # Source-level holdout filter: every candidate window contains an NCAA, so
        # if any chain residue is a holdout the parent's downstream holdout check
        # would reject the file anyway -- drop the source entirely.
        if self.holdout_resnames:
            chain_names = {r.get("res_name") for r in residues}
            if chain_names & self.holdout_resnames:
                return None

        n_res = len(residues)
        if n_res < self.window_size_min + self.min_target_size:
            return None

        import numpy as _np

        ca_coords = _np.array(
            [r["backbone"].get("CA", (float("nan"), float("nan"), float("nan"))) for r in residues],
            dtype=float,
        )

        candidates = self._enumerate_candidates(n_res, ncaa_indices, ca_coords, residues)
        if not candidates:
            return None

        return {
            "residues": residues,
            "ncaa_indices": ncaa_indices,
            "ca_coords": ca_coords,
            "candidates": candidates,
        }

    def _load_source(self, idx: int) -> dict | None:
        """Return the heavy parse dict for source ``idx``, or None if unusable.

        The dict holds ``residues, ncaa_indices, ca_coords, candidates``.

        Memory-bounded: the heavy entry is held in a bounded LRU
        (``_source_cache``, capacity ``source_cache_maxsize``) while the
        lightweight candidate list is held forever in ``_source_candidates``. On
        a heavy-cache miss this re-parses from disk (deterministic) and inserts
        into the LRU, evicting the least-recently-used entry when over capacity.
        Negative results are recorded as ``None`` in ``_source_candidates`` only
        (cheap, persistent) -- never in the LRU.
        """
        # LRU hit: refresh recency and return.
        cached = self._source_cache.get(idx)
        if cached is not None:
            self._source_cache.move_to_end(idx)
            return cached

        # Known-bad source (persistent negative cache): skip re-parse.
        if idx in self._source_candidates and self._source_candidates[idx] is None:
            return None

        entry = self._parse_source(idx)
        if entry is None:
            self._source_candidates[idx] = None
            return None

        # Persist the lightweight candidates forever; cache the heavy entry in the LRU.
        self._source_candidates[idx] = entry["candidates"]
        self._source_cache[idx] = entry
        self._source_cache.move_to_end(idx)
        while len(self._source_cache) > self.source_cache_maxsize:
            self._source_cache.popitem(last=False)
        return entry

    def _get_candidates(self, idx: int) -> "list[tuple[int, int, int]] | None":
        """Return the lightweight candidate list for source ``idx`` (None if unusable).

        Uses the persistent ``_source_candidates`` map when present (no heavy parse
        retained); otherwise triggers ``_load_source`` once to populate it. Lets
        ``_build_valid_indices`` / index builders enumerate every source's candidates
        without ballooning the bounded heavy LRU -- only the last ``source_cache_maxsize``
        heavy parses stay resident, while ALL candidate lists persist.
        """
        if idx in self._source_candidates:
            return self._source_candidates[idx]
        # Not yet seen: parse once (populates _source_candidates as a side effect).
        entry = self._load_source(idx)
        return None if entry is None else entry["candidates"]

    def _enumerate_candidates(
        self,
        n_res: int,
        ncaa_indices: list[int],
        ca_coords: "np.ndarray",
        residues: list[dict],
    ) -> list[tuple[int, int, int]]:
        """Return every ``(window_size, ncaa_idx, start)`` whose binder window
        survives every window-dependent filter:

        - contains the NCAA at ``ncaa_idx``;
        - has a radial-cropped target with at least ``min_target_size`` residues;
        - has ``window_size <= max_binder_length`` (parent's post-window length
          check would otherwise reject the binder);
        - contains no residue with ``len(sidechain) > max_sidechain_atoms`` (those
          atoms would be silently dropped by ``residues_to_tensors`` and corrupt
          the GT). This unconditional slot-cap exclusion replaces mainline's
          ``filter_oversized`` flag -- on this branch the slot cap is always the
          tensor-shape invariant, not an opt-in.
        """
        if self.max_binder_length is not None:
            effective_ws_max = min(self.window_size_max, int(self.max_binder_length))
        else:
            effective_ws_max = self.window_size_max
        if effective_ws_max < self.window_size_min:
            return []

        import numpy as _np

        oversized = _np.fromiter(
            (len(r.get("sidechain", [])) > self.max_sidechain_atoms for r in residues),
            dtype=bool,
            count=n_res,
        )
        # Prefix sum over the oversized mask: a window [start:end) contains an oversized
        # residue iff oversized_prefix[end] - oversized_prefix[start] > 0. O(1) per window.
        oversized_prefix = _np.concatenate(([0], _np.cumsum(oversized.astype(_np.int64))))

        # ------------------------------------------------------------------
        # Vectorized radial-crop precompute (replaces 1 fresh O(B*T) crop PER candidate).

        # The per-candidate cost used to dominate dataset construction: for a single
        # source ~77 candidate windows each rebuilt a fresh binder_ca x target_ca distance
        # matrix via `_radial_crop_target_indices`. At 153k sources x 4 dataset instances
        # (train+eval, holdout-filter + _build_valid_indices) that O(N_src * 77 * B * T)
        # python/numpy work was the dataset-build "hang".

        # Instead compute the FULL N x N CA-CA within-radius boolean matrix ONCE per source.
        # For any window [start:end), the number of kept target residues is
        # (within[start:end, :].any(axis=0) & finite & not-in-window).sum()
        # which is a cheap slice + reduction -- no per-candidate matrix rebuild. The result
        # is IDENTICAL to looping `_radial_crop_target_indices` (same distance test, same
        # NaN handling, same "within radius of ANY binder CA" semantics).
        # ------------------------------------------------------------------
        within = None  # (N, N) bool: within[b, t] == ||CA_b - CA_t|| <= radius (both finite)
        finite = None  # (N,) bool: residue has a finite CA
        if self.target_radius > 0:
            finite = _np.isfinite(ca_coords).all(axis=-1)  # (N,)
            r2 = float(self.target_radius) ** 2
            diffs = ca_coords[:, None, :] - ca_coords[None, :, :]  # (N, N, 3)
            d2 = (diffs**2).sum(axis=-1)  # (N, N)
            within = (d2 <= r2) & finite[:, None] & finite[None, :]

        # The ONLY window invariant is "the window CONTAINS the NCAA" (plus the target-size /
        # slot-cap / length filters below). We keep EVERY containing window and sample uniformly
        # at pick time -- no central-band preference, no terminus fallback. Two reasons:
        # (1) Edge placements are legitimate training signal (NCAA with conditioning context on
        # only one side mirrors real chain-terminus interfaces); biasing them out narrows the
        # distribution for no payoff now that the anchor is carried by the design mask.
        # (2) The old centered-vs-fallback choice was a GLOBAL switch over all NCAAs/sizes: if any
        # central window existed it returned ONLY the centered subset, silently dropping a
        # second NCAA's valid (but non-central) windows from `candidates`, so
        # `_build_ncaa_position_index` never enumerated that NCAA. With no centering every
        # containing window is retained per NCAA, so every anchorable NCAA is enumerated.
        # Mainline keeps all containing windows and biases at K-draw time; on this branch the design
        # mask (not the window) carries the anchor, so we simply keep all of them and sample uniformly.
        candidates: list[tuple[int, int, int]] = []
        for window_size in range(self.window_size_min, effective_ws_max + 1):
            if window_size > n_res:
                break  # all larger sizes also won't fit
            if n_res - window_size < self.min_target_size:
                # Even the full rest-of-chain is too small for this (and every larger) window.
                break
            for ncaa_idx in ncaa_indices:
                lo = max(0, ncaa_idx - window_size + 1)
                hi = min(n_res - window_size, ncaa_idx)
                if hi < lo:
                    continue
                for start in range(lo, hi + 1):
                    end = start + window_size
                    # Reject windows containing any oversized residue (slot cap) -- O(1) via prefix sum.
                    if oversized_prefix[end] - oversized_prefix[start] > 0:
                        continue
                    if self.target_radius > 0:
                        # Kept targets = residues OUTSIDE the window that are within radius of ANY
                        # in-window CA AND have a finite CA. `within[start:end]` is (W, N); reducing
                        # over the binder axis gives per-target "within radius of any binder CA".
                        near_any = within[start:end].any(axis=0)  # (N,)
                        near_any[start:end] = False  # exclude in-window residues from the target set
                        n_keep = int(near_any.sum())  # finite already folded into `within`
                        if n_keep < self.min_target_size:
                            continue
                    candidates.append((window_size, ncaa_idx, start))
        return candidates

    # ------------------------------------------------------------------
    # Windowing: pick one precomputed candidate per call.
    # ------------------------------------------------------------------

    def _pick_candidate(self, candidates: list[tuple[int, int, int]], pdb_stem: str) -> tuple[int, int, int]:
        """Return one candidate per call. Hash-seeded from ``pdb_stem`` when
        ``deterministic_binder_k`` is set (stable window for val/test); random
        otherwise (fresh window every access). Every candidate is valid by
        construction, so this never fails for a non-empty list.
        """
        import random as _rand

        if self.deterministic_binder_k:
            import hashlib

            d = hashlib.sha256(pdb_stem.encode() + b"|cand").digest()
            choice = int.from_bytes(d[:8], "big") % len(candidates)
            return candidates[choice]
        return _rand.choice(candidates)

    def _pick_candidate_for_anchor(
        self, candidates: list[tuple[int, int, int]], anchor_pos: int, pdb_stem: str
    ) -> tuple[int, int, int]:
        """Pick a candidate window whose anchored NCAA is exactly ``anchor_pos`` (enumeration eval).

        Restricts the candidate pool to windows containing the NCAA at ``anchor_pos`` and defers
        to ``_pick_candidate`` for the determinism / uniform-sampling choice among them.

        Falls back to the unrestricted pick if ``anchor_pos`` has no candidate (shouldn't happen
        for a position emitted by ``_build_ncaa_position_index``, which only emits NCAA positions
        that survive the same window filters -- guard defensively).
        """
        matching = [c for c in candidates if c[1] == anchor_pos]
        if matching:
            return self._pick_candidate(matching, pdb_stem)
        return self._pick_candidate(candidates, pdb_stem)

    # ------------------------------------------------------------------
    # Override: per-call windowing in place of the pre-built 2-chain parse.
    # ------------------------------------------------------------------

    def _parse_pdb_cached(self, idx: int) -> tuple | None:
        """Slice a fresh ``(binder_window, radial-cropped target)`` pair from the
        cached source parse. Same 4-tuple return shape as the parent
        (``binder_chain_id, target_chain_id, binder_residues, target_residues``),
        so the inherited ``_load_pdb`` tensorizes + design-masks it identically.

        Returns shallow copies of the windowed residue dicts so that
        ``_maybe_add_pseudo_backbone``'s in-place backbone mutation (when
        ``pseudo_backbone`` is on) never corrupts the cached source residues.
        Returns None only when the source itself is unusable.
        """
        source = self._load_source(idx)
        if source is None:
            return None
        candidates: list[tuple[int, int, int]] = source["candidates"]

        pdb_path = self.pdb_files[idx]
        # ENUMERATION MODE (eval): `__getitem__` set `_forced_anchor` to a source-chain NCAA
        # position. Pick a candidate window that contains it (uniform among containing windows),
        # so the systematic per-NCAA eval anchors deterministically at that exact site.
        forced_anchor = self._forced_anchor
        self._forced_anchor = None
        if forced_anchor is not None:
            window_size, _ncaa_idx, start = self._pick_candidate_for_anchor(candidates, forced_anchor, pdb_path.stem)
        else:
            window_size, _ncaa_idx, start = self._pick_candidate(candidates, pdb_path.stem)
        end = start + window_size
        # Window-relative index of the anchored NCAA (the candidate's ncaa_idx, which the
        # design-mask reads back via `_ncaa_window_pos` to guarantee the NCAA is designed).
        self._last_ncaa_window_pos = _ncaa_idx - start

        residues: list[dict] = source["residues"]
        ca_coords = source["ca_coords"]
        n_res = len(residues)
        binder_residues = residues[start:end]
        target_candidate_idx = list(range(start)) + list(range(end, n_res))

        if self.target_radius > 0:
            binder_ca = ca_coords[start:end]
            target_ca = ca_coords[target_candidate_idx]
            keep = _radial_crop_target_indices(binder_ca, target_ca, self.target_radius)
            target_residues = [residues[target_candidate_idx[i]] for i in keep]
        else:
            target_residues = [residues[i] for i in target_candidate_idx]

        # Defensive shallow copies: the inherited _load_pdb may mutate
        # res["backbone"] in place via _maybe_add_pseudo_backbone. Tuples inside
        # are immutable, so copying the dict + list is sufficient to isolate this
        # call's mutations from the cached source.
        def _copy_residue(r: dict) -> dict:
            out = dict(r)
            if "backbone" in out:
                out["backbone"] = dict(out["backbone"])
            if "sidechain" in out:
                out["sidechain"] = list(out["sidechain"])
            return out

        binder_residues = [_copy_residue(r) for r in binder_residues]
        target_residues = [_copy_residue(r) for r in target_residues]

        # Chain IDs match the pre-built 2-chain convention so downstream consumers
        # see the same shape they always have.
        return ("n", "1", binder_residues, target_residues)

    # ------------------------------------------------------------------
    # Override: evict the parent's per-idx cache in non-deterministic mode so
    # every __getitem__ produces a fresh window. The parent caches _load_pdb
    # output in _cached_data keyed by idx; without eviction the first sampled
    # window would become permanent, defeating online windowing.
    # ------------------------------------------------------------------

    def _load_pdb(self, idx: int) -> dict | None:
        # `_last_ncaa_window_pos` is set as a side effect of `_parse_pdb_cached` (called inside
        # the parent `_load_pdb`). Stash it, then thread it into the returned dict so the parent's
        # `__getitem__` / `iter_valid` design-mask sampler reads it back as `forced_include`.
        self._last_ncaa_window_pos = None
        # ENUMERATION CACHE-REUSE FIX: the parent `_load_pdb` short-circuits to the cached dict
        # (`if idx in self._cached_data: return ...`) BEFORE it ever calls `_parse_pdb_cached`.
        # In deterministic_binder_k mode we DON'T evict afterwards (window stability), so a source
        # with multiple NCAAs would return the FIRST cached window for every enumerated anchor:
        # `_parse_pdb_cached` (which consumes `_forced_anchor` and sets `_last_ncaa_window_pos`)
        # never runs for the 2nd+ anchor, the forced design anchor is lost, and len==N sources
        # collapse to one repeated window. Evict here whenever an anchor is pending so the parent
        # re-parses THIS anchor's window with the correct `_ncaa_window_pos`.
        if self._forced_anchor is not None:
            self._cached_data.pop(idx, None)
        data = super()._load_pdb(idx)
        ncaa_window_pos = self._last_ncaa_window_pos
        if not self.deterministic_binder_k:
            self._cached_data.pop(idx, None)
        if data is not None and ncaa_window_pos is not None:
            # _load_pdb may return a cached dict (deterministic mode); copy so we never poison
            # the cache with a per-access key, and so different anchors per access stay isolated.
            data = dict(data)
            data["_ncaa_window_pos"] = ncaa_window_pos
        return data

    # ------------------------------------------------------------------
    # Override: validation pass. Every admitted source has >=1 valid candidate
    # (enforced in _load_source), so _load_pdb cannot return None for any index
    # in _valid_indices. We reuse the parent's scan but key it off _load_source
    # (pure function of file content + config) so the valid set is identical
    # across DDP ranks and train/eval twins.
    # ------------------------------------------------------------------

    def _enumerate_indices(self, indices: "list[int]") -> "list[tuple[int, list | None]]":
        """Parse + enumerate candidates for ``indices`` serially. Returns ``(idx, candidates)``
        pairs (``candidates is None`` for unusable sources). Standalone (no cache mutation) so it
        can run in a worker process; the parent merges the results into the caches.
        """
        out = []
        for i in indices:
            entry = self._parse_source(i)
            out.append((i, None if entry is None else entry["candidates"]))
        return out

    def _build_valid_indices(self) -> None:
        if self._valid_indices is not None:
            return
        # CHANGE 1 -- DISK CACHE: a verified hit skips the (~15 min) build entirely. The key
        # hashes every enumeration-relevant input, so any keyed change misses and rebuilds.
        if self._load_index_cache():
            return
        # SINGLE-BUILDER LOCK: on distributed all 14 ranks miss the (per-pod) cache at once and
        # would each parse all ~153k PDBs in parallel (14x NFS storm). With a SHARED writable cache
        # dir (ATOMWEAVER_INDEX_CACHE_DIR on NFS) we serialize via an flock'd lockfile: exactly one rank
        # builds + saves; the rest block on the lock, then load the freshly-written cache. With a
        # per-pod / unwritable cache (no shared dir, lock can't be taken) we fall back to every rank
        # building independently -- still correct, just not deduplicated.
        if self._build_valid_indices_under_lock():
            return
        self._run_index_build()

    def _build_valid_indices_under_lock(self) -> bool:
        """Try the single-builder path under an ``flock``'d lockfile in the cache dir.

        Returns True iff this call fully populated the index (either it built+saved while holding the
        lock, or it loaded a cache that the lock holder wrote). Returns False if no lock could be
        taken (no writable cache dir, or ``fcntl`` unavailable) so the caller does an independent
        build.

        CORRECTNESS UNDER SPAWN/RAY: the lock is an OS-level ``fcntl.flock`` on a real fd, held only
        for the duration of THIS process's build -- it is NOT inherited across the spawn boundary
        (the index-build spawn pool re-execs fresh interpreters that never touch this fd) and it is
        released when the fd is closed (explicit close + automatic on process exit), so a crashing
        builder never deadlocks the waiters. Every rank is a SEPARATE OS process (torchrun/Ray Train
        ranks are processes, not threads), so flock genuinely arbitrates between them. After we win
        the lock we RE-CHECK the cache: if a previous holder already wrote it we just load (no
        duplicate build); waiters that blocked on the lock fall into this same re-check and load.
        """
        try:
            import fcntl as _fcntl
        except ImportError:
            return False  # non-POSIX: no flock, every rank builds independently
        cache_dir = self._index_cache_dir()
        if cache_dir is None:
            return False  # no writable cache dir -> can't coordinate; independent build
        # One lock PER cache key (distinct configs build distinct caches concurrently, no contention).
        lock_path = cache_dir / f".build.{self._index_cache_key()}.lock"
        try:
            lock_fd = open(lock_path, "w")  # noqa: SIM115 -- held across the build, closed in finally
        except OSError:
            return False
        try:
            _fcntl.flock(lock_fd.fileno(), _fcntl.LOCK_EX)  # blocks until we own it
            # RE-CHECK under the lock: a prior holder may have already written the cache while we
            # waited -- load it instead of rebuilding (the whole point of the single-builder path).
            if self._load_index_cache():
                return True
            # We are THE builder: parse, finalize, and save (save persists for the waiters behind us).
            self._run_index_build()
            return True
        except OSError:
            return False  # flock failed (e.g. NFS without lock support): fall back to independent build
        finally:
            import contextlib as _contextlib

            with _contextlib.suppress(OSError):
                _fcntl.flock(lock_fd.fileno(), _fcntl.LOCK_UN)
            lock_fd.close()

    def _run_index_build(self) -> None:
        """Parse+enumerate every source, finalize the CSR store, set ``_valid_indices``, and save.

        The validity scan must parse + enumerate every one of the (up to ~153k) source PDBs once.
        ``_parse_source`` is a PURE function of file content + config, so the scan is embarrassingly
        parallel. A single serial pass over 153k large-protein PDBs took ~15+ min on one core (the
        dataset-build "hang" at full monomer scale); fan the parse out across processes and merge the
        lightweight candidate lists back. Only the cheap candidate lists cross the process boundary
        (never the heavy residues/coords), so memory stays bounded.
        """
        n = len(self.pdb_files)
        n_workers = self._index_build_workers()
        if n_workers <= 1 or n < 2000:
            # Small pool or workers disabled: serial path (also the test fallback).
            results = self._enumerate_indices(list(range(n)))
        else:
            results = self._build_valid_indices_parallel(n, n_workers)

        # FINALIZE the candidate index into the COW-safe numpy CSR store. `results` covers EVERY
        # source in range(n), so the packed arrays are complete (status != -1 for all). This is the
        # host-RAM fix: after this the per-source candidates live in 3 shared C buffers, not a
        # 153k-entry Python dict/list/tuple graph that forked workers would each COW-duplicate.
        items = {int(i): cands for i, cands in results}
        self._source_candidates.finalize_from(items, n)
        self._valid_indices = sorted(i for i, cands in results if cands is not None)
        # Persist for the next launch (atomic tmp+rename; best-effort, never raises).
        self._save_index_cache()

    def _build_valid_indices_parallel(self, n: int, n_workers: int) -> "list[tuple[int, list | None]]":
        """CHANGE 2 -- SPAWN-CONTEXT parallel build (cold/cache-miss path).

        Uses a SPAWN-context ``ProcessPoolExecutor`` (``mp_context=get_context("spawn")``) so the
        cold build runs PARALLEL even under torchrun/Ray: spawn workers re-import the module and do
        NOT inherit the parent's CUDA context, so there is no fork-after-CUDA-init deadlock (the
        reason the old build was forced serial). Spawn workers also don't share parent host RAM via
        copy-on-write, so this additionally avoids the COW host-RAM blowup of a forked pool.

        The worker is the module-level (picklable) ``_index_build_worker``; each task ships a small
        ``(cfg, pdb_paths)`` tuple (the parse config + that chunk's PDB paths) -- no heavy dataset
        state crosses the process boundary. On any pool failure we fall back to the serial path.
        """
        import multiprocessing as _mp
        from concurrent.futures import ProcessPoolExecutor

        cfg = self._index_build_config()
        # Chunk the index space so each task amortizes process startup + IPC over many sources.
        chunk = max(1, (n + n_workers * 4 - 1) // (n_workers * 4))
        chunk_bounds = [(s, min(s + chunk, n)) for s in range(0, n, chunk)]
        tasks = [(cfg, self.pdb_files[s:e]) for s, e in chunk_bounds]
        try:
            ctx = _mp.get_context("spawn")
            results: list[tuple[int, list | None]] = []
            with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as ex:
                # Worker returns chunk-LOCAL indices; remap to global via the chunk's start offset.
                for (s, _e), part in zip(chunk_bounds, ex.map(_index_build_worker, tasks), strict=False):
                    for local_i, cands in part:
                        results.append((s + local_i, cands))
            return results
        except Exception:
            return self._enumerate_indices(list(range(n)))

    def _index_build_workers(self) -> int:
        """Number of SPAWN processes for the parallel valid-index parse (env-overridable).

        ``ATOMWEAVER_INDEX_BUILD_WORKERS`` overrides; default is a bounded ``min(8, cpu-1)`` to keep
        the NFS/CPU load bounded (parsing is I/O + Python-bound, so oversubscription is unhelpful).
        Set to 0/1 to force the serial path (e.g. inside a dataloader worker, where nested process
        pools are unsafe).

        Spawn makes the pool fork-SAFE: a spawn worker re-execs a fresh interpreter and does NOT
        inherit the parent's CUDA context, so the old CUDA-fork deadlock no longer applies. We
        therefore ALLOW parallel even with a live CUDA context and under torchrun/Ray Train. We
        only cap the worker count (we do NOT serialize per rank): each rank's bounded ``min(8,
        cpu-1)`` pool is short-lived and the disk cache means only the FIRST cold launch pays it at
        all -- subsequent launches hit the cache and spawn no pool.
        """
        import os as _os

        env = _os.environ.get("ATOMWEAVER_INDEX_BUILD_WORKERS")
        if env is not None:
            try:
                return max(1, int(env))
            except ValueError:
                pass
        cpu = _os.cpu_count() or 1
        return max(1, min(8, cpu - 1))

    # ------------------------------------------------------------------
    # Enumeration mode (eval): one sample per (PDB, NCAA position).
    # ------------------------------------------------------------------

    def _build_ncaa_position_index(self) -> None:
        """Populate ``_ncaa_position_index`` for enumeration mode (one tuple per scoreable NCAA).

        Emits one ``(pdb_idx, ncaa_pos)`` tuple per scoreable NCAA residue across every admitted
        source PDB. Two filters drop positions the discretizer DB can't score (they'd pollute the
        C-bucket denominator):

        NAME FILTER: ``res_name`` in ``enumeration_exclude_resnames`` (operator-supplied,
        kept in sync with the eval DB; catches residues whose CCD ideal exceeds the slot cap
        but whose parsed PDB form would pass the size filter).

        SIZE FILTER: parsed sidechain heavy-atom count > ``max_sidechain_atoms``.

        Adapted from mainline: same two-filter logic, but anchored at a source-chain position
        (consumed by ``_pick_candidate_for_anchor`` to pick the window) rather than a pre-built
        binder position. Walks ``_valid_indices`` (sources with >=1 usable window). Distinct
        from ``_valid_indices`` (one entry per PDB) -- this carries one entry per (PDB, NCAA).
        """
        if self._ncaa_position_index is not None:
            return
        if not self.enumerate_ncaa_positions:
            return
        self._build_valid_indices()
        self._ncaa_position_index = []
        skipped_oversized: dict[str, int] = {}
        skipped_excluded_name: dict[str, int] = {}
        for pdb_idx in self._valid_indices:
            source = self._load_source(pdb_idx)
            if source is None:
                continue
            residues = source["residues"]
            # Restrict to the NCAA positions the windower would actually anchor (post-holdout /
            # post-oversize pruning already applied in `_load_source`/`_enumerate_candidates`).
            anchorable = {c[1] for c in source["candidates"]}
            for pos in sorted(anchorable):
                res = residues[pos]
                rn = res.get("res_name")
                if rn in STANDARD_AA:
                    continue
                if rn in self.enumeration_exclude_resnames:
                    skipped_excluded_name[rn] = skipped_excluded_name.get(rn, 0) + 1
                    continue
                n_sc = len(res.get("sidechain", []))
                if n_sc > self.max_sidechain_atoms:
                    skipped_oversized[rn] = skipped_oversized.get(rn, 0) + 1
                    continue
                self._ncaa_position_index.append((pdb_idx, pos))
        if skipped_oversized or skipped_excluded_name:
            import sys

            if skipped_excluded_name:
                details = ", ".join(f"{rn}x{c}" for rn, c in sorted(skipped_excluded_name.items()))
                print(
                    f"  [enumerate_ncaa_positions] skipped {sum(skipped_excluded_name.values())} "
                    f"NCAA position(s) by name (enumeration_exclude_resnames; "
                    f"DB has no reference for these): {details}",
                    file=sys.stderr,
                )
            if skipped_oversized:
                details = ", ".join(f"{rn}x{c}" for rn, c in sorted(skipped_oversized.items()))
                print(
                    f"  [enumerate_ncaa_positions] skipped {sum(skipped_oversized.values())} "
                    f"NCAA position(s) with sidechain heavy-atom count > "
                    f"max_sidechain_atoms={self.max_sidechain_atoms} "
                    f"(unrepresentable in the discretizer DB): {details}",
                    file=sys.stderr,
                )

    def __len__(self) -> int:
        """Number of samples -- switches on ``enumerate_ncaa_positions``.

        Default: number of valid sources (one window per source). Enumeration mode: total
        scoreable NCAA positions across every admitted source. Both build their index on first
        call (so the count is the REAL valid count, never an estimate -- see HIGH-3).
        """
        if self.enumerate_ncaa_positions:
            self._build_ncaa_position_index()
            return len(self._ncaa_position_index or [])
        self._build_valid_indices()
        return len(self._valid_indices)

    def _design_mask_anchor_kwargs(
        self, ncaa_window_pos: int | None, interface_positions: "list[int] | None" = None
    ) -> dict:
        """Weighted NCAA design-anchor kwargs for the TRAINING path.

        Up-weights (does NOT force) the stashed window-relative NCAA index in the parent's
        ``_sample_design_mask`` weighted-subset draw, via ``weighted_include`` /
        ``weighted_include_weight=ncaa_design_anchor_weight`` (default 4.0). The K budget is preserved
        exactly; only WHICH K sites are chosen is biased. The NCAA's inclusion probability is
        ``w / (w + L - 1)`` at K=1 and 1.0 at K=L. When it is not picked it stays as clean pinned
        conditioning context (bidirectional objective).

        Only reached via the inherited (training) ``__getitem__`` / ``iter_valid``. The enumeration
        eval path builds its own design mask and HARD-FORCES the anchor (prob 1.0) through a separate
        ``forced_include`` path, bypassing this weighted bias entirely.

        ``interface_positions`` is accepted for base-signature compatibility but IGNORED here: the
        monomer NCAA path anchors on the NCAA site (interface anchoring is not wired to this dataset,
        and ``_load_pdb`` here never computes interface_positions).
        """
        if ncaa_window_pos is None:
            return {}
        return {
            "weighted_include": ncaa_window_pos,
            "weighted_include_weight": self.ncaa_design_anchor_weight,
        }

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Get one sample (enumeration mode anchors deterministically at the enumerated NCAA).

        Enumeration mode pins ``_forced_anchor`` to the enumerated NCAA source position so the
        window + design mask deterministically anchor at that site, and HARD-FORCES the NCAA into
        the designed set (prob 1.0) via ``forced_include`` -- a SEPARATE path from the training
        weighted bias, so systematic per-NCAA eval is unaffected by ``ncaa_design_anchor_weight``.

        Training path (``super().__getitem__``): the NCAA is UP-WEIGHTED (not forced) in the design-
        mask draw by ``ncaa_design_anchor_weight`` (default 4.0); whether it lands in the designed set
        depends on K, L, and the weight. With K-masking OFF (mix_prob=None) the whole window is always
        designed (NCAA included) regardless of the weight. To reproduce mainline's strict K=1 single-
        site NCAA eval, build this dataset in enumeration mode with ``random_binder_k_mix_prob=0.0``
        and ``random_binder_k_min=1`` so each partial design is K=1 pinned at the NCAA.
        """
        if self.enumerate_ncaa_positions:
            self._build_ncaa_position_index()
            pdb_idx, ncaa_pos = self._ncaa_position_index[idx]
            self._forced_anchor = ncaa_pos
            data = self._load_pdb(pdb_idx)
            if data is None:
                return data
            data = dict(data)
            # ENUMERATION: HARD-FORCE the anchor (prob 1.0) via forced_include -- distinct from the
            # training weighted_include bias so per-NCAA eval is unaffected by the training weight.
            forced = data.pop("_ncaa_window_pos", None)
            # The designable-site gate outranks forced_include. forced_include only decides WHICH
            # sites the K-budget covers; it is not a licence to evaluate recovery against ground
            # truth we have declared untrustworthy. So an enumerated NCAA whose own deposited
            # composition contradicts its label is gated out, and this sample yields an all-False
            # design mask rather than a scored one -- the honest outcome, but a silent one, hence
            # the warning.
            design_mask = apply_designable_gate(
                self._sample_design_mask(data["backbone_coords"].shape[0], forced_include=forced),
                data.get("designable_mask"),
            )
            if forced is not None and not bool(design_mask.any()):
                # Counted alongside the training path's emptied masks (see _gated_design_mask)
                # so one instance reports one total. Warned per sample rather than with backoff:
                # eval volume is small and each skipped target changes a reported leg-C/D number.
                self._emptied_design_mask_count += 1
                _LOG.warning(
                    "Enumerated NCAA at window position %s in %s is not designable (deposited "
                    "composition contradicts its label); design mask is empty for this sample.",
                    forced,
                    data.get("pdb_name", "<unknown>"),
                )
            data["design_mask"] = design_mask
            return data
        return super().__getitem__(idx)


# Bind the dataset's UNBOUND parse/enumerate methods onto the picklable spawn-worker shim
# (forward reference resolved now that the class exists). The shim therefore executes the exact
# same enumeration code as the in-process serial path -- guaranteeing parallel == serial results.
_IndexBuildShim._parse_source = OnTheFlyNcaaDataset._parse_source
_IndexBuildShim._enumerate_candidates = OnTheFlyNcaaDataset._enumerate_candidates


def collate_binders(
    batch: list[dict],
    name_to_idx: dict[str, int] | None = None,
    no_target: bool = False,
    shift_target: bool = False,
) -> dict[str, torch.Tensor]:
    """
    Collate function for batching binder-target complexes.

    Pads all sequences to the maximum length in the batch.

    Parameters
    ----------
    batch : list[dict]
        List of binder-target complex data dictionaries.
    name_to_idx : dict[str, int], optional
        Mapping from 3-letter residue codes to database indices.
        If provided, residue_indices will be included in output.
    no_target : bool
        Baseline A: Mask out all target atoms (test backbone-only conditioning).
    shift_target : bool
        Baseline B: Shift target rigidly by random vector (test memorization vs. interactions).

    Returns
    -------
    collated : dict[str, torch.Tensor]
        Batched tensors with padding for both binder and target.
    """
    max_binder_len = max(b["backbone_coords"].shape[0] for b in batch)
    max_sc = batch[0]["sidechain_coords"].shape[1]
    batch_size = len(batch)

    # Check if target data is present
    has_target = "target_backbone_coords" in batch[0]
    if has_target:
        max_target_len = max(b["target_backbone_coords"].shape[0] for b in batch)

    # Initialize padded tensors for binder
    backbone_coords = torch.zeros(batch_size, max_binder_len, 4, 3)
    backbone_mask = torch.zeros(batch_size, max_binder_len, 4, dtype=torch.bool)
    sidechain_coords = torch.zeros(batch_size, max_binder_len, max_sc, 3)
    sidechain_mask = torch.zeros(batch_size, max_binder_len, max_sc, dtype=torch.bool)
    sidechain_element_types = torch.full((batch_size, max_binder_len, max_sc), 0, dtype=torch.long)  # 0 = PAD
    seq_mask = torch.zeros(batch_size, max_binder_len, dtype=torch.bool)
    # design_mask: which binder residues are being DESIGNED (vs GT-pinned binder context).
    # Defaults to True at real positions (= design-all) if an item lacks it (no K-masking).
    # That default is itself a generated design mask, so it goes through apply_designable_gate
    # like every other one -- otherwise a dataset that never sets design_mask would design
    # untrustworthy sites by omission, which is exactly the hole this MR is closing.
    design_mask = torch.zeros(batch_size, max_binder_len, dtype=torch.bool)
    # designable_mask is carried through to the batch (padded False, i.e. "not designable", so
    # padding can never be mistaken for a designable site) so downstream eval harnesses that
    # build their OWN design masks from the batch can apply the same gate. Absent from the batch
    # when no item supplies one.
    has_designable = any(b.get("designable_mask") is not None for b in batch)
    designable_mask = torch.zeros(batch_size, max_binder_len, dtype=torch.bool) if has_designable else None
    # bei_env (OPTIONAL): per-residue B/E/I env code (0=Interface, 1=Buried, 2=Exposed) for the per-env
    # FCC slopes. Only present when the dataset was built with fcc_per_env; absent for every other dataset
    # (and the model then falls back to the scalar fcc_slope). `any(...)` so a mixed ConcatDataset batch
    # still carries it if ANY item has it. Padding + any item lacking bei_env stay at the -1 "no env /
    # missing" sentinel (NOT Exposed) -- the model reads -1 rows with the scalar fcc_slope, so a batch
    # source that doesn't provide bei_env can never silently inherit the Exposed slope.
    has_bei_env = any("bei_env" in b for b in batch)
    if has_bei_env:
        bei_env = torch.full((batch_size, max_binder_len), -1, dtype=torch.long)  # -1 = no env / missing

    # Residue indices for discretization loss
    if name_to_idx is not None:
        residue_indices = torch.zeros(batch_size, max_binder_len, dtype=torch.long)
        # True only where the residue is an actual disc-DB class. Out-of-DB residues
        # (no name_to_idx key) fall back to index 0 above, so this mask lets the disc
        # loss skip them rather than mis-train them as residue-0. Coord/element losses
        # still see them via seq_mask (real geometry is useful to learn from).
        residue_in_disc_db = torch.zeros(batch_size, max_binder_len, dtype=torch.bool)

    # Initialize padded tensors for target
    if has_target:
        target_backbone_coords = torch.zeros(batch_size, max_target_len, 4, 3)
        target_backbone_mask = torch.zeros(batch_size, max_target_len, 4, dtype=torch.bool)
        target_seq_mask = torch.zeros(batch_size, max_target_len, dtype=torch.bool)
        target_ca_coords = torch.zeros(batch_size, max_target_len, 3)
        # Residue type indices for target (for cross-attention conditioning)
        target_residue_types = torch.zeros(batch_size, max_target_len, dtype=torch.long)

        # Target sidechain coords/mask (for PDB saving)
        has_tgt_sc = batch[0].get("target_sidechain_coords") is not None
        max_target_sc = max(b["target_sidechain_coords"].shape[1] for b in batch) if has_tgt_sc else 0
        if max_target_sc > 0:
            target_sidechain_coords = torch.zeros(batch_size, max_target_len, max_target_sc, 3)
            target_sidechain_mask = torch.zeros(batch_size, max_target_len, max_target_sc, dtype=torch.bool)

        # For flattened target atoms, find max number of atoms
        max_target_atoms = max(b["target_coords"].shape[0] for b in batch)
        target_coords = torch.zeros(batch_size, max_target_atoms, 3)
        target_mask = torch.zeros(batch_size, max_target_atoms, dtype=torch.bool)
        # Per-atom features for SE(3) transformer
        target_atom_residue_idx = torch.zeros(batch_size, max_target_atoms, dtype=torch.long)
        target_atom_type = torch.zeros(batch_size, max_target_atoms, dtype=torch.long)
        target_atom_element_type = torch.full((batch_size, max_target_atoms), -1, dtype=torch.long)
        target_atom_residue_type = torch.full((batch_size, max_target_atoms), -1, dtype=torch.long)
        target_atom_is_backbone = torch.zeros(batch_size, max_target_atoms, dtype=torch.bool)

    res_names = []
    target_res_names = []
    pdb_names = []

    for i, b in enumerate(batch):
        # Binder data
        binder_len = b["backbone_coords"].shape[0]
        backbone_coords[i, :binder_len] = b["backbone_coords"]
        backbone_mask[i, :binder_len] = b["backbone_mask"]
        sidechain_coords[i, :binder_len] = b["sidechain_coords"]
        sidechain_mask[i, :binder_len] = b["sidechain_mask"]
        if "sidechain_element_types" in b:
            sidechain_element_types[i, :binder_len] = b["sidechain_element_types"]
        seq_mask[i, :binder_len] = True
        _dsg = b.get("designable_mask")
        if designable_mask is not None:
            if _dsg is None:
                # An item with no designable_mask has no opinion -> all real positions designable.
                designable_mask[i, :binder_len] = True
            else:
                _dn = min(binder_len, _dsg.shape[0])
                designable_mask[i, :_dn] = _dsg[:_dn]
                designable_mask[i, _dn:binder_len] = True  # unclaimed tail: no opinion
        # Normalise the item's design mask to `binder_len` FIRST. A hand-built item may supply a
        # shorter or longer one; the unclaimed tail stays False (design nothing we were not told
        # about), which is what the zero-initialised row already gave. Normalising here rather
        # than after the gate is what lets `apply_designable_gate` demand equal lengths: it now
        # receives two masks that are both exactly `binder_len`, so a genuine misalignment
        # upstream raises there instead of silently gating a prefix.
        _dm_in = b.get("design_mask")
        _dm = torch.ones(binder_len, dtype=torch.bool)
        if _dm_in is not None:
            _mn = min(binder_len, _dm_in.shape[0])
            _dm[:_mn] = _dm_in[:_mn].bool()
            _dm[_mn:] = False
        # Gate against the row we just normalised above (item mask over the claimed prefix, "no
        # opinion" over the tail), not the raw item mask, for the same reason.
        # Idempotent when the item came from __getitem__ (already gated); load-bearing for the
        # all-True default above and for any item whose producer skipped the gate.
        _dm = apply_designable_gate(_dm, None if designable_mask is None else designable_mask[i, :binder_len])
        design_mask[i, :binder_len] = _dm
        if has_bei_env and b.get("bei_env") is not None:
            bei_env[i, :binder_len] = b["bei_env"]
        res_names.append(b["res_names"])
        pdb_names.append(b.get("pdb_name", ""))

        # Convert res_names to indices for discretization loss
        if name_to_idx is not None:
            for j, name in enumerate(b["res_names"]):
                residue_indices[i, j] = name_to_idx.get(name, 0)
                residue_in_disc_db[i, j] = name in name_to_idx
            # A residue whose deposited composition contradicts its label is a bad matching
            # exemplar even when it is still worth designing, so gate it out of the disc loss
            # here rather than at the design mask.
            _exact = b.get("exact_composition_mask")
            if _exact is not None:
                _n = min(binder_len, _exact.shape[0])
                residue_in_disc_db[i, :_n] &= _exact[:_n]

        # Target data
        if has_target:
            target_len = b["target_backbone_coords"].shape[0]
            target_backbone_coords[i, :target_len] = b["target_backbone_coords"]
            target_backbone_mask[i, :target_len] = b["target_backbone_mask"]
            target_seq_mask[i, :target_len] = True
            target_ca_coords[i, :target_len] = b["target_ca_coords"]
            target_res_names.append(b["target_res_names"])
            if max_target_sc > 0 and "target_sidechain_coords" in b:
                sc_slots = b["target_sidechain_coords"].shape[1]
                target_sidechain_coords[i, :target_len, :sc_slots] = b["target_sidechain_coords"]
                target_sidechain_mask[i, :target_len, :sc_slots] = b["target_sidechain_mask"]

            # Convert target res_names to residue type indices
            # Use standard amino acid mapping (same as RESIDUE_TO_IDX in models.py)
            from atomweaver.joint_diffusion.models import RESIDUE_TO_IDX

            for j, name in enumerate(b["target_res_names"]):
                target_residue_types[i, j] = RESIDUE_TO_IDX.get(name, RESIDUE_TO_IDX["UNK"])

            # Flattened target atoms with per-atom features
            n_target_atoms = b["target_coords"].shape[0]
            target_coords[i, :n_target_atoms] = b["target_coords"]
            target_mask[i, :n_target_atoms] = b["target_mask"]
            if "target_atom_residue_idx" in b:
                target_atom_residue_idx[i, :n_target_atoms] = b["target_atom_residue_idx"]
                target_atom_type[i, :n_target_atoms] = b["target_atom_type"]
                target_atom_element_type[i, :n_target_atoms] = b["target_atom_element_type"]
                target_atom_residue_type[i, :n_target_atoms] = b["target_atom_residue_type"]
                target_atom_is_backbone[i, :n_target_atoms] = b["target_atom_is_backbone"]

    result = {
        "backbone_coords": backbone_coords,
        "backbone_mask": backbone_mask,
        "sidechain_coords": sidechain_coords,
        "sidechain_mask": sidechain_mask,
        "sidechain_element_types": sidechain_element_types,
        "seq_mask": seq_mask,
        "design_mask": design_mask,
        **({"designable_mask": designable_mask} if designable_mask is not None else {}),
        "res_names": res_names,
        "pdb_names": pdb_names,
    }

    if has_bei_env:
        result["bei_env"] = bei_env

    if name_to_idx is not None:
        result["residue_indices"] = residue_indices
        result["residue_in_disc_db"] = residue_in_disc_db

    if has_target:
        # Baseline A: Mask out all target atoms (set masks to False)
        if no_target:
            target_mask = torch.zeros_like(target_mask)
            target_backbone_mask = torch.zeros_like(target_backbone_mask)
            target_seq_mask = torch.zeros_like(target_seq_mask)

        # Baseline B: Shift target coordinates rigidly by random vector
        if shift_target:
            # Generate random shift vector: uniform in range [-15, 15] Angstroms per axis
            # This ensures target is far enough to break any memorized interactions
            shift_vector = torch.rand(3) * 30.0 - 15.0  # [-15, 15] for each axis
            target_backbone_coords = target_backbone_coords + shift_vector
            target_ca_coords = target_ca_coords + shift_vector
            target_coords = target_coords + shift_vector

        result["target_backbone_coords"] = target_backbone_coords
        result["target_backbone_mask"] = target_backbone_mask
        result["target_seq_mask"] = target_seq_mask
        result["target_ca_coords"] = target_ca_coords
        result["target_res_names"] = target_res_names
        result["target_residue_types"] = target_residue_types
        # Flattened target atoms for EGNN graph
        result["target_coords"] = target_coords
        result["target_mask"] = target_mask
        # Per-atom features for SE(3) transformer
        result["target_atom_residue_idx"] = target_atom_residue_idx
        result["target_atom_type"] = target_atom_type
        result["target_atom_element_type"] = target_atom_element_type
        result["target_atom_residue_type"] = target_atom_residue_type
        result["target_atom_is_backbone"] = target_atom_is_backbone
        if max_target_sc > 0:
            result["target_sidechain_coords"] = target_sidechain_coords
            result["target_sidechain_mask"] = target_sidechain_mask

    return result


# Backwards compatibility alias
collate_peptides = collate_binders
