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
import warnings
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch.utils.data import Dataset

from .diffusion import NUM_ELEMENT_TYPES  # vocab knob single source of truth (validated in diffusion.py)

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy as np  # for "np.ndarray" string annotations in the on-the-fly NCAA windowing helpers


# Candidate tuple width: every per-source window candidate is (window_size, ncaa_idx, start).
_CAND_FIELDS = 3


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
            element = element_col or infer_element_from_atom_name(atom_name)

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


class BinderDataset(Dataset):
    """Validated peptide-target structures for inference, cached after parsing."""

    def __init__(
        self,
        pdb_dir: str | Path,
        max_sidechain_atoms: int = 14,
        max_binder_length: int | None = 32,
        max_target_length: int | None = 256,
        file_list: Sequence[str] | None = None,
        max_files: int | None = None,
        min_sidechain_atoms: int = 1,
        reserved_slot0: bool = False,
        proximity_target_crop: bool = True,
    ):
        self.pdb_dir = Path(pdb_dir)
        self.max_sidechain_atoms = max_sidechain_atoms
        self.max_binder_length = max_binder_length
        self.max_target_length = max_target_length
        self.min_sidechain_atoms = min_sidechain_atoms
        self.reserved_slot0 = reserved_slot0
        self.proximity_target_crop = proximity_target_crop
        self.pdb_files = (
            [self.pdb_dir / name for name in file_list] if file_list is not None else sorted(self.pdb_dir.glob("*.pdb"))
        )
        if max_files is not None:
            self.pdb_files = self.pdb_files[:max_files]
        self._cached_data: dict[int, dict] = {}
        self._valid_indices = [i for i in range(len(self.pdb_files)) if self._load_pdb(i) is not None]

    def __len__(self) -> int:
        return len(self._valid_indices)

    def __getitem__(self, idx: int) -> dict:
        data = dict(self._cached_data[self._valid_indices[idx]])
        design = torch.ones(data["backbone_coords"].shape[0], dtype=torch.bool)
        data["design_mask"] = apply_designable_gate(design, data.get("designable_mask"))
        return data

    def _parse_pdb_cached(self, idx: int) -> tuple | None:
        """Parse the peptide and target chains of one input PDB."""
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

            if self.reserved_slot0:
                for residue in binder_residues:
                    _apply_reserved_slot0(residue)

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

            self._cached_data[idx] = data
            return data

        except (OSError, ValueError, KeyError):
            return None


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

    # `_parse_source` / `_enumerate_candidates` are bound to the dataset's UNBOUND methods
    # just after the class definition (forward reference), so the shim runs the exact same
    # enumeration code as the in-process serial path -- no logic duplication, no drift.


# Bind the dataset's UNBOUND parse/enumerate methods onto the picklable spawn-worker shim
# (forward reference resolved now that the class exists). The shim therefore executes the exact
# same enumeration code as the in-process serial path -- guaranteeing parallel == serial results.


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
