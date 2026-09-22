"""
Data loading utilities for residue databases.

Provides functions to load pre-built residue databases for use with
ResidueDatabaseMatcher and other US-align components.
"""

from __future__ import annotations

import torch


def _chirality_from_reference(
    coords: torch.Tensor,
    mask: torch.Tensor,
    backbone_indices: dict[str, int],
) -> int | None:
    """Geometric L/D chirality sign for a single residue reference structure.

    Identifies the C-beta **geometrically** (the side-chain atom ~1.0-2.0 A from
    CA, off the backbone) rather than assuming a fixed atom slot, because for
    many NCAAs the C-beta is not at slot 0 (see the slot-misalignment finding).
    The sign is the handedness of the scalar triple product
    ``dot(cross(N-CA, C-CA), Cb-CA)``: +1 for L-amino acids, -1 for D.

    Parameters
    ----------
    coords : torch.Tensor
        Reference atom coordinates of shape ``(num_atoms, 3)``.
    mask : torch.Tensor
        Boolean validity mask of shape ``(num_atoms,)``.
    backbone_indices : dict[str, int]
        Atom-axis indices for the ``N``/``CA``/``C``/``O`` backbone atoms.

    Returns
    -------
    int or None
        ``+1`` (L) or ``-1`` (D); ``None`` when chirality is undefined -- no
        C-beta (e.g. glycine), an alpha,alpha-disubstituted CA (e.g. AIB/DIV)
        where the side is ambiguous, or a non-alpha backbone (e.g. the
        beta/gamma-amino acids BAL/B3L/B3Q, whose labelled CA is NOT a standard
        alpha stereocenter). Callers should default ``None`` to ``+1``.
    """
    if any(k not in backbone_indices for k in ("N", "CA", "C")):
        return None
    n_i, ca_i, c_i = backbone_indices["N"], backbone_indices["CA"], backbone_indices["C"]
    o_i = backbone_indices.get("O", -1)
    backbone = {n_i, ca_i, c_i}
    if o_i >= 0:
        backbone.add(o_i)

    ca = coords[ca_i]
    # Alpha-carbon guard. A true alpha stereocenter requires CA to be COVALENTLY BONDED
    # to BOTH the backbone amide N and the carbonyl C. beta-/gamma-amino acids (beta-alanine
    # BAL, the beta3 residues B3L/B3Q, ...) interpose an extra CH2 between CA and one of those
    # partners, so that partner sits ~2.5 A away and the labelled CA carries no defined
    # handedness. Without this guard the geometric triple product below happily returns a
    # spurious +/-1 for such residues (BAL read -1, a false D). Undefined -> None.
    if float((coords[n_i] - ca).norm()) >= 1.7 or float((coords[c_i] - ca).norm()) >= 1.7:
        return None
    # Candidate C-beta atoms: valid, non-backbone, bonded distance (~1.0-2.0 A) from CA.
    candidates = []
    for a in range(coords.shape[0]):
        if not bool(mask[a]) or a in backbone:
            continue
        dist = float((coords[a] - ca).norm())
        if 1.0 < dist < 2.0:
            candidates.append(a)
    if not candidates:
        return None  # no C-beta (glycine, capping groups) -> undefined
    if len(candidates) > 1:
        return None  # alpha,alpha-disubstituted (AIB/DIV): handedness ambiguous -> undefined

    cb = coords[candidates[0]]
    n, c = coords[n_i], coords[c_i]
    triple = torch.dot(torch.linalg.cross(n - ca, c - ca), cb - ca)
    return 1 if float(triple) >= 0.0 else -1


def build_chirality_lookup(db: dict) -> dict[str, int]:
    """Build a per-residue-code chirality sign lookup from a rotamer database.

    Derives the L/D sign **geometrically** from each residue's reference C-beta
    (see :func:`_chirality_from_reference`). The result maps every residue
    identifier (``ccd_code`` and all CCD ``pdb_ids``) to ``+1`` (L) or ``-1`` (D).
    Residues whose chirality is undefined (glycine, alpha,alpha-disubstituted,
    capping groups) are omitted, so a missing key should default to ``+1``.

    Parameters
    ----------
    db : dict
        Loaded residue database with
        ``coords``, ``masks``, ``metadata`` and per-entry ``backbone_indices``.

    Returns
    -------
    dict[str, int]
        Mapping residue code -> chirality sign (+1 or -1). L-residues and
        unknown codes are not guaranteed present; treat absence as +1.
    """
    coords = db["coords"]
    masks = db["masks"]
    metadata = db["metadata"]
    # coords/masks are keyed per-rotamer; metadata is per-residue-type. Map each
    # residue type to its first rotamer row to read a representative geometry.
    lookup: dict[str, int] = {}
    row = 0
    for m in metadata:
        n_rot = int(m.get("num_rotamers", 1)) or 1
        bb = m.get("backbone_indices")
        if isinstance(bb, dict):
            sign = _chirality_from_reference(coords[row], masks[row], bb)
            if sign is not None:
                codes = [m.get("ccd_code"), *(m.get("pdb_ids", []) or [])]
                for code in codes:
                    if code:
                        lookup[str(code)] = sign
        row += n_rot

    return lookup


def chirality_signs_for_codes(
    res_codes: list[str],
    lookup: dict[str, int],
) -> list[int]:
    """Map a list of residue codes to chirality signs, defaulting unknowns to +1 (L).

    Parameters
    ----------
    res_codes : list[str]
        Per-position 3-letter residue codes.
    lookup : dict[str, int]
        Mapping from :func:`build_chirality_lookup`.

    Returns
    -------
    list[int]
        One sign per code; ``+1`` for L / unknown, ``-1`` for D.
    """
    return [lookup.get(code, 1) for code in res_codes]
