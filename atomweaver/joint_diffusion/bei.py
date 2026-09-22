"""Shared sidechain-exposure classifier: legacy exclusive B/E/I + the new two-axis scheme.

Two selectable definitions live here, chosen per-call via ``bei_def``:

``"aron_overlap"`` (DEFAULT, the new run)
    Ports the self-occupancy ``residue_burial_summaries`` + ``residue_contacts_longer_chain``
    (an internal prototype) as TWO INDEPENDENT, OVERLAPPING axes
    per residue -- a residue may be positive on both; there is NO forced single exclusive label at
    the classifier level:

      * ``burial`` in {``"buried"``, ``"partial"``, ``"exposed"``} from the sidechain's RELATIVE SASA
        ``relative_sasa = sidechain_SASA_in_full_context / sidechain_SASA_in_isolation`` where the
        "sidechain" atom set is **CA + true sidechain heavy atoms** (backbone N/C/O excluded, H excluded;
        GLYCINE = CA alone -- this CA-inclusive set is the key glycine fix). Full context = the whole
        peptide+target heavy-atom system (target INCLUDED so target-shielding lowers SASA); isolation =
        the same sidechain-role atoms with the rest of the system absent. Thresholds
        ``rel < 0.05 -> buried``; ``rel < 0.25 -> partial``; ``else exposed`` (all three tracked).
      * ``is_contacting`` (bool): any sidechain-role heavy atom within 4.0 Å of any receptor (target)
        heavy atom, AND a DIRECTION filter -- ``Cα->pseudo-Cβ · Cα->nearest-target-atom > 0`` (points at the
        target). The pseudo-Cβ is the chirality-aware idealized-Cβ ray reused from
        :func:`atomweaver.joint_diffusion.models.compute_pseudo_cb_direction`.

    Two DELIBERATE deviations: (1) SASA uses biotite's element-only ``vdw_radii="Single"``
    radii (NCAA-agnostic), NOT his AMBER/ProtOr set; (2) the direction filter above is an addition to
    the bare 4.0 Å proximity contact.

``"legacy_sidechain_excl"``
    The prior EXCLUSIVE Interface/Buried/Exposed definition (sidechain ΔSASA >= 16 Å² -> Interface;
    tip self-packing -> Buried; else Exposed). Kept byte-identical so historical numbers reproduce.

Both defs fall back to a direction-vector method when biotite is unimportable.

``classify_bei_batch`` returns a single collapsed ``{p: "Interface"|"Buried"|"Exposed"}`` label under
BOTH defs (interface-first precedence for the new def) so existing consumers keep their contract;
``classify_bei_two_axis`` exposes the full overlapping two-axis dict for callers that want both axes.
"""

import torch

# --- legacy ("legacy_sidechain_excl") constants -----------------------------------------------------
DELTA_IFACE = 16.0  # Å²: sidechain area occluded BY the target above which -> Interface
REL_EXPOSED = 0.30  # complex-SASA / peptide-alone-SASA above which (non-iface) -> Exposed
TIP_NBR_BURY = 12  # own-chain heavy atoms within R_TIP_BURY of the sidechain TIP -> Buried
R_TIP_BURY = 6.0  # Å: neighborhood radius around the sidechain tip for self-packing burial
# --- direction-vector FALLBACK constants (used only if biotite SASA is unavailable) ---
R_CONTACT = 5.0  # Å: sidechain-tip within this of the target -> tip "near target"
DOT_IFACE = 0.0  # Cα->sidechain · Cα->nearest-target > this AND tip near target -> Interface
BEI_LABELS = ("Interface", "Buried", "Exposed")
_ELEM_NAME_BEI = {1: "C", 2: "N", 3: "O", 4: "S"}  # model element encoding (PAD=0 skipped)
_BB_ELEM_BEI = ("N", "C", "C", "O")  # backbone slot -> element (N, CA, C, O)

# --- new two-axis ("aron_overlap") constants (the self-occupancy thresholds) ------------------------------
BURIAL_CLASSES = ("buried", "partial", "exposed")
BURIED_REL_SASA = 0.05  # relative_sasa below this -> buried
EXPOSED_REL_SASA = 0.25  # relative_sasa below this (and >= buried) -> partial; else exposed
CONTACT_DIST = 4.0  # Å: sidechain-role heavy atom within this of a target heavy atom -> proximity contact
DIRECTION_DOT = 0.0  # Cα->pseudo-Cβ · Cα->nearest-target > this (points toward target) gates is_contacting
SASA_N_POINTS = 1000  # biotite sasa point_number for the new def (matches the reference)

DEFAULT_BEI_DEF = "aron_overlap"  # NEW def is the default (the new run); legacy stays reproducible

try:
    import biotite.structure as _bt_struc  # noqa: F401

    _HAS_BIOTITE = True
except Exception:
    _HAS_BIOTITE = False


def _sasa_per_atom_bei(coords, elements, probe=1.4, n_points=200):
    """Per-atom SASA for a bare atom cloud via biotite, using element-only VdW radii.

    ``vdw_radii="Single"`` assigns one radius per ELEMENT (no res_name/atom_name topology lookup),
    which is correct for a coordinate cloud with no canonical residue identity.
    """
    import biotite.structure as struc
    import numpy as _np

    n = len(coords)
    arr = struc.AtomArray(n)
    arr.coord = coords.astype(_np.float32)
    arr.element = _np.array(elements, dtype="U2")
    arr.chain_id = _np.array(["A"] * n)
    arr.res_id = _np.arange(n)
    arr.res_name = _np.array(["UNK"] * n)
    arr.atom_name = _np.array(["X"] * n)
    return struc.sasa(arr, probe_radius=probe, point_number=n_points, vdw_radii="Single")


def _tip_self_neighbors(p, tip, sc_coords, sc_mask, backbone_coords, backbone_mask, r_tip_bury):
    """Own-chain heavy atoms within ``r_tip_bury`` of the sidechain TIP, excluding self + i±1."""
    n_res = sc_coords.shape[0]
    own = []
    for q in range(n_res):
        if abs(q - p) <= 1:
            continue
        bbm = backbone_mask[q].bool()
        if bool(bbm.any()):
            own.append(backbone_coords[q][bbm])
        scm = sc_mask[q].bool()
        if bool(scm.any()):
            own.append(sc_coords[q][scm])
    if not own:
        return 0
    oc = torch.cat(own, dim=0)
    return int((torch.cdist(tip.unsqueeze(0), oc) < r_tip_bury).sum().item())


# =====================================================================================================
# NEW two-axis ("aron_overlap") classifier
# =====================================================================================================


def _bucket_relative_sasa(rel: float, buried_rel: float, exposed_rel: float) -> str:
    """Bucket a relative SASA into ``buried`` / ``partial`` / ``exposed`` (the classify_relative_sasa)."""
    if rel < buried_rel:
        return "buried"
    if rel < exposed_rel:
        return "partial"
    return "exposed"


def _pseudo_cb_dir(bb4: torch.Tensor, chir: torch.Tensor | None):
    """Chirality-aware CA->pseudo-Cβ unit vector from one residue's (N, CA, C, O) backbone.

    Reuses the corrected direction-cone construction verbatim from
    :func:`atomweaver.joint_diffusion.models.compute_pseudo_cb_direction` (imported lazily to keep
    ``bei`` importable without the heavy ``models`` module and to avoid an import cycle).
    """
    from atomweaver.joint_diffusion.models import compute_pseudo_cb_direction

    return compute_pseudo_cb_direction(bb4, chirality=chir)


def _residue_masked_heavy(p, sc_coords, sc_mask, backbone_coords):
    """The residue's sidechain-role heavy atoms = CA + true sidechain heavy atoms (glycine = CA alone)."""
    ca = backbone_coords[p, 1].unsqueeze(0)
    sm = sc_mask[p].bool()
    if bool(sm.any()):
        return torch.cat([ca, sc_coords[p][sm]], dim=0)
    return ca


def _is_contacting(
    p,
    sc_coords,
    sc_mask,
    backbone_coords,
    t_xyz,
    contact_dist: float,
    direction_dot: float,
    chir: torch.Tensor | None,
) -> bool:
    """Proximity (any sidechain-role heavy atom within ``contact_dist`` of a target heavy atom)
    AND the direction filter (Cα->pseudo-Cβ · Cα->nearest-target > ``direction_dot``).
    """
    if t_xyz is None or t_xyz.numel() == 0:
        return False
    ca = backbone_coords[p, 1]
    masked = _residue_masked_heavy(p, sc_coords, sc_mask, backbone_coords)
    if not bool((torch.cdist(masked, t_xyz) <= contact_dist).any().item()):
        return False
    # direction filter: pseudo-Cβ ray must point toward the nearest target atom
    d_ca = torch.norm(t_xyz - ca, dim=1)
    nearest_t = t_xyz[int(torch.argmin(d_ca).item())]
    t_vec = nearest_t - ca
    tn = torch.norm(t_vec)
    if tn < 1e-6:
        return True  # target atom sits on CA; proximity already satisfied, direction undefined
    cb_dir = _pseudo_cb_dir(backbone_coords[p], chir)
    return bool((cb_dir @ (t_vec / tn)).item() > direction_dot)


def _classify_two_axis_direction(
    p,
    sc_coords,
    sc_mask,
    backbone_coords,
    backbone_mask,
    target_coords,
    target_mask,
    contact_dist,
    direction_dot,
    chir,
    tip_nbr_bury=TIP_NBR_BURY,
    r_tip_bury=R_TIP_BURY,
) -> dict:
    """biotite-unavailable fallback for one residue's two axes (no SASA).

    ``is_contacting`` uses the SAME 4 Å proximity + pseudo-Cβ direction filter as the SASA path.
    ``burial`` is approximated from tip self-packing (relative SASA is unavailable): dense own-chain
    neighborhood -> ``buried``, moderate -> ``partial``, sparse -> ``exposed``; ``relative_sasa`` is NaN.
    """
    t_xyz = None
    if target_coords is not None and target_coords.numel() > 0:
        tm = target_mask.bool() if target_mask is not None else torch.ones(target_coords.shape[0], dtype=torch.bool)
        t_sel = target_coords[tm]
        if t_sel.numel() > 0:
            t_xyz = t_sel
    is_contact = _is_contacting(p, sc_coords, sc_mask, backbone_coords, t_xyz, contact_dist, direction_dot, chir)

    ca = backbone_coords[p, 1]
    sm = sc_mask[p].bool()
    if bool(sm.any()):
        res_atoms = sc_coords[p][sm]
        tip = res_atoms[int(torch.argmax(torch.norm(res_atoms - ca, dim=1)).item())]
    else:
        tip = ca  # glycine -> pack around CA
    tip_nbrs = _tip_self_neighbors(p, tip, sc_coords, sc_mask, backbone_coords, backbone_mask, r_tip_bury)
    if tip_nbrs >= tip_nbr_bury:
        burial = "buried"
    elif tip_nbrs >= tip_nbr_bury / 2:
        burial = "partial"
    else:
        burial = "exposed"
    return {"burial": burial, "is_contacting": is_contact, "relative_sasa": float("nan")}


def classify_bei_two_axis(
    sc_coords: torch.Tensor,
    sc_mask: torch.Tensor,
    sc_el: torch.Tensor | None,
    backbone_coords: torch.Tensor,
    backbone_mask: torch.Tensor,
    target_coords: torch.Tensor | None = None,
    target_mask: torch.Tensor | None = None,
    target_el: torch.Tensor | None = None,
    chirality: torch.Tensor | None = None,
    buried_rel: float = BURIED_REL_SASA,
    exposed_rel: float = EXPOSED_REL_SASA,
    contact_dist: float = CONTACT_DIST,
    direction_dot: float = DIRECTION_DOT,
    n_points: int = SASA_N_POINTS,
) -> dict:
    """Classify EVERY peptide residue on the TWO OVERLAPPING axes (the scheme + direction filter).

    Returns ``{p: {"burial": "buried"|"partial"|"exposed", "is_contacting": bool,
    "relative_sasa": float}}`` for p in range(L). Overlap is intended -- a residue can be both buried
    and contacting; no exclusive label is forced here (see :func:`collapse_bei_env` for the callers
    that need a single env code). Computed once per complex (SASA needs the whole peptide+target
    system); a pure property of the GT structure.

    ``chirality`` is an optional per-residue L/D sign tensor (``+1`` L / ``-1`` D) for the pseudo-Cβ
    direction; ``None`` treats all residues as L (glycine is achiral so either handedness gives ~the
    same ray). Falls back to :func:`_classify_two_axis_direction` when biotite is unavailable.
    """
    L = sc_coords.shape[0]

    def _chir(p):
        return None if chirality is None else chirality[p]

    if not _HAS_BIOTITE:
        return {
            p: _classify_two_axis_direction(
                p,
                sc_coords,
                sc_mask,
                backbone_coords,
                backbone_mask,
                target_coords,
                target_mask,
                contact_dist,
                direction_dot,
                _chir(p),
            )
            for p in range(L)
        }
    import numpy as _np

    t_xyz = None
    tm = None
    if target_coords is not None and target_coords.numel() > 0:
        tm = target_mask.bool() if target_mask is not None else torch.ones(target_coords.shape[0], dtype=torch.bool)
        t_sel = target_coords[tm]
        if t_sel.numel() > 0:
            t_xyz = t_sel

    # ---- assemble the peptide heavy-atom cloud; track residue index + sidechain-role membership.
    # sidechain-role = CA (backbone slot 1) + every true sidechain heavy atom (glycine = CA only).
    pc, pe, pr, p_scrole = [], [], [], []
    for p in range(L):
        bm = backbone_mask[p].bool()
        for j in torch.where(bm)[0].tolist():
            pc.append(backbone_coords[p, j].cpu().numpy())
            pe.append(_BB_ELEM_BEI[j] if j < len(_BB_ELEM_BEI) else "C")
            pr.append(p)
            p_scrole.append(j == 1)  # CA -> sidechain-role
        sm = sc_mask[p].bool()
        for j in torch.where(sm)[0].tolist():
            pc.append(sc_coords[p, j].cpu().numpy())
            e = int(sc_el[p, j].item()) if sc_el is not None else 1
            pe.append(_ELEM_NAME_BEI.get(e, "C"))
            pr.append(p)
            p_scrole.append(True)
    if not pc:
        return {p: {"burial": "exposed", "is_contacting": False, "relative_sasa": 1.0} for p in range(L)}
    pc = _np.asarray(pc, dtype=_np.float32)
    pr = _np.asarray(pr)
    p_scrole = _np.asarray(p_scrole)

    # ---- full-context SASA (peptide + target), per peptide atom
    try:
        if t_xyz is not None:
            tc = t_xyz.cpu().numpy().astype(_np.float32)
            te = (
                [_ELEM_NAME_BEI.get(int(x), "C") for x in target_el[tm].tolist()]
                if target_el is not None
                else ["C"] * len(tc)
            )
            sasa_ctx = _sasa_per_atom_bei(_np.concatenate([pc, tc], 0), pe + te, n_points=n_points)[: len(pc)]
        else:
            sasa_ctx = _sasa_per_atom_bei(pc, pe, n_points=n_points)
    except Exception:
        return {
            p: _classify_two_axis_direction(
                p,
                sc_coords,
                sc_mask,
                backbone_coords,
                backbone_mask,
                target_coords,
                target_mask,
                contact_dist,
                direction_dot,
                _chir(p),
            )
            for p in range(L)
        }

    out = {}
    for p in range(L):
        role = (pr == p) & p_scrole
        role_idx = _np.nonzero(role)[0]
        if role_idx.size == 0:
            out[p] = {"burial": "exposed", "is_contacting": False, "relative_sasa": 1.0}
            continue
        role_coords = pc[role_idx]
        role_elems = [pe[i] for i in role_idx.tolist()]
        try:
            sasa_iso = _sasa_per_atom_bei(role_coords, role_elems, n_points=n_points)
            iso_total = float(sasa_iso.sum())
        except Exception:
            iso_total = 0.0
        ctx_total = float(sasa_ctx[role_idx].sum())
        rel = ctx_total / iso_total if iso_total > 1e-6 else 1.0
        burial = _bucket_relative_sasa(rel, buried_rel, exposed_rel)
        is_contact = _is_contacting(
            p, sc_coords, sc_mask, backbone_coords, t_xyz, contact_dist, direction_dot, _chir(p)
        )
        out[p] = {"burial": burial, "is_contacting": is_contact, "relative_sasa": rel}
    return out


def collapse_bei_env(entry: dict) -> str:
    """Collapse one two-axis entry to a single env label with INTERFACE-FIRST precedence.

    ``is_contacting -> "Interface"``; else ``burial == "buried" -> "Buried"``; else ``"Exposed"``
    (``partial`` folds into Exposed). Used by callers that still need the exclusive B/E/I env code.
    """
    if entry.get("is_contacting"):
        return "Interface"
    if entry.get("burial") == "buried":
        return "Buried"
    return "Exposed"


# =====================================================================================================
# Dispatcher: keeps the {p: "Interface"|"Buried"|"Exposed"} contract for existing consumers
# =====================================================================================================


def classify_bei_batch(
    sc_coords: torch.Tensor,
    sc_mask: torch.Tensor,
    sc_el: torch.Tensor | None,
    backbone_coords: torch.Tensor,
    backbone_mask: torch.Tensor,
    target_coords: torch.Tensor | None = None,
    target_mask: torch.Tensor | None = None,
    target_el: torch.Tensor | None = None,
    delta_iface: float = DELTA_IFACE,
    rel_exposed: float = REL_EXPOSED,
    tip_nbr_bury: int = TIP_NBR_BURY,
    r_tip_bury: float = R_TIP_BURY,
    bei_def: str = DEFAULT_BEI_DEF,
    chirality: torch.Tensor | None = None,
) -> dict:
    """Classify every peptide residue and return ``{p: "Interface"|"Buried"|"Exposed"}``.

    ``bei_def="aron_overlap"`` (default): run the two-axis classifier and collapse each residue with
    interface-first precedence (:func:`collapse_bei_env`). ``bei_def="legacy_sidechain_excl"``: the
    prior exclusive sidechain-ΔSASA definition (byte-identical, for reproducing historical numbers).

    Callers that need BOTH overlapping axes should call :func:`classify_bei_two_axis` directly.
    """
    if bei_def == "legacy_sidechain_excl":
        return _classify_bei_batch_legacy(
            sc_coords,
            sc_mask,
            sc_el,
            backbone_coords,
            backbone_mask,
            target_coords=target_coords,
            target_mask=target_mask,
            target_el=target_el,
            delta_iface=delta_iface,
            rel_exposed=rel_exposed,
            tip_nbr_bury=tip_nbr_bury,
            r_tip_bury=r_tip_bury,
        )
    if bei_def != "aron_overlap":
        raise ValueError(f"bei_def must be 'aron_overlap' or 'legacy_sidechain_excl', got {bei_def!r}")
    axes = classify_bei_two_axis(
        sc_coords,
        sc_mask,
        sc_el,
        backbone_coords,
        backbone_mask,
        target_coords=target_coords,
        target_mask=target_mask,
        target_el=target_el,
        chirality=chirality,
    )
    return {p: collapse_bei_env(a) for p, a in axes.items()}


# =====================================================================================================
# LEGACY exclusive ("legacy_sidechain_excl") classifier -- byte-identical to the earlier definition
# =====================================================================================================


def _classify_bei_batch_legacy(
    sc_coords: torch.Tensor,
    sc_mask: torch.Tensor,
    sc_el: torch.Tensor | None,
    backbone_coords: torch.Tensor,
    backbone_mask: torch.Tensor,
    target_coords: torch.Tensor | None = None,
    target_mask: torch.Tensor | None = None,
    target_el: torch.Tensor | None = None,
    delta_iface: float = DELTA_IFACE,
    rel_exposed: float = REL_EXPOSED,
    tip_nbr_bury: int = TIP_NBR_BURY,
    r_tip_bury: float = R_TIP_BURY,
) -> dict:
    """LEGACY exclusive Interface/Buried/Exposed classifier (sidechain ΔSASA >= 16 Å² -> Interface).

    Returns ``{p: label}`` for p in range(L). Computed once per complex. Preserved verbatim so
    historical numbers reproduce; reachable via ``classify_bei_batch(..., bei_def="legacy_sidechain_excl")``.

    SASA path (biotite available):
      relSASA = sasa_complex / sasa_alone;  ΔSASA = sasa_alone - sasa_complex
      tip_nbrs = own-chain heavy atoms within ``r_tip_bury`` of the sidechain TIP
      Buried    if tip_nbrs >= tip_nbr_bury and ΔSASA < delta_iface
      Interface if ΔSASA >= delta_iface
      Exposed   if relSASA >= rel_exposed
      else Buried.
    Fallback (no biotite): direction-vector (see classify_bei_direction).
    """
    L = sc_coords.shape[0]
    if not _HAS_BIOTITE:
        return {
            p: classify_bei_direction(p, sc_coords, sc_mask, backbone_coords, backbone_mask, target_coords, target_mask)
            for p in range(L)
        }
    import numpy as _np

    # ---- assemble the peptide atom cloud (backbone + sidechains), tracking residue & sidechain flags
    pc, pe, pr, p_is_sc = [], [], [], []
    for p in range(L):
        bm = backbone_mask[p].bool()
        for j in torch.where(bm)[0].tolist():
            pc.append(backbone_coords[p, j].cpu().numpy())
            pe.append("C")
            pr.append(p)
            p_is_sc.append(False)
        sm = sc_mask[p].bool()
        for j in torch.where(sm)[0].tolist():
            pc.append(sc_coords[p, j].cpu().numpy())
            e = int(sc_el[p, j].item()) if sc_el is not None else 1
            pe.append(_ELEM_NAME_BEI.get(e, "C"))
            pr.append(p)
            p_is_sc.append(True)
    if not pc:
        return dict.fromkeys(range(L), "Exposed")
    pc = _np.array(pc, dtype=_np.float32)
    pr = _np.array(pr)
    p_is_sc = _np.array(p_is_sc)

    try:
        sasa_alone = _sasa_per_atom_bei(pc, pe)
        if target_coords is not None and target_coords.numel() > 0:
            tm = target_mask.bool() if target_mask is not None else torch.ones(target_coords.shape[0], dtype=torch.bool)
            tc = target_coords[tm].cpu().numpy().astype(_np.float32)
            te = (
                [_ELEM_NAME_BEI.get(int(x), "C") for x in target_el[tm].tolist()]
                if target_el is not None
                else ["C"] * len(tc)
            )
            sasa_complex = _sasa_per_atom_bei(_np.concatenate([pc, tc], 0), pe + te)[: len(pc)]
        else:
            sasa_complex = sasa_alone
    except Exception:
        return {
            p: classify_bei_direction(p, sc_coords, sc_mask, backbone_coords, backbone_mask, target_coords, target_mask)
            for p in range(L)
        }

    labels = {}
    for p in range(L):
        m = (pr == p) & p_is_sc
        if not m.any():
            labels[p] = "Exposed"  # glycine / no sidechain -> nothing to bury
            continue
        s_alone = float(sasa_alone[m].sum())
        s_complex = float(sasa_complex[m].sum())
        dsasa = s_alone - s_complex
        rel_complex = s_complex / max(s_alone, 1e-3)
        ca = backbone_coords[p, 1]
        res_atoms = sc_coords[p][sc_mask[p].bool()]
        tip = res_atoms[int(torch.argmax(torch.norm(res_atoms - ca, dim=1)).item())]
        tip_nbrs = _tip_self_neighbors(p, tip, sc_coords, sc_mask, backbone_coords, backbone_mask, r_tip_bury)
        if tip_nbrs >= tip_nbr_bury and dsasa < delta_iface:
            labels[p] = "Buried"
        elif dsasa >= delta_iface:
            labels[p] = "Interface"
        elif rel_complex >= rel_exposed:
            labels[p] = "Exposed"
        else:
            labels[p] = "Buried"
    return labels


def classify_bei_direction(
    p: int,
    sc_coords: torch.Tensor,
    sc_mask: torch.Tensor,
    backbone_coords: torch.Tensor,
    backbone_mask: torch.Tensor,
    target_coords: torch.Tensor | None = None,
    target_mask: torch.Tensor | None = None,
    r_contact: float = R_CONTACT,
    dot_iface: float = DOT_IFACE,
    tip_nbr_bury: int = TIP_NBR_BURY,
    r_tip_bury: float = R_TIP_BURY,
) -> str:
    """Direction-vector fallback (no SASA) for the LEGACY exclusive def. DIRECTIONALITY-aware.

    v = Cα->sidechain-centroid (where the sidechain points); t = Cα->nearest target atom.
      Interface : v·t̂ > dot_iface  (points toward target)  AND sidechain tip within r_contact of target
      Exposed   : points away from the target (v·t̂ <= dot_iface) and tip not target-contacting
      Buried    : sidechain tip packed in a dense own-chain neighborhood, regardless of direction
    """
    ca = backbone_coords[p, 1]
    res_sc_mask = sc_mask[p].bool()
    if not bool(res_sc_mask.any()):
        return "Exposed"  # glycine / no sidechain
    res_atoms = sc_coords[p][res_sc_mask]
    centroid = res_atoms.mean(dim=0)
    v = centroid - ca
    if torch.norm(v) < 1e-6:
        return "Exposed"
    v_hat = v / torch.norm(v)
    tip = res_atoms[int(torch.argmax(torch.norm(res_atoms - ca, dim=1)).item())]
    tip_nbrs = _tip_self_neighbors(p, tip, sc_coords, sc_mask, backbone_coords, backbone_mask, r_tip_bury)

    points_toward = False
    tip_near_target = False
    if target_coords is not None and target_coords.numel() > 0:
        tc = target_coords[target_mask.bool()] if target_mask is not None else target_coords
        if tc.numel() > 0:
            d_ca = torch.norm(tc - ca, dim=1)
            nearest_t = tc[int(torch.argmin(d_ca).item())]
            t = nearest_t - ca
            if torch.norm(t) > 1e-6:
                t_hat = t / torch.norm(t)
                points_toward = bool((v_hat @ t_hat).item() > dot_iface)
            tip_near_target = bool((torch.cdist(tip.unsqueeze(0), tc) < r_contact).any().item())

    if points_toward and tip_near_target:
        return "Interface"
    if tip_nbrs >= tip_nbr_bury:
        return "Buried"
    return "Exposed"
