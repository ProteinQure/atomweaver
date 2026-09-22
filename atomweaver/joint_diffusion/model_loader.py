"""Checkpoint -> evaluation model loader for AtomWeaver public inference.

Rebuilds an :class:`~atomweaver.joint_diffusion.models.InverseFoldingDiffusion` from a
Lightning checkpoint's ``state_dict`` + ``hyper_parameters`` and wraps it in a minimal
eval-only holder exposing exactly what the sampler/eval needs (``.model``,
``.name_to_idx``, ``._ckpt_reserved_slot0``). This replaces the training-time
``DiffusionLightningModule`` (not shipped in the public package): the LightningModule's
constructor performs NO mutation of ``model`` and ``model.sample`` never reads its extra
attributes, so this wrapper reproduces the eval behaviour exactly while dropping the
entire training path.

Extracted from the training pipeline's checkpoint loader and residue-name lookup for the
public inference package; signatures are unchanged.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn


class _EvalModule(nn.Module):
    """Minimal eval-only stand-in for the training ``DiffusionLightningModule``.

    Holds the rebuilt model plus the residue-name lookup. Being an ``nn.Module`` with
    ``model`` as a registered submodule gives ``.to(device)`` / ``.eval()`` for free, so
    call sites (``lm.model``, ``lm.name_to_idx``, ``lm._ckpt_reserved_slot0``) are
    identical to before.
    """

    def __init__(self, model: nn.Module, name_to_idx: "dict[str, int] | None"):
        super().__init__()
        self.model = model
        self.name_to_idx = name_to_idx
        self._ckpt_reserved_slot0 = None


_CANONICAL_NAME_TO_CODE = [
    ("Alanine", "ALA"),
    ("Cysteine", "CYS"),
    ("Aspartic", "ASP"),
    ("Glutamic acid", "GLU"),
    ("Phenylalanine", "PHE"),
    ("Glycine", "GLY"),
    ("Histidine", "HIS"),
    ("Isoleucine", "ILE"),
    ("Lysine", "LYS"),
    ("Leucine", "LEU"),
    ("Methionine", "MET"),
    ("Asparagine", "ASN"),
    ("Proline", "PRO"),
    ("Glutamine", "GLN"),
    ("Arginine", "ARG"),
    ("Serine", "SER"),
    ("Threonine", "THR"),
    ("Valine", "VAL"),
    ("Tryptophan", "TRP"),
    ("Tyrosine", "TYR"),
]


def build_name_to_idx(db_metadata: list) -> dict[str, int]:
    """Map residue name / pdb_id / 3-letter code -> residue DB row index.

    Extracted verbatim from the discretization-loss setup so the pretrain-only volumetric-decoy-CE path can
    build the SAME lookup (used by the collate to fill ``batch['residue_indices']``) without constructing the
    full discretization loss.
    """
    name_to_idx: dict[str, int] = {}
    for i, meta in enumerate(db_metadata):
        for pdb_id in meta.get("pdb_ids", []):
            if pdb_id and pdb_id not in name_to_idx:
                name_to_idx[pdb_id] = i
        name = meta.get("name", "")
        for aa_name, code in _CANONICAL_NAME_TO_CODE:
            if aa_name in name and code not in name_to_idx:
                name_to_idx[code] = i
    return name_to_idx


def _load_lightning_module_for_eval(
    checkpoint_path: Path,
    residue_db: str,
    device: str,
    coord_process_type: str | None = None,
):
    """Reconstruct an eval-only model holder from a checkpoint.

    Returns an :class:`_EvalModule` (minimal stand-in for the training
    ``DiffusionLightningModule``) exposing ``.model`` / ``.name_to_idx`` /
    ``._ckpt_reserved_slot0`` -- the only attributes the sampler/eval reads.
    """
    from atomweaver.joint_diffusion.models import InverseFoldingDiffusion

    # Accept a str path from public callers: several helpers below use Path methods
    # (e.g. ``checkpoint_path.as_posix()``), which crash on a bare string.
    checkpoint_path = Path(checkpoint_path)

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("state_dict", {})
    hparams = ckpt.get("hyper_parameters", {})

    # LEGACY-WHEN-ABSENT sigma sharpness (used by the volumetric_sigma_element_scale kwarg below). A pre-Fourier
    # checkpoint (no volumetric_fourier_frequencies key) that stored volumetric_sigma_element_scale=None was
    # TRAINED with the blurry ~1.0 A mean-normalized per-element sigma; the current default is the sharp self-occupancy
    # 0.6 A. Rebuild such a ckpt with the legacy scale so its volumetric head -- and, for option-ii ckpts, the
    # sampling CONTEXT it feeds -- is byte-faithful (else its recovery would silently drift). A new ckpt
    # (fourier key present) with None keeps the sharp default; any ckpt that stored an explicit scale wins.
    from atomweaver.joint_diffusion.volumetric_head import legacy_default_sigma_element_scale

    _legacy_sigma_scale = (
        legacy_default_sigma_element_scale() if "volumetric_fourier_frequencies" not in hparams else None
    )

    model_state = {k.replace("model.", ""): v for k, v in state_dict.items() if k.startswith("model.")}

    timesteps = hparams.get("timesteps")
    if timesteps is None and "model.diffusion.betas" in state_dict:
        timesteps = int(state_dict["model.diffusion.betas"].shape[0])
    if timesteps is None:
        timesteps = 250

    # Infer max_sidechain_atoms from the ckpt's flow shell buffers (hparams often omits it).
    # v60-lineage uses 16, legacy 14 -- without this the eval builds a 14-slot model and a max16
    # checkpoint fails to load (shell_radii shape 16 vs 14).
    _msc = hparams.get("max_sidechain_atoms")
    for _k in ("model.coord_flow._shell_radii", "model.coord_flow._slot_occupancy"):
        if _msc is None and _k in state_dict:
            _msc = int(state_dict[_k].shape[0])
    if _msc is None:
        _msc = 14

    hidden_dim = hparams.get("hidden_dim", 128)
    num_layers = hparams.get("num_layers", 4)
    use_cluster_particle = hparams.get("use_cluster_particle_diffusion")
    if use_cluster_particle is None:
        cluster_hints = [
            hparams.get("cluster_assignment_loss_weight", 0.0),
            hparams.get("cluster_cohesion_loss_weight", 0.0),
            hparams.get("cluster_structure_loss_weight", 0.0),
            hparams.get("cluster_contrastive_loss_weight", 0.0),
        ]
        cluster_multiplier_hints = [
            hparams.get("cluster_discretization_weight_multiplier", 1.0),
            hparams.get("cluster_pocket_weight_multiplier", 1.0),
        ]
        use_cluster_particle = (
            any(value and value > 0 for value in cluster_hints)
            or any(value and value != 1.0 for value in cluster_multiplier_hints)
            or "cluster_split_merge" in checkpoint_path.as_posix()
        )

    if coord_process_type is None:
        if "coord_process_type" not in hparams:
            raise ValueError(
                "coord_process_type missing from ckpt hyper_parameters -- refusing to silently default "
                "the coordinate generative process (ddpm vs flow_matching). A wrong guess corrupts every "
                "sampled coordinate. Pass coord_process_type=... for a known-legacy ckpt, or re-save the "
                "ckpt with this hparam."
            )
        coord_process_type = hparams["coord_process_type"]
    model = InverseFoldingDiffusion(
        max_sidechain_atoms=_msc,
        timesteps=timesteps,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        coord_process_type=coord_process_type,
        flow_ghost_power=hparams.get("flow_ghost_power", 2.0),
        flow_real_proximal_power=hparams.get("flow_real_proximal_power", 2.0),
        flow_real_distal_power=hparams.get("flow_real_distal_power", 1.0),
        flow_use_conditional_groupwise=hparams.get("flow_use_conditional_groupwise", False),
        coord_loss_weight=hparams.get("coord_loss_weight", 1.25),
        count_correlation_loss_weight=hparams.get("count_correlation_loss_weight", 20.0),
        count_loss_type=hparams.get("count_loss_type", "correlation"),
        count_perturb_prob=hparams.get("count_perturb_prob", 0.0),
        count_corr_tau=hparams.get("count_corr_tau", 0.5),
        count_tau_floor=hparams.get("count_tau_floor", 0.0),
        mixture_lr_threshold=hparams.get("mixture_lr_threshold", 0.0),
        use_target_conditioning=hparams.get("use_target_conditioning", True),
        use_cluster_particle_diffusion=use_cluster_particle,
        target_condition_scale=hparams.get("target_condition_scale", 1.0),
        cluster_target_condition_scale=hparams.get("cluster_target_condition_scale", 1.0),
        ghost_weight=hparams.get("ghost_weight", 0.0),
        pad_sampling_init=hparams.get("pad_sampling_init", "bare"),
        element_fn_weight=hparams.get("element_fn_weight", 1.0),
        element_loss_non_pad_only=hparams.get("element_loss_non_pad_only", False),
        num_cross_attn_layers=hparams.get("num_cross_attn_layers", 5),
        num_cross_attn_heads=hparams.get("num_cross_attn_heads", 8),
        use_film=hparams.get("use_film", True),
        use_bidirectional_target_conditioning=hparams.get("use_bidirectional_target_conditioning", True),
        use_sidechain_target_residue_attention=hparams.get("use_sidechain_target_residue_attention", True),
        sidechain_target_cutoff=hparams.get("sidechain_target_cutoff", hparams.get("binder_target_cutoff", 15.0)),
        backbone_target_cutoff=hparams.get("backbone_target_cutoff", hparams.get("binder_target_cutoff", 25.0)),
        # LEGACY-WHEN-ABSENT: a checkpoint that predates the flag was TRAINED with the 0-10 A RBF span even
        # though it carries backbone_target_cutoff in hparams. Default False here so an old ckpt rebuilds with
        # the legacy span; a new ckpt that saved rbf_span_to_cutoff=True keeps its widened span (its hparam wins).
        rbf_span_to_cutoff=hparams.get("rbf_span_to_cutoff", False),
        multi_count_max_plus=hparams.get("multi_count_max_plus", 1),
        decoupled_count=hparams.get("decoupled_count", False),
        count_ramp_threshold=hparams.get("count_ramp_threshold", 0.9),
        count_overdispersion=hparams.get("count_overdispersion", 1.5),
        use_jackie=hparams.get("use_jackie", False),
        oracle_atom_counts=hparams.get("oracle_atom_counts", False),
        occupancy_loss_weight=hparams.get("occupancy_loss_weight", 0.0),
        mixture_gate_weight=hparams.get("mixture_gate_weight", 0.0),
        ghost_var_floor=hparams.get("ghost_var_floor", 0.3),
        mixture_loss_weight=hparams.get("mixture_loss_weight", 0.0),
        mixture_gate_max_noise=hparams.get("mixture_gate_max_noise", 1.0),
        mixture_override_pad=hparams.get("mixture_override_pad", False),
        disc_detach_mask=hparams.get("disc_detach_mask", False),
        all_carbon_sampling=hparams.get("all_carbon_sampling", False),
        non_pad_element_sampling=hparams.get("non_pad_element_sampling", False),
        late_element_resolution=hparams.get("late_element_resolution", False),
        late_element_cutoff=hparams.get("late_element_cutoff", 0.25),
        contact_weight_min=hparams.get("contact_weight_min", 1.0),
        contact_weight_scale=hparams.get("contact_weight_scale", 5.0),
        contact_weight_boost=hparams.get("contact_weight_boost", 0.0),
        contact_weight_coord=hparams.get("contact_weight_coord", True),
        contact_weight_count=hparams.get("contact_weight_count", False),
        fill_corrected_coord_weight=hparams.get("fill_corrected_coord_weight", 0.0),
        fcc_slope=hparams.get("fcc_slope", 0.097),
        fcc_tau=hparams.get("fcc_tau", 0.5),
        fcc_per_env=hparams.get("fcc_per_env", False),
        fcc_slope_buried=hparams.get("fcc_slope_buried", 0.067),
        fcc_slope_exposed=hparams.get("fcc_slope_exposed", 0.082),
        occupancy_match_loss_weight=hparams.get("occupancy_match_loss_weight", 0.0),
        occupancy_match_sigma=hparams.get("occupancy_match_sigma", 1.0),
        occupancy_match_tau=hparams.get("occupancy_match_tau", 0.5),
        use_volumetric_head=hparams.get("use_volumetric_head", False),
        use_available_volume=hparams.get("use_available_volume", False),
        use_single_site_context=hparams.get("use_single_site_context", False),
        volumetric_ss_context_p_max=hparams.get("volumetric_ss_context_p_max", 0.0),
        volumetric_ss_context_start_epoch=hparams.get("volumetric_ss_context_start_epoch", 40),
        volumetric_loss_weight=hparams.get("volumetric_loss_weight", 0.0),
        volumetric_decoy_ce_weight=hparams.get("volumetric_decoy_ce_weight", 0.0),
        volumetric_decoy_ce_temperature=hparams.get("volumetric_decoy_ce_temperature", 1.0),
        volumetric_context_radius=hparams.get("volumetric_context_radius", 10.0),
        volumetric_n_query=hparams.get("volumetric_n_query", 128),
        volumetric_sigma=hparams.get("volumetric_sigma", 1.0),
        volumetric_per_element_sigma=hparams.get("volumetric_per_element_sigma", False),
        volumetric_sigma_element_scale=(
            hparams["volumetric_sigma_element_scale"]
            if hparams.get("volumetric_sigma_element_scale") is not None
            else _legacy_sigma_scale
        ),
        volumetric_empty_weight=hparams.get("volumetric_empty_weight", 1.75),
        # LEGACY-WHEN-ABSENT (backward-compat for pre-Fourier/pre-softplus checkpoints). A checkpoint whose
        # hparams DO NOT carry these keys was trained with the LEGACY volumetric head: raw-coord query (NO
        # Fourier lift => fourier_frequencies=0, hidden+3 density trunk), a SIGNED (non-softplus) field, and no
        # trunk dropout. Defaulting to the NEW production values (64 / softplus-on) here would build a Fourier
        # head whose state_dict cannot load the old legacy-head weights. A NEW checkpoint that SAVED e.g.
        # volumetric_fourier_frequencies=64 still rebuilds Fourier (its stored hparam wins over this default).
        volumetric_fourier_frequencies=hparams.get("volumetric_fourier_frequencies", 0),
        volumetric_fourier_scale=hparams.get("volumetric_fourier_scale", 10.0),
        volumetric_dropout=hparams.get("volumetric_dropout", 0.0),
        volumetric_use_softplus=hparams.get("volumetric_use_softplus", False),
        volumetric_per_query_context=hparams.get("volumetric_per_query_context", False),
        volumetric_context_k=hparams.get("volumetric_context_k", 128),
        volumetric_context_heads=hparams.get("volumetric_context_heads", 4),
        volumetric_context_chunk=hparams.get("volumetric_context_chunk", 32),
        volumetric_atom_anchored_queries=hparams.get("volumetric_atom_anchored_queries", False),
        volumetric_scale_anchor_weight=hparams.get("volumetric_scale_anchor_weight", 0.0),
        use_volumetric_self_consistency=hparams.get("use_volumetric_self_consistency", False),
        self_consistency_weight=hparams.get("self_consistency_weight", 1.0),
        self_consistency_ramp_start=hparams.get("self_consistency_ramp_start", 0.2),
        self_consistency_ramp_end=hparams.get("self_consistency_ramp_end", 0.6),
        use_volumetric_deep_inject=hparams.get("use_volumetric_deep_inject", False),
        use_volumetric_density_inject=hparams.get("use_volumetric_density_inject", False),
        use_target_deep_inject=hparams.get("use_target_deep_inject", False),
        use_backbone_deep_inject=hparams.get("use_backbone_deep_inject", False),
        frame_v2_deep_inject_detach=hparams.get("frame_v2_deep_inject_detach", False),
        use_volumetric_existence_coupling=hparams.get("use_volumetric_existence_coupling", False),
        # Self-contained rebuild: a FULL model checkpoint already carries the head weights (loaded via
        # model.load_state_dict below), so do NOT re-load the standalone Phase-1 file here -- otherwise a
        # trained checkpoint would wrongly REQUIRE that original file to still exist on disk. The stored
        # hparams keep volumetric_head_pretrained as provenance; we just don't act on it during rebuild.
        # (The training-launch path in main() still passes the real path so the warm-start DOES load.)
        volumetric_head_pretrained="",
        freeze_volumetric_head=hparams.get("freeze_volumetric_head", False),
        # Rebuild the head-only (no-denoiser) architecture when the checkpoint was a Phase-1 pretrain, so its
        # state_dict loads cleanly. Default False = full model (byte-identical for every existing checkpoint).
        volumetric_pretrain_only=hparams.get("volumetric_pretrain_only", False),
        volumetric_target_include_context=hparams.get("volumetric_target_include_context", False),
        # Loss-target selector (affects only which supervision target the integrated volumetric loss uses; no
        # tensors depend on it). Recorded for faithful rebuild. Default False = historical self-only target.
        volumetric_loss_supervision_target=hparams.get("volumetric_loss_supervision_target", False),
        use_slot_attention=hparams.get("use_slot_attention", False),
        count_embed_mode=hparams.get("count_embed_mode", "linear"),
        num_slot_attn_layers=hparams.get("num_slot_attn_layers", 2),
        num_slot_attn_heads=hparams.get("num_slot_attn_heads", 4),
        use_directional_slot_attention=hparams.get("use_directional_slot_attention", False),
        n_scouts=hparams.get("n_scouts", 3),
        position_based_element_powers=hparams.get("position_based_element_powers", True),
        preal_gate_target=hparams.get("preal_gate_target", "none"),
        residue_count_loss_weight=hparams.get("residue_count_loss_weight", 0.0),
        atom_count_loss_weight=hparams.get("atom_count_loss_weight", 0.5),
        count_ranking_loss_weight=hparams.get("count_ranking_loss_weight", 0.0),
        count_pearson_loss_weight=hparams.get("count_pearson_loss_weight", 0.0),
        noise_dependent_ghost_weight=hparams.get("noise_dependent_ghost_weight", False),
        element_pad_prior=hparams.get("element_pad_prior", None),
        count_extreme_alpha=hparams.get("count_extreme_alpha", 0.0),
        asymmetric_perturb=hparams.get("asymmetric_perturb", False),
        mixture_head_dropout=hparams.get("mixture_head_dropout", 0.0),
        rc_head_tau=hparams.get("rc_head_tau", -1.0),
        dynamic_lrt=hparams.get("dynamic_lrt", False),
        dynamic_lrt_weight=hparams.get("dynamic_lrt_weight", 2.0),
        dynamic_lrt_clamp=hparams.get("dynamic_lrt_clamp", 0.0),
        dynamic_lrt_loss_type=hparams.get("dynamic_lrt_loss_type", "l1"),
        dynamic_lrt_rank_weight=hparams.get("dynamic_lrt_rank_weight", 0.0),
        dynamic_lrt_reg_weight=hparams.get("dynamic_lrt_reg_weight", 0.0),
        dlrt_analytical_scale=hparams.get("dlrt_analytical_scale", 0.0),
        pseudo_cb_direction=hparams.get("pseudo_cb_direction", False),
        d_aa_l_source_prob=hparams.get(
            "d_aa_l_source_prob", 1.0 if hparams.get("always_L_source_prior", False) else 0.0
        ),
        element_flow_matching=hparams.get("element_flow_matching", False),
        split_element_existence=hparams.get("split_element_existence", False),
        split_absorbing_element=hparams.get("split_absorbing_element", True),
        split_element_flow=hparams.get("split_element_flow", False),
        split_element_flow_temp=hparams.get("split_element_flow_temp", 0.0),
        existence_absorbing=hparams.get("existence_absorbing", False),
        existence_source_value=hparams.get("existence_source_value", 0.5),
        existence_power=hparams.get("existence_power", 1.0),
        existence_threshold=hparams.get("existence_threshold", 0.5),
        existence_velocity_scale=hparams.get("existence_velocity_scale", 1.0),
        atom_mask_loss_weight=hparams.get("atom_mask_loss_weight", 0.5),
        use_donut_source=hparams.get("flow_donut_source", False),
        donut_thickness_ratio=hparams.get("donut_thickness_ratio", 0.3),
        use_empirical_shell_thickness=hparams.get("use_empirical_shell_thickness", False),
        shell_target_var_scale=hparams.get("shell_target_var_scale", 1.0),
        donut_element_init=hparams.get("donut_element_init", "pad"),
        absorbing_mask=hparams.get("absorbing_mask", True),
        use_element_velocity_coupling=hparams.get("use_element_velocity_coupling", False),
        evc_scheduled_sampling_prob=hparams.get("evc_scheduled_sampling_prob", 0.0),
        evc_ss_corruption=hparams.get("evc_ss_corruption", False),
        ss_underfill_bias=hparams.get("ss_underfill_bias", 0.7),
        evc_ss_noised_element_prob=hparams.get("evc_ss_noised_element_prob", 0.0),
        evc_ss_corrupt_selfcond=hparams.get("evc_ss_corrupt_selfcond", False),
        use_bond_attention=hparams.get("use_bond_attention", False),
        zero_init_bond_attention=hparams.get("zero_init_bond_attention", False),
        bond_loss_weight=hparams.get("bond_loss_weight", 1.0),
        pocket_contact_loss_weight=hparams.get("pocket_contact_loss_weight", 0.0),
        valence_loss_weight=hparams.get("valence_loss_weight", 0.0),
        use_cross_residue_packing=hparams.get("use_cross_residue_packing", False),
        packing_contact_loss_weight=hparams.get("packing_contact_loss_weight", 1.0),
        use_shape_prior=hparams.get("use_shape_prior", False),
        use_backbone_dihedral=hparams.get("use_backbone_dihedral", False),
        use_burial_feature=hparams.get("use_burial_feature", False),
        use_residue_frame_stream=hparams.get("use_residue_frame_stream", False),
        residue_frame_layers=hparams.get("residue_frame_layers", 2),
        use_residue_frame_deep_inject=hparams.get("use_residue_frame_deep_inject", False),
        residue_frame_supervision_weight=hparams.get("residue_frame_supervision_weight", 1.0),
        residue_frame_consistency_weight=hparams.get("residue_frame_consistency_weight", 1.0),
        residue_frame_consistency_ramp_frac=hparams.get("residue_frame_consistency_ramp_frac", 0.3),
        use_residue_frame_stream_v2=hparams.get("use_residue_frame_stream_v2", False),
        residue_frame_v2_layers=hparams.get("residue_frame_v2_layers", 2),
        frame_v2_clean_input=hparams.get("frame_v2_clean_input", True),
        use_residue_frame_v2_deep_inject=hparams.get("use_residue_frame_v2_deep_inject", False),
        use_bond_angle_deep_inject=hparams.get("use_bond_angle_deep_inject", False),
        frame_v2_centroid_weight=hparams.get("frame_v2_centroid_weight", 1.0),
        frame_v2_chi1_weight=hparams.get("frame_v2_chi1_weight", 1.0),
        use_stereochem_head=hparams.get("use_stereochem_head", False),
        frame_v2_stereo_weight=hparams.get("frame_v2_stereo_weight", 1.0),
        use_stereochem_t_resolution=hparams.get("use_stereochem_t_resolution", False),
        stereo_t_resolution_weight=hparams.get("stereo_t_resolution_weight", 1.0),
        use_stereochem_t_resolution_feedback=hparams.get("use_stereochem_t_resolution_feedback", False),
        t_resolution_feedback_ramp_start=hparams.get("t_resolution_feedback_ramp_start", 0.1),
        t_resolution_feedback_ramp_end=hparams.get("t_resolution_feedback_ramp_end", 0.5),
        use_stereochem_gated_init=hparams.get("use_stereochem_gated_init", False),
        gated_init_ramp_start=hparams.get("gated_init_ramp_start", 0.1),
        gated_init_ramp_end=hparams.get("gated_init_ramp_end", 0.5),
        use_polarity_head=hparams.get("use_polarity_head", False),
        polarity_loss_weight=hparams.get("polarity_loss_weight", 0.1),
        use_glycine_head=hparams.get("use_glycine_head", False),
        glycine_loss_weight=hparams.get("glycine_loss_weight", 0.1),
        glycine_pad_cap=hparams.get("glycine_pad_cap", 4.0),
        use_neighbor_x0_packing=hparams.get("use_neighbor_x0_packing", False),
        neighbor_x0_packing_recycles=hparams.get("neighbor_x0_packing_recycles", 2),
        neighbor_x0_packing_radius=hparams.get("neighbor_x0_packing_radius", 8.0),
        neighbor_x0_corrupt_add_prob=hparams.get("neighbor_x0_corrupt_add_prob", 0.0),
        neighbor_x0_corrupt_drop_prob=hparams.get("neighbor_x0_corrupt_drop_prob", 0.0),
        neighbor_x0_corrupt_noise_prob=hparams.get("neighbor_x0_corrupt_noise_prob", 0.0),
        neighbor_x0_corrupt_coord_noise=hparams.get("neighbor_x0_corrupt_coord_noise", 0.0),
        neighbor_x0_corrupt_disconnect_prob=hparams.get("neighbor_x0_corrupt_disconnect_prob", 0.0),
        neighbor_self_dropout_prob=hparams.get("neighbor_self_dropout_prob", 0.0),
        neighbor_x0_highnoise_weight=hparams.get("neighbor_x0_highnoise_weight", 1.0),
        neighbor_x0_packing_prob_start=hparams.get("neighbor_x0_packing_prob_start", 0.0),
        neighbor_x0_packing_prob_end=hparams.get("neighbor_x0_packing_prob_end", 0.8),
        neighbor_x0_packing_ramp_epochs=hparams.get("neighbor_x0_packing_ramp_epochs", 50),
        neighbor_x0_packing_random_recycles=hparams.get("neighbor_x0_packing_random_recycles", False),
        neighbor_x0_packing_max_recycles=hparams.get("neighbor_x0_packing_max_recycles", 5),
        neighbor_x0_packing_recycle_weights=hparams.get("neighbor_x0_packing_recycle_weights") or None,
        use_transition_weighted_loss=hparams.get("use_transition_weighted_loss", False),
        transition_weight_sticky=hparams.get("transition_weight_sticky", 2.5),
        transition_weight_revised=hparams.get("transition_weight_revised", 1.25),
        transition_weight_regression=hparams.get("transition_weight_regression", 2.0),
        transition_weight_cap=hparams.get("transition_weight_cap", 3.0),
        transition_coord_tol=hparams.get("transition_coord_tol", 1.0),
        transition_coord_same_tol=hparams.get("transition_coord_same_tol", 0.25),
        transition_coord_mag_scale=hparams.get("transition_coord_mag_scale", 2.0),
        cross_residue_packing_spatial=hparams.get("cross_residue_packing_spatial", False),
        cross_residue_packing_k=hparams.get("cross_residue_packing_k", 4),
        cross_residue_packing_radius=hparams.get("cross_residue_packing_radius", 10.0),
        cross_residue_packing_min_seq_sep=hparams.get("cross_residue_packing_min_seq_sep", 2),
        main_path_underfill_prob=hparams.get("main_path_underfill_prob", 0.0),
        main_path_underfill_bias=hparams.get("main_path_underfill_bias", 0.85),
        shape_prior_n_anchors=hparams.get("shape_prior_n_anchors", 4),
        shape_prior_loss_weight=hparams.get("shape_prior_loss_weight", 1.0),
        use_bond_co_diffusion=hparams.get("use_bond_co_diffusion", False),
        bond_co_diffusion_weight=hparams.get("bond_co_diffusion_weight", 1.0),
        use_coord_self_conditioning=hparams.get("use_coord_self_conditioning", False),
        use_plan_latent=hparams.get("use_plan_latent", False),
        plan_latent_loss_weight=hparams.get("plan_latent_loss_weight", 1.0),
        use_global_latent_matching=hparams.get("use_global_latent_matching", False),
        global_latent_weight=hparams.get("global_latent_weight", 1.0),
        global_latent_gate_t_min=hparams.get("global_latent_gate_t_min", 0.5),
        global_latent_gate_t_full=hparams.get("global_latent_gate_t_full", 1.0),
        global_latent_embed_dim=hparams.get("global_latent_embed_dim", 64),
        global_latent_qupid_lookup=hparams.get("global_latent_qupid_lookup", ""),
        global_latent_confidence=hparams.get("global_latent_confidence", True),
        global_latent_confidence_weight=hparams.get("global_latent_confidence_weight", 0.1),
        global_latent_condition_main_stream=hparams.get("global_latent_condition_main_stream", True),
        global_latent_film_layers=hparams.get("global_latent_film_layers", "last"),
        latent_self_dropout_prob=hparams.get("latent_self_dropout_prob", 0.0),
        distal_threshold_cap=hparams.get("distal_threshold_cap", 0.0),
        distal_jitter_cap=hparams.get("distal_jitter_cap", 0.0),
        distal_shell_ramp_epochs=hparams.get("distal_shell_ramp_epochs", 0),
        graft_init_std=hparams.get("graft_init_std", 0.0),
        use_interaction_intent=hparams.get("use_interaction_intent", False),
        interaction_intent_loss_weight=hparams.get("interaction_intent_loss_weight", 1.0),
        interaction_intent_num_classes=hparams.get("interaction_intent_num_classes", 4),
        interaction_intent_velocity_bias=hparams.get("interaction_intent_velocity_bias", True),
        num_edge_types=hparams.get("num_edge_types", 3),
        activation_checkpointing=False,  # Never checkpoint during eval (no memory pressure, want speed)
        use_chirality=hparams.get("use_chirality", False),
        pairwise_distance_loss_weight=hparams.get("pairwise_distance_loss_weight", 0.0),
        bonded_geometry_loss_weight=hparams.get("bonded_geometry_loss_weight", 0.0),
        rotamer_rmsd_loss_weight=hparams.get("rotamer_rmsd_loss_weight", 0.0),
        bond_window_loss_weight=hparams.get("bond_window_loss_weight", 0.0),
        bond_angle_loss_weight=hparams.get("bond_angle_loss_weight", 0.0),
        bond_length_loss_weight=hparams.get("bond_length_loss_weight", 0.0),
        scale_loss_weight=hparams.get("scale_loss_weight", 0.0),
        dist_from_ca_loss_weight=hparams.get("dist_from_ca_loss_weight", 0.0),
    )
    model.load_state_dict(model_state, strict=False)

    # Residue-name -> DB-row lookup, built exactly as the training module built it (from the
    # SAME residue DB). Only used by the eval collate to fill batch["residue_indices"].
    db = torch.load(residue_db, weights_only=False)
    name_to_idx = build_name_to_idx(db["metadata"])

    lightning_module = _EvalModule(model=model, name_to_idx=name_to_idx)
    # Preserve the RAW stored slot-layout hparam (None when the ckpt predates it) so eval-side loaders
    # can AUTO-DETECT the reserved-slot0 layout and warn on older ckpts. reserved_slot0 absent -> None.
    lightning_module._ckpt_reserved_slot0 = hparams.get("reserved_slot0", None)
    lightning_module = lightning_module.to(device)
    lightning_module.eval()
    return lightning_module
