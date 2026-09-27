"""Radius graphs and learned edge types used by the released SE(3) transformer.

Graph nodes combine peptide side chains, peptide backbone atoms, and target
atoms. Edge ordering is preserved because it affects message aggregation.
"""

from __future__ import annotations

import torch
import torch.nn as nn

# Edge type constants
EDGE_TYPE_INTRA_RESIDUE = 0  # Within same binder residue
EDGE_TYPE_INTER_RESIDUE = 1  # Across binder residues
EDGE_TYPE_BINDER_TARGET = 2  # Binder <-> target
NUM_EDGE_TYPES = 3


class EdgeTypeEmbedding(nn.Module):
    """
    Embedding layer for edge types.

    Converts edge type indices to learned embeddings that can be passed
    to SE(3) transformer layers as edge features.

    Parameters
    ----------
    num_types : int
        Number of edge types (default: 3 for intra/inter/binder-target).
    embed_dim : int
        Dimension of edge embeddings.
    """

    def __init__(self, num_types: int = NUM_EDGE_TYPES, embed_dim: int = 16):
        super().__init__()
        self.embed = nn.Embedding(num_types, embed_dim)

    def forward(self, edge_type: torch.Tensor) -> torch.Tensor:
        """
        Embed edge types.

        Parameters
        ----------
        edge_type : torch.Tensor
            Edge type indices of shape (E,).

        Returns
        -------
        edge_attr : torch.Tensor
            Edge embeddings of shape (E, embed_dim).
        """
        return self.embed(edge_type)


def build_multi_type_radius_graph(
    binder_sc_coords: torch.Tensor,
    binder_sc_residue_idx: torch.Tensor,
    binder_bb_coords: torch.Tensor | None = None,
    binder_bb_residue_idx: torch.Tensor | None = None,
    target_coords: torch.Tensor | None = None,
    intra_residue_cutoff: float = 8.0,
    inter_residue_cutoff: float = 8.0,
    sidechain_target_cutoff: float = 10.0,
    backbone_target_cutoff: float = 25.0,
    ca_ca_prefilter: float = 20.0,
    binder_ca_coords: torch.Tensor | None = None,
    target_ca_coords: torch.Tensor | None = None,
    binder_ca_only_coords: torch.Tensor | None = None,
    binder_ca_only_bb_indices: torch.Tensor | None = None,
    num_edge_types: int = 3,
) -> tuple[torch.Tensor, torch.Tensor, int, int, int]:
    """
    Build a radius graph with multiple edge types for full geometric context.

    Creates edges for:
    - Intra-residue: atoms within the same binder residue
    - Inter-residue: atoms across different binder residues
    - Binder-target: binder atoms to target atoms, with optional split cutoffs

    When ``binder_ca_only_coords`` is provided, binder-target edges use split
    cutoffs: SC->target at ``sidechain_target_cutoff`` and CA->target at
    ``backbone_target_cutoff``. Otherwise, all binder->target edges use
    ``sidechain_target_cutoff`` (legacy single-cutoff mode).

    Parameters
    ----------
    binder_sc_coords : torch.Tensor
        Binder sidechain atom coordinates of shape (N_sc, 3).
    binder_sc_residue_idx : torch.Tensor
        Residue index for each binder sidechain atom of shape (N_sc,).
    binder_bb_coords : torch.Tensor, optional
        Binder backbone atom coordinates of shape (N_bb, 3).
    binder_bb_residue_idx : torch.Tensor, optional
        Residue index for each binder backbone atom of shape (N_bb,).
    target_coords : torch.Tensor, optional
        Target atom coordinates of shape (N_target, 3).
    intra_residue_cutoff : float
        Distance cutoff for intra-residue edges (default 8.0 Å).
    inter_residue_cutoff : float
        Distance cutoff for inter-residue edges (default 8.0 Å).
    sidechain_target_cutoff : float
        Distance cutoff for SC->target edges (default 10.0 Å).
    backbone_target_cutoff : float
        Distance cutoff for CA->target edges (default 25.0 Å). Only used when
        ``binder_ca_only_coords`` is provided (split cutoff mode).
    ca_ca_prefilter : float
        CA-CA distance prefilter for binder-target edges (default 20.0 Å).
        Only target atoms near binder are considered.
    binder_ca_coords : torch.Tensor, optional
        Binder CA coordinates for prefiltering, shape (L_binder, 3).
    target_ca_coords : torch.Tensor, optional
        Target CA coordinates for prefiltering, shape (L_target, 3).
    binder_ca_only_coords : torch.Tensor, optional
        CA atom coordinates extracted from the backbone block, shape (N_ca, 3).
        When provided, enables split cutoff mode.
    binder_ca_only_bb_indices : torch.Tensor, optional
        Indices of CA atoms within the backbone block, shape (N_ca,). Used to
        map CA rows back to correct node indices in the combined graph.

    Returns
    -------
    edge_index : torch.Tensor
        Edge indices of shape (2, E).
    edge_type : torch.Tensor
        Edge type for each edge of shape (E,).
        0 = intra-residue, 1 = inter-residue, 2 = binder-target
    n_binder_sc : int
        Number of binder sidechain atoms.
    n_binder_bb : int
        Number of binder backbone atoms.
    n_target : int
        Number of target atoms.
    """
    device = binder_sc_coords.device
    n_binder_sc = binder_sc_coords.size(0)
    n_binder_bb = binder_bb_coords.size(0) if binder_bb_coords is not None else 0
    n_target = target_coords.size(0) if target_coords is not None else 0

    edge_list = []
    edge_type_list = []

    # --- Intra-residue edges (binder sidechain only) ---
    if n_binder_sc > 0:
        dist_sc = torch.cdist(binder_sc_coords, binder_sc_coords)
        for res_idx in binder_sc_residue_idx.unique():
            res_mask = binder_sc_residue_idx == res_idx
            res_indices = torch.where(res_mask)[0]
            if len(res_indices) < 2:
                continue
            # Submatrix for this residue
            sub_dist = dist_sc[res_mask][:, res_mask]
            row, col = torch.where((sub_dist < intra_residue_cutoff) & (sub_dist > 0))
            if len(row) > 0:
                edge_list.append(torch.stack([res_indices[row], res_indices[col]]))
                edge_type_list.append(torch.full((len(row),), EDGE_TYPE_INTRA_RESIDUE, device=device))

    # --- Inter-residue edges (binder sidechain to sidechain, different residues) ---
    if n_binder_sc > 1:
        dist_sc = torch.cdist(binder_sc_coords, binder_sc_coords)
        row, col = torch.where((dist_sc < inter_residue_cutoff) & (dist_sc > 0))
        # Filter to different residues
        diff_res_mask = binder_sc_residue_idx[row] != binder_sc_residue_idx[col]
        row, col = row[diff_res_mask], col[diff_res_mask]
        if len(row) > 0:
            edge_list.append(torch.stack([row, col]))
            edge_type_list.append(torch.full((len(row),), EDGE_TYPE_INTER_RESIDUE, device=device))

    # --- Binder backbone to binder sidechain edges (inter-residue type) ---
    if n_binder_bb > 0 and n_binder_sc > 0:
        dist_bb_sc = torch.cdist(binder_bb_coords, binder_sc_coords)
        row, col = torch.where(dist_bb_sc < inter_residue_cutoff)
        # Filter to different residues (backbone atom i shouldn't connect to its own sidechain)
        diff_res_mask = binder_bb_residue_idx[row] != binder_sc_residue_idx[col]
        row_filt, col_filt = row[diff_res_mask], col[diff_res_mask]
        if len(row_filt) > 0:
            # Offset backbone indices to come after sidechain
            bb_offset = n_binder_sc
            edge_list.append(torch.stack([row_filt + bb_offset, col_filt]))
            edge_list.append(torch.stack([col_filt, row_filt + bb_offset]))  # Symmetric
            edge_type_list.append(torch.full((len(row_filt),), EDGE_TYPE_INTER_RESIDUE, device=device))
            edge_type_list.append(torch.full((len(row_filt),), EDGE_TYPE_INTER_RESIDUE, device=device))

    # --- Binder-target edges ---
    if n_target > 0 and (n_binder_sc > 0 or n_binder_bb > 0):
        # Optional CA-CA prefiltering
        target_mask = torch.ones(n_target, dtype=torch.bool, device=device)
        if binder_ca_coords is not None and target_ca_coords is not None:
            # Find target residues within ca_ca_prefilter of any binder CA
            ca_ca_dist = torch.cdist(target_ca_coords, binder_ca_coords)
            min_dist_to_binder = ca_ca_dist.min(dim=1).values
            target_mask = min_dist_to_binder < ca_ca_prefilter

        target_indices_filtered = torch.where(target_mask)[0]
        if len(target_indices_filtered) > 0:
            target_coords_filtered = target_coords[target_mask]
            target_offset = n_binder_sc + n_binder_bb

            if binder_ca_only_coords is not None and binder_ca_only_bb_indices is not None:
                # Split cutoff mode: SC->target tight, CA->target broad
                # SC->target edges
                if n_binder_sc > 0:
                    dist_sc_t = torch.cdist(binder_sc_coords, target_coords_filtered)
                    row, col = torch.where(dist_sc_t < sidechain_target_cutoff)
                    if len(row) > 0:
                        col_global = target_indices_filtered[col] + target_offset
                        edge_list.append(torch.stack([row, col_global]))
                        edge_type_list.append(torch.full((len(row),), EDGE_TYPE_BINDER_TARGET, device=device))
                        edge_list.append(torch.stack([col_global, row]))
                        edge_type_list.append(torch.full((len(row),), EDGE_TYPE_BINDER_TARGET, device=device))

                # CA->target edges
                if len(binder_ca_only_coords) > 0:
                    dist_ca_t = torch.cdist(binder_ca_only_coords, target_coords_filtered)
                    row, col = torch.where(dist_ca_t < backbone_target_cutoff)
                    if len(row) > 0:
                        row_global = binder_ca_only_bb_indices[row] + n_binder_sc
                        col_global = target_indices_filtered[col] + target_offset
                        edge_list.append(torch.stack([row_global, col_global]))
                        edge_type_list.append(torch.full((len(row),), EDGE_TYPE_BINDER_TARGET, device=device))
                        edge_list.append(torch.stack([col_global, row_global]))
                        edge_type_list.append(torch.full((len(row),), EDGE_TYPE_BINDER_TARGET, device=device))
            else:
                # Legacy single-cutoff mode: all binder -> target
                if n_binder_bb > 0:
                    binder_all_coords = torch.cat([binder_sc_coords, binder_bb_coords], dim=0)
                else:
                    binder_all_coords = binder_sc_coords

                dist_bt = torch.cdist(binder_all_coords, target_coords_filtered)
                row, col = torch.where(dist_bt < sidechain_target_cutoff)
                if len(row) > 0:
                    col_global = target_indices_filtered[col] + target_offset
                    edge_list.append(torch.stack([row, col_global]))
                    edge_type_list.append(torch.full((len(row),), EDGE_TYPE_BINDER_TARGET, device=device))
                    edge_list.append(torch.stack([col_global, row]))
                    edge_type_list.append(torch.full((len(row),), EDGE_TYPE_BINDER_TARGET, device=device))

    # Concatenate all edges
    if edge_list:
        edge_index = torch.cat(edge_list, dim=1)
        edge_type = torch.cat(edge_type_list, dim=0)
    else:
        edge_index = torch.zeros(2, 0, dtype=torch.long, device=device)
        edge_type = torch.zeros(0, dtype=torch.long, device=device)

    # Optional: collapse to 2-way edge types (intra-residue vs extra-residue).
    # Maps inter-residue (1) and binder-target (2) both -> 1, keeping intra-residue (0) as 0.
    # This is forward-compatible with monomer / small-molecule pretraining where the
    # inter-residue vs binder-target distinction doesn't exist naturally.
    if num_edge_types == 2 and edge_type.numel() > 0:
        edge_type = (edge_type > 0).long()

    return edge_index, edge_type, n_binder_sc, n_binder_bb, n_target
