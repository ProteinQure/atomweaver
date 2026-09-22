"""
E(n) Equivariant Graph Neural Network (EGNN) for SE(3)-equivariant molecular modeling.

This module implements EGNN as described in:
    Satorras et al., "E(n) Equivariant Graph Neural Networks" (2021)
    https://arxiv.org/abs/2102.09844

EGNN is chosen as the backbone for our diffusion model due to its simplicity and
efficiency. However, there are several alternative SE(3)-equivariant architectures
worth considering for future work:

Alternative Architectures
-------------------------
1. **SE3-Transformer / Tensor Field Networks (TFN)**
   - Paper: Fuchs et al., "SE(3)-Transformers" (NeurIPS 2020)
   - Pros: More expressive than EGNN. Uses spherical harmonics to represent
     higher-order geometric features (directions, angles, etc.). Can capture
     angular relationships explicitly through irreducible representations.
   - Cons: Computationally expensive due to spherical harmonics and Clebsch-Gordan
     coefficient computations. Memory-hungry. Complex implementation requiring
     specialized libraries (e3nn). Slower training and inference.
   - Use when: You need to model angular dependencies, bond angles, or dihedral
     information explicitly. Good for tasks where geometric detail matters.

2. **Equiformer**
   - Paper: Liao & Smidt, "Equiformer" (2022); used by EquiFold for protein folding
   - Pros: State-of-the-art on many molecular benchmarks. Combines SE3-Transformer
     architecture with MLP attention (from GATv2) and non-linear message passing.
     Depthwise tensor products for efficiency.
   - Cons: Even more complex than SE3-Transformer. Heavy dependency on e3nn library.
     Requires careful hyperparameter tuning.
   - Use when: You need maximum expressivity and have compute budget. Good for
     protein structure prediction tasks.

3. **Frame Averaging / Vector Neurons**
   - Paper: Puny et al., "Frame Averaging for Invariant and Equivariant Learning" (2022)
   - Pros: Can make ANY architecture equivariant by averaging predictions over
     random rotations/translations. Conceptually simple.
   - Cons: Requires multiple forward passes (expensive at inference). Approximation
     errors from finite sampling. Not truly equivariant, just approximately so.
   - Use when: You want to retrofit equivariance onto an existing non-equivariant
     model without rewriting it.

4. **GemNet / PaiNN / SchNet variants**
   - Various papers from the molecular ML community
   - Pros: Well-tested on molecular property prediction. Good trade-offs between
     expressivity and efficiency.
   - Cons: Often specialized for specific tasks (e.g., energy prediction).
   - Use when: Your task aligns with their original design goals.

Why EGNN for this project
-------------------------
- Simple to implement from scratch (no e3nn or other heavy dependencies)
- For atom clouds where relationships are primarily distance-based, EGNN's
  expressivity is often sufficient
- DiffPepBuilder and similar peptide diffusion work demonstrates EGNN works
  well for this domain
- Fast training iteration for research exploration

Future Research Directions
--------------------------
**Groupwise / Hierarchical Diffusion Processes**

A promising direction is to use different noise schedules for atoms based on their
distance from the backbone:

1. **Backbone-proximal atoms (e.g., Cb carbons)**: Diffuse LAST (denoise FIRST)
   - These define the "broad direction" of the side chain
   - Less noise in forward process -> emerge from noise early in reverse
   - Resolved first so distal atoms can be conditioned on them

2. **Distal atoms (side chain tips, rings, etc.)**: Diffuse FIRST (denoise LAST)
   - Fine structural details that depend on proximal atom positions
   - More noise in forward process -> stay noisy longer in reverse
   - Resolved last, conditioned on already-placed Cb

This hierarchical approach offers several advantages:
- **Tractability**: For most of the diffusion process, we can focus on coarse
  structure (Cb) without worrying about resolving distal atom positions
- **Physical intuition**: Mirrors the natural hierarchy backbone -> Cb -> side chain
- **Reduced complexity**: Early reverse steps predict Cb direction, later steps
  refine distal atomic positions
- **Better gradients**: Proximal atoms have more stable gradients since they're
  less noisy during training

Implementation ideas:
- Blockwise diffusion with different noise schedules per atom group
- Cascaded diffusion: first predict Cβ positions, then condition distal atoms
- Continuous noise schedule that varies with atom depth (distance from backbone)
- Could use atom "depth" as an additional conditioning signal

This is similar in spirit to cascaded diffusion models for images (coarse-to-fine)
but grounded in molecular structure.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class EGNNLayer(nn.Module):
    """
    Single E(n) Equivariant Graph Neural Network layer.

    Updates node features (invariant) and coordinates (equivariant) via message passing.
    The key insight is that coordinate updates are computed as weighted sums of
    relative position vectors, which preserves equivariance.

    Parameters
    ----------
    node_dim : int
        Dimension of node features.
    edge_dim : int, optional
        Dimension of edge features. If None, no edge features are used.
    hidden_dim : int, optional
        Hidden dimension for MLPs. Defaults to node_dim.
    act : nn.Module, optional
        Activation function. Defaults to SiLU.
    residual : bool
        Whether to use residual connections for node features.
    normalize : bool
        Whether to normalize coordinate updates (improves stability).
    coords_agg : str
        Aggregation method for coordinate updates: 'mean' or 'sum'.
    """

    def __init__(
        self,
        node_dim: int,
        edge_dim: int | None = None,
        hidden_dim: int | None = None,
        act: nn.Module | None = None,
        residual: bool = True,
        normalize: bool = True,
        coords_agg: str = "mean",
    ):
        super().__init__()

        self.node_dim = node_dim
        self.hidden_dim = hidden_dim or node_dim
        self.edge_dim = edge_dim or 0
        self.residual = residual
        self.normalize = normalize
        self.coords_agg = coords_agg

        act = act or nn.SiLU()

        # Edge MLP: computes messages from node pairs + distance + edge features
        edge_input_dim = 2 * node_dim + 1 + self.edge_dim  # h_i, h_j, ||x_i - x_j||^2, edge_attr
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_input_dim, self.hidden_dim),
            act,
            nn.Linear(self.hidden_dim, self.hidden_dim),
            act,
        )

        # Node MLP: updates node features from aggregated messages
        self.node_mlp = nn.Sequential(
            nn.Linear(node_dim + self.hidden_dim, self.hidden_dim),
            act,
            nn.Linear(self.hidden_dim, node_dim),
        )

        # Coordinate MLP: computes scalar weights for coordinate updates
        self.coord_mlp = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            act,
            nn.Linear(self.hidden_dim, 1, bias=False),
        )

        # Optional: attention-like weighting
        self.att_mlp = nn.Sequential(
            nn.Linear(self.hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        h: torch.Tensor,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Parameters
        ----------
        h : torch.Tensor
            Node features of shape (N, node_dim).
        x : torch.Tensor
            Node coordinates of shape (N, 3).
        edge_index : torch.Tensor
            Edge indices of shape (2, E) where E is number of edges.
        edge_attr : torch.Tensor, optional
            Edge features of shape (E, edge_dim).

        Returns
        -------
        h_out : torch.Tensor
            Updated node features of shape (N, node_dim).
        x_out : torch.Tensor
            Updated coordinates of shape (N, 3).
        """
        row, col = edge_index  # row = source, col = target

        # Compute relative positions and distances
        rel_pos = x[row] - x[col]  # (E, 3)
        dist_sq = (rel_pos**2).sum(dim=-1, keepdim=True)  # (E, 1)

        # Build edge input
        edge_input = [h[row], h[col], dist_sq]
        if edge_attr is not None:
            edge_input.append(edge_attr)
        edge_input = torch.cat(edge_input, dim=-1)

        # Compute edge messages
        m_ij = self.edge_mlp(edge_input)  # (E, hidden_dim)

        # Compute attention weights
        att = self.att_mlp(m_ij)  # (E, 1)
        m_ij = m_ij * att

        # Aggregate messages to nodes
        m_i = torch.zeros(h.size(0), self.hidden_dim, device=h.device, dtype=h.dtype)
        m_i.scatter_add_(0, col.unsqueeze(-1).expand(-1, self.hidden_dim), m_ij)

        # Update node features
        h_out = self.node_mlp(torch.cat([h, m_i], dim=-1))
        if self.residual:
            h_out = h + h_out

        # Compute coordinate updates
        coord_weights = self.coord_mlp(m_ij)  # (E, 1)

        if self.normalize:
            # Normalize by distance to improve stability
            dist = torch.sqrt(dist_sq + 1e-8)
            rel_pos_norm = rel_pos / dist
            coord_update = rel_pos_norm * coord_weights
        else:
            coord_update = rel_pos * coord_weights

        # Aggregate coordinate updates
        x_update = torch.zeros_like(x)
        x_update.scatter_add_(0, col.unsqueeze(-1).expand(-1, 3), coord_update)

        if self.coords_agg == "mean":
            # Count number of incoming edges per node
            counts = torch.zeros(x.size(0), 1, device=x.device, dtype=x.dtype)
            counts.scatter_add_(0, col.unsqueeze(-1), torch.ones_like(coord_weights))
            counts = counts.clamp(min=1)
            x_update = x_update / counts

        x_out = x + x_update

        return h_out, x_out


class EGNN(nn.Module):
    """
    E(n) Equivariant Graph Neural Network.

    A stack of EGNN layers for processing molecular graphs with SE(3) equivariance.

    Parameters
    ----------
    node_dim : int
        Dimension of node features.
    hidden_dim : int
        Hidden dimension for MLPs.
    out_dim : int
        Output dimension for node features.
    edge_dim : int, optional
        Dimension of edge features.
    num_layers : int
        Number of EGNN layers.
    residual : bool
        Whether to use residual connections.
    normalize : bool
        Whether to normalize coordinate updates.
    """

    def __init__(
        self,
        node_dim: int,
        hidden_dim: int,
        out_dim: int,
        edge_dim: int | None = None,
        num_layers: int = 4,
        residual: bool = True,
        normalize: bool = True,
    ):
        super().__init__()

        self.node_dim = node_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

        # Input projection
        self.input_proj = nn.Linear(node_dim, hidden_dim)

        # EGNN layers
        self.layers = nn.ModuleList(
            [
                EGNNLayer(
                    node_dim=hidden_dim,
                    edge_dim=edge_dim,
                    hidden_dim=hidden_dim,
                    residual=residual,
                    normalize=normalize,
                )
                for _ in range(num_layers)
            ]
        )

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, out_dim)

    def forward(
        self,
        h: torch.Tensor,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Parameters
        ----------
        h : torch.Tensor
            Node features of shape (N, node_dim).
        x : torch.Tensor
            Node coordinates of shape (N, 3).
        edge_index : torch.Tensor
            Edge indices of shape (2, E).
        edge_attr : torch.Tensor, optional
            Edge features of shape (E, edge_dim).

        Returns
        -------
        h_out : torch.Tensor
            Output node features of shape (N, out_dim).
        x_out : torch.Tensor
            Output coordinates of shape (N, 3).
        """
        h = self.input_proj(h)

        for layer in self.layers:
            h, x = layer(h, x, edge_index, edge_attr)

        h = self.output_proj(h)

        return h, x


# Edge type constants
EDGE_TYPE_INTRA_RESIDUE = 0  # Within same binder residue
EDGE_TYPE_INTER_RESIDUE = 1  # Across binder residues
EDGE_TYPE_BINDER_TARGET = 2  # Binder <-> target
NUM_EDGE_TYPES = 3


class EdgeTypeEmbedding(nn.Module):
    """
    Embedding layer for edge types.

    Converts edge type indices to learned embeddings that can be passed
    to EGNN layers as edge features.

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
