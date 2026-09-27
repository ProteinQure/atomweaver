"""
SE(3) Equivariant Transformer for molecular modeling.

This module implements an SE(3)-equivariant attention mechanism that learns
all-to-all interactions between atoms while preserving geometric equivariance.

Layers update node FEATURES ONLY, over a fixed input geometry; coordinates are a pure
pass-through. Equivariance holds because features depend on coordinates only through invariant
pairwise distances, which are computed from the (constant) input coordinates at every layer.

Key differences from EGNN:
- Uses attention (softmax over queries/keys) instead of fixed message aggregation
- All-to-all attention allows learning complex interaction patterns
- Does NOT mutate coordinates (EGNN updates them); geometry is fixed across the stack
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def _should_checkpoint_layer(i: int, stride: float) -> bool:
    """
    Decide whether SE3 layer ``i`` is activation-checkpointed for a given stride.

    Uses a floor-based formula that generalizes the integer ``i % stride == 0``
    rule to fractional strides while reducing *exactly* to it for integer
    ``stride``. A new "checkpoint bucket" opens whenever ``floor(i / stride)``
    advances, so layer ``i`` is checkpointed iff it is the first layer in its
    bucket.

    Examples (10 layers, ``i`` in ``range(10)``)
    --------
    - ``stride=1.0`` -> all 10 layers
    - ``stride=2.0`` -> ``{0, 2, 4, 6, 8}`` (identical to ``i % 2 == 0``)
    - ``stride=3.0`` -> ``{0, 3, 6, 9}``
    - ``stride=1.5`` -> ``{0, 2, 3, 5, 6, 8, 9}`` (7 layers, stores 3)

    Parameters
    ----------
    i : int
        Zero-based layer index.
    stride : float
        Checkpoint stride (>= 1.0). Larger -> fewer checkpointed layers.

    Returns
    -------
    bool
        Whether layer ``i`` should be activation-checkpointed.
    """
    return math.floor(i / stride) != math.floor((i - 1) / stride)


class SE3AttentionLayer(nn.Module):
    """
    SE(3) Equivariant Attention Layer.

    Uses multi-head attention where:
    - Query/Key/Value are computed from node features (invariant)
    - Attention scores are modulated by geometric features (distances)

    The layer updates node FEATURES ONLY, over a fixed geometry: coordinates are a pure
    pass-through (identity). Distances feeding the attention bias are computed from the input
    coordinates, which stay constant across the stack, so every layer attends over the same true
    geometry the radius graph was built on. Equivariance is preserved because features depend on
    coordinates only through invariant pairwise distances.

    Parameters
    ----------
    node_dim : int
        Dimension of node features.
    num_heads : int
        Number of attention heads.
    edge_dim : int, optional
        Dimension of edge features (e.g., edge type embeddings).
    dropout : float
        Attention dropout rate.
    """

    def __init__(
        self,
        node_dim: int,
        num_heads: int = 4,
        edge_dim: int | None = None,
        dropout: float = 0.0,
        rbf_max_dist: float | None = None,
    ):
        super().__init__()

        self.node_dim = node_dim
        self.num_heads = num_heads
        self.head_dim = node_dim // num_heads
        self.edge_dim = edge_dim or 0
        self.scale = self.head_dim**-0.5

        # Query, Key, Value projections
        self.q_proj = nn.Linear(node_dim, node_dim)
        self.k_proj = nn.Linear(node_dim, node_dim)
        self.v_proj = nn.Linear(node_dim, node_dim)
        self.out_proj = nn.Linear(node_dim, node_dim)

        # Geometric attention bias: distance -> attention bias per head
        # Uses distance features: [1/d, 1/d^2, radial basis functions]
        self.n_dist_features = 16  # Number of radial basis functions
        dist_input_dim = 2 + self.n_dist_features + self.edge_dim  # 1/d, 1/d^2, RBFs, edge_attr
        self.dist_to_bias = nn.Sequential(
            nn.Linear(dist_input_dim, node_dim),
            nn.SiLU(),
            nn.Linear(node_dim, num_heads),
        )

        # Layer norm and dropout
        self.norm1 = nn.LayerNorm(node_dim)
        self.norm2 = nn.LayerNorm(node_dim)
        self.dropout = nn.Dropout(dropout)

        # FFN for node features
        self.ffn = nn.Sequential(
            nn.Linear(node_dim, node_dim * 4),
            nn.SiLU(),
            nn.Linear(node_dim * 4, node_dim),
        )

        # RBF centers and widths for distance encoding.

        # The RBF span must cover the LARGEST radial extent of edges the graph actually carries.
        # The live graph includes CA->target context edges out to ``backbone_target_cutoff`` (25 Å /
        # 15 Å depending on run), but a hardcoded 0-10 Å span (width 1.0) saturates every Gaussian to
        # ~0 past ~12 Å, leaving the bias net distance-blind on exactly the 12-25 Å edges that tell the
        # binder how far the target is. When ``rbf_max_dist`` is supplied we widen the span to that
        # cutoff and scale the Gaussian width to the new center spacing (span / (n-1)).
        # The number of centers is UNCHANGED (``n_dist_features``), so
        # ``dist_to_bias``'s input dim -- and every learned weight -- is identical: this is a buffer-only
        # change that loads cleanly from a base checkpoint.

        # The buffer is registered NON-persistent so it is (a) recomputed from config at construction
        # and (b) NOT clobbered by a base checkpoint's stale 0-10 ``rbf_centers`` on a strict=False
        # resume -- which would otherwise silently revert the widened span.
        if rbf_max_dist is None:
            # Legacy default: byte-identical to the historical 0-10 Å / width-1.0 encoding.
            rbf_span = 10.0
            rbf_width = 1.0
        else:
            rbf_span = float(rbf_max_dist)
            rbf_width = max(rbf_span / max(self.n_dist_features - 1, 1), 1e-3)
        self.register_buffer("rbf_centers", torch.linspace(0.0, rbf_span, self.n_dist_features), persistent=False)
        self.rbf_width = rbf_width

    def _compute_distance_features(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None,
    ) -> torch.Tensor:
        """
        Compute geometric features from pairwise distances.

        Parameters
        ----------
        x : torch.Tensor
            Coordinates of shape (N, 3).
        edge_index : torch.Tensor
            Edge indices of shape (2, E).
        edge_attr : torch.Tensor, optional
            Edge attributes of shape (E, edge_dim).

        Returns
        -------
        dist_features : torch.Tensor
            Distance features of shape (E, dist_input_dim).
        """
        row, col = edge_index
        rel_pos = x[col] - x[row]  # (E, 3)
        dist = torch.norm(rel_pos, dim=-1, keepdim=True).clamp(min=1e-6)  # (E, 1)

        # Distance-based features
        inv_dist = 1.0 / dist  # (E, 1)
        inv_dist_sq = 1.0 / (dist**2)  # (E, 1)

        # Radial basis functions
        rbf = torch.exp(-((dist - self.rbf_centers) ** 2) / (2 * self.rbf_width**2))  # (E, n_rbf)

        features = [inv_dist, inv_dist_sq, rbf]
        if edge_attr is not None:
            features.append(edge_attr)

        return torch.cat(features, dim=-1)

    def forward(
        self,
        h: torch.Tensor,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with SE(3) equivariant attention.

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
            Updated node features of shape (N, node_dim).
        x_out : torch.Tensor
            Coordinates of shape (N, 3), returned unchanged (identity pass-through).
        """
        n_nodes = h.shape[0]
        row, col = edge_index

        # Compute distance features for attention bias
        dist_features = self._compute_distance_features(x, edge_index, edge_attr)
        attn_bias = self.dist_to_bias(dist_features)  # (E, num_heads)

        # Compute Q, K, V
        q = self.q_proj(h).view(n_nodes, self.num_heads, self.head_dim)  # (N, H, D)
        k = self.k_proj(h).view(n_nodes, self.num_heads, self.head_dim)  # (N, H, D)
        v = self.v_proj(h).view(n_nodes, self.num_heads, self.head_dim)  # (N, H, D)

        # Compute attention scores for each edge
        # attn_ij = (q_i · k_j) / sqrt(d) + bias_ij
        q_i = q[row]  # (E, H, D)
        k_j = k[col]  # (E, H, D)
        attn_scores = (q_i * k_j).sum(dim=-1) * self.scale  # (E, H)
        attn_scores = attn_scores + attn_bias  # Add geometric bias

        # Softmax over neighbors for each node
        # We need to group edges by source node and softmax within each group
        attn_weights = self._sparse_softmax(attn_scores, row, n_nodes)  # (E, H)
        attn_weights = self.dropout(attn_weights)

        # Aggregate values
        v_j = v[col]  # (E, H, D)
        weighted_v = attn_weights.unsqueeze(-1) * v_j  # (E, H, D)

        # Sum over neighbors for each node
        h_update = torch.zeros(n_nodes, self.num_heads, self.head_dim, device=h.device)
        h_update.index_add_(0, row, weighted_v)
        h_update = h_update.view(n_nodes, -1)  # (N, node_dim)

        # Output projection and residual
        h_update = self.out_proj(h_update)
        h = self.norm1(h + self.dropout(h_update))

        # FFN with residual
        h = self.norm2(h + self.dropout(self.ffn(h)))

        # Coordinates pass through UNCHANGED (identity). We deliberately do NOT mutate x here.
        # Rationale: the denoiser reads its velocity solely from the invariant feature stream and
        # DISCARDS the stack's returned coordinates. The old EGNN-style `x = x + 0.1 * x_update`
        # therefore had exactly one live effect: it drifted `x` across depth, so each layer's
        # `_compute_distance_features(x, ...)` saw a running (drifted) geometry while the edge list
        # was built once by radius on the true input `x_t`. That decoupled neighbor GEOMETRY from
        # neighbor SELECTION -- a depth-dependent, unsupervised internal self-inconsistency. Keeping
        # x fixed makes every layer attend over the same true `x_t` geometry the radius graph was
        # built on, restoring intra-pass geometric consistency.
        return h, x

    def _sparse_softmax(
        self,
        scores: torch.Tensor,
        indices: torch.Tensor,
        n_nodes: int,
    ) -> torch.Tensor:
        """
        Compute softmax over sparse adjacency (grouped by source node).

        Parameters
        ----------
        scores : torch.Tensor
            Attention scores of shape (E, H).
        indices : torch.Tensor
            Source node indices of shape (E,).
        n_nodes : int
            Total number of nodes.

        Returns
        -------
        weights : torch.Tensor
            Softmax weights of shape (E, H).
        """
        # Subtract max for numerical stability (per node, per head)
        max_scores = torch.zeros(n_nodes, scores.shape[1], device=scores.device)
        max_scores.index_reduce_(0, indices, scores, reduce="amax", include_self=False)
        max_scores = max_scores[indices]  # (E, H)
        scores = scores - max_scores

        # Exp
        exp_scores = torch.exp(scores)

        # Sum per node
        sum_exp = torch.zeros(n_nodes, scores.shape[1], device=scores.device)
        sum_exp.index_add_(0, indices, exp_scores)
        sum_exp = sum_exp[indices].clamp(min=1e-8)  # (E, H)

        return exp_scores / sum_exp


class SE3Transformer(nn.Module):
    """
    Stack of SE(3) Equivariant Transformer layers.

    Parameters
    ----------
    node_dim : int
        Dimension of node features.
    hidden_dim : int
        Hidden dimension (should equal node_dim for residual connections).
    out_dim : int
        Output dimension.
    edge_dim : int, optional
        Dimension of edge features.
    num_layers : int
        Number of transformer layers.
    num_heads : int
        Number of attention heads per layer.
    dropout : float
        Dropout rate.
    """

    def __init__(
        self,
        node_dim: int,
        hidden_dim: int,
        out_dim: int,
        edge_dim: int | None = None,
        num_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.0,
        activation_checkpointing: bool = False,
        activation_checkpoint_stride: float = 1.0,
        rbf_max_dist: float | None = None,
    ):
        super().__init__()

        self.node_dim = node_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.num_layers = num_layers
        self.activation_checkpointing = activation_checkpointing
        # Selective ("less extreme") checkpointing: with stride s>1 we recompute only every
        # s-th layer, leaving the rest's activations cached. stride=1 -> checkpoint every layer
        # (full, ~30% wallclock tax); larger stride trades less memory savings for less recompute.
        # Tune the smallest stride that still fits the batch (e.g. bs=1 on a 24 GB card).
        self.activation_checkpoint_stride = max(1.0, float(activation_checkpoint_stride))

        # Input projection if needed
        self.input_proj = nn.Linear(node_dim, hidden_dim) if node_dim != hidden_dim else nn.Identity()

        # Transformer layers
        self.layers = nn.ModuleList(
            [
                SE3AttentionLayer(
                    node_dim=hidden_dim,
                    num_heads=num_heads,
                    edge_dim=edge_dim,
                    dropout=dropout,
                    rbf_max_dist=rbf_max_dist,
                )
                for _ in range(num_layers)
            ]
        )

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, out_dim) if hidden_dim != out_dim else nn.Identity()

    def forward(
        self,
        h: torch.Tensor,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
        layer_conditioning: list[torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through all transformer layers.

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
        layer_conditioning : list of torch.Tensor, optional
            Per-layer node conditioning to add into the node features *before* each layer (the
            clean-context per-layer interweave). Length must equal ``num_layers``; element ``i`` has
            shape (N, hidden_dim). ``None`` (the default) leaves the pre-interweave path byte-identical;
            the caller supplies already-**zero-init-projected** tensors, so the sum is an exact no-op at
            init and ramps as the projections learn.

        Returns
        -------
        h_out : torch.Tensor
            Output node features of shape (N, out_dim).
        x_out : torch.Tensor
            Coordinates of shape (N, 3), returned unchanged (identity pass-through through every layer).
        """
        h = self.input_proj(h)

        for i, layer in enumerate(self.layers):
            if layer_conditioning is not None:
                # Zero-init-projected clean-context conditioning: identity at init, ramps as it learns.
                h = h + layer_conditioning[i]
            checkpoint_this_layer = (
                self.activation_checkpointing
                and self.training
                and _should_checkpoint_layer(i, self.activation_checkpoint_stride)
            )
            if checkpoint_this_layer:
                # Recompute activations during backward to save memory.
                # use_reentrant=False is the recommended mode for new code.
                h, x = torch.utils.checkpoint.checkpoint(layer, h, x, edge_index, edge_attr, use_reentrant=False)
            else:
                h, x = layer(h, x, edge_index, edge_attr)

        h = self.output_proj(h)

        return h, x
