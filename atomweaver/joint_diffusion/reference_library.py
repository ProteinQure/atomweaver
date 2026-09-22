"""Reference geometry and residue identities shared by read-out tools."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .matching import GeometricMatcher


@dataclass
class ReferenceLibrary:
    """Reference tensors and their ordered residue-type mapping."""

    data: dict
    ccd_to_idx: dict[str, int]
    rotamer_to_type: torch.Tensor
    representable: torch.Tensor

    @classmethod
    def load(cls, path: str) -> ReferenceLibrary:
        """Load the library using the released 14-heavy-atom candidate budget."""
        data = torch.load(path, map_location="cpu", weights_only=True)
        ccd_to_idx = {}
        rotamer_types = []
        representable = []
        for index, meta in enumerate(data["metadata"]):
            rotamer_types.extend([index] * int(meta.get("num_rotamers", 1)))
            for code in meta.get("pdb_ids", []) or []:
                if code:
                    ccd_to_idx.setdefault(code, index)
            representable.append(max(int(meta.get("num_atoms", 0)) - 4, 0) <= 14)
        return cls(data, ccd_to_idx, torch.tensor(rotamer_types), torch.tensor(representable))

    def matcher(
        self,
        *,
        device: str = "cpu",
        atom_mismatch_penalty: float = 0.5,
        element_mismatch_penalty: float = 0.3,
        chirality_mismatch_penalty: float = 50.0,
        repack_prediction: bool = True,
    ) -> GeometricMatcher:
        """Build an inference scorer without creating any training-loss machinery."""
        return (
            GeometricMatcher(
                residue_database=self.data["coords"],
                residue_masks=self.data["masks"],
                backbone_indices=self.data.get("backbone_indices"),
                element_types=self.data.get("element_types"),
                rotamer_to_type=self.rotamer_to_type,
                atom_mismatch_penalty=atom_mismatch_penalty,
                element_mismatch_penalty=element_mismatch_penalty,
                chirality_mismatch_penalty=chirality_mismatch_penalty,
                repack_prediction=repack_prediction,
            )
            .to(device)
            .eval()
        )
