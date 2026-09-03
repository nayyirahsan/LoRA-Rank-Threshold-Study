"""Oracle low-rank truncation of the full fine-tuning update (hypothesis H2).

For every LoRA-targeted linear layer, take dW = W_FT - W_base, SVD it once, and build
models whose weights are W_base + SVD_r(dW). Evaluating those models at each rank r
separates two explanations for LoRA's rank threshold:

  oracle(r) ~ LoRA(r)   ->  capacity: no rank-r update can do better
  oracle(r) >> LoRA(r)  ->  optimization: a good rank-r update exists but LoRA doesn't find it

Only the LoRA target modules are truncated. Everything else (embeddings, norms) stays at
base, the same parameters LoRA leaves frozen. So oracle(full rank) is the "linear-only FT"
ceiling, and the gap between it and real FT is the cost of not training those parameters.

Memory: factors are kept in float32 on `device`. Float16 would lose the small tail
singular directions. That's ~2.5GB for Qwen3-0.6B.
"""
from __future__ import annotations

import torch
from torch import nn

TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def target_linears(model: nn.Module, targets=TARGET_MODULES) -> dict[str, nn.Linear]:
    return {
        name: m for name, m in model.named_modules() if isinstance(m, nn.Linear) and name.split(".")[-1] in targets
    }


def _energy(singular_values: torch.Tensor) -> torch.Tensor:
    # Squared singular values in float64 on the CPU. This is at most a few thousand numbers. With the SVD on
    # CUDA, the searchsorted threshold used to be created on the CPU, and the first grid_final oracle
    # sweep died on every run with a device-mismatch error. The move and the cast must be separate
    # steps: a single .to("cpu", torch.float64) on an MPS tensor returned zeros (MPS has no float64).
    return singular_values.detach().to("cpu").to(torch.float64).square()


def energy_rank(singular_values: torch.Tensor, fraction: float = 0.9) -> int:
    """Smallest r whose top-r singular values hold `fraction` of the squared Frobenius norm."""
    energy = _energy(singular_values)
    total = energy.sum()
    if total == 0:
        return 0
    cumulative = energy.cumsum(0) / total
    return int(torch.searchsorted(cumulative, torch.tensor(fraction, dtype=cumulative.dtype)).item()) + 1


def energy_captured(singular_values: torch.Tensor, rank: int) -> float:
    energy = _energy(singular_values)
    total = energy.sum()
    return float(energy[:rank].sum() / total) if total > 0 else 1.0


class DeltaSVD:
    """SVD factors of dW for every target linear, computed once and reused for every rank."""

    def __init__(self, base: nn.Module, tuned: nn.Module, targets=TARGET_MODULES, device: str = "cpu"):
        base_linears, tuned_linears = target_linears(base, targets), target_linears(tuned, targets)
        if base_linears.keys() != tuned_linears.keys():
            raise ValueError("base and tuned models have different target modules")
        self.device = device
        self.base_weights: dict[str, torch.Tensor] = {}
        self.us: dict[str, torch.Tensor] = {}   # U * S, shape (out, k)
        self.vh: dict[str, torch.Tensor] = {}   # Vh, shape (k, in)
        self.singular_values: dict[str, torch.Tensor] = {}
        with torch.no_grad():
            for name, base_lin in base_linears.items():
                w_base = base_lin.weight.detach().to(device=device, dtype=torch.float32)
                w_tuned = tuned_linears[name].weight.detach().to(device=device, dtype=torch.float32)
                u, s, vh = torch.linalg.svd(w_tuned - w_base, full_matrices=False)
                self.base_weights[name] = w_base
                self.us[name] = u * s
                self.vh[name] = vh
                self.singular_values[name] = s

    @property
    def max_rank(self) -> int:
        return max(s.numel() for s in self.singular_values.values())

    def energy_ranks(self, fraction: float = 0.9) -> dict[str, int]:
        return {name: energy_rank(s, fraction) for name, s in self.singular_values.items()}

    @torch.no_grad()
    def apply(self, model: nn.Module, rank: int) -> None:
        """Set model's target weights, in place, to W_base + SVD_rank(dW). Non-target parameters are untouched."""
        for name, lin in target_linears(model).items():
            r = min(rank, self.singular_values[name].numel())
            w = self.base_weights[name] + self.us[name][:, :r] @ self.vh[name][:r]
            lin.weight.copy_(w.to(device=lin.weight.device, dtype=lin.weight.dtype))
