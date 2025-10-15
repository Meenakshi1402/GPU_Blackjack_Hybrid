"""
engine.py
---------
High-level GPU engine that controls blackjack simulation:
  - Prepares data (RNG states, rule tensors, result buffers)
  - Launches the Triton kernel (simulate_hands_kernel)
  - Collects and aggregates results from GPU
"""
import torch, math
from dataclasses import dataclass
from .rng import make_rng_states
from .device_strategy import rules_43x10_tensor
from .kernels import simulate_hands_kernel

@dataclass
class GPUStats:
    wins: int
    losses: int
    pushes: int
    total_pl: float
 """
    Run blackjack simulation on GPU using Triton.

    Args:
        ngames: number of blackjack hands to simulate
        seed: RNG seed for reproducibility
        block: number of GPU threads per block (Triton compile-time constant)
 """
def run_gpu_blackjack(ngames: int = 1_000_000, seed: int = 42, block: int = 8192):
    device = "cuda"
    rules = rules_43x10_tensor(device)
    n = ngames
    grid = (math.ceil(n / block),)

    # prepare buffers
    rng = make_rng_states(grid[0] * block, seed).to(device)
    results = torch.zeros(grid[0] * block, dtype=torch.float32, device=device)
    counts = torch.zeros(3, dtype=torch.int32, device=device)

    simulate_hands_kernel[grid](
    rng,
    rules.flatten(),
    results,
    counts,
    n,
    PAYOFF_BJ=1.5,      #  pass plain Python floats (not torch tensors)
    PAYOFF_PUSH=0.0,
    BLOCK=block,
)


    # gather results
    wins, losses, pushes = counts.cpu().tolist()
    total_pl = results[:n].sum().item()
    return GPUStats(wins=wins, losses=losses, pushes=pushes, total_pl=total_pl)
