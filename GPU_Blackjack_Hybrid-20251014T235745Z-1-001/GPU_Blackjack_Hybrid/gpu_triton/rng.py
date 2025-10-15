"""
rng.py
------
Generates per-thread random number generator (RNG) states
for use by the Triton blackjack kernel.
Each GPU thread receives a unique 32-bit seed derived from:
  - thread index
  - global seed (user input)
This ensures statistically independent streams per thread.
"""
import torch

def make_rng_states(n_threads: int, seed: int) -> torch.IntTensor:
     """
    Create random initial states for each GPU thread.
    Uses MurmurHash3-style bit mixing for strong entropy.

    Args:
        n_threads: total number of GPU threads
        seed: global seed for reproducibility

    Returns:
        torch.IntTensor of shape (n_threads,)
    """
    s = torch.arange(n_threads, dtype=torch.int64) ^ seed
    s ^= (s >> 33)
    s *= 0xff51afd7ed558ccd
    s ^= (s >> 33)
    s *= 0xc4ceb9fe1a85ec53
    s ^= (s >> 33) # Keep lower 32 bits (for Triton int32 compatibility)
    s = (s & 0xFFFFFFFF).int()
    s[s == 0] = 1  # Ensure no zero seeds (XORShift breaks with zero state)
    return s
