"""
kernels.py
----------
Defines the Triton GPU kernel for blackjack simulation.
Each GPU thread simulates one blackjack hand in parallel:
  - Generates random cards using XORShift32 RNG
  - Computes hand values
  - Determines win/loss/push outcome
  - Writes results back to GPU memory
"""
import triton
import triton.language as tl

@triton.jit
def _xorshift32(x):
    x ^= (x << 13)
    x ^= (x >> 17)
    x ^= (x << 5)
    return x

@triton.jit
def _rand_rank(rng_state):
    s = _xorshift32(rng_state)
    val = (s % 13) + 1          # 1..13 (Ace..King)
    return s, val

@triton.jit
def _is_face(rank: tl.int32):
    return (rank >= 11) & (rank <= 13)

@triton.jit
def _card_value(rank: tl.int32):
    # Face cards = 10, Ace = 1 (no soft-ace logic here)
    return tl.where(_is_face(rank), 10, rank)

@triton.jit
def simulate_hands_kernel(
    rng_ptr,               # int32* RNG states
    rules_ptr,             # int32* [43*10] table (unused placeholder)
    results_ptr,           # fp32*  per-lane profit/loss
    counts_ptr,            # int32* 3-element counter array
    n_games: tl.int32,     # total hands to simulate
    PAYOFF_BJ: tl.float32, # kept for signature compatibility
    PAYOFF_PUSH: tl.float32,
    BLOCK: tl.constexpr,
):
    pid  = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_games

    # ---- Load RNG states ----
    rng = tl.load(rng_ptr + offs, mask=mask)

    # ---- Deal cards ----
    rng, pr1 = _rand_rank(rng)
    rng, pr2 = _rand_rank(rng)
    rng, dr1 = _rand_rank(rng)
    rng, dr2 = _rand_rank(rng)

    # ---- Compute totals (split function calls for Triton stability) ----
    pr1_val = _card_value(pr1)
    pr2_val = _card_value(pr2)
    pv = pr1_val + pr2_val

    dr1_val = _card_value(dr1)
    dr2_val = _card_value(dr2)
    dv = dr1_val + dr2_val

    # ---- Determine outcomes ----
    win  = (pv <= 21) & ((dv > 21) | (pv > dv))
    lose = (pv > 21) | ((dv <= 21) & (pv < dv))
    push = (pv <= 21) & (dv <= 21) & (pv == dv)

    # ---- Profit/loss per lane ----
    pl = tl.zeros([BLOCK], dtype=tl.float32)
    pl = tl.where(win,  1.0, pl)
    pl = tl.where(lose, -1.0, pl)
    pl = tl.where(push, PAYOFF_PUSH, pl)

    # ---- Store results ----
    tl.store(results_ptr + offs, pl, mask=mask)

    # ---- Aggregate counters (atomic adds) ----
    wins_i   = tl.sum(win.to(tl.int32), axis=0)
    losses_i = tl.sum(lose.to(tl.int32), axis=0)
    pushes_i = tl.sum(push.to(tl.int32), axis=0)
    tl.atomic_add(counts_ptr + 0, wins_i)
    tl.atomic_add(counts_ptr + 1, losses_i)
    tl.atomic_add(counts_ptr + 2, pushes_i)

    # ---- Save RNG state ----
    tl.store(rng_ptr + offs, rng, mask=mask)
