"""
tests_smoke.py
--------------
Lightweight smoke test for GPU Blackjack simulation.
Verifies end-to-end functionality on a small sample size.
"""

from engine import run_gpu_blackjack

def main():
    print(" Running GPU Blackjack smoke test...")
    stats = run_gpu_blackjack(ngames=100_000, seed=2025)
    print(" Simulation complete.")
    print(f"Results:\n  Wins: {stats.wins}\n  Losses: {stats.losses}\n  Pushes: {stats.pushes}\n  Total P/L: {stats.total_pl:.2f}")

if __name__ == "__main__":
    main()
