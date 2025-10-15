import time
from gpu_triton.engine import GPUStats, run_gpu_blackjack

def run_large_gpu_blackjack(total_games=70_000_000, batch_size=1_000_000, seed=42):
    """
    Run large-scale GPU Blackjack simulations in multiple batches.

    Each batch launches one GPU kernel that runs 'batch_size' blackjack games in parallel.
    The CPU loop controls the batching while the GPU does the heavy computation.
    """
    total = GPUStats(0, 0, 0, 0.0)
    num_batches = total_games // batch_size

    print(f"\n Starting GPU Blackjack simulation")
    print(f"Total games: {total_games:,} | Batch size: {batch_size:,} | Batches: {num_batches}")
    print("=" * 75)

    start_time = time.time()

    for i in range(0, total_games, batch_size):
        batch_num = (i // batch_size) + 1
        batch_start = time.time()

        # 🔹 GPU kernel launch (runs one full batch in parallel)
        s = run_gpu_blackjack(batch_size, seed + i)

        # 🔹 Aggregate results on CPU
        total = GPUStats(
            wins=total.wins + s.wins,
            losses=total.losses + s.losses,
            pushes=total.pushes + s.pushes,
            total_pl=total.total_pl + s.total_pl,
        )

        batch_time = time.time() - batch_start
        throughput = batch_size / batch_time  ## Throughput = how many games per second your GPU is processing.
        print(f" Batch {batch_num}/{num_batches} | Time: {batch_time:.2f}s | "
              f"Throughput: {throughput:,.0f} games/sec")

    total_time = time.time() - start_time
    overall_throughput = total_games / total_time

    print("=" * 75)
    print(" Simulation complete!")
    print(f" Total time: {total_time:.2f} seconds (~{total_time/60:.2f} minutes)")
    print(f" Overall speed: {overall_throughput:,.0f} games/sec")
    print("=" * 75)

    return total


if __name__ == "__main__":
    # You can change total_games here for a smaller or larger run
    big = run_large_gpu_blackjack(total_games=5_000_000, batch_size=1_000_000)
    print("\nFinal Aggregated Results:")
    print(big)
