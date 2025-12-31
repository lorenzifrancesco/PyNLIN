import numpy as np
from scipy.optimize import fsolve, root
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from analysis.fiber_analysis.load_fiber_values import load_phase_delay
from matplotlib import pyplot as plt
from numpy import polyval
from itertools import product
import pynlin
import analysis.utils.cfg as cfg
import time, random
import heapq

def main():
    """Demo brute-force heap selection of top random samples."""
    time_start = time.time()
    acc = 0

    # Use a min-heap to maintain top 1000 results
    # We use negative values since heapq is a min-heap but we want max values
    top_results = []
    max_heap_size = 1000

    for i in range(1_000_000):
        acc = random.random()
        
        # Add current result to our top results collection
        if len(top_results) < max_heap_size:
            # If we haven't reached 1000 results yet, just add it
            heapq.heappush(top_results, acc)
        else:
            # If current result is better than the worst in our top 1000
            if acc > top_results[0]:
                heapq.heapreplace(top_results, acc)
        
        if i % 100000 == 0:  # Print progress every 100k iterations
            current_min = top_results[0] if top_results else 0
            print(f"Iteration {i}: Current acc = {acc:.6f}, Min in top 1000 = {current_min:.6f}")

    # Convert heap to sorted list (best to worst)
    final_top_results = sorted(top_results, reverse=True)

    print(f"\nTop 10 results:")
    for idx, result in enumerate(final_top_results[:10], 1):
        print(f"{idx:2d}. {result:.6f}")

    print(f"\nResult #{max_heap_size} (worst in top 1000): {final_top_results[-1]:.6f}")
    print(f"Best result: {final_top_results[0]:.6f}")
    print(f"Total results preserved: {len(final_top_results)}")

    time_end = time.time()
    print(f"Time taken: {time_end - time_start:.2f} seconds")


if __name__ == "__main__":
    main()
