#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Benchmark: dense matrix (10000x10000) times 1000 vectors, comparing sequential (n=1)
vs batched multiplication (n vectors at a time), for all n in [2..1000] with 1000 % n == 0.

Outputs:
- times[n]: wall time for processing all 1000 vectors using batch size n
- ratio[n] = times[n] / times[1]
- plot: ratio vs n

⚠️ Resource note:
A dense 10000x10000 float64 matrix is ~800 MB (plus overhead). float32 is ~400 MB.
Each matmul is expensive: 10000^2 * n flops per batch. This benchmark can take a long time.
"""

import os
import time
import gc
import numpy as np
import matplotlib.pyplot as plt


def set_blas_threads(nthreads: int = 1):
    """
    Best-effort control of BLAS threading for more stable timings.
    Must be set BEFORE NumPy loads BLAS in some setups; still helpful to do early.
    """
    for var in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ.setdefault(var, str(nthreads))


def benchmark_matvecs(
    m: int = 10_000,
    num_vecs: int = 1000,
    dtype=np.float32,
    seed: int = 0,
    warmup: bool = True,
):
    """
    Returns:
      ns: list of batch sizes (including 1)
      times: dict {n: seconds}
      ratios: dict {n: times[n]/times[1]}
    """
    assert num_vecs % 1 == 0
    rng = np.random.default_rng(seed)

    # Create random matrix A
    # Tip: use float32 to reduce RAM; also consider order='F' for some BLAS, but keep default here.
    print(f"Allocating A ({m}x{m}, {dtype}), this can be large...")
    A = rng.standard_normal((m, m), dtype=dtype)

    # Small warm-up to trigger BLAS kernels / cache effects
    if warmup:
        print("Warm-up matmul...")
        xw = rng.standard_normal((m, 4), dtype=dtype)
        _ = A @ xw
        del xw, _
        gc.collect()

    # Candidate batch sizes: 1 plus divisors of num_vecs in [2..num_vecs]
    ns = [1] + [n for n in range(2, num_vecs + 1) if (num_vecs % n == 0)]
    times = {}

    def run_for_batch(n: int) -> float:
        """
        Process all num_vecs vectors using batches of size n:
        total batches = num_vecs/n
        Each batch does: Y = A @ X, where X shape = (m, n)
        We generate X per-batch to avoid storing all 1000 vectors at once.
        """
        batches = num_vecs // n
        t0 = time.perf_counter()
        for _ in range(batches):
            X = rng.standard_normal((m, n), dtype=dtype)
            Y = A @ X
            # Prevent overly aggressive dead-code elimination (rare in Python, but keep a trivial use)
            # and force materialization.
            _sink = float(np.sum(Y[0, :1]))
            del X, Y, _sink
        t1 = time.perf_counter()
        return t1 - t0

    # Measure n=1 first (sequential case)
    print("Benchmarking n=1 (sequential matvec)...")
    times[1] = run_for_batch(1)
    print(f"  time[1] = {times[1]:.6f} s")

    # Measure remaining n
    for n in ns[1:]:
        gc.collect()
        print(f"Benchmarking n={n} (batched)...")
        tn = run_for_batch(n)
        times[n] = tn
        print(f"  time[{n}] = {tn:.6f} s   ratio={tn/times[1]:.4f}")

    ratios = {n: times[n] / times[1] for n in ns}
    return ns, times, ratios


def plot_ratios(ns, ratios, title="Time ratio (batch n) / time (n=1)"):
    xs = np.array(ns, dtype=int)
    ys = np.array([ratios[n] for n in ns], dtype=float)

    plt.figure()
    plt.plot(xs, ys, marker="o", linestyle="-")
    plt.xlabel("Batch size n (1000 % n == 0)")
    plt.ylabel("time[n] / time[1]")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Optional: stabilize threading for repeatability (set to >1 if you want max throughput)
    set_blas_threads(nthreads=1)

    # Parameters
    M = 10_000
    NUM_VECS = 1000
    DTYPE = np.float32  # change to np.float64 if you have enough RAM and want float64

    ns, times, ratios = benchmark_matvecs(
        m=M,
        num_vecs=NUM_VECS,
        dtype=DTYPE,
        seed=0,
        warmup=True,
    )

    plot_ratios(ns, ratios)

