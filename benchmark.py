"""
Author: Saad Bhatti
Desc:
Headless benchmark comparing the object/quad-tree flock (flock.py) against the
vectorized SoA flock (flock_array.py). Rendering is disabled so we measure only the
per-frame physics cost. Runs across a range of flock sizes and reports ms/frame and
the speedup.

Run:  python benchmark.py
      python benchmark.py --sizes 350 1000 2000 5000 --frames 60 --gpu

The --gpu flag swaps the array backend to CuPy (Phase 2); it falls back to NumPy
with a warning if CuPy/the GPU is unavailable.
"""

import os
import sys
import time
import argparse

# Headless SDL so pygame.display works without a window (the flock code calls
# pygame.display.get_surface()).
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import numpy as np
import pygame

WIDTH, HEIGHT = 1200, 600
SEED = 1234
MOUSE = (0, 0)  # same fixed "mouse" fed to both backends for a fair comparison


def time_object_flock(n, frames, warmup):
    """Per-frame time for the object-based quad-tree flock, rendering disabled."""
    from boid import Boid
    from flock import Flock

    # Neutralize rendering so we isolate physics.
    orig_draw = Boid.draw
    Boid.draw = lambda self: None
    try:
        import random
        random.seed(SEED)
        flock = Flock(n)

        def one_frame():
            flock.draw(useQtree=True, showQtree=False, useCohesion=True,
                       useSeperation=True, useAlignment=True)

        for _ in range(warmup):
            one_frame()
        t0 = time.perf_counter()
        for _ in range(frames):
            one_frame()
        elapsed = time.perf_counter() - t0
    finally:
        Boid.draw = orig_draw
    return elapsed / frames


def time_array_flock(n, frames, warmup, xp):
    """Per-frame time for the vectorized flock on backend `xp`. Returns ms/frame,
    or None if the GPU ran out of memory (the dense N*N matrices can exceed a
    small card's VRAM)."""
    from flock_array import FlockArray

    on_gpu = xp is not np

    def sync():
        # CuPy is async; force completion so timings are real.
        if on_gpu:
            xp.cuda.runtime.deviceSynchronize()

    try:
        flock = FlockArray(n, WIDTH, HEIGHT, seed=SEED, xp=xp)
        for _ in range(warmup):
            flock.step(mouse_pos=MOUSE)
        sync()
        t0 = time.perf_counter()
        for _ in range(frames):
            flock.step(mouse_pos=MOUSE)
        sync()
        elapsed = time.perf_counter() - t0
        return (elapsed / frames) * 1000
    except Exception as e:
        if on_gpu and type(e).__name__ == "OutOfMemoryError":
            return None
        raise
    finally:
        if on_gpu:
            # Release pooled device memory so the next (larger) size starts clean.
            xp.get_default_memory_pool().free_all_blocks()


def get_backend(use_gpu):
    if not use_gpu:
        return np, "NumPy (CPU)"
    try:
        import cuda_setup  # register nvidia-*-cu11 wheel DLL dirs before cupy loads
        import cupy as cp
        cp.cuda.runtime.getDeviceProperties(0)
        # touch the device to confirm it actually works
        _ = (cp.arange(8) * 2).sum()
        cp.cuda.runtime.deviceSynchronize()
        name = cp.cuda.runtime.getDeviceProperties(0)["name"].decode()
        return cp, "CuPy (GPU: {})".format(name)
    except Exception as e:
        print("  [!] GPU unavailable ({}: {}); falling back to NumPy.".format(
            type(e).__name__, str(e).splitlines()[0]))
        return np, "NumPy (CPU, GPU fallback)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+",
                    default=[350, 1000, 2000, 5000])
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--gpu", action="store_true",
                    help="use CuPy for the array backend (Phase 2)")
    args = ap.parse_args()

    pygame.init()
    pygame.display.set_mode((WIDTH, HEIGHT))

    xp, backend_name = get_backend(args.gpu)

    print("Frames/sample: {}  (warmup {})".format(args.frames, args.warmup))
    print("Array backend: {}\n".format(backend_name))
    header = "{:>8}  {:>14}  {:>14}  {:>10}".format(
        "N", "quadtree ms", "array ms", "speedup")
    print(header)
    print("-" * len(header))

    for n in args.sizes:
        obj_ms = time_object_flock(n, args.frames, args.warmup) * 1000
        arr_ms = time_array_flock(n, args.frames, args.warmup, xp)
        if arr_ms is None:
            print("{:>8}  {:>14.3f}  {:>14}  {:>10}".format(
                n, obj_ms, "OOM", "-"))
            continue
        speedup = obj_ms / arr_ms if arr_ms > 0 else float("inf")
        print("{:>8}  {:>14.3f}  {:>14.3f}  {:>9.2f}x".format(
            n, obj_ms, arr_ms, speedup))

    pygame.quit()


if __name__ == "__main__":
    main()
