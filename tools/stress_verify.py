"""Large-scale (>= 1M elements per instruction) local hardware stress verification for all 22 gfx1201 variants.

Usage:
    uv run --no-project --with numpy python tools/stress_verify.py [--waves-per-mode 2048] [--co PATH]
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from amdgpu_matrix_model.model import wmma_wave
from amdgpu_matrix_model.specs import SPECS
from tools.capture_evidence import gen_random_chunk
from tools.hw_run import run_jobs
from tools.probe_asm import build_co


def _check_chunk(args: tuple) -> tuple[int, int]:
    spec_name, a, b, c, d_hw, idx, ik, nlo, nhi, cl = args
    n_mm = 0
    for w in range(len(a)):
        pred = wmma_wave(
            spec_name,
            a[w].reshape(-1).tolist(),
            b[w].reshape(-1).tolist(),
            c[w].reshape(-1).tolist(),
            idx_words=(idx[w].reshape(-1).tolist() if idx is not None else None),
            index_key=ik,
            neg_lo=nlo,
            neg_hi=nhi,
            clamp=cl,
        )
        got = d_hw[w].reshape(-1).tolist()
        if pred != got:
            n_mm += sum(1 for x, y in zip(pred, got) if x != y)
    return len(a) * 256, n_mm


def check_job(
    spec_name: str,
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    d_hw: np.ndarray,
    idx: np.ndarray | None,
    ik: int,
    nlo: tuple[int, int, int],
    nhi: tuple[int, int, int],
    cl: bool,
) -> tuple[int, int]:
    workers = min(os.cpu_count() or 4, 16, max(1, len(a) // 16))
    if workers <= 1:
        return _check_chunk((spec_name, a, b, c, d_hw, idx, ik, nlo, nhi, cl))
    splits = np.array_split(np.arange(len(a)), workers)
    tasks = [
        (spec_name, a[s], b[s], c[s], d_hw[s], (idx[s] if idx is not None else None), ik, nlo, nhi, cl)
        for s in splits
        if len(s) > 0
    ]
    tot = mm = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for t, m in ex.map(_check_chunk, tasks):
            tot += t
            mm += m
    return tot, mm


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--waves-per-mode", type=int, default=2048, help="Waves per random mode (default 2048 -> 1M/variant)")
    ap.add_argument("--seed", type=int, default=90000)
    ap.add_argument("--co", type=Path, default=None, help="Pre-assembled probes_gfx1201.co (default: assemble locally)")
    args = ap.parse_args()

    co = args.co if args.co is not None else build_co()
    seed = args.seed
    jobs, meta = [], {}

    for spec in SPECS.values():
        if not spec.sparse and not spec.is_int:
            for mode in ("finite", "raw"):
                a, b, c, _ = gen_random_chunk(spec, args.waves_per_mode, seed, mode)
                seed += 1
                jid = f"{spec.name}__{mode}"
                jobs.append({"id": jid, "spec": spec.name, "a": a, "b": b, "c": c})
                meta[jid] = (spec.name, a, b, c, None, 0, (0, 0, 0), (0, 0, 0), False)
        elif not spec.sparse and spec.is_int:
            nw = max(1, args.waves_per_mode // 4)
            for sa in (0, 1):
                for sb in (0, 1):
                    for cl in (False, True):
                        tag = "" if (sa == 0 and sb == 0 and not cl) else f"_s{sa}{sb}" + ("_clamp" if cl else "")
                        kern = f"{spec.name}{tag}"
                        a, b, c, _ = gen_random_chunk(spec, nw, seed, "clamp_stress" if cl else "raw")
                        seed += 1
                        jobs.append({"id": kern, "spec": spec.name, "kernel": kern, "a": a, "b": b, "c": c})
                        meta[kern] = (spec.name, a, b, c, None, 0, (sa, sb, 0), (0, 0, 0), cl)
        elif spec.sparse and not spec.is_int:
            nw = max(1, args.waves_per_mode // 2)
            for ik in (0, 1):
                kern = f"{spec.name}_ik1" if ik == 1 else spec.name
                for mode in ("finite", "raw"):
                    a, b, c, idx = gen_random_chunk(spec, nw, seed, mode)
                    seed += 1
                    jid = f"{kern}__{mode}"
                    jobs.append({"id": jid, "spec": spec.name, "kernel": kern, "a": a, "b": b, "c": c, "idx": idx})
                    meta[jid] = (spec.name, a, b, c, idx, ik, (0, 0, 0), (0, 0, 0), False)
        else:
            keys = (0, 1) if spec.has_index_key else (0,)
            nw = max(1, args.waves_per_mode // (8 if spec.has_index_key else 4))
            for ik in keys:
                ik_tag = "_ik1" if (spec.has_index_key and ik == 1) else ""
                for sa in (0, 1):
                    for sb in (0, 1):
                        for cl in (False, True):
                            stag = "" if (sa == 0 and sb == 0 and not cl) else f"_s{sa}{sb}" + ("_clamp" if cl else "")
                            kern = f"{spec.name}{ik_tag}{stag}"
                            a, b, c, idx = gen_random_chunk(spec, nw, seed, "clamp_stress" if cl else "raw")
                            seed += 1
                            jobs.append({"id": kern, "spec": spec.name, "kernel": kern, "a": a, "b": b, "c": c, "idx": idx})
                            meta[kern] = (spec.name, a, b, c, idx, ik, (sa, sb, 0), (0, 0, 0), cl)

    t0 = time.monotonic()
    hw_res = run_jobs(jobs, co_path=co)
    print(f"GPU stress capture completed in {time.monotonic() - t0:.2f}s ({len(jobs)} jobs)")

    grand_tot = grand_mm = 0
    for jid, (spec_name, a, b, c, idx, ik, nlo, nhi, cl) in meta.items():
        tot, mm = check_job(spec_name, a, b, c, hw_res[jid], idx, ik, nlo, nhi, cl)
        grand_tot += tot
        grand_mm += mm
        print(f"[{'PASS' if mm == 0 else 'FAIL'}] {jid:46s}: {tot - mm:7d}/{tot:7d} ({mm} mm)")
    print(f"\nSTRESS TOTAL: {grand_tot - grand_mm:,}/{grand_tot:,} bit-exact ({grand_mm} mismatches)")
    sys.exit(0 if grand_mm == 0 else 1)


if __name__ == "__main__":
    main()
