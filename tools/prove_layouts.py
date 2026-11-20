"""Milestone 1: prove wave32 operand layouts on gfx1201 hardware using exact small integers.

Tests every one of the 22 instructions (11 dense WMMA + 11 sparse SWMMAC, including
both index_key:0 and index_key:1 on K=32 SWMMAC = 32 hardware kernels).
"""

from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from amdgpu_matrix_model.layout import pack_a, pack_b, pack_cd, pack_idx, unpack_cd
from amdgpu_matrix_model.specs import SPECS, WmmaSpec
from tools.hw_run import run_jobs
from tools.probe_asm import build_co

# Valid 2:4 index pairs (idx0 < idx1)
_VALID_PAIRS = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))


def encode_small_int(val: int, fmt: str) -> int:
    """Encode a small exact integer into raw format bits."""
    if fmt == "i32":
        return val & 0xFFFFFFFF
    if fmt == "iu8":
        return val & 0xFF
    if fmt == "iu4":
        return val & 0xF
    if fmt == "f32":
        return struct.unpack("<I", struct.pack("<f", float(val)))[0]
    if fmt == "bf16":
        return struct.unpack("<I", struct.pack("<f", float(val)))[0] >> 16
    if fmt == "f16":
        return int(np.float16(val).view(np.uint16))
    if fmt == "fp8":
        if val == 0:
            return 0
        sign = 0x80 if val < 0 else 0
        v = abs(val)
        e = v.bit_length() - 1
        frac = ((v << (3 - e)) if e <= 3 else (v >> (e - 3))) & 0x7
        return sign | ((e + 7) << 3) | frac
    if fmt == "bf8":
        if val == 0:
            return 0
        sign = 0x80 if val < 0 else 0
        v = abs(val)
        e = v.bit_length() - 1
        frac = ((v << (2 - e)) if e <= 2 else (v >> (e - 2))) & 0x3
        return sign | ((e + 15) << 2) | frac
    raise ValueError(fmt)


def decode_exact_int(raw: int, fmt: str) -> int:
    """Decode an exact small integer from D format bits."""
    if fmt == "i32":
        return raw - (1 << 32) if raw >= (1 << 31) else raw
    if fmt == "f32":
        f = struct.unpack("<f", struct.pack("<I", raw))[0]
        assert f == int(f), f"non-integer f32 {f} ({raw:#x})"
        return int(f)
    if fmt == "bf16":
        f = struct.unpack("<f", struct.pack("<I", raw << 16))[0]
        assert f == int(f), f"non-integer bf16 {f} ({raw:#x})"
        return int(f)
    if fmt == "f16":
        f = float(np.uint16(raw).view(np.float16))
        assert f == int(f), f"non-integer f16 {f} ({raw:#x})"
        return int(f)
    raise ValueError(fmt)


def build_dense_layout_job(spec: WmmaSpec) -> tuple[dict, list[list[list[int]]]]:
    """Build waves that uniquely prove every A[r,k], B[k,c], C[r,c] -> D[r,c] coordinate."""
    a_waves, b_waves, c_waves, expected = [], [], [], []

    # Wave 0: C -> D identity (0..255 all exact in f32, f16, bf16, i32)
    z_a = [[0] * spec.k_a for _ in range(16)]
    z_b = [[0] * 16 for _ in range(spec.k_b)]
    c0_int = [[16 * r + c for c in range(16)] for r in range(16)]
    c0_raw = [[encode_small_int(v, spec.d_fmt) for v in row] for row in c0_int]
    a_waves.append(pack_a(spec, z_a))
    b_waves.append(pack_b(spec, z_b))
    c_waves.append(pack_cd(spec, c0_raw))
    expected.append(c0_int)

    # Waves 1..1+4*K: per-k 4-bit row/col signature probes using {1, 2} (exact in every format)
    z_c = pack_cd(spec, [[0] * 16 for _ in range(16)])
    one_a, two_a = encode_small_int(1, spec.a_fmt), encode_small_int(2, spec.a_fmt)
    one_b, two_b = encode_small_int(1, spec.b_fmt), encode_small_int(2, spec.b_fmt)
    for kk in range(spec.k):
        for bit in range(4):
            a_raw = [[0] * spec.k_a for _ in range(16)]
            b_raw = [[0] * 16 for _ in range(spec.k_b)]
            a_int = [2 if ((r >> bit) & 1) else 1 for r in range(16)]
            b_int = [2 if ((c >> bit) & 1) else 1 for c in range(16)]
            for r in range(16):
                a_raw[r][kk] = two_a if a_int[r] == 2 else one_a
            for c in range(16):
                b_raw[kk][c] = two_b if b_int[c] == 2 else one_b
            a_waves.append(pack_a(spec, a_raw))
            b_waves.append(pack_b(spec, b_raw))
            c_waves.append(z_c)
            expected.append([[a_int[r] * b_int[c] for c in range(16)] for r in range(16)])

    job = {
        "id": spec.name,
        "spec": spec.name,
        "a": np.array(a_waves, dtype=np.uint32),
        "b": np.array(b_waves, dtype=np.uint32),
        "c": np.array(c_waves, dtype=np.uint32),
    }
    return job, expected


def build_sparse_layout_job(spec: WmmaSpec, index_key: int = 0) -> tuple[dict, list[list[list[int]]]]:
    """Build waves that prove A (compressed), B (dense), C/D, and row-distinct index_set layout."""
    a_waves, b_waves, c_waves, idx_waves, expected = [], [], [], [], []
    kernel = f"{spec.name}_ik{index_key}" if (spec.has_index_key and index_key == 1) else spec.name

    # Row-distinct valid 2:4 index table (16 x k_a)
    n_groups = spec.k // 4
    idx_table = [[0] * spec.k_a for _ in range(16)]
    for r in range(16):
        for g in range(n_groups):
            p0, p1 = _VALID_PAIRS[(r * 3 + g * 5 + index_key) % len(_VALID_PAIRS)]
            idx_table[r][2 * g] = p0
            idx_table[r][2 * g + 1] = p1
    # Also fill unused half of index VGPR with junk to prove index_key isolates the active half
    idx_packed = pack_idx(spec, idx_table, index_key=index_key)
    if spec.has_index_key:
        other_shift = 0 if index_key == 1 else 16
        idx_packed = [w | (0x4444 << other_shift) for w in idx_packed]

    # Wave 0: C -> D identity
    z_a = [[0] * spec.k_a for _ in range(16)]
    z_b = [[0] * 16 for _ in range(spec.k_b)]
    c0_int = [[16 * r + c for c in range(16)] for r in range(16)]
    c0_raw = [[encode_small_int(v, spec.d_fmt) for v in row] for row in c0_int]
    a_waves.append(pack_a(spec, z_a))
    b_waves.append(pack_b(spec, z_b))
    c_waves.append(pack_cd(spec, c0_raw))
    idx_waves.append(idx_packed)
    expected.append(c0_int)

    # For each compressed ka in 0..k_a-1 and each col bit in 0..3:
    # A[r, ka] = 1 for all r; B[4*g + s, c] = (s + 1) * (2 if (c>>bit)&1 else 1) for s in 0..3 (values in 1..8?
    # Wait: bf8 holds 1,2,3,4,6,8 exact, NOT 5 or 7; iu4 unsigned holds 0..15.
    # Better: use B[4*g + s, c] = 2 if (s == sel and (c>>bit)&1) ... even simpler:
    # For each ka, run 2 waves:
    #   Wave A (tests exact selected offset s in 0..3 for every row r):
    #     A[r, ka] = 1, B[4*g + s, c] = (s + 1) in {1, 2, 3, 4} (all exact in EVERY format incl. bf8 and iu4!)
    #   Wave B (tests row/col identity on ka with 4 bit planes):
    #     A[r, ka] = 2 if (r>>bit)&1 else 1, B[4*g + s, c] = 2 if (c>>bit)&1 else 1 for all s in 0..3
    # That tests: (1) every row's exact 2-bit index in every group, (2) every A[r, ka] slot, (3) every B[kb, c] slot!
    z_c = pack_cd(spec, [[0] * 16 for _ in range(16)])
    one_a, two_a = encode_small_int(1, spec.a_fmt), encode_small_int(2, spec.a_fmt)
    one_b, two_b = encode_small_int(1, spec.b_fmt), encode_small_int(2, spec.b_fmt)

    for ka in range(spec.k_a):
        g = ka // 2
        # Index selection probe (distinct values 1,2,3,4 per offset in group g)
        a_raw = [[0] * spec.k_a for _ in range(16)]
        b_raw = [[0] * 16 for _ in range(spec.k_b)]
        for r in range(16):
            a_raw[r][ka] = one_a
        for s in range(4):
            val_b = encode_small_int(s + 1, spec.b_fmt)
            b_raw[4 * g + s] = [val_b] * 16
        a_waves.append(pack_a(spec, a_raw))
        b_waves.append(pack_b(spec, b_raw))
        c_waves.append(z_c)
        idx_waves.append(idx_packed)
        expected.append([[idx_table[r][ka] + 1 for _ in range(16)] for r in range(16)])

        # Row/col bit-plane probes (proves A[r, ka] and B[4*g + s, c] coordinates)
        for bit in range(4):
            a_raw = [[0] * spec.k_a for _ in range(16)]
            b_raw = [[0] * 16 for _ in range(spec.k_b)]
            a_int = [2 if ((r >> bit) & 1) else 1 for r in range(16)]
            b_int = [2 if ((c >> bit) & 1) else 1 for c in range(16)]
            for r in range(16):
                a_raw[r][ka] = two_a if a_int[r] == 2 else one_a
            for s in range(4):
                for c in range(16):
                    b_raw[4 * g + s][c] = two_b if b_int[c] == 2 else one_b
            a_waves.append(pack_a(spec, a_raw))
            b_waves.append(pack_b(spec, b_raw))
            c_waves.append(z_c)
            idx_waves.append(idx_packed)
            expected.append([[a_int[r] * b_int[c] for c in range(16)] for r in range(16)])

    job = {
        "id": kernel,
        "spec": spec.name,
        "kernel": kernel,
        "a": np.array(a_waves, dtype=np.uint32),
        "b": np.array(b_waves, dtype=np.uint32),
        "c": np.array(c_waves, dtype=np.uint32),
        "idx": np.array(idx_waves, dtype=np.uint32),
    }
    return job, expected


def main() -> None:
    ap = argparse.ArgumentParser(description="Prove wave32 layouts for all 22 gfx1201 matrix instructions.")
    ap.add_argument("--co", type=Path, default=None, help="Pre-assembled probes_gfx1201.co (default: assemble locally)")
    args = ap.parse_args()
    co = args.co if args.co is not None else build_co()
    print(f"Using code object: {co}")

    jobs = []
    expected_map: dict[str, tuple[WmmaSpec, list[list[list[int]]]]] = {}

    for spec in SPECS.values():
        if not spec.sparse:
            job, exp = build_dense_layout_job(spec)
            jobs.append(job)
            expected_map[job["id"]] = (spec, exp)
        else:
            keys = (0, 1) if spec.has_index_key else (0,)
            for ik in keys:
                job, exp = build_sparse_layout_job(spec, index_key=ik)
                jobs.append(job)
                expected_map[job["id"]] = (spec, exp)

    res = run_jobs(jobs, co_path=co)
    total_elems = 0
    for jid, (spec, exp_waves) in expected_map.items():
        d_waves = res[jid]
        assert len(d_waves) == len(exp_waves)
        for w_idx, exp in enumerate(exp_waves):
            got_raw = unpack_cd(spec, d_waves[w_idx].reshape(-1))
            got = [[decode_exact_int(got_raw[r][c], spec.d_fmt) for c in range(16)] for r in range(16)]
            if got != exp:
                raise AssertionError(f"{jid} wave {w_idx} layout mismatch!\nexp[0]={exp[0]}\ngot[0]={got[0]}")
        n_el = len(exp_waves) * 256
        total_elems += n_el
        print(f"PASS layout: {jid:36s} ({len(exp_waves):3d} waves, {n_el:6d} elems)")
    print(f"ALL {len(expected_map)} kernels (all 22 gfx1201 instructions) layout-proved: {total_elems} exact elements.")


if __name__ == "__main__":
    main()
