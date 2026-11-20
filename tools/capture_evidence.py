"""Capture and freeze compact binary hardware evidence (AMDMAT02) for all 22 gfx1201 instructions.

Random chunks store only their SplitMix64 seed, generator mode, and hardware output D words.
Hand-built adversarial chunks store explicit A, B, C inputs plus hardware output D words.
"""

from __future__ import annotations

import argparse
import array
import hashlib
import json
import platform
import struct
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from amdgpu_matrix_model.layout import pack_a, pack_b, pack_cd
from amdgpu_matrix_model.specs import SPECS, WmmaSpec
from amdgpu_matrix_model.validation import generate_chunk_inputs, validate, write_evidence_file
from tools.hw_run import run_jobs
from tools.probe_asm import build_co


def _f32_bits(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", x))[0]


def _f16_bits(x: float) -> int:
    return int(np.float16(x).view(np.uint16))


def _bf16_bits(x: float) -> int:
    return _f32_bits(x) >> 16


def _enc_fp8_pow2(exp: int, neg: bool = False) -> int:
    s = 0x80 if neg else 0
    if exp >= -6:
        return s | ((exp + 7) << 3)
    return s | {-7: 4, -8: 2, -9: 1}[exp]


def _enc_bf8_pow2(exp: int, neg: bool = False) -> int:
    s = 0x80 if neg else 0
    if exp >= -14:
        return s | ((exp + 15) << 2)
    return s | {-15: 2, -16: 1}[exp]


def _split_pow2_8(prod_exp: int, fa: str, fb: str, neg: bool = False) -> tuple[int, int]:
    ra = (-9, 8) if fa == "fp8" else (-16, 15)
    rb = (-9, 8) if fb == "fp8" else (-16, 15)
    ea = max(ra[0], min(ra[1], prod_exp // 2))
    eb = prod_exp - ea
    if eb < rb[0]:
        eb, ea = rb[0], prod_exp - rb[0]
    elif eb > rb[1]:
        eb, ea = rb[1], prod_exp - rb[1]
    a_b = _enc_fp8_pow2(ea, neg) if fa == "fp8" else _enc_bf8_pow2(ea, neg)
    b_b = _enc_fp8_pow2(eb, False) if fb == "fp8" else _enc_bf8_pow2(eb, False)
    return a_b, b_b


def _pack_diag_cases(spec: WmmaSpec, cases: list[tuple[list[int], list[int], int]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    a_w, b_w, c_w = [], [], []
    for at in range(0, len(cases), 16):
        chunk = cases[at : at + 16]
        am = [[0] * spec.k_a for _ in range(16)]
        bm = [[0] * 16 for _ in range(spec.k_b)]
        cm = [[0] * 16 for _ in range(16)]
        for t, (av, bv, cv) in enumerate(chunk):
            am[t] = av
            for k in range(spec.k_b):
                bm[k][t] = bv[k]
            cm[t][t] = cv
        a_w.append(pack_a(spec, am))
        b_w.append(pack_b(spec, bm))
        c_w.append(pack_cd(spec, cm))
    return (
        np.array(a_w, dtype=np.uint32).reshape(-1, 32, spec.a_dwords),
        np.array(b_w, dtype=np.uint32).reshape(-1, 32, spec.b_dwords),
        np.array(c_w, dtype=np.uint32).reshape(-1, 32, spec.cd_dwords),
    )


def gen_random_chunk(spec: WmmaSpec, n_waves: int, seed: int, gen: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    a_l, b_l, c_l, idx_l = generate_chunk_inputs(spec, n_waves, seed, gen)
    a = np.array(a_l, dtype=np.uint32).reshape(n_waves, 32, spec.a_dwords)
    b = np.array(b_l, dtype=np.uint32).reshape(n_waves, 32, spec.b_dwords)
    c = np.array(c_l, dtype=np.uint32).reshape(n_waves, 32, spec.cd_dwords)
    idx = np.array(idx_l, dtype=np.uint32).reshape(n_waves, 32) if idx_l is not None else None
    return a, b, c, idx


def gen_adversarial_dense_float(spec: WmmaSpec) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Targeted adversarial edge cases covering every discovered arithmetic subtlety."""
    cases: list[tuple[list[int], list[int], int]] = []

    if spec.is_fp8_family:
        af, bf = spec.a_fmt, spec.b_fmt
        a1 = _enc_fp8_pow2(0) if af == "fp8" else _enc_bf8_pow2(0)
        b1 = _enc_fp8_pow2(0) if bf == "fp8" else _enc_bf8_pow2(0)
        a_inf = 0x7C if af == "bf8" else None
        b_inf = 0x7C if bf == "bf8" else None

        cases.append(([0x80] * 16, [0x00] * 16, 0x8000_0000))
        cases.append(([0x00] * 16, [0x00] * 16, 0x8000_0000))
        for pos in (0, 4, 8, 12):
            av, bv = [0] * 16, [0] * 16
            av[pos], bv[pos] = a1, b1
            av[pos + 1], bv[pos + 1] = a1 ^ 0x80, b1
            cases.append((av, bv, 0x8000_0000))
        max_e = (8 if af == "fp8" else 15) + (8 if bf == "fp8" else 15)
        min_e = (-9 if af == "fp8" else -16) + (-9 if bf == "fp8" else -16)
        pa, pb = _split_pow2_8(min(max_e, 16), af, bf, False)
        sa, sb = _split_pow2_8(max(min_e, min(max_e, 16) - 26), af, bf, True)
        for p1, p2 in ((1, 2), (4, 5), (2, 5), (9, 10)):
            av, bv = [0] * 16, [0] * 16
            av[0], bv[0] = pa, pb
            av[p1], bv[p1] = sa, sb
            av[p2], bv[p2] = sa, sb
            cases.append((av, bv, 0))
            cases.append((av, bv, _f32_bits(-(2.0 ** (min(max_e, 16) - 27)))))
        for cv in (0, 0x7F80_0000, 0xFF80_0000, 0x7F80_0123, 0xFFC0_1234):
            for pos in (0, 7, 8, 15):
                av, bv = [0] * 16, [0] * 16
                av[pos], bv[pos] = 0x7F, b1
                cases.append((av, bv, cv))
                if a_inf is not None:
                    av, bv = [0] * 16, [0] * 16
                    av[pos], bv[pos] = a_inf, 0
                    cases.append((av, bv, cv))
                if a_inf is not None and b_inf is not None:
                    av, bv = [0] * 16, [0] * 16
                    av[pos], bv[pos] = a_inf, b_inf
                    cases.append((av, bv, cv))
                    av2, bv2 = [0] * 16, [0] * 16
                    av2[pos], bv2[pos] = a_inf | 0x80, b_inf
                    cases.append((av2, bv2, cv))
    else:
        is_bf16 = spec.a_fmt == "bf16"
        enc = _bf16_bits if is_bf16 else _f16_bits
        pinf = 0x7F80 if is_bf16 else 0x7C00
        ninf = 0xFF80 if is_bf16 else 0xFC00
        snan = 0x7F81 if is_bf16 else 0x7C01
        one = enc(1.0)
        c_nzero = 0x8000_0000 if spec.d_fmt == "f32" else 0x8000
        c_snan = 0x7F80_0123 if spec.d_fmt == "f32" else (0x7F85 if spec.d_fmt == "bf16" else 0x7C05)
        c_ninf = 0xFF80_0000 if spec.d_fmt == "f32" else (0xFF80 if spec.d_fmt == "bf16" else 0xFC00)

        cases.append(([0x8000] * 16, [0x0000] * 16, c_nzero))
        cases.append(([0x0000] * 16, [0x0000] * 16, c_nzero))
        for pos in (0, 4, 8, 12):
            for av_v, bv_v, cv in [
                (pinf, pinf, 0),
                (pinf, ninf, 0),
                (ninf, ninf, 0),
                (pinf, 0, 0),
                (0, pinf, c_snan),
                (snan, one, c_snan),
                (pinf, one, c_ninf),
                (pinf, one, c_snan),
            ]:
                av, bv = [0] * 16, [0] * 16
                av[pos], bv[pos] = av_v, bv_v
                cases.append((av, bv, cv))
        if spec.d_fmt == "f16":
            cases.append(([_f16_bits(2**-11), _f16_bits(-(2**-13))] + [0] * 14, [one, _f16_bits(2**-12)] + [0] * 14, 0x3C01))
            cases.append(([_f16_bits(2**-11), _f16_bits(2**-13)] + [0] * 14, [one, _f16_bits(2**-12)] + [0] * 14, 0x3C00))
            cases.append(([_f16_bits(32768.0), _f16_bits(32752.0), _f16_bits(-(2**-9))] + [0] * 13, [one, one, one] + [0] * 13, 0))
            for t in (39, 40, 41, 42):
                ea = -(t // 2)
                eb = -t - ea
                cases.append(([_f16_bits(2**-12), _f16_bits(2.0**ea)] + [0] * 14, [_f16_bits(2**-13), _f16_bits(2.0**eb)] + [0] * 14, 0))
        elif spec.d_fmt == "bf16":
            cases.append(([_bf16_bits(-(2**-13))] + [0] * 15, [_bf16_bits(2**-12)] + [0] * 15, 0x3F81))
            cases.append(([_bf16_bits(-(2**-12))] + [0] * 15, [_bf16_bits(2**-12)] + [0] * 15, 0x3F81))
            cases.append(([_bf16_bits(-(2**-12))] + [0] * 15, [_bf16_bits(2**-11)] + [0] * 15, 0x3F81))
            cases.append(([_bf16_bits(2**-13)] + [0] * 15, [_bf16_bits(2**-12)] + [0] * 15, 0xBF81))
        else:
            big_e = 60 if is_bf16 else 14
            for i, j, l in ((0, 8, 4), (0, 4, 8), (8, 4, 12), (4, 12, 0), (0, 1, 2), (0, 2, 1)):
                av, bv = [0] * 16, [0] * 16
                av[i], bv[i] = enc(2.0**big_e), enc(2.0**big_e)
                av[j], bv[j] = enc(-(2.0**big_e)), enc(2.0**big_e)
                av[l], bv[l] = one, one
                cases.append((av, bv, 0))

    return _pack_diag_cases(spec, cases)


def _detect_hardware() -> dict[str, str]:
    hip_ver = "unknown"
    ver_h = Path("/opt/rocm/include/hip/hip_version.h")
    if ver_h.is_file():
        vals = {}
        for line in ver_h.read_text().splitlines():
            parts = line.split()
            if len(parts) >= 3 and parts[0] == "#define" and parts[1].startswith("HIP_VERSION_"):
                vals[parts[1]] = parts[2]
        if all(k in vals for k in ("HIP_VERSION_MAJOR", "HIP_VERSION_MINOR", "HIP_VERSION_PATCH")):
            hip_ver = f"{vals['HIP_VERSION_MAJOR']}.{vals['HIP_VERSION_MINOR']}.{vals['HIP_VERSION_PATCH']}"
    return {
        "gpu": "AMD Radeon RX 9070 XT",
        "gfx_target": "gfx1201",
        "hip_version": hip_ver,
        "kernel_driver": platform.release(),
        "capture_date": date.today().isoformat(),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Capture frozen gfx1201 hardware evidence for all 22 instructions.")
    ap.add_argument("--co", type=Path, default=None, help="Pre-assembled probes_gfx1201.co (default: assemble locally)")
    args = ap.parse_args()
    co = args.co if args.co is not None else build_co()
    ev_dir = ROOT / "evidence"
    ev_dir.mkdir(parents=True, exist_ok=True)

    spec_chunks: dict[str, list[dict]] = {}
    all_jobs: list[dict] = []
    seed = 202601

    for spec in SPECS.values():
        chunks = []
        if not spec.sparse and not spec.is_int:
            for gen in ("finite", "raw"):
                a, b, c, _ = gen_random_chunk(spec, 200, seed, gen)
                jid = f"{spec.name}__{gen}"
                chunks.append({"tag": gen, "gen": gen, "seed": seed, "waves": 200, "jid": jid})
                all_jobs.append({"id": jid, "spec": spec.name, "a": a, "b": b, "c": c})
                seed += 1
            a_adv, b_adv, c_adv = gen_adversarial_dense_float(spec)
            jid = f"{spec.name}__adv"
            chunks.append(
                {"tag": "adv", "gen": "explicit", "waves": len(a_adv), "jid": jid, "a": a_adv, "b": b_adv, "c": c_adv}
            )
            all_jobs.append({"id": jid, "spec": spec.name, "a": a_adv, "b": b_adv, "c": c_adv})

            mod_list = (
                [("neg_c", (0, 0, 1), (0, 0, 0)), ("abs_c", (0, 0, 0), (0, 0, 1))]
                if spec.is_fp8_family
                else [
                    ("neg_a", (1, 0, 0), (1, 0, 0)),
                    ("neg_lo_a", (1, 0, 0), (0, 0, 0)),
                    ("neg_hi_b", (0, 0, 0), (0, 1, 0)),
                    ("neg_c", (0, 0, 1), (0, 0, 0)),
                    ("abs_c", (0, 0, 0), (0, 0, 1)),
                    ("negabs_c", (0, 0, 1), (0, 0, 1)),
                ]
            )
            a_m, b_m, c_m, _ = gen_random_chunk(spec, 8, seed, "raw")
            m_seed = seed
            seed += 1
            for mtag, nlo, nhi in mod_list:
                jid = f"{spec.name}__{mtag}"
                chunks.append(
                    {
                        "tag": mtag,
                        "gen": "raw",
                        "seed": m_seed,
                        "waves": 8,
                        "neg_lo": list(nlo),
                        "neg_hi": list(nhi),
                        "jid": jid,
                    }
                )
                all_jobs.append({"id": jid, "spec": spec.name, "kernel": f"{spec.name}_{mtag}", "a": a_m, "b": b_m, "c": c_m})

        elif not spec.sparse and spec.is_int:
            for sa in (0, 1):
                for sb in (0, 1):
                    for cl in (False, True):
                        tag = f"s{sa}{sb}" + ("_clamp" if cl else "")
                        kern = spec.name if (sa == 0 and sb == 0 and not cl) else f"{spec.name}_{tag}"
                        gen = "clamp_stress" if cl else "raw"
                        a, b, c, _ = gen_random_chunk(spec, 50, seed, gen)
                        jid = f"{spec.name}__{tag}"
                        chunks.append(
                            {
                                "tag": tag,
                                "gen": gen,
                                "seed": seed,
                                "waves": 50,
                                "neg_lo": [sa, sb, 0],
                                "clamp": cl,
                                "jid": jid,
                            }
                        )
                        all_jobs.append({"id": jid, "spec": spec.name, "kernel": kern, "a": a, "b": b, "c": c})
                        seed += 1

        elif spec.sparse and not spec.is_int:
            for ik in (0, 1):
                ik_tag = "_ik1" if ik == 1 else ""
                kern = f"{spec.name}{ik_tag}"
                for gen in ("finite", "raw"):
                    a, b, c, idx = gen_random_chunk(spec, 100, seed, gen)
                    tag = f"ik{ik}_{gen}"
                    jid = f"{spec.name}__{tag}"
                    chunks.append({"tag": tag, "gen": gen, "seed": seed, "waves": 100, "index_key": ik, "jid": jid})
                    all_jobs.append({"id": jid, "spec": spec.name, "kernel": kern, "a": a, "b": b, "c": c, "idx": idx})
                    seed += 1

        else:
            keys = (0, 1) if spec.has_index_key else (0,)
            nw = 25 if spec.has_index_key else 50
            for ik in keys:
                ik_tag = "_ik1" if (spec.has_index_key and ik == 1) else ""
                for sa in (0, 1):
                    for sb in (0, 1):
                        for cl in (False, True):
                            stag = "" if (sa == 0 and sb == 0 and not cl) else f"_s{sa}{sb}" + ("_clamp" if cl else "")
                            kern = f"{spec.name}{ik_tag}{stag}"
                            tag = f"ik{ik}_s{sa}{sb}" + ("_clamp" if cl else "")
                            gen = "clamp_stress" if cl else "raw"
                            a, b, c, idx = gen_random_chunk(spec, nw, seed, gen)
                            jid = f"{spec.name}__{tag}"
                            chunks.append(
                                {
                                    "tag": tag,
                                    "gen": gen,
                                    "seed": seed,
                                    "waves": nw,
                                    "index_key": ik,
                                    "neg_lo": [sa, sb, 0],
                                    "clamp": cl,
                                    "jid": jid,
                                }
                            )
                            all_jobs.append({"id": jid, "spec": spec.name, "kernel": kern, "a": a, "b": b, "c": c, "idx": idx})
                            seed += 1

        spec_chunks[spec.name] = chunks

    print(f"Capturing {len(all_jobs)} GPU jobs across all {len(SPECS)} instructions...")
    t0 = time.monotonic()
    hw_res = run_jobs(all_jobs, co_path=co)
    print(f"Hardware capture finished in {time.monotonic() - t0:.2f}s.")

    manifest = {"target": "gfx1201", "hardware": _detect_hardware(), "variants": []}
    total_bytes = 0
    total_elems = 0
    for spec in SPECS.values():
        hdr_chunks = []
        words = array.array("I")
        v_elems = 0
        for ch in spec_chunks[spec.name]:
            d_hw = hw_res[ch["jid"]]
            v_elems += ch["waves"] * 256
            hdr_entry: dict = {"tag": ch["tag"], "gen": ch["gen"], "waves": ch["waves"]}
            if ch["gen"] != "explicit":
                hdr_entry["seed"] = ch["seed"]
            if ch.get("index_key", 0):
                hdr_entry["index_key"] = ch["index_key"]
            if any(ch.get("neg_lo", (0, 0, 0))):
                hdr_entry["neg_lo"] = ch["neg_lo"]
            if any(ch.get("neg_hi", (0, 0, 0))):
                hdr_entry["neg_hi"] = ch["neg_hi"]
            if ch.get("clamp", False):
                hdr_entry["clamp"] = True
            hdr_chunks.append(hdr_entry)

            if ch["gen"] == "explicit":
                words.extend(np.asarray(ch["a"], dtype=np.uint32).reshape(-1).tolist())
                words.extend(np.asarray(ch["b"], dtype=np.uint32).reshape(-1).tolist())
                words.extend(np.asarray(ch["c"], dtype=np.uint32).reshape(-1).tolist())
                if spec.sparse:
                    words.extend(np.asarray(ch["idx"], dtype=np.uint32).reshape(-1).tolist())
            words.extend(np.asarray(d_hw, dtype=np.uint32).reshape(-1).tolist())

        out_path = ev_dir / f"{spec.name}.mmbin"
        write_evidence_file(out_path, {"target": "gfx1201", "spec": spec.name, "chunks": hdr_chunks}, words)
        sz = out_path.stat().st_size
        sha = hashlib.sha256(out_path.read_bytes()).hexdigest()
        total_bytes += sz
        total_elems += v_elems
        manifest["variants"].append(
            {
                "spec": spec.name,
                "file": out_path.name,
                "elements": v_elems,
                "waves": v_elems // 256,
                "bytes": sz,
                "sha256": sha,
            }
        )
        print(f"  wrote {out_path.name:38s}: {v_elems:7d} elems, {sz / 1024:7.1f} KiB")

    manifest["total_elements"] = total_elems
    manifest["total_bytes"] = total_bytes
    (ev_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Total frozen evidence: {total_elems:,} elements, {total_bytes / (1024 * 1024):.2f} MiB")

    rc = validate(ev_dir)
    if rc != 0:
        sys.exit(rc)


if __name__ == "__main__":
    main()
