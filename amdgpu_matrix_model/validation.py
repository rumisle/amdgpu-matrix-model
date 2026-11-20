"""GPU-free validator checking the model bit-for-bit against all frozen gfx1201 hardware evidence.

Zero runtime dependencies (Python standard library only).

Usage:
    python -m amdgpu_matrix_model.validation [--evidence DIR] [--variant NAME] [--workers N]
"""

from __future__ import annotations

import argparse
import array
import json
import os
import struct
import sys
import time
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from .model import wmma_wave
from .specs import SPECS, WmmaSpec

MAGIC = b"AMDMAT02"
STATE_MAGIC = b"AMDST001"
_MASK64 = 0xFFFF_FFFF_FFFF_FFFF
_VALID_24_NIBBLES = (0x4, 0x8, 0xC, 0x9, 0xD, 0xE)


class SplitMix64:
    """Standard 64-bit SplitMix64 PRNG (zero external dependencies)."""

    __slots__ = ("state",)

    def __init__(self, seed: int) -> None:
        self.state = seed & _MASK64

    def next_u64(self) -> int:
        s = (self.state + 0x9E37_79B9_7F4A_7C15) & _MASK64
        self.state = s
        z = ((s ^ (s >> 30)) * 0xBF58_476D_1CE4_E5B9) & _MASK64
        z = ((z ^ (z >> 27)) * 0x94D0_49BB_1331_11EB) & _MASK64
        return (z ^ (z >> 31)) & _MASK64

    def next_u32(self) -> int:
        return self.next_u64() & 0xFFFF_FFFF


def _transform_finite_word(u: int, fmt: str, idx: int, count: int) -> int:
    """Map one raw uint32 word deterministically into a finite (incl. subnormal) packed word."""
    if fmt == "f16":
        lo = u & 0xFFFF
        hi = u >> 16
        if (lo & 0x7C00) == 0x7C00:
            lo &= 0x83FF
        if (hi & 0x7C00) == 0x7C00:
            hi &= 0x83FF
        return lo | (hi << 16)
    if fmt == "bf16":
        half = count >> 1
        out = 0
        for sh in (0, 16):
            h = (u >> sh) & 0xFFFF
            if (h & 0x0F) == 0:
                h &= 0x807F
            elif idx < half:
                exp = 96 + (((h >> 7) & 0xFF) & 0x3F)
                h = (h & 0x807F) | (exp << 7)
            elif (h & 0x7F80) == 0x7F80:
                h &= 0x807F
            out |= h << sh
        return out
    if fmt == "fp8":
        out = 0
        for sh in (0, 8, 16, 24):
            b = (u >> sh) & 0xFF
            if (b & 0x7F) == 0x7F:
                b &= 0xFE
            out |= b << sh
        return out
    if fmt == "bf8":
        out = 0
        for sh in (0, 8, 16, 24):
            b = (u >> sh) & 0xFF
            if (b & 0x7C) == 0x7C:
                b &= 0x83
            out |= b << sh
        return out
    # f32
    q3 = (count * 3) >> 2
    if (u & 0x0F) == 0:
        return u & 0x807F_FFFF
    if idx < q3:
        exp = 96 + (((u >> 23) & 0xFF) & 0x3F)
        return (u & 0x807F_FFFF) | (exp << 23)
    if (u & 0x7F80_0000) == 0x7F80_0000:
        return u & 0x807F_FFFF
    return u


def _gen_sparse_idx(rng: SplitMix64, waves: int) -> list[int]:
    """First half of waves use valid 2:4 index nibbles; second half use raw uint32 words."""
    half_words = (waves // 2) * 32
    total_words = waves * 32
    out = [0] * total_words
    for i in range(half_words):
        u = rng.next_u64()
        w = 0
        for nib in range(8):
            w |= _VALID_24_NIBBLES[((u >> (8 * nib)) & 0xFF) % 6] << (4 * nib)
        out[i] = w
    for i in range(half_words, total_words):
        out[i] = rng.next_u32()
    return out


def generate_chunk_inputs(
    spec: WmmaSpec, waves: int, seed: int, gen: str
) -> tuple[list[int], list[int], list[int], list[int] | None]:
    """Deterministically generate (A, B, C, IDX) uint32 word lists from (seed, gen) using SplitMix64."""
    rng = SplitMix64(seed)
    na = waves * 32 * spec.a_dwords
    nb = waves * 32 * spec.b_dwords
    nc = waves * 32 * spec.cd_dwords

    if gen == "raw":
        a = [rng.next_u32() for _ in range(na)]
        b = [rng.next_u32() for _ in range(nb)]
        c = [rng.next_u32() for _ in range(nc)]
    elif gen == "finite":
        a = [_transform_finite_word(rng.next_u32(), spec.a_fmt, i, na) for i in range(na)]
        b = [_transform_finite_word(rng.next_u32(), spec.b_fmt, i, nb) for i in range(nb)]
        c = [_transform_finite_word(rng.next_u32(), spec.d_fmt, i, nc) for i in range(nc)]
    elif gen == "clamp_stress":
        a = [rng.next_u32() for _ in range(na)]
        b = [rng.next_u32() for _ in range(nb)]
        half = nc >> 1
        c = [
            (((0x7FFF_FFFF if i < half else 0x8000_0000) + ((rng.next_u32() & 0xFFFF) - 32768)) & 0xFFFF_FFFF)
            for i in range(nc)
        ]
    else:
        raise ValueError(f"Unknown chunk generator: {gen!r}")

    idx = _gen_sparse_idx(rng, waves) if spec.sparse else None
    return a, b, c, idx


def default_evidence_dir() -> Path:
    here = Path(__file__).resolve().parent
    for cand in (here.parent / "evidence", here / "evidence", Path.cwd() / "evidence"):
        if cand.is_dir():
            return cand
    return here.parent / "evidence"


def read_evidence_file(path: Path) -> tuple[dict, array.array]:
    """Read a compact `.mmbin` evidence file into (header_dict, uint32_array)."""
    raw = path.read_bytes()
    if len(raw) < 12 or raw[:8] != MAGIC:
        raise ValueError(f"{path}: invalid magic header (expected {MAGIC!r})")
    hdr_len = struct.unpack_from("<I", raw, 8)[0]
    hdr_end = 12 + hdr_len
    header = json.loads(raw[12:hdr_end].decode("utf-8"))
    payload = zlib.decompress(raw[hdr_end:])
    words = array.array("I")
    words.frombytes(payload)
    if sys.byteorder != "little":
        words.byteswap()
    return header, words


def write_evidence_file(path: Path, header: dict, words: array.array) -> None:
    """Write a compact `.mmbin` evidence file (magic + JSON header + zlib u32 payload)."""
    hdr_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    out_words = array.array("I", words)
    if sys.byteorder != "little":
        out_words.byteswap()
    comp = zlib.compress(out_words.tobytes(), level=6)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(MAGIC + struct.pack("<I", len(hdr_bytes)) + hdr_bytes + comp)


def validate_state_file(path: Path) -> tuple[int, int]:
    """Validate frozen hardware state probes (MODE, raw clamp, sentinel/op_sel, EXEC masking)."""
    raw = path.read_bytes()
    if len(raw) < 8 or raw[:8] != STATE_MAGIC:
        raise ValueError(f"{path}: invalid state magic")
    doc = json.loads(zlib.decompress(raw[8:]).decode("utf-8"))
    checks = 0
    mismatches = 0

    by_mode = {row["write"]: row for row in doc["mode_ctrl"]}
    ctrl_asserts = [
        by_mode[0x00F0]["f32_tie"] == 0x3F80_0000 and by_mode[0x00F1]["f32_tie"] == 0x3F80_0001,
        by_mode[0x00F0]["f16_tie"] == 0x3C00 and by_mode[0x00F4]["f16_tie"] == 0x3C01,
        by_mode[0x00F0]["f32_sub"] == 0x0080_0000 and by_mode[0x00C0]["f32_sub"] == 0x0000_0000,
        by_mode[0x00F0]["f16_sub"] == 0x0400 and by_mode[0x0030]["f16_sub"] == 0x0000,
        by_mode[0xFFF0]["read"] == 0x0000_00F0,
    ]
    checks += len(ctrl_asserts)
    mismatches += sum(1 for ok in ctrl_asserts if not ok)

    for entry in doc["mode_wmma"]:
        spec = SPECS[entry["spec"]]
        wpm = int(entry["waves_per_mode"])
        a_per, b_per, cd_per = 32 * spec.a_dwords, 32 * spec.b_dwords, 32 * spec.cd_dwords
        expected_waves = []
        for w in range(wpm):
            expected_waves.extend(
                wmma_wave(
                    spec.name,
                    entry["a"][w * a_per : (w + 1) * a_per],
                    entry["b"][w * b_per : (w + 1) * b_per],
                    entry["c"][w * cd_per : (w + 1) * cd_per],
                )
            )
        for d_hw in entry["d_by_mode"]:
            checks += wpm * 256
            if d_hw != expected_waves:
                mismatches += 1

    for entry in doc["raw_clamp"]:
        pred = wmma_wave(entry["spec"], entry["a"], entry["b"], entry["c"])
        checks += 256
        if entry["d_hw"] != pred:
            mismatches += 1

    for entry in doc["sentinel_opsel"]:
        pred = wmma_wave(entry["spec"], entry["a"], entry["b"], entry["c"])
        d8 = entry["d_sentinel_8dw"]
        lo4 = [w for lane in range(32) for w in d8[lane * 8 : lane * 8 + 4]]
        hi4 = [w for lane in range(32) for w in d8[lane * 8 + 4 : lane * 8 + 8]]
        checks += 256 * 3
        if lo4 != pred or any(w != 0xDEAD_BEEF for w in hi4):
            mismatches += 1
        if entry["d_rawopsel_7"] != pred:
            mismatches += 1
        if entry["d_rawopsel_hi0"] != pred:
            mismatches += 1

    for entry in doc["exec_mask"]:
        pred = wmma_wave(entry["spec"], entry["a"], entry["b"], entry["c"], idx_words=entry["idx"])
        for d_hw in entry["d_by_mask"]:
            checks += 256
            if d_hw != pred:
                mismatches += 1

    return checks, mismatches


def _validate_file(path_str: str) -> tuple[str, int, int, list[str], float]:
    t0 = time.monotonic()
    path = Path(path_str)
    header, words = read_evidence_file(path)
    spec_name = header["spec"]
    spec = SPECS[spec_name]

    a_per = 32 * spec.a_dwords
    b_per = 32 * spec.b_dwords
    cd_per = 32 * spec.cd_dwords
    idx_per = 32 if spec.sparse else 0

    pos = 0
    total_elems = 0
    mismatches = 0
    details: list[str] = []

    for chunk in header["chunks"]:
        waves = int(chunk["waves"])
        tag = chunk["tag"]
        gen = chunk.get("gen", "explicit")
        ik = int(chunk.get("index_key", 0))
        nlo = tuple(chunk.get("neg_lo", (0, 0, 0)))
        nhi = tuple(chunk.get("neg_hi", (0, 0, 0)))
        cl = bool(chunk.get("clamp", False))

        c_len = waves * cd_per
        if gen == "explicit":
            a_len = waves * a_per
            b_len = waves * b_per
            i_len = waves * idx_per
            a_slice = words[pos : pos + a_len]
            pos += a_len
            b_slice = words[pos : pos + b_len]
            pos += b_len
            c_slice = words[pos : pos + c_len]
            pos += c_len
            idx_slice = words[pos : pos + i_len] if spec.sparse else None
            pos += i_len
        else:
            a_slice, b_slice, c_slice, idx_slice = generate_chunk_inputs(spec, waves, int(chunk["seed"]), gen)

        d_slice = words[pos : pos + c_len]
        pos += c_len

        for w in range(waves):
            aw = a_slice[w * a_per : (w + 1) * a_per]
            bw = b_slice[w * b_per : (w + 1) * b_per]
            cw = c_slice[w * cd_per : (w + 1) * cd_per]
            dw = d_slice[w * cd_per : (w + 1) * cd_per]
            iw = idx_slice[w * idx_per : (w + 1) * idx_per] if idx_slice is not None else None
            pred = wmma_wave(
                spec_name,
                aw,
                bw,
                cw,
                idx_words=iw,
                index_key=ik,
                neg_lo=nlo,
                neg_hi=nhi,
                clamp=cl,
            )
            total_elems += 256
            if list(dw) != pred:
                if spec.d_bits == 32:
                    for k_w, (hw_w, pr_w) in enumerate(zip(dw, pred)):
                        if hw_w != pr_w:
                            mismatches += 1
                            if len(details) < 5:
                                details.append(f"{tag} w={w} dw={k_w}: hw={hw_w:#010x} pred={pr_w:#010x}")
                else:
                    for k_w, (hw_w, pr_w) in enumerate(zip(dw, pred)):
                        if hw_w != pr_w:
                            if (hw_w & 0xFFFF) != (pr_w & 0xFFFF):
                                mismatches += 1
                            if (hw_w >> 16) != (pr_w >> 16):
                                mismatches += 1
                            if len(details) < 5:
                                details.append(f"{tag} w={w} dw={k_w}: hw={hw_w:#010x} pred={pr_w:#010x}")

    if pos != len(words):
        raise ValueError(f"{path}: trailing words ({pos} != {len(words)})")
    return spec_name, total_elems, mismatches, details, time.monotonic() - t0


def validate(
    evidence_dir: Path | None = None,
    variants: list[str] | None = None,
    workers: int | None = None,
) -> int:
    ev_dir = evidence_dir or default_evidence_dir()
    files = sorted(ev_dir.glob("*.mmbin")) if ev_dir.is_dir() else []
    if variants:
        want = set(variants)
        files = [f for f in files if f.stem in want]
    if not files:
        print(
            f"No .mmbin evidence files found in {ev_dir}.\n"
            "Frozen evidence is shipped in the repository checkout (not inside the installed wheel).\n"
            "Run from a repository checkout or pass --evidence PATH_TO_EVIDENCE_DIR.",
            file=sys.stderr,
        )
        return 2

    n_workers = workers if workers is not None else min(os.cpu_count() or 1, len(files), 16)
    t0 = time.monotonic()
    if n_workers <= 1 or len(files) == 1:
        results = [_validate_file(str(f)) for f in files]
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            results = list(ex.map(_validate_file, [str(f) for f in files]))

    grand_elems = 0
    grand_mm = 0
    for spec_name, elems, mm, details, dt in results:
        grand_elems += elems
        grand_mm += mm
        status = "PASS" if mm == 0 else "FAIL"
        print(f"[{status}] {spec_name:34s}: {elems - mm:7d}/{elems:7d} bit-exact ({mm} mismatches) [{dt:.2f}s]")
        for d in details:
            print(f"   {d}")

    state_path = ev_dir / "state_gfx1201.mmstate"
    if not variants and state_path.is_file():
        t_s = time.monotonic()
        st_checks, state_mm = validate_state_file(state_path)
        st_status = "PASS" if state_mm == 0 else "FAIL"
        print(
            f"[{st_status}] {'state_gfx1201 (MODE/clamp/op_sel/EXEC)':34s}: "
            f"{st_checks - state_mm:7d}/{st_checks:7d} checks exact ({state_mm} mismatches) [{time.monotonic() - t_s:.2f}s]"
        )
        grand_mm += state_mm

    print(
        f"\nValidated {len(results)}/{len(SPECS)} instructions, "
        f"{grand_elems - grand_mm:,}/{grand_elems:,} elements bit-exact "
        f"({grand_mm} mismatches) in {time.monotonic() - t0:.2f}s."
    )
    return 0 if grand_mm == 0 else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Validate gfx1201 WMMA/SWMMAC model against frozen hardware evidence.")
    ap.add_argument("--evidence", type=Path, default=None, help="Directory containing .mmbin files")
    ap.add_argument("--variant", action="append", default=None, help="Specific instruction name(s) to validate")
    ap.add_argument("--workers", type=int, default=None, help="Parallel worker processes (default: auto)")
    args = ap.parse_args(argv)
    return validate(args.evidence, args.variant, args.workers)


if __name__ == "__main__":
    sys.exit(main())
