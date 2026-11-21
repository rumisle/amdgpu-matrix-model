"""Capture and freeze compact binary hardware evidence (AMDMAT02) for all 22 gfx1201 instructions.

`plan()` is the complete corpus definition. Random chunks store only their SplitMix64 seed, generator
and the hardware D words; explicit chunks store their A, B, C (, IDX) inputs too.

    capture_evidence.py            capture every planned chunk and rewrite all files
    capture_evidence.py --update   keep every stored chunk whose header (and explicit inputs) still
                                   match the plan, capture only new or changed chunks, and rewrite
                                   only the files that change

The manifest records, per chunk, the capture session and the SHA-256 of its inputs.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from amdgpu_matrix_model.evidence import (  # noqa: E402
    MANIFEST_FORMAT,
    STATE_FILE,
    TARGET,
    Chunk,
    generate_chunk_inputs,
    inputs_sha256,
    iter_chunk_data,
    read_manifest,
    read_matrix_file,
    sha256_file,
    write_matrix_file,
)
from amdgpu_matrix_model.layout import pack_a, pack_b, pack_cd  # noqa: E402
from amdgpu_matrix_model.specs import SPECS, WmmaSpec  # noqa: E402
from tools import cases  # noqa: E402
from tools.cases import bf16_bits as _bf16_bits  # noqa: E402
from tools.cases import f16_bits as _f16_bits  # noqa: E402
from tools.cases import f32_bits as _f32_bits  # noqa: E402
from tools.probe_asm import AB_MODS, C_MODS, SPARSE_AB_MODS, build_co, kernel_name  # noqa: E402

EVIDENCE = ROOT / "evidence"


@dataclass(eq=False)
class Planned:
    spec: str
    chunk: Chunk
    a: np.ndarray
    b: np.ndarray
    c: np.ndarray
    idx: np.ndarray | None
    kernel: str
    d: np.ndarray | None = None
    session: str | None = None

    def flat_inputs(self) -> tuple[list[int], list[int], list[int], list[int] | None]:
        f = lambda x: np.asarray(x, dtype=np.uint32).reshape(-1).tolist()  # noqa: E731
        return f(self.a), f(self.b), f(self.c), (f(self.idx) if self.idx is not None else None)


def _random(spec: WmmaSpec, chunk: Chunk) -> Planned:
    a, b, c, idx = generate_chunk_inputs(spec, chunk.waves, chunk.seed, chunk.gen)
    w = chunk.waves
    arr = lambda v, dw: np.array(v, dtype=np.uint32).reshape(w, 32, dw)  # noqa: E731
    return Planned(
        spec.name,
        chunk,
        arr(a, spec.a_dwords),
        arr(b, spec.b_dwords),
        arr(c, spec.cd_dwords),
        np.array(idx, dtype=np.uint32).reshape(w, 32) if idx is not None else None,
        kernel_name(spec, chunk.index_key, chunk.neg_lo, chunk.neg_hi, chunk.clamp),
    )


def _explicit(spec: WmmaSpec, tag: str, packed, index_key: int = 0, neg_lo=(0, 0, 0), neg_hi=(0, 0, 0)) -> Planned:
    a, b, c, idx = packed
    chunk = Chunk(tag, "explicit", len(a), None, index_key, tuple(neg_lo), tuple(neg_hi), False)
    return Planned(spec.name, chunk, a, b, c, idx, kernel_name(spec, index_key, chunk.neg_lo, chunk.neg_hi, False))


def _pack_diag_cases(spec: WmmaSpec, case_list: list[tuple[list[int], list[int], int]]):
    return cases.pack_cases(spec, [(a, b, None, c) for a, b, c in case_list])


def _enc_fp8_pow2(exp: int, neg: bool = False) -> int:
    return cases.pow2("fp8", exp, neg)


def _enc_bf8_pow2(exp: int, neg: bool = False) -> int:
    return cases.pow2("bf8", exp, neg)


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


def gen_adversarial_dense_float(spec: WmmaSpec):
    """The original hand-built special-value and alignment cases (first capture, unchanged inputs).

    Slot positions are logical K labels (RDNA4 ISA layout). The loops over (0, 8, 4, 12) visit the
    first slot of steps 0, 2, 1, 3 and the triples below place their three products in three different
    steps; both orders are kept from the first capture so that the stored inputs stay bit-identical.
    """
    case_list: list[tuple[list[int], list[int], int]] = []

    if spec.is_fp8_family:
        af, bf = spec.a_fmt, spec.b_fmt
        a1 = _enc_fp8_pow2(0) if af == "fp8" else _enc_bf8_pow2(0)
        b1 = _enc_fp8_pow2(0) if bf == "fp8" else _enc_bf8_pow2(0)
        a_inf = 0x7C if af == "bf8" else None
        b_inf = 0x7C if bf == "bf8" else None

        case_list.append(([0x80] * 16, [0x00] * 16, 0x8000_0000))
        case_list.append(([0x00] * 16, [0x00] * 16, 0x8000_0000))
        for pos in (0, 4, 8, 12):
            av, bv = [0] * 16, [0] * 16
            av[pos], bv[pos] = a1, b1
            av[pos + 1], bv[pos + 1] = a1 ^ 0x80, b1
            case_list.append((av, bv, 0x8000_0000))
        max_e = (8 if af == "fp8" else 15) + (8 if bf == "fp8" else 15)
        min_e = (-9 if af == "fp8" else -16) + (-9 if bf == "fp8" else -16)
        pa, pb = _split_pow2_8(min(max_e, 16), af, bf, False)
        sa, sb = _split_pow2_8(max(min_e, min(max_e, 16) - 26), af, bf, True)
        for p1, p2 in ((1, 2), (4, 5), (2, 5), (9, 10)):
            av, bv = [0] * 16, [0] * 16
            av[0], bv[0] = pa, pb
            av[p1], bv[p1] = sa, sb
            av[p2], bv[p2] = sa, sb
            case_list.append((av, bv, 0))
            case_list.append((av, bv, _f32_bits(-(2.0 ** (min(max_e, 16) - 27)))))
        for cv in (0, 0x7F80_0000, 0xFF80_0000, 0x7F80_0123, 0xFFC0_1234):
            for pos in (0, 7, 8, 15):
                av, bv = [0] * 16, [0] * 16
                av[pos], bv[pos] = 0x7F, b1
                case_list.append((av, bv, cv))
                if a_inf is not None:
                    av, bv = [0] * 16, [0] * 16
                    av[pos], bv[pos] = a_inf, 0
                    case_list.append((av, bv, cv))
                if a_inf is not None and b_inf is not None:
                    av, bv = [0] * 16, [0] * 16
                    av[pos], bv[pos] = a_inf, b_inf
                    case_list.append((av, bv, cv))
                    av2, bv2 = [0] * 16, [0] * 16
                    av2[pos], bv2[pos] = a_inf | 0x80, b_inf
                    case_list.append((av2, bv2, cv))
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

        case_list.append(([0x8000] * 16, [0x0000] * 16, c_nzero))
        case_list.append(([0x0000] * 16, [0x0000] * 16, c_nzero))
        for pos in (0, 8, 4, 12):
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
                case_list.append((av, bv, cv))
        if spec.d_fmt == "f16":
            case_list.append(([_f16_bits(2**-11), _f16_bits(-(2**-13))] + [0] * 14, [one, _f16_bits(2**-12)] + [0] * 14, 0x3C01))
            case_list.append(([_f16_bits(2**-11), _f16_bits(2**-13)] + [0] * 14, [one, _f16_bits(2**-12)] + [0] * 14, 0x3C00))
            case_list.append(([_f16_bits(32768.0), _f16_bits(32752.0), _f16_bits(-(2**-9))] + [0] * 13, [one, one, one] + [0] * 13, 0))
            for t in (39, 40, 41, 42):
                ea = -(t // 2)
                eb = -t - ea
                case_list.append(([_f16_bits(2**-12), _f16_bits(2.0**ea)] + [0] * 14, [_f16_bits(2**-13), _f16_bits(2.0**eb)] + [0] * 14, 0))
        elif spec.d_fmt == "bf16":
            case_list.append(([_bf16_bits(-(2**-13))] + [0] * 15, [_bf16_bits(2**-12)] + [0] * 15, 0x3F81))
            case_list.append(([_bf16_bits(-(2**-12))] + [0] * 15, [_bf16_bits(2**-12)] + [0] * 15, 0x3F81))
            case_list.append(([_bf16_bits(-(2**-12))] + [0] * 15, [_bf16_bits(2**-11)] + [0] * 15, 0x3F81))
            case_list.append(([_bf16_bits(2**-13)] + [0] * 15, [_bf16_bits(2**-12)] + [0] * 15, 0xBF81))
        else:
            big_e = 60 if is_bf16 else 14
            for i, j, l in ((0, 4, 8), (0, 8, 4), (4, 8, 12), (8, 12, 0), (0, 1, 2), (0, 2, 1)):
                av, bv = [0] * 16, [0] * 16
                av[i], bv[i] = enc(2.0**big_e), enc(2.0**big_e)
                av[j], bv[j] = enc(-(2.0**big_e)), enc(2.0**big_e)
                av[l], bv[l] = one, one
                case_list.append((av, bv, 0))

    return _pack_diag_cases(spec, case_list)


def _first_capture(spec: WmmaSpec, seed: int) -> tuple[list[Planned], int]:
    """Chunks of the first capture, with the same seed sequence."""
    out: list[Planned] = []
    if not spec.sparse and not spec.is_int:
        for gen in ("finite", "raw"):
            out.append(_random(spec, Chunk(gen, gen, 200, seed)))
            seed += 1
        out.append(_explicit(spec, "adv", gen_adversarial_dense_float(spec)))
        if spec.is_fp8_family:
            mods = [("neg_c", *C_MODS[0]), ("abs_c", *C_MODS[1])]
        else:
            tags = ("neg_a", "neg_lo_a", "neg_hi_b")
            mods = [(t, *m) for t, m in zip(tags, AB_MODS)]
            mods += [(t, *m) for t, m in zip(("neg_c", "abs_c", "negabs_c"), C_MODS)]
        for tag, lo, hi in mods:
            out.append(_random(spec, Chunk(tag, "raw", 8, seed, 0, lo, hi)))
        seed += 1
    elif not spec.sparse:
        for sa in (0, 1):
            for sb in (0, 1):
                for cl in (False, True):
                    tag = f"s{sa}{sb}" + ("_clamp" if cl else "")
                    out.append(_random(spec, Chunk(tag, "clamp_stress" if cl else "raw", 50, seed, 0, (sa, sb, 0), (0, 0, 0), cl)))
                    seed += 1
    elif not spec.is_int:
        for ik in (0, 1):
            for gen in ("finite", "raw"):
                out.append(_random(spec, Chunk(f"ik{ik}_{gen}", gen, 100, seed, ik)))
                seed += 1
    else:
        keys = (0, 1) if spec.has_index_key else (0,)
        nw = 25 if spec.has_index_key else 50
        for ik in keys:
            for sa in (0, 1):
                for sb in (0, 1):
                    for cl in (False, True):
                        tag = f"ik{ik}_s{sa}{sb}" + ("_clamp" if cl else "")
                        gen = "clamp_stress" if cl else "raw"
                        out.append(_random(spec, Chunk(tag, gen, nw, seed, ik, (sa, sb, 0), (0, 0, 0), cl)))
                        seed += 1
    return out, seed


def _targeted(spec: WmmaSpec, seed: int) -> tuple[list[Planned], int]:
    """Second capture: step order, cancellation, tiny accumulators, signed zeros, sparse modifiers."""
    out: list[Planned] = []
    if spec.is_int or spec.is_fp8_family:
        return out, seed
    out.append(_explicit(spec, "k_order", cases.pack_cases(spec, cases.k_order_cases(spec))))
    out.append(_explicit(spec, "cancel", cases.pack_cases(spec, cases.cancel_cases(spec))))
    out.append(_explicit(spec, "tiny_acc", cases.pack_cases(spec, cases.tiny_acc_cases(spec))))
    out.append(_explicit(spec, "signed_zero", cases.pack_cases(spec, cases.signed_zero_cases(spec))))
    if spec.sparse:
        witness = cases.modifier_witness_cases(spec)
        for ik in (0, 1):
            for lo, hi in (((0, 0, 0), (0, 0, 0)),) + SPARSE_AB_MODS:
                mtag = "".join(map(str, lo)) + "_" + "".join(map(str, hi))
                packed = cases.pack_cases(spec, witness, index_key=ik, junk_other_half=True)
                out.append(_explicit(spec, f"ik{ik}_mod_witness_{mtag}", packed, ik, lo, hi))
                if any(lo) or any(hi):
                    out.append(_random(spec, Chunk(f"ik{ik}_mod_{mtag}", "finite", 8, seed, ik, lo, hi)))
                    seed += 1
    return out, seed


def plan() -> dict[str, list[Planned]]:
    """The complete corpus: per instruction, the first-capture chunks followed by the targeted ones."""
    out: dict[str, list[Planned]] = {}
    seed = 202601
    for spec in SPECS.values():
        out[spec.name], seed = _first_capture(spec, seed)
    seed = 303001
    for spec in SPECS.values():
        more, seed = _targeted(spec, seed)
        out[spec.name] += more
    return out


def _old_sessions(manifest_path: Path) -> tuple[list[dict], dict[tuple[str, str], str]]:
    """Capture sessions and per-chunk (and state file) session ids from an existing manifest (format 1 or 2)."""
    if not manifest_path.is_file():
        return [], {}
    m = read_manifest(manifest_path)
    if m.get("format") == MANIFEST_FORMAT:
        by_chunk = {(f["file"], ch["tag"]): ch["session"] for f in m["files"] if f["kind"] == "matrix" for ch in f["chunks"]}
        by_chunk.update({(f["file"], f["sha256"]): f["session"] for f in m["files"] if f["kind"] == "state"})
        return m["sessions"], by_chunk
    # Format 1 (first release): one session, chunk list implied by the files.
    hw = m["hardware"]
    sess = {
        "id": hw["capture_date"],
        "date": hw["capture_date"],
        "hardware": {"device_name": hw["gpu"], "isa": hw["gfx_target"], "hip_version": hw["hip_version"]},
        "code_object_sha256": None,
        "tool_revision": None,
        "note": "first capture; code objects and capture logs were not retained",
    }
    return [sess], {}


def main() -> None:
    ap = argparse.ArgumentParser(description="Capture frozen gfx1201 hardware evidence for all 22 instructions.")
    ap.add_argument("--co", type=Path, default=None, help="Pre-assembled probes_gfx1201.co (default: assemble locally)")
    ap.add_argument("--update", action="store_true", help="Reuse stored chunks that still match the plan")
    ap.add_argument("--dry-run", action="store_true", help="Only report what would be captured")
    args = ap.parse_args()

    planned = plan()
    sessions, old_chunk_session = _old_sessions(EVIDENCE / "manifest.json")
    default_old = sessions[0]["id"] if sessions else None
    todo: list[Planned] = []
    changed: set[str] = set()
    for name, chunks in planned.items():
        path = EVIDENCE / f"{name}.mmbin"
        stored: dict[str, tuple] = {}
        if args.update and path.is_file():
            mf = read_matrix_file(path, name)
            stored = {ch.tag: (ch, a, b, c, idx, d) for ch, a, b, c, idx, d in iter_chunk_data(mf)}
        kept = 0
        for p in chunks:
            hit = stored.get(p.chunk.tag)
            if hit is not None and hit[0] == p.chunk and inputs_sha256(*hit[1:5]) == inputs_sha256(*p.flat_inputs()):
                p.d = np.array(hit[5], dtype=np.uint32)
                p.session = old_chunk_session.get((path.name, p.chunk.tag), default_old)
                kept += 1
            else:
                todo.append(p)
        if kept != len(chunks) or set(stored) - {p.chunk.tag for p in chunks}:
            changed.add(name)
        print(f"  {name:34s}: {kept:3d} stored chunks kept, {len(chunks) - kept:3d} to capture")
    waves = sum(p.chunk.waves for p in todo)
    print(f"{len(todo)} chunks ({waves} waves) to capture; {len(changed)} files change")
    if args.dry_run:
        return
    if todo:
        from tools.hw_run import capture_session, run_jobs

        co = args.co if args.co is not None else build_co()
        session = capture_session(co, "matrix")
        sessions = [s for s in sessions if s["id"] != session["id"]] + [session]
        jobs = []
        for i, p in enumerate(todo):
            job = {"id": f"j{i}", "spec": p.spec, "kernel": p.kernel, "a": p.a, "b": p.b, "c": p.c}
            if p.idx is not None:
                job["idx"] = p.idx
            jobs.append(job)
        t0 = time.monotonic()
        res = run_jobs(jobs, co_path=co)
        print(f"Hardware capture finished in {time.monotonic() - t0:.2f}s.")
        for i, p in enumerate(todo):
            p.d = res[f"j{i}"]
            p.session = session["id"]

    files = []
    for name, chunks in planned.items():
        spec = SPECS[name]
        path = EVIDENCE / f"{name}.mmbin"
        if name in changed or not args.update:
            words: list[int] = []
            for p in chunks:
                a, b, c, idx = p.flat_inputs()
                if p.chunk.gen == "explicit":
                    words += a + b + c + (idx or [])
                words += np.asarray(p.d, dtype=np.uint32).reshape(-1).tolist()
            write_matrix_file(path, spec, [p.chunk for p in chunks], words)
        mf = read_matrix_file(path, name)
        assert [c.tag for c in mf.chunks] == [p.chunk.tag for p in chunks]
        files.append(
            {
                "file": path.name,
                "kind": "matrix",
                "spec": name,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "waves": sum(c.waves for c in mf.chunks),
                "elements": 256 * sum(c.waves for c in mf.chunks),
                "chunks": [
                    {"tag": p.chunk.tag, "waves": p.chunk.waves, "inputs_sha256": inputs_sha256(*p.flat_inputs()), "session": p.session}
                    for p in chunks
                ],
            }
        )
    write_manifest(files, sessions, old_chunk_session)


def write_manifest(matrix_files: list[dict], sessions: list[dict], old: dict) -> None:
    """Write evidence/manifest.json from the matrix entries and the current state file.

    The state file's session comes from build/state_session.json (written by probe_state.py) when its
    SHA-256 matches, otherwise from the previous manifest entry for the same file content.
    """
    from amdgpu_matrix_model.validation import state_summary

    state_path = EVIDENCE / STATE_FILE
    st = state_summary(state_path)
    sha = sha256_file(state_path)
    side = ROOT / "build" / "state_session.json"
    sess = json.loads(side.read_text()) if side.is_file() else None
    if sess is not None and sess.pop("file_sha256") == sha:
        sessions = [s for s in sessions if s["id"] != sess["id"]] + [sess]
        state_session = sess["id"]
    elif (STATE_FILE, sha) in old:
        state_session = old[(STATE_FILE, sha)]
    else:
        raise SystemExit("state file has no known capture session (run tools/probe_state.py --capture)")
    used = {c["session"] for f in matrix_files for c in f["chunks"]} | {state_session}
    sessions = [s for s in sessions if s["id"] in used]
    files = matrix_files + [
        {"file": STATE_FILE, "kind": "state", "bytes": state_path.stat().st_size, "sha256": sha, "session": state_session, **st}
    ]
    manifest = {
        "format": MANIFEST_FORMAT,
        "target": TARGET,
        "sessions": sessions,
        "files": files,
        "totals": {
            "matrix_files": len(matrix_files),
            "matrix_waves": sum(f["waves"] for f in matrix_files),
            "matrix_elements": sum(f["elements"] for f in matrix_files),
            "state_records": st["records"],
            "state_checks": st["checks"],
            "bytes": sum(f["bytes"] for f in files),
        },
    }
    path = EVIDENCE / "manifest.json"
    path.write_text(json.dumps(manifest, indent=1) + "\n")
    print(f"Wrote {path} (sha256 {sha256_file(path)})")


if __name__ == "__main__":
    main()
