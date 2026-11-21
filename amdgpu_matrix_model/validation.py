"""GPU-free validator: checks a model bit for bit against the frozen gfx1201 hardware evidence.

Full validation (the default) is a release gate. It passes only if

- the manifest has the pinned SHA-256 of this release (or the one given with --manifest-sha256),
- the evidence directory holds exactly the release inventory (22 `.mmbin` files and the state file),
  each with the size and SHA-256 the manifest lists,
- every file parses under the strict schema of EVIDENCE.md, names the instruction its file name and
  manifest entry name, and has exactly the manifest's chunks (tags, waves, input SHA-256),
- every predicted D word list has exactly the hardware's length and every element is equal,
- every state record of the release inventory is present and every recorded output is reproduced.

`--variant NAME` validates only the named instructions (still after the full integrity check) and
reports PARTIAL. Exit status: 0 pass, 1 mismatches, 2 malformed, missing or unexpected evidence.

Usage:
    python -m amdgpu_matrix_model.validation [--evidence DIR] [--variant NAME ...] [--workers N]
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from .evidence import (
    CONTROL_OPERANDS,
    CONTROL_RESULTS,
    CONTROL_WRITES,
    MANIFEST_FORMAT,
    STATE_FILE,
    STATE_PREFILL,
    STATE_STORE_DWORDS,
    TARGET,
    EvidenceError,
    decode_vop3p,
    expected_files,
    inputs_sha256,
    is_u,
    iter_chunk_data,
    probe_fields,
    probe_runs,
    read_manifest,
    read_matrix_file,
    read_state_file,
    sha256_file,
    state_probes,
)
from .model import wmma_wave
from .specs import SPECS, WmmaSpec

# SHA-256 of evidence/manifest.json of this release. The manifest in turn pins every evidence file.
RELEASE_MANIFEST_SHA256 = "6240d170cffb038489b32ffa3d192af5312eade08f1aa3087467eb68f93bf8a5"

WaveModel = Callable[..., list[int]]  # signature of model.wmma_wave


@dataclass
class Tally:
    """Separate counters; every pair is (checked, mismatched) in one unit."""

    elements: int = 0  # D elements (one per f32/i32 word, two per f16/bf16 word)
    element_mm: int = 0
    preserved: int = 0  # destination words past D that must keep their prefill value
    preserved_mm: int = 0
    control: int = 0  # MODE control-row values
    control_mm: int = 0
    details: list[str] = field(default_factory=list)

    @property
    def mismatches(self) -> int:
        return self.element_mm + self.preserved_mm + self.control_mm

    def note(self, msg: str) -> None:
        if len(self.details) < 8:
            self.details.append(msg)

    def add(self, other: Tally) -> None:
        for k in ("elements", "element_mm", "preserved", "preserved_mm", "control", "control_mm"):
            setattr(self, k, getattr(self, k) + getattr(other, k))


def compare_d(spec: WmmaSpec, hw: list[int], pred: list[int], tally: Tally, where: str) -> None:
    """Compare one wave's D words element by element.

    The prediction must be a list of exactly len(hw) raw u32 words: Python ints in 0..2**32-1
    (bool, float and out-of-range values are errors, whether or not they compare equal).
    """
    if type(pred) is not list or len(pred) != len(hw):
        raise EvidenceError(f"{where}: model returned {len(pred) if hasattr(pred, '__len__') else pred!r} words, hardware has {len(hw)}")
    for i, p in enumerate(pred):
        if not is_u(p, 32):
            raise EvidenceError(f"{where}: model word {i} is {p!r}, not a u32 int")
    per = 32 // spec.d_bits
    mask = (1 << spec.d_bits) - 1
    tally.elements += per * len(hw)
    for i, (h, p) in enumerate(zip(hw, pred)):
        if h != p:
            n = sum(((h >> (s * spec.d_bits)) & mask) != ((p >> (s * spec.d_bits)) & mask) for s in range(per))
            tally.element_mm += n
            tally.note(f"{where} word {i}: hw={h:#010x} model={p:#010x}")


# --- matrix files ------------------------------------------------------------------------------


def validate_matrix_file(path: Path, entry: dict, model: WaveModel = wmma_wave) -> Tally:
    """Validate one `.mmbin` against its manifest entry and the model."""
    mf = read_matrix_file(path, entry["spec"])
    spec = mf.spec
    if [c.tag for c in mf.chunks] != [c["tag"] for c in entry["chunks"]]:
        raise EvidenceError(f"{path.name}: chunk list differs from the manifest")
    tally = Tally()
    a_per, b_per, cd_per = 32 * spec.a_dwords, 32 * spec.b_dwords, 32 * spec.cd_dwords
    for (ch, a, b, c, idx, d), m in zip(iter_chunk_data(mf), entry["chunks"]):
        if ch.waves != m["waves"] or inputs_sha256(a, b, c, idx) != m["inputs_sha256"]:
            raise EvidenceError(f"{path.name}:{ch.tag}: waves or input SHA-256 differ from the manifest")
        for w in range(ch.waves):
            pred = model(
                spec.name,
                a[w * a_per : (w + 1) * a_per],
                b[w * b_per : (w + 1) * b_per],
                c[w * cd_per : (w + 1) * cd_per],
                idx_words=idx[w * 32 : (w + 1) * 32] if idx is not None else None,
                index_key=ch.index_key,
                neg_lo=ch.neg_lo,
                neg_hi=ch.neg_hi,
                clamp=ch.clamp,
            )
            compare_d(spec, d[w * cd_per : (w + 1) * cd_per], pred, tally, f"{ch.tag} wave {w}")
    if tally.elements != 256 * sum(c.waves for c in mf.chunks) or tally.elements != entry["elements"]:
        raise EvidenceError(f"{path.name}: compared {tally.elements} elements, manifest lists {entry['elements']}")
    return tally


# --- state file --------------------------------------------------------------------------------


def expected_control(write: int) -> dict[str, int]:
    """MODE readbacks and VALU results implied by the RDNA4 MODE fields for a value written to [15:0].

    Only MODE[7:0] is writable (FP_ROUND [3:0]: f32 [1:0], f16/f64 [3:2]; FP_DENORM [7:4]: f32 [5:4],
    f16/f64 [7:6]; denorm bit 0 keeps input denormals, bit 1 output denormals; round 1 = toward +inf).
    """
    m = write & 0xFF
    return {
        "init": 0x00F0,
        "read": m,
        "read_after_s_round_mode_15": (m & 0xF0) | 0x0F,
        "read_after_s_denorm_mode_15": 0xFF,
        "f32_add_rne_tie": 0x3F80_0001 if (m & 3) == 1 else 0x3F80_0000,
        "f32_add_denorm_in": 0x0080_0000 if (m >> 4) & 1 else 0,
        "f32_sub_denorm_out": 0x0040_0000 if (m >> 5) & 1 else 0,
        "f16_add_rne_tie": 0x3C01 if ((m >> 2) & 3) == 1 else 0x3C00,
        "f16_add_denorm_in": 0x0400 if (m >> 6) & 1 else 0,
        "f16_sub_denorm_out": 0x0200 if (m >> 7) & 1 else 0,
    }


def _check_keys(rec: dict, keys: set[str], what: str) -> None:
    if not isinstance(rec, dict) or set(rec) != keys:
        raise EvidenceError(f"{what}: fields {sorted(rec) if isinstance(rec, dict) else rec!r} != {sorted(keys)}")


def _words(v, n: int, what: str) -> list[int]:
    if not isinstance(v, list) or len(v) != n or not all(is_u(x, 32) for x in v):
        raise EvidenceError(f"{what}: needs exactly {n} u32 words")
    return v


_WMMA_KEYS = {
    "kind", "probe", "spec", "encoding", "index_key", "neg_lo", "neg_hi", "clamp", "op_sel", "op_sel_hi",
    "store_dwords", "prefill", "waves", "a", "b", "c", "idx", "runs",
}  # fmt: skip


def check_state_schema(doc: dict) -> tuple[list[dict], list[dict]]:
    """Check every record and the exact release inventory; return (control rows, wmma records)."""
    controls, wmmas = [], []
    for n, rec in enumerate(doc["records"]):
        kind = rec.get("kind") if isinstance(rec, dict) else None
        if kind == "mode_control":
            _check_keys(rec, {"kind", "write", "operands", "results"}, f"record {n}")
            if rec["operands"] != {k: list(v) for k, v in CONTROL_OPERANDS.items()}:
                raise EvidenceError(f"record {n}: control operands differ from EVIDENCE.md")
            _check_keys(rec["results"], set(CONTROL_RESULTS), f"record {n} results")
            if not is_u(rec["write"], 16) or not all(is_u(v, 32) for v in rec["results"].values()):
                raise EvidenceError(f"record {n}: control values out of range")
            controls.append(rec)
        elif kind == "wmma":
            _check_keys(rec, _WMMA_KEYS, f"record {n}")
            spec = SPECS.get(rec["spec"])
            if spec is None or rec["probe"] not in state_probes(spec):
                raise EvidenceError(f"record {n}: unknown probe {rec['probe']!r} for {rec['spec']!r}")
            where = f"record {n} ({rec['probe']} {spec.name})"
            want = probe_fields(spec, rec["probe"])
            if {k: rec[k] for k in want} != want:
                raise EvidenceError(f"{where}: fields differ from the probe definition")
            enc = rec["encoding"]
            if not isinstance(enc, list) or len(enc) != 2 or not all(is_u(x, 32) for x in enc):
                raise EvidenceError(f"{where}: encoding must be two u32 words")
            f = decode_vop3p(*enc)
            op_sel = (f["op_sel"] & ~1) if spec.sparse else f["op_sel"]
            key = f["op_sel"] & 1 if spec.sparse else 0
            if (f["encoding"], f["op"]) != (0xCC, spec.opcode) or [f["neg_lo"], f["neg_hi"], f["clamp"], op_sel, f["op_sel_hi"], key] != [
                rec["neg_lo"], rec["neg_hi"], rec["clamp"], rec["op_sel"], rec["op_sel_hi"], rec["index_key"]
            ]:  # fmt: skip
                raise EvidenceError(f"{where}: encoding does not decode to the recorded fields")
            if rec["store_dwords"] != STATE_STORE_DWORDS or rec["prefill"] != STATE_PREFILL:
                raise EvidenceError(f"{where}: store_dwords / prefill differ from EVIDENCE.md")
            w = rec["waves"]
            if not is_u(w, 8) or w < 1:
                raise EvidenceError(f"{where}: bad wave count")
            _words(rec["a"], w * 32 * spec.a_dwords, f"{where} a")
            _words(rec["b"], w * 32 * spec.b_dwords, f"{where} b")
            _words(rec["c"], w * 32 * spec.cd_dwords, f"{where} c")
            if spec.sparse:
                _words(rec["idx"], w * 32, f"{where} idx")
            elif rec["idx"] is not None:
                raise EvidenceError(f"{where}: dense forms have idx null")
            runs = rec["runs"]
            if not isinstance(runs, list) or [(r.get("mode"), r.get("exec_lo")) for r in runs] != probe_runs(rec["probe"]):
                raise EvidenceError(f"{where}: runs differ from the probe definition")
            for r in runs:
                _check_keys(r, {"mode", "exec_lo", "d"}, where)
                _words(r["d"], w * 32 * STATE_STORE_DWORDS, f"{where} d")
            wmmas.append(rec)
        else:
            raise EvidenceError(f"record {n}: unknown kind {kind!r}")
    if [r["write"] for r in controls] != list(CONTROL_WRITES):
        raise EvidenceError("state: MODE control rows differ from the release inventory")
    have = [(r["spec"], r["probe"]) for r in wmmas]
    want = [(s.name, p) for s in SPECS.values() for p in state_probes(s)]
    if have != want:
        raise EvidenceError(f"state: WMMA records differ from the release inventory ({len(have)} vs {len(want)})")
    return controls, wmmas


def state_summary(path: Path) -> dict:
    """Record and check counts of a state file (schema-checked; no model involved)."""
    controls, wmmas = check_state_schema(read_state_file(path))
    checks = len(controls) * len(CONTROL_RESULTS)
    for rec in wmmas:
        spec = SPECS[rec["spec"]]
        per_wave = 256 + 32 * (STATE_STORE_DWORDS - spec.cd_dwords)
        checks += per_wave * rec["waves"] * len(rec["runs"])
    return {"records": len(controls) + len(wmmas), "checks": checks}


def validate_state_file(path: Path, model: WaveModel = wmma_wave) -> Tally:
    """Reproduce every state record: control rows from MODE semantics, WMMA runs with the model."""
    controls, wmmas = check_state_schema(read_state_file(path))
    tally = Tally()
    for rec in controls:
        exp = expected_control(rec["write"])
        for k, v in exp.items():
            tally.control += 1
            if rec["results"][k] != v:
                tally.control_mm += 1
                tally.note(f"control write={rec['write']:#06x} {k}: hw={rec['results'][k]:#x} expected={v:#x}")
    for rec in wmmas:
        spec = SPECS[rec["spec"]]
        a_per, b_per, cd_per = 32 * spec.a_dwords, 32 * spec.b_dwords, 32 * spec.cd_dwords
        for r in rec["runs"]:
            for w in range(rec["waves"]):
                pred = model(
                    spec.name,
                    rec["a"][w * a_per : (w + 1) * a_per],
                    rec["b"][w * b_per : (w + 1) * b_per],
                    rec["c"][w * cd_per : (w + 1) * cd_per],
                    idx_words=rec["idx"][w * 32 : (w + 1) * 32] if spec.sparse else None,
                    index_key=rec["index_key"],
                    neg_lo=rec["neg_lo"],
                    neg_hi=rec["neg_hi"],
                    clamp=rec["clamp"],
                    op_sel=rec["op_sel"],
                    op_sel_hi=rec["op_sel_hi"],
                    mode=r["mode"],
                    exec_lo=r["exec_lo"],
                )
                stored = r["d"][w * 32 * STATE_STORE_DWORDS : (w + 1) * 32 * STATE_STORE_DWORDS]
                d_hw, kept = [], []
                for lane in range(32):
                    row = stored[lane * STATE_STORE_DWORDS : (lane + 1) * STATE_STORE_DWORDS]
                    d_hw += row[: spec.cd_dwords]
                    kept += row[spec.cd_dwords :]
                where = f"state {rec['probe']} {spec.name} mode={r['mode']:#06x} exec={r['exec_lo']:#010x} wave {w}"
                compare_d(spec, d_hw, pred, tally, where)
                tally.preserved += len(kept)
                bad = sum(x != STATE_PREFILL for x in kept)
                tally.preserved_mm += bad
                if bad:
                    tally.note(f"{where}: {bad} words past D were overwritten")
    return tally


# --- manifest and driver -----------------------------------------------------------------------


def default_evidence_dir() -> Path:
    here = Path(__file__).resolve().parent
    for cand in (here.parent / "evidence", Path.cwd() / "evidence"):
        if cand.is_dir():
            return cand
    return here.parent / "evidence"


_SESSION_KEYS = {"id", "date", "hardware", "code_object_sha256", "tool_revision"}
_MATRIX_ENTRY = {"file", "kind", "spec", "bytes", "sha256", "waves", "elements", "chunks"}
_STATE_ENTRY = {"file", "kind", "bytes", "sha256", "session", "records", "checks"}
_TOTALS = {"matrix_files", "matrix_waves", "matrix_elements", "state_records", "state_checks", "bytes"}
_HEX64 = re.compile(r"[0-9a-f]{64}")


def check_manifest_schema(m) -> None:
    """Every manifest field, type and kind (EVIDENCE.md); counts are checked by check_inventory."""
    if not isinstance(m, dict) or set(m) != {"format", "target", "sessions", "files", "totals"}:
        raise EvidenceError("manifest.json: top-level fields differ from EVIDENCE.md")
    if m["format"] != MANIFEST_FORMAT or m["target"] != TARGET:
        raise EvidenceError("manifest.json: unsupported format or target")
    if not isinstance(m["sessions"], list) or not m["sessions"]:
        raise EvidenceError("manifest.json: sessions must be a non-empty list")
    for s in m["sessions"]:
        if not isinstance(s, dict) or not _SESSION_KEYS <= set(s) <= _SESSION_KEYS | {"note"} or not isinstance(s["id"], str):
            raise EvidenceError("manifest.json: malformed session")
    if len({s["id"] for s in m["sessions"]}) != len(m["sessions"]):
        raise EvidenceError("manifest.json: duplicate session id")
    if not isinstance(m["files"], list):
        raise EvidenceError("manifest.json: files must be a list")
    for f in m["files"]:
        kind = f.get("kind") if isinstance(f, dict) else None
        keys = {"matrix": _MATRIX_ENTRY, "state": _STATE_ENTRY}.get(kind)
        if keys is None or set(f) != keys:
            raise EvidenceError(f"manifest.json: entry {f.get('file') if isinstance(f, dict) else f!r} has kind or fields outside the schema")
        if not isinstance(f["sha256"], str) or not _HEX64.fullmatch(f["sha256"]):
            raise EvidenceError(f"{f['file']}: manifest sha256 is not 64 lowercase hex digits")
        counts = ("bytes", "waves", "elements") if kind == "matrix" else ("bytes", "records", "checks")
        if not all(is_u(f[k], 63) for k in counts):
            raise EvidenceError(f"{f['file']}: manifest counts must be non-negative integers")
        if kind == "matrix":
            if not isinstance(f["chunks"], list) or not f["chunks"]:
                raise EvidenceError(f"{f['file']}: manifest chunks must be a non-empty list")
            for c in f["chunks"]:
                if not isinstance(c, dict) or set(c) != {"tag", "waves", "inputs_sha256", "session"} or not is_u(c["waves"], 32):
                    raise EvidenceError(f"{f['file']}: malformed manifest chunk")
                if not isinstance(c["inputs_sha256"], str) or not _HEX64.fullmatch(c["inputs_sha256"]):
                    raise EvidenceError(f"{f['file']}: manifest inputs_sha256 is not 64 lowercase hex digits")
    t = m["totals"]
    if not isinstance(t, dict) or set(t) != _TOTALS or not all(is_u(v, 63) for v in t.values()):
        raise EvidenceError("manifest.json: totals fields differ from EVIDENCE.md")


def check_inventory(ev_dir: Path, manifest_sha256: str) -> dict:
    """Pinned manifest, full manifest schema, exact file inventory, sizes, hashes and every count."""
    mpath = ev_dir / "manifest.json"
    if not mpath.is_file():
        raise EvidenceError(f"{mpath} not found")
    if sha256_file(mpath) != manifest_sha256:
        raise EvidenceError(f"manifest.json SHA-256 {sha256_file(mpath)} is not the trusted {manifest_sha256}")
    m = read_manifest(mpath)
    check_manifest_schema(m)
    names = [f["file"] for f in m["files"]]
    if names != expected_files() or [f["kind"] for f in m["files"]] != ["matrix"] * len(SPECS) + ["state"]:
        raise EvidenceError("manifest.json does not list exactly the release inventory")
    present = sorted(p.name for p in ev_dir.iterdir() if p.suffix in (".mmbin", ".mmstate"))
    if present != sorted(names):
        raise EvidenceError(f"evidence directory holds {present} instead of the manifest's files")
    sessions = {s["id"] for s in m["sessions"]}
    for f in m["files"]:
        p = ev_dir / f["file"]
        if p.stat().st_size != f["bytes"] or sha256_file(p) != f["sha256"]:
            raise EvidenceError(f"{f['file']}: size or SHA-256 differs from the manifest")
        if f["kind"] == "matrix":
            if f["file"] != f"{f['spec']}.mmbin" or any(c["session"] not in sessions for c in f["chunks"]):
                raise EvidenceError(f"{f['file']}: manifest entry names another instruction or session")
            if f["elements"] != 256 * f["waves"] or f["waves"] != sum(c["waves"] for c in f["chunks"]):
                raise EvidenceError(f"{f['file']}: manifest counts are inconsistent")
        else:
            if f["session"] not in sessions:
                raise EvidenceError(f"{f['file']}: manifest entry names an unknown session")
            if {k: f[k] for k in ("records", "checks")} != state_summary(p):
                raise EvidenceError(f"{f['file']}: manifest records/checks differ from the state file")
    mats, st = m["files"][:-1], m["files"][-1]
    want = {
        "matrix_files": len(mats),
        "matrix_waves": sum(f["waves"] for f in mats),
        "matrix_elements": sum(f["elements"] for f in mats),
        "state_records": st["records"],
        "state_checks": st["checks"],
        "bytes": sum(f["bytes"] for f in m["files"]),
    }
    if m["totals"] != want:
        raise EvidenceError("manifest.json: totals do not add up")
    return m


def _run_file(args: tuple[str, dict]) -> tuple[str, Tally, str | None, float]:
    t0 = time.monotonic()
    path, entry = args
    try:
        return entry["spec"], validate_matrix_file(Path(path), entry), None, time.monotonic() - t0
    except EvidenceError as e:
        return entry["spec"], Tally(), str(e), time.monotonic() - t0


def validate(
    evidence_dir: Path | None = None,
    variants: list[str] | None = None,
    workers: int | None = None,
    manifest_sha256: str = RELEASE_MANIFEST_SHA256,
    model: WaveModel = wmma_wave,
    out=sys.stdout,
) -> int:
    ev_dir = evidence_dir or default_evidence_dir()
    t0 = time.monotonic()
    try:
        manifest = check_inventory(ev_dir, manifest_sha256)
    except (EvidenceError, OSError) as e:
        print(f"[ERROR] {e}", file=out)
        return 2
    print(f"[PASS] manifest {manifest_sha256[:16]}...: {len(manifest['files'])} files present, sizes and SHA-256 match", file=out)
    entries = {f["spec"]: f for f in manifest["files"] if f["kind"] == "matrix"}
    selected = list(SPECS) if not variants else variants
    unknown = [v for v in selected if v not in SPECS and v != "state"]
    if unknown:
        print(f"[ERROR] unknown instruction(s): {', '.join(unknown)}", file=out)
        return 2
    full = not variants
    jobs = [(str(ev_dir / entries[s]["file"]), entries[s]) for s in selected if s in SPECS]
    n_workers = workers if workers is not None else min(os.cpu_count() or 1, max(1, len(jobs)), 16)
    if model is not wmma_wave or n_workers <= 1 or len(jobs) <= 1:
        results = []
        for path, entry in jobs:
            t1 = time.monotonic()
            try:
                results.append((entry["spec"], validate_matrix_file(Path(path), entry, model), None, time.monotonic() - t1))
            except EvidenceError as e:
                results.append((entry["spec"], Tally(), str(e), time.monotonic() - t1))
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            results = list(ex.map(_run_file, jobs))

    total, errors = Tally(), 0
    for name, tl, err, dt in results:
        if err:
            errors += 1
            print(f"[ERROR] {name:34s}: {err}", file=out)
            continue
        total.add(tl)
        status = "PASS" if tl.mismatches == 0 else "FAIL"
        print(f"[{status}] {name:34s}: {tl.elements - tl.element_mm:7d}/{tl.elements:7d} elements bit-exact [{dt:.2f}s]", file=out)
        for d in tl.details:
            print(f"   {d}", file=out)

    state_done = full or "state" in selected
    if state_done:
        t1 = time.monotonic()
        try:
            st = validate_state_file(ev_dir / STATE_FILE, model)
        except EvidenceError as e:
            errors += 1
            print(f"[ERROR] {STATE_FILE}: {e}", file=out)
        else:
            status = "PASS" if st.mismatches == 0 else "FAIL"
            print(
                f"[{status}] {'state_gfx1201':34s}: {st.elements - st.element_mm:,}/{st.elements:,} D elements, "
                f"{st.preserved - st.preserved_mm:,}/{st.preserved:,} preserved words, "
                f"{st.control - st.control_mm}/{st.control} MODE control values [{time.monotonic() - t1:.2f}s]",
                file=out,
            )
            for d in st.details:
                print(f"   {d}", file=out)
            total.add(st)

    n_inst = sum(1 for _, _, err, _ in results if err is None)
    scope = "FULL" if full else f"PARTIAL ({len(jobs)} of {len(SPECS)} instructions{', state' if state_done else ''})"
    print(
        f"\n{scope}: {n_inst}/{len(jobs)} instruction files and {'the' if state_done else 'no'} state file checked; "
        f"{total.elements - total.element_mm:,}/{total.elements:,} D elements bit-exact, "
        f"{total.preserved - total.preserved_mm:,}/{total.preserved:,} preserved words, "
        f"{total.control - total.control_mm}/{total.control} control values; "
        f"{total.mismatches} mismatches, {errors} errors in {time.monotonic() - t0:.2f}s.",
        file=out,
    )
    if errors:
        return 2
    return 0 if total.mismatches == 0 else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Validate the gfx1201 WMMA/SWMMAC model against frozen hardware evidence.")
    ap.add_argument("--evidence", type=Path, default=None, help="Evidence directory (default: ./evidence of the checkout)")
    ap.add_argument("--variant", action="append", default=None, help="Validate only this instruction ('state' for the state file); repeatable")
    ap.add_argument("--workers", type=int, default=None, help="Parallel worker processes (default: auto)")
    ap.add_argument("--manifest-sha256", default=RELEASE_MANIFEST_SHA256, help="Trusted manifest SHA-256 (default: this release)")
    args = ap.parse_args(argv)
    return validate(args.evidence, args.variant, args.workers, args.manifest_sha256)


if __name__ == "__main__":
    sys.exit(main())
