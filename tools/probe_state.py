"""Hardware state probes (MODE, raw CLAMP / OPSEL bits, destination extent, EXEC) -> AMDST002 state file.

Every WMMA / SWMMAC probe runs one exact 64-bit VOP3P encoding (assembled by llvm-mc, then with raw
fields set) and records that encoding, its inputs, the per-run state (MODE value written before the
instruction, EXEC mask at issue) and all stored destination VGPRs. Destination VGPRs that are not
instruction inputs are prefilled with 0xDEADBEEF, and 8 VGPRs per lane are stored for every form, so
words past the D operand show whether the instruction wrote them.

    probe_state.py --build-co PATH      assemble the probe code object (needs llvm-mc and ld.lld)
    probe_state.py --co PATH --capture  run on the gfx1201 GPU and write evidence/state_gfx1201.mmstate
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from amdgpu_matrix_model.evidence import (  # noqa: E402
    CONTROL_OPERANDS,
    CONTROL_RESULTS,
    CONTROL_WRITES,
    DEFAULT_MODE,
    EXEC_MASKS,
    STATE_FILE,
    STATE_PREFILL,
    STATE_STORE_DWORDS,
    WMMA_MODES,
    decode_vop3p,
    generate_chunk_inputs,
    probe_fields,
    probe_runs,
    state_probes,
    write_state_file,
)
from amdgpu_matrix_model.layout import pack_a, pack_b, pack_cd, pack_idx  # noqa: E402
from amdgpu_matrix_model.specs import SPECS, WmmaSpec  # noqa: E402
from tools.abi import wrap_kernels  # noqa: E402
from tools.cases import bf16_bits, f16_bits, f32_bits, pow2  # noqa: E402
from tools.probe_asm import _emit_loads, _emit_ptr_add, _emit_stores, _vreg  # noqa: E402

PREFILL = STATE_PREFILL
STORE = STATE_STORE_DWORDS
_VOFF = {1: "v1", 2: "v2", 4: "v3", 8: "v4", 16: "v5"}

CTRL_MODES, WMMA_MODES, EXEC_MASKS = CONTROL_WRITES, WMMA_MODES, EXEC_MASKS
CTRL_OPERANDS, CTRL_RESULTS = CONTROL_OPERANDS, CONTROL_RESULTS


def _llvm_mc_encoding(line: str) -> tuple[int, int]:
    r = subprocess.run(
        ["llvm-mc", "-triple=amdgcn-amd-amdhsa", "-mcpu=gfx1201", "-show-encoding"],
        input=line + "\n",
        capture_output=True,
        text=True,
        check=True,
    )
    for out_line in r.stdout.splitlines():
        if "encoding:" in out_line:
            b = bytes(int(x, 16) for x in out_line.split("encoding:")[1].split("[")[1].split("]")[0].split(","))
            assert len(b) == 8, line
            return struct.unpack("<II", b)
    raise RuntimeError(f"no encoding for {line!r}:\n{r.stderr}")


def base_encoding(spec: WmmaSpec) -> tuple[int, int]:
    """Default VOP3P encoding with the probe kernel's registers (A v8, B v12, C v20, D v28 or v20, IDX v36).

    OPSEL_HI is 7 as llvm-mc encodes it; build_state_co checks every encoding against llvm-mc.
    """
    vdst, src2 = (20, 256 + 36) if spec.sparse else (28, 256 + 20)
    dw0 = (0xCC << 24) | (spec.opcode << 16) | (1 << 14) | vdst
    dw1 = (0x3 << 27) | (src2 << 18) | ((256 + 12) << 9) | (256 + 8)
    return dw0, dw1


def llvm_mc_base_encoding(spec: WmmaSpec) -> tuple[int, int]:
    a, b = _vreg(8, spec.a_dwords), _vreg(12, spec.b_dwords)
    if spec.sparse:
        return _llvm_mc_encoding(f"{spec.name} {_vreg(20, spec.cd_dwords)}, {a}, {b}, v36")
    return _llvm_mc_encoding(f"{spec.name} {_vreg(28, spec.cd_dwords)}, {a}, {b}, {_vreg(20, spec.cd_dwords)}")


def set_fields(dw0: int, dw1: int, *, clamp: bool | None = None, op_sel: int | None = None, op_sel_hi: int | None = None) -> tuple[int, int]:
    """Set raw VOP3P fields: CLAMP dw0[15], OPSEL dw0[13:11], OPSEL_HI {dw0[14], dw1[28:27]}."""
    if clamp is not None:
        dw0 = (dw0 & ~(1 << 15)) | (int(clamp) << 15)
    if op_sel is not None:
        dw0 = (dw0 & ~(0x7 << 11)) | (op_sel << 11)
    if op_sel_hi is not None:
        dw0 = (dw0 & ~(1 << 14)) | (((op_sel_hi >> 2) & 1) << 14)
        dw1 = (dw1 & ~(0x3 << 27)) | ((op_sel_hi & 0x3) << 27)
    return dw0, dw1


def _prologue(spec: WmmaSpec, store: int) -> list[str]:
    lines = [
        "\ts_load_b256 s[4:11], s[0:1], 0x0",
        "\ts_load_b64 s[12:13], s[0:1], 0x20",
        "\ts_load_b64 s[20:21], s[0:1], 0x28",
        "\ts_mov_b32 s14, ttmp9",
        "\ts_mov_b32 s15, 0",
        "\tv_lshlrev_b32 v1, 2, v0",
        "\tv_lshlrev_b32 v2, 3, v0",
        "\tv_lshlrev_b32 v3, 4, v0",
        "\tv_lshlrev_b32 v4, 5, v0",
        "\tv_lshlrev_b32 v5, 6, v0",
        "\ts_wait_kmcnt 0x0",
        *_emit_ptr_add(4, spec.a_dwords),
        *_emit_ptr_add(6, spec.b_dwords),
        *_emit_ptr_add(8, spec.cd_dwords),
        *_emit_ptr_add(10, store),
        *_emit_ptr_add(12, 1),
        "\ts_lshl_b64 s[16:17], s[14:15], 2",
        "\ts_add_co_u32 s20, s20, s16",
        "\ts_add_co_ci_u32 s21, s21, s17",
        "\ts_load_b32 s18, s[20:21], 0x0",
    ]
    return lines


def wmma_kernel(spec: WmmaSpec, dw0: int, dw1: int, *, set_mode: bool, set_exec: bool) -> str:
    lines = _prologue(spec, STORE)
    lines += _emit_loads(8, spec.a_dwords, _VOFF[spec.a_dwords], "s[4:5]")
    lines += _emit_loads(12, spec.b_dwords, _VOFF[spec.b_dwords], "s[6:7]")
    lines += _emit_loads(20, spec.cd_dwords, _VOFF[spec.cd_dwords], "s[8:9]")
    if spec.sparse:
        lines += _emit_loads(36, 1, "v1", "s[12:13]")
    d_base = 20 if spec.sparse else 28
    first_free = d_base + (spec.cd_dwords if spec.sparse else 0)
    lines += [f"\tv_mov_b32 v{r}, {PREFILL:#x}" for r in range(first_free, d_base + STORE)]
    lines += ["\ts_wait_kmcnt 0x0", "\ts_wait_loadcnt 0x0"]
    if set_mode:
        lines.append("\ts_setreg_b32 hwreg(HW_REG_WAVE_MODE, 0, 16), s18")
    if set_exec:
        lines.append("\ts_mov_b32 exec_lo, s18")
    lines += [f"\t.long {dw0:#010x}, {dw1:#010x}", "\tv_nop"]
    if set_exec:
        lines.append("\ts_mov_b32 exec_lo, -1")
    lines += _emit_stores(d_base, STORE, _VOFF[STORE], "s[10:11]")
    lines += ["\ts_wait_storecnt 0x0", "\ts_endpgm"]
    return "\n".join(lines) + "\n"


def control_kernel() -> str:
    """MODE readbacks and same-wave VALU positive controls (16 result dwords per lane)."""
    spec = SPECS["v_swmmac_f32_16x16x32_f16"]  # buffer shapes only: A 4, B 8, C 8 dwords per lane
    lines = _prologue(spec, 16)
    lines += _emit_loads(20, 8, "v4", "s[8:9]")
    lines += _emit_loads(8, 4, "v3", "s[4:5]")
    lines += [
        "\ts_wait_kmcnt 0x0",
        "\ts_wait_loadcnt 0x0",
        "\ts_getreg_b32 s19, hwreg(HW_REG_WAVE_MODE)",
        "\tv_mov_b32 v40, s19",
        "\ts_setreg_b32 hwreg(HW_REG_WAVE_MODE, 0, 16), s18",
        "\ts_getreg_b32 s19, hwreg(HW_REG_WAVE_MODE)",
        "\tv_mov_b32 v41, s19",
        "\tv_add_f32 v44, v20, v21",
        "\tv_add_f32 v45, v22, v23",
        "\tv_sub_f32 v46, v24, v25",
        "\tv_lshrrev_b32 v30, 16, v26",
        "\tv_add_f16 v47, v26, v30",
        "\tv_lshrrev_b32 v31, 16, v27",
        "\tv_add_f16 v48, v27, v31",
        "\tv_lshrrev_b32 v32, 16, v8",
        "\tv_sub_f16 v49, v8, v32",
        "\tv_and_b32 v47, 0xffff, v47",
        "\tv_and_b32 v48, 0xffff, v48",
        "\tv_and_b32 v49, 0xffff, v49",
        "\ts_round_mode 0xf",
        "\ts_getreg_b32 s19, hwreg(HW_REG_WAVE_MODE)",
        "\tv_mov_b32 v42, s19",
        "\ts_denorm_mode 0xf",
        "\ts_getreg_b32 s19, hwreg(HW_REG_WAVE_MODE)",
        "\tv_mov_b32 v43, s19",
        *[f"\tv_mov_b32 v{r}, 0" for r in range(50, 56)],
        *_emit_stores(40, 16, "v5", "s[10:11]"),
        "\ts_wait_storecnt 0x0",
        "\ts_endpgm",
    ]
    return "\n".join(lines) + "\n"


def probe_list() -> list[dict]:
    """Every WMMA probe of the release inventory (evidence.state_probes) with its exact encoding."""
    out = []
    for spec in SPECS.values():
        base = base_encoding(spec)
        for probe in state_probes(spec):
            f = probe_fields(spec, probe)
            enc = set_fields(*base, clamp=f["clamp"], op_sel=f["op_sel"], op_sel_hi=f["op_sel_hi"])
            out.append(dict(probe=probe, spec=spec, enc=enc, set_mode=probe == "mode", set_exec=probe == "exec", kernel=f"{probe}_{spec.name}"))
    return out


def build_state_co(out_path: Path) -> Path:
    for spec in SPECS.values():
        assert base_encoding(spec) == llvm_mc_base_encoding(spec), spec.name
    kernels = [("probe_mode_control", control_kernel())]
    kernels += [(p["kernel"], wmma_kernel(p["spec"], *p["enc"], set_mode=p["set_mode"], set_exec=p["set_exec"])) for p in probe_list()]
    asm = wrap_kernels(kernels, kernarg=48, vgprs=64, sgprs=24)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        s_path, o_path = Path(tmp) / "state.s", Path(tmp) / "state.o"
        s_path.write_text(asm)
        for cmd in (
            ["llvm-mc", "-triple=amdgcn-amd-amdhsa", "-mcpu=gfx1201", "-filetype=obj", str(s_path), "-o", str(o_path)],
            ["ld.lld", "-shared", str(o_path), "-o", str(out_path)],
        ):
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode:
                raise RuntimeError(f"{cmd[0]} failed:\n{r.stderr}")
    return out_path


# --- inputs ------------------------------------------------------------------------------------


def _one(fmt: str) -> int:
    return pow2(fmt, 0)


def _c_val(fmt: str, v: float) -> int:
    if v != v:
        return {"f32": 0x7F80_0123, "f16": 0x7C05, "bf16": 0x7F85}[fmt]
    return {"f32": f32_bits, "f16": f16_bits, "bf16": bf16_bits}[fmt](v)


def _wave(spec: WmmaSpec, am, bm, cm, im) -> tuple[list[int], list[int], list[int], list[int] | None]:
    return pack_a(spec, am), pack_b(spec, bm), pack_cd(spec, cm), (pack_idx(spec, im) if spec.sparse else None)


def _empty(spec: WmmaSpec):
    im = [[k % 2 for k in range(spec.k_a)] for _ in range(16)] if spec.sparse else None
    return [[0] * spec.k_a for _ in range(16)], [[0] * 16 for _ in range(spec.k_b)], [[0] * 16 for _ in range(16)], im


def _bk(spec: WmmaSpec, ka: int) -> int:
    """Dense B row used by compressed ka under the default index pairs (0, 1)."""
    return 4 * (ka // 2) + ka % 2 if spec.sparse else ka


def mode_wave(spec: WmmaSpec):
    """Rounding ties, tiny inexact terms, subnormal inputs and outputs, NaN and Inf in one wave."""
    am, bm, cm, im = _empty(spec)
    af, bf, df = spec.a_fmt, spec.b_fmt, spec.d_fmt
    one, neg1 = _one(af), pow2(af, 0, True)
    tiny = pow2(af, -12) if af in ("f16", "bf16") else pow2(af, -9)
    sub_a = 1
    snan = {"f16": 0x7C01, "bf16": 0x7F81, "fp8": 0x7F, "bf8": 0x7D}[af]
    inf = {"f16": 0x7C00, "bf16": 0x7F80, "fp8": 0x7E, "bf8": 0x7C}[af]  # fp8 has no Inf: 0x7E is 448
    for r in range(16):
        am[r][0] = one if r % 2 == 0 else neg1
        am[r][1] = tiny if r < 8 else tiny ^ (1 << (spec.a_bits - 1))
        am[r][2] = sub_a
        if r == 12:
            am[r][0] = snan
        elif r == 13:
            am[r][0] = inf
        elif r == 14:
            am[r][:3] = [sub_a, 0, 0]
        elif r == 15:
            am[r][:3] = [0, 0, 0]
    for c in range(16):
        bm[_bk(spec, 0)][c] = _one(bf) if c < 12 else (1 if c < 14 else 0)
        bm[_bk(spec, 1)][c] = pow2(bf, -12) if bf in ("f16", "bf16") else pow2(bf, -9)
        bm[_bk(spec, 2)][c] = 1
    c_one = _c_val(df, 1.0)
    sign = 1 << (spec.d_bits - 1)
    vals = [c_one, c_one ^ sign, 1, 1 ^ sign, 0, _c_val(df, float("nan")), c_one ^ 1, (c_one ^ sign) ^ 1]
    for r in range(16):
        for c in range(16):
            cm[r][c] = vals[(r + c) % 8]
    return _wave(spec, am, bm, cm, im)


def clamp_wave(spec: WmmaSpec):
    """C and A*B on both sides of [0, 1], signed zeros, infinities and NaN."""
    am, bm, cm, im = _empty(spec)
    c_vals = [-5.0, -0.5, -0.0, 0.0, 0.25, 0.5, 1.0, 2.0, 100.0, float("inf"), float("-inf"), float("nan")]
    a_exps = [(-1, False), (-1, True), (1, False), (1, True), (-3, False), (2, True)]
    for r in range(16):
        e, neg = a_exps[r % len(a_exps)]
        am[r][0] = pow2(spec.a_fmt, e, neg) if r < 12 else 0
    for c in range(16):
        bm[_bk(spec, 0)][c] = _one(spec.b_fmt)
    for r in range(16):
        for c in range(16):
            cm[r][c] = _c_val(spec.d_fmt, c_vals[(r * 16 + c) % len(c_vals)])
    return _wave(spec, am, bm, cm, im)


def random_wave(spec: WmmaSpec, seed: int):
    a, b, c, idx = generate_chunk_inputs(spec, 1, seed, "raw" if spec.is_int else "finite")
    if spec.sparse:
        idx = generate_chunk_inputs(spec, 2, seed, "raw")[3][:32]  # valid 2:4 index words
    return a, b, c, idx


def _concat(waves):
    a, b, c = ([x for w in waves for x in w[i]] for i in range(3))
    idx = [x for w in waves for x in w[3]] if waves[0][3] is not None else None
    return a, b, c, idx


def probe_inputs(p: dict, seed: int):
    spec = p["spec"]
    if p["probe"] == "mode":
        waves = [random_wave(spec, seed), random_wave(spec, seed + 1)] if spec.is_int else [mode_wave(spec), random_wave(spec, seed)]
    elif p["probe"] == "exec":
        waves = [random_wave(spec, seed)]
    else:
        waves = [clamp_wave(spec), mode_wave(spec), random_wave(spec, seed)]
    return _concat(waves), len(waves)


# --- capture -----------------------------------------------------------------------------------


def _decoded_fields(spec: WmmaSpec, dw0: int, dw1: int) -> dict:
    f = decode_vop3p(dw0, dw1)
    return {
        "index_key": f["op_sel"] & 1 if spec.sparse else 0,
        "neg_lo": f["neg_lo"],
        "neg_hi": f["neg_hi"],
        "clamp": f["clamp"],
        "op_sel": 0 if spec.sparse else f["op_sel"],
        "op_sel_hi": f["op_sel_hi"],
    }


def capture(co: Path) -> None:
    from tools.hw_run import capture_session, run_jobs

    session = capture_session(co, "state")

    probes = probe_list()
    jobs, meta = [], []
    n_ctrl = len(CTRL_MODES)
    c_ctrl = np.zeros((n_ctrl, 32, 8), dtype=np.uint32)
    a_ctrl = np.zeros((n_ctrl, 32, 4), dtype=np.uint32)
    ops = list(CTRL_OPERANDS.values())
    for i in range(3):
        c_ctrl[:, :, 2 * i], c_ctrl[:, :, 2 * i + 1] = ops[i]
    c_ctrl[:, :, 6] = ops[3][0] | (ops[3][1] << 16)
    c_ctrl[:, :, 7] = ops[4][0] | (ops[4][1] << 16)
    a_ctrl[:, :, 0] = ops[5][0] | (ops[5][1] << 16)
    jobs.append(
        {
            "id": "ctrl",
            "spec": "v_swmmac_f32_16x16x32_f16",
            "kernel": "probe_mode_control",
            "a": a_ctrl,
            "b": np.zeros((n_ctrl, 32, 8), dtype=np.uint32),
            "c": c_ctrl,
            "idx": np.zeros((n_ctrl, 32), dtype=np.uint32),
            "param": np.array(CTRL_MODES, dtype=np.uint32),
            "d_dwords": 16,
        }
    )
    seed = 404001
    for i, p in enumerate(probes):
        spec = p["spec"]
        (a, b, c, idx), w = probe_inputs(p, seed)
        seed += 3
        runs = [{"mode": m, "exec_lo": e} for m, e in probe_runs(p["probe"])]
        param = [r["mode"] if p["set_mode"] else r["exec_lo"] for r in runs for _ in range(w)]
        n = len(runs)
        arr = lambda v, dw: np.tile(np.array(v, dtype=np.uint32).reshape(w, 32, dw), (n, 1, 1))  # noqa: E731
        job = {"id": f"p{i}", "spec": spec.name, "kernel": p["kernel"], "a": arr(a, spec.a_dwords), "b": arr(b, spec.b_dwords), "c": arr(c, spec.cd_dwords), "param": np.array(param, dtype=np.uint32), "d_dwords": STORE}
        if spec.sparse:
            job["idx"] = np.tile(np.array(idx, dtype=np.uint32).reshape(w, 32), (n, 1))
        jobs.append(job)
        meta.append((p, (a, b, c, idx), w, runs))

    res = run_jobs(jobs, co_path=co)
    records: list[dict] = []
    d_ctrl = res["ctrl"]
    for wi, m in enumerate(CTRL_MODES):
        lanes = d_ctrl[wi]
        assert (lanes == lanes[0]).all(), f"mode control results differ between lanes for {m:#x}"
        records.append(
            {
                "kind": "mode_control",
                "write": m,
                "operands": {k: list(v) for k, v in CTRL_OPERANDS.items()},
                "results": {k: int(lanes[0][i]) for i, k in enumerate(CTRL_RESULTS)},
            }
        )
    for i, (p, (a, b, c, idx), w, runs) in enumerate(meta):
        spec = p["spec"]
        d = res[f"p{i}"].reshape(len(runs), w * 32 * STORE)
        records.append(
            {
                "kind": "wmma",
                "probe": p["probe"],
                "spec": spec.name,
                "encoding": list(p["enc"]),
                **_decoded_fields(spec, *p["enc"]),
                "store_dwords": STORE,
                "prefill": PREFILL,
                "waves": w,
                "a": a,
                "b": b,
                "c": c,
                "idx": idx,
                "runs": [{**r, "d": d[ri].tolist()} for ri, r in enumerate(runs)],
            }
        )
    out = ROOT / "evidence" / STATE_FILE
    write_state_file(out, records)
    session["file_sha256"] = hashlib.sha256(out.read_bytes()).hexdigest()
    side = ROOT / "build" / "state_session.json"
    side.parent.mkdir(parents=True, exist_ok=True)
    side.write_text(json.dumps(session, indent=1) + "\n")
    print(f"Wrote {out} ({out.stat().st_size} bytes, {len(records)} records); session {session}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--build-co", type=Path, default=None, help="Assemble the state probe code object to PATH and exit")
    ap.add_argument("--co", type=Path, default=None, help="Pre-assembled state probe code object")
    ap.add_argument("--capture", action="store_true", help="Run the probes and write the state evidence file")
    args = ap.parse_args()
    if args.build_co is not None:
        build_state_co(args.build_co)
        print(f"Assembled {args.build_co}")
        return
    co = args.co if args.co is not None else build_state_co(ROOT / "build" / "state_probes_gfx1201.co")
    if args.capture:
        capture(co)
    else:
        ap.error("nothing to do: pass --capture (and --co) or --build-co")


if __name__ == "__main__":
    main()
