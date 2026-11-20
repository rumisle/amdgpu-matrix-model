"""Hardware state probes: HW_REG_WAVE_MODE, raw float clamp/op_sel bits, f16/bf16 D sentinel, and EXEC masking."""

from __future__ import annotations

import argparse
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from amdgpu_matrix_model.layout import pack_a, pack_b, pack_cd, pack_idx, unpack_cd
from amdgpu_matrix_model.specs import SPECS
from tools.abi import wrap_kernels
from tools.hw_run import run_jobs
from tools.probe_asm import _emit_loads, _emit_ptr_add, _emit_stores, _vreg


def f32_bits(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", x))[0]


def f16_bits(x: float) -> int:
    return int(np.float16(x).view(np.uint16))


def bf16_bits(x: float) -> int:
    return f32_bits(x) >> 16


def _asm_single_inst(line: str) -> tuple[int, int]:
    """Assemble one 64-bit VOP3P instruction with llvm-mc and return (dw0, dw1)."""
    r = subprocess.run(
        ["llvm-mc", "-triple=amdgcn-amd-amdhsa", "-mcpu=gfx1201", "-show-encoding"],
        input=line + "\n",
        capture_output=True,
        text=True,
        check=True,
    )
    for out_line in r.stdout.splitlines():
        if "encoding:" in out_line:
            hex_bytes = out_line.split("encoding:")[1].split("[")[1].split("]")[0].split(",")
            b = bytes(int(x.strip(), 16) for x in hex_bytes)
            assert len(b) == 8, f"expected 8 bytes for {line}, got {len(b)}"
            return struct.unpack("<II", b)
    raise RuntimeError(f"no encoding in:\n{r.stdout}\n{r.stderr}")


def build_state_co() -> Path:
    """Assemble kernels for MODE probing, raw clamp/op_sel probing, sentinel checks, and EXEC masking."""
    kernels: list[tuple[str, str]] = []

    # 1. Mode bit probe + control v_add_f32 / v_add_f16 kernel:
    #    kernarg (40 bytes): a_ptr, b_ptr, c_ptr, d_ptr, idx_ptr
    #    Reads mode_val from c_ptr[0] (wave-uniform scalar via first dword of wave's c buffer? No:
    #    pass mode_val per wave in idx_ptr[wg * 32 + 0]!! We ALREADY have 5 pointers in hw_run.py!)
    #    Wait: hw_run.py takes (a, b, c, idx) and allocates d of same size as c!
    #    Let's reuse that exact 5-pointer ABI so hw_run.py needs ZERO changes!
    #    - For mode_ctrl kernel (spec="v_wmma_f32_16x16x16_f16" shape: cd_dwords=8, idx=1 dword/lane):
    #      idx[wg, 0] = new_mode_16 (written to HW_REG_WAVE_MODE[15:0])
    #      d[wg, lane, 0] = initial HW_REG_WAVE_MODE
    #      d[wg, lane, 1] = readback HW_REG_WAVE_MODE after s_setreg_b32(0, 16)
    #      d[wg, lane, 2] = v_add_f32(c[0], c[1]) under new mode
    #      d[wg, lane, 3] = v_add_f32(c[2], c[3]) under new mode
    #      d[wg, lane, 4] = v_add_f16(c[4].lo, c[4].hi) under new mode
    #      d[wg, lane, 5] = v_add_f16(c[5].lo, c[5].hi) under new mode
    #      d[wg, lane, 6] = readback after s_round_mode 0xF (only if idx[wg,1]==1 else 0)
    #      d[wg, lane, 7] = readback after s_denorm_mode 0xF (only if idx[wg,1]==1 else 0)
    ctrl_body = "\n".join(
        [
            "\ts_load_b256 s[4:11], s[0:1], 0x0",
            "\ts_load_b64 s[12:13], s[0:1], 0x20",
            "\ts_mov_b32 s14, ttmp9",
            "\ts_mov_b32 s15, 0",
            "\tv_lshlrev_b32 v1, 2, v0",
            "\tv_lshlrev_b32 v4, 5, v0",
            "\ts_wait_kmcnt 0x0",
            *_emit_ptr_add(8, 8),
            *_emit_ptr_add(10, 8),
            *_emit_ptr_add(12, 1),
            "\ts_load_b32 s18, s[12:13], 0x0",  # mode value to write to bits [15:0]
            *_emit_loads(20, 8, "v4", "s[8:9]"),
            "\ts_wait_kmcnt 0x0",
            "\ts_wait_loadcnt 0x0",
            "\ts_getreg_b32 s19, hwreg(HW_REG_WAVE_MODE)",  # initial mode
            "\tv_mov_b32 v28, s19",
            "\ts_setreg_b32 hwreg(HW_REG_WAVE_MODE, 0, 16), s18",
            "\ts_getreg_b32 s20, hwreg(HW_REG_WAVE_MODE)",  # readback after setreg
            "\tv_mov_b32 v29, s20",
            "\tv_add_f32 v30, v20, v21",
            "\tv_add_f32 v31, v22, v23",
            "\tv_lshrrev_b32 v26, 16, v24",
            "\tv_add_f16 v32, v24, v26",
            "\tv_lshrrev_b32 v27, 16, v25",
            "\tv_add_f16 v33, v25, v27",
            "\ts_round_mode 0xf",
            "\ts_getreg_b32 s21, hwreg(HW_REG_WAVE_MODE)",
            "\tv_mov_b32 v34, s21",
            "\ts_denorm_mode 0xf",
            "\ts_getreg_b32 s22, hwreg(HW_REG_WAVE_MODE)",
            "\tv_mov_b32 v35, s22",
            *_emit_stores(28, 8, "v4", "s[10:11]"),
            "\ts_wait_storecnt 0x0",
            "\ts_endpgm",
        ]
    )
    kernels.append(("probe_mode_ctrl", ctrl_body))

    # 2. MODE-controlled WMMA kernel for every dense float WMMA variant:
    #    Reads 16-bit mode from idx_ptr[wg * 32 + 0] and sets HW_REG_WAVE_MODE[15:0] before WMMA.
    float_dense = [
        "v_wmma_f32_16x16x16_f16",
        "v_wmma_f32_16x16x16_bf16",
        "v_wmma_f16_16x16x16_f16",
        "v_wmma_bf16_16x16x16_bf16",
        "v_wmma_f32_16x16x16_fp8_fp8",
        "v_wmma_f32_16x16x16_fp8_bf8",
        "v_wmma_f32_16x16x16_bf8_fp8",
        "v_wmma_f32_16x16x16_bf8_bf8",
    ]
    for name in float_dense:
        spec = SPECS[name]
        voff_a = {1: "v1", 2: "v2", 4: "v3", 8: "v4"}[spec.a_dwords]
        voff_b = {1: "v1", 2: "v2", 4: "v3", 8: "v4"}[spec.b_dwords]
        voff_c = {1: "v1", 2: "v2", 4: "v3", 8: "v4"}[spec.cd_dwords]
        a_reg = _vreg(8, spec.a_dwords)
        b_reg = _vreg(12, spec.b_dwords)
        c_reg = _vreg(20, spec.cd_dwords)
        d_reg = _vreg(28, spec.cd_dwords)
        body = "\n".join(
            [
                "\ts_load_b256 s[4:11], s[0:1], 0x0",
                "\ts_load_b64 s[12:13], s[0:1], 0x20",
                "\ts_mov_b32 s14, ttmp9",
                "\ts_mov_b32 s15, 0",
                "\tv_lshlrev_b32 v1, 2, v0",
                "\tv_lshlrev_b32 v2, 3, v0",
                "\tv_lshlrev_b32 v3, 4, v0",
                "\tv_lshlrev_b32 v4, 5, v0",
                "\ts_wait_kmcnt 0x0",
                *_emit_ptr_add(4, spec.a_dwords),
                *_emit_ptr_add(6, spec.b_dwords),
                *_emit_ptr_add(8, spec.cd_dwords),
                *_emit_ptr_add(10, spec.cd_dwords),
                *_emit_ptr_add(12, 1),
                "\ts_load_b32 s18, s[12:13], 0x0",
                *_emit_loads(8, spec.a_dwords, voff_a, "s[4:5]"),
                *_emit_loads(12, spec.b_dwords, voff_b, "s[6:7]"),
                *_emit_loads(20, spec.cd_dwords, voff_c, "s[8:9]"),
                "\ts_wait_kmcnt 0x0",
                "\ts_wait_loadcnt 0x0",
                "\ts_setreg_b32 hwreg(HW_REG_WAVE_MODE, 0, 16), s18",
                f"\t{name} {d_reg}, {a_reg}, {b_reg}, {c_reg}",
                "\tv_nop",
                *_emit_stores(28, spec.cd_dwords, voff_c, "s[10:11]"),
                "\ts_wait_storecnt 0x0",
                "\ts_endpgm",
            ]
        )
        kernels.append((f"mode_{name}", body))

    # 3. Raw VOP3P CLAMP=1 (bit 15) and OPSEL probes on float WMMA (+ sentinel pre-fill of v[28:35])
    for name in (
        "v_wmma_f32_16x16x16_f16",
        "v_wmma_f32_16x16x16_bf16",
        "v_wmma_f16_16x16x16_f16",
        "v_wmma_bf16_16x16x16_bf16",
        "v_wmma_f32_16x16x16_fp8_fp8",
        "v_wmma_f32_16x16x16_bf8_bf8",
    ):
        spec = SPECS[name]
        voff_a = {1: "v1", 2: "v2", 4: "v3", 8: "v4"}[spec.a_dwords]
        voff_b = {1: "v1", 2: "v2", 4: "v3", 8: "v4"}[spec.b_dwords]
        voff_c = {1: "v1", 2: "v2", 4: "v3", 8: "v4"}[spec.cd_dwords]
        a_reg = _vreg(8, spec.a_dwords)
        b_reg = _vreg(12, spec.b_dwords)
        c_reg = _vreg(20, spec.cd_dwords)
        d_reg = _vreg(28, spec.cd_dwords)
        dw0, dw1 = _asm_single_inst(f"{name} {d_reg}, {a_reg}, {b_reg}, {c_reg}")

        # Raw CLAMP=1 (bit 15 of dw0)
        dw0_clamp = dw0 | (1 << 15)
        variants = [("rawclamp", dw0_clamp, dw1)]
        if spec.d_fmt in ("f16", "bf16"):
            # Test raw OPSEL bits [13:11]: e.g. opsel=0b111 (bits 11,12,13), opsel_hi_2=0 (bit 14 cleared), and sentinel check
            variants.append(("sentinel", dw0, dw1))
            variants.append(("rawopsel_7", dw0 | (0x7 << 11), dw1))
            variants.append(("rawopsel_hi0", dw0 & ~(1 << 14), dw1 & ~(0x3 << 27)))

        for tag, w0, w1 in variants:
            # Always store 8 dwords for sentinel check (uses v[28:35] prefilled with 0xDEADBEEF);
            # for others store spec.cd_dwords
            store_dw = 8 if tag == "sentinel" else spec.cd_dwords
            voff_d = "v4" if store_dw == 8 else voff_c
            sentinel_init = [f"\tv_mov_b32 v{r}, 0xdeadbeef" for r in range(28, 36)]
            body = "\n".join(
                [
                    "\ts_load_b256 s[4:11], s[0:1], 0x0",
                    "\ts_mov_b32 s14, ttmp9",
                    "\ts_mov_b32 s15, 0",
                    "\tv_lshlrev_b32 v1, 2, v0",
                    "\tv_lshlrev_b32 v2, 3, v0",
                    "\tv_lshlrev_b32 v3, 4, v0",
                    "\tv_lshlrev_b32 v4, 5, v0",
                    "\ts_wait_kmcnt 0x0",
                    *_emit_ptr_add(4, spec.a_dwords),
                    *_emit_ptr_add(6, spec.b_dwords),
                    *_emit_ptr_add(8, spec.cd_dwords),
                    *_emit_ptr_add(10, store_dw),
                    *_emit_loads(8, spec.a_dwords, voff_a, "s[4:5]"),
                    *_emit_loads(12, spec.b_dwords, voff_b, "s[6:7]"),
                    *_emit_loads(20, spec.cd_dwords, voff_c, "s[8:9]"),
                    *sentinel_init,
                    "\ts_wait_loadcnt 0x0",
                    f"\t.long {w0:#010x}, {w1:#010x}",
                    "\tv_nop",
                    *_emit_stores(28, store_dw, voff_d, "s[10:11]"),
                    "\ts_wait_storecnt 0x0",
                    "\ts_endpgm",
                ]
            )
            kernels.append((f"{tag}_{name}", body))

    # 4. EXEC masking probe kernels (dense f32_f16, f16_f16, f32_fp8_fp8, i32_iu8, and sparse f32_f16)
    #    Reads exec_lo mask from idx_ptr[wg*32 + 0] (for sparse, reads exec_lo from idx_ptr[wg*32 + 0] and uses valid idx 0x44444444 in v36!).
    for name in (
        "v_wmma_f32_16x16x16_f16",
        "v_wmma_f16_16x16x16_f16",
        "v_wmma_f32_16x16x16_fp8_fp8",
        "v_wmma_i32_16x16x16_iu8",
        "v_swmmac_f32_16x16x32_f16",
    ):
        spec = SPECS[name]
        voff_a = {1: "v1", 2: "v2", 4: "v3", 8: "v4"}[spec.a_dwords]
        voff_b = {1: "v1", 2: "v2", 4: "v3", 8: "v4"}[spec.b_dwords]
        voff_c = {1: "v1", 2: "v2", 4: "v3", 8: "v4"}[spec.cd_dwords]
        a_reg = _vreg(8, spec.a_dwords)
        b_reg = _vreg(12, spec.b_dwords)
        c_reg = _vreg(20, spec.cd_dwords)
        d_reg = _vreg(28, spec.cd_dwords)
        if spec.sparse:
            # For swmmac, VDST is both C and D -> load C into v[20:27], set v36 = 0x44444444 (selects 0,1 per group),
            # then mask EXEC, run swmmac, restore EXEC, store v[20:27].
            exec_inst = [
                "\tv_mov_b32 v36, 0x44444444",
                "\ts_mov_b32 exec_lo, s18",
                f"\t{name} {c_reg}, {a_reg}, {b_reg}, v36",
                "\tv_nop",
                "\ts_mov_b32 exec_lo, -1",
                *_emit_stores(20, spec.cd_dwords, voff_c, "s[10:11]"),
            ]
        else:
            exec_inst = [
                *[f"\tv_mov_b32 v{r}, 0xdeadbeef" for r in range(28, 28 + spec.cd_dwords)],
                "\ts_mov_b32 exec_lo, s18",
                f"\t{name} {d_reg}, {a_reg}, {b_reg}, {c_reg}",
                "\tv_nop",
                "\ts_mov_b32 exec_lo, -1",
                *_emit_stores(28, spec.cd_dwords, voff_c, "s[10:11]"),
            ]
        body = "\n".join(
            [
                "\ts_load_b256 s[4:11], s[0:1], 0x0",
                "\ts_load_b64 s[12:13], s[0:1], 0x20",
                "\ts_mov_b32 s14, ttmp9",
                "\ts_mov_b32 s15, 0",
                "\tv_lshlrev_b32 v1, 2, v0",
                "\tv_lshlrev_b32 v2, 3, v0",
                "\tv_lshlrev_b32 v3, 4, v0",
                "\tv_lshlrev_b32 v4, 5, v0",
                "\ts_wait_kmcnt 0x0",
                *_emit_ptr_add(4, spec.a_dwords),
                *_emit_ptr_add(6, spec.b_dwords),
                *_emit_ptr_add(8, spec.cd_dwords),
                *_emit_ptr_add(10, spec.cd_dwords),
                *_emit_ptr_add(12, 1),
                "\ts_load_b32 s18, s[12:13], 0x0",
                *_emit_loads(8, spec.a_dwords, voff_a, "s[4:5]"),
                *_emit_loads(12, spec.b_dwords, voff_b, "s[6:7]"),
                *_emit_loads(20, spec.cd_dwords, voff_c, "s[8:9]"),
                "\ts_wait_kmcnt 0x0",
                "\ts_wait_loadcnt 0x0",
                *exec_inst,
                "\ts_wait_storecnt 0x0",
                "\ts_endpgm",
            ]
        )
        kernels.append((f"exec_{name}", body))

    out_path = ROOT / "build" / "state_probes_gfx1201.co"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    asm = wrap_kernels(kernels, kernarg=40, vgprs=40, sgprs=24)
    with tempfile.TemporaryDirectory() as tmp:
        s_path = Path(tmp) / "state.s"
        o_path = Path(tmp) / "state.o"
        s_path.write_text(asm)
        r = subprocess.run(
            ["llvm-mc", "-triple=amdgcn-amd-amdhsa", "-mcpu=gfx1201", "-filetype=obj", str(s_path), "-o", str(o_path)],
            capture_output=True,
            text=True,
        )
        if r.returncode:
            raise RuntimeError(f"llvm-mc failed:\n{r.stderr}")
        r = subprocess.run(["ld.lld", "-shared", str(o_path), "-o", str(out_path)], capture_output=True, text=True)
        if r.returncode:
            raise RuntimeError(f"ld.lld failed:\n{r.stderr}")
    return out_path


def run_probes() -> None:
    ap = argparse.ArgumentParser(description="Probe gfx1201 MODE, float clamp/op_sel, f16/bf16 D sentinel, and EXEC masking.")
    ap.add_argument("--co", type=Path, default=None, help="Pre-assembled state_probes_gfx1201.co (default: assemble locally)")
    ap.add_argument("--build-co", type=Path, default=None, help="Assemble state_probes_gfx1201.co and exit")
    ap.add_argument("--capture", action="store_true", help="Write evidence/state_gfx1201.mmstate")
    args = ap.parse_args()

    if args.build_co is not None:
        p = build_state_co()
        if args.build_co != p:
            args.build_co.parent.mkdir(parents=True, exist_ok=True)
            args.build_co.write_bytes(p.read_bytes())
        print(f"Assembled {args.build_co}")
        return

    co = args.co if args.co is not None else build_state_co()
    print(f"Using code object: {co}")

    # --- 1. Mode bitfield & positive control probe ---
    # Test modes:
    #   bits[1:0] = fp_round_32 (0=RNE, 1=+inf, 2=-inf, 3=RTZ)
    #   bits[3:2] = fp_round_16_64
    #   bits[5:4] = fp_denorm_32 (0=flush_both, 1=flush_out, 2=flush_in, 3=allow)
    #   bits[7:6] = fp_denorm_16_64
    #   bits[8..15] = probe writability (0x0000, 0x0100, 0x0200, 0xFF00)
    ctrl_modes = [
        0x00F0,  # default: RNE, allow all denorms
        0x00F1,  # f32 +inf
        0x00F2,  # f32 -inf
        0x00F3,  # f32 RTZ
        0x00F4,  # f16 +inf
        0x00F8,  # f16 -inf
        0x00FC,  # f16 RTZ
        0x00C0,  # f32 flush in+out
        0x00D0,  # f32 flush out only
        0x00E0,  # f32 flush in only
        0x0030,  # f16 flush in+out
        0x0070,  # f16 flush out only
        0x00B0,  # f16 flush in only
        0x01F0,  # bit 8 (DX10_CLAMP?)
        0x02F0,  # bit 9 (IEEE?)
        0xFFF0,  # all upper bits [15:8] = 1 (see which bits are writable!)
    ]
    n_cm = len(ctrl_modes)
    c_ctrl = np.zeros((n_cm, 32, 8), dtype=np.uint32)
    # v20+v21 = 1.0 + 2^-24 (tests f32 rounding!); v22+v23 = 2^-127 + 2^-127 (subnormal f32 -> 2^-126 normal!)
    # Wait: also test subnormal input -> subnormal output: v22 = 0x00000001 (2^-149), v23 = 0x00000001
    c_ctrl[:, :, 0] = f32_bits(1.0)
    c_ctrl[:, :, 1] = f32_bits(2.0**-24)
    c_ctrl[:, :, 2] = 0x0040_0000  # 2^-127 (f32 subnormal; sum = 2^-126 normal if denorm-in allowed, 0 if flushed!)
    c_ctrl[:, :, 3] = 0x0040_0000
    # v24 = (2^-11 << 16) | 1.0 (f16 rounding!); v25 = (0x0200 << 16) | 0x0200 (f16 subnormal 2^-15 + 2^-15 = 2^-14!)
    c_ctrl[:, :, 4] = (f16_bits(2.0**-11) << 16) | f16_bits(1.0)
    c_ctrl[:, :, 5] = (0x0200 << 16) | 0x0200
    idx_ctrl = np.zeros((n_cm, 32), dtype=np.uint32)
    for wi, mv in enumerate(ctrl_modes):
        idx_ctrl[wi, :] = mv

    # Note: hw_run expects spec with matching dwords; use v_swmmac_f32_16x16x32_f16? No, ctrl only uses c/d (8 dw) & idx (1 dw).
    # Pass a/b matching v_swmmac_f32_16x16x32_f16 (4, 8) since sparse=True makes hw_run pass idx!
    jobs = [
        {
            "id": "ctrl",
            "spec": "v_swmmac_f32_16x16x32_f16",
            "kernel": "probe_mode_ctrl",
            "a": np.zeros((n_cm, 32, 4), dtype=np.uint32),
            "b": np.zeros((n_cm, 32, 8), dtype=np.uint32),
            "c": c_ctrl,
            "idx": idx_ctrl,
        }
    ]

    # --- 2. MODE test across ALL 8 dense float WMMA variants ---
    # Test 16 mode values covering ALL 4 f32 round, ALL 4 f16 round, ALL 4 f32 denorm, ALL 4 f16 denorm, and bits 8..11!
    wmma_modes = [
        0x00F0,  # baseline: RNE, allow all denorms
        0x00FF,  # f32 RTZ + f16 RTZ + allow denorms
        0x00F5,  # f32 +inf + f16 +inf + allow denorms
        0x00FA,  # f32 -inf + f16 -inf + allow denorms
        0x0000,  # RNE + flush ALL denorms (in+out, f32 + f16/f64)
        0x0055,  # +inf + flush out only (f32 + f16/f64)
        0x00AA,  # -inf + flush in only (f32 + f16/f64)
        0x000F,  # RTZ + flush ALL denorms
        0x01F0,  # bit 8 set
        0x02F0,  # bit 9 set
        0x03F0,  # bits 8,9 set
        0x0FF0,  # bits 8..11 set
        0x0000 | 0x0F00,  # bits 8..11 set + flush all denorms
        0xFFF0,  # all upper bits set
    ]
    float_dense = [
        "v_wmma_f32_16x16x16_f16",
        "v_wmma_f32_16x16x16_bf16",
        "v_wmma_f16_16x16x16_f16",
        "v_wmma_bf16_16x16x16_bf16",
        "v_wmma_f32_16x16x16_fp8_fp8",
        "v_wmma_f32_16x16x16_fp8_bf8",
        "v_wmma_f32_16x16x16_bf8_fp8",
        "v_wmma_f32_16x16x16_bf8_bf8",
    ]
    mode_inputs = {}
    rng = np.random.default_rng(12345)
    for name in float_dense:
        spec = SPECS[name]
        # Build 4 rich waves (1024 elements) containing ties, tiny inexact, subnormals (in & out), NaNs, Infs, random finite:
        # Repeat those 4 waves for EVERY mode in wmma_modes so we directly diff against baseline mode 0x00F0!
        base_a = np.zeros((4, 32, spec.a_dwords), dtype=np.uint32)
        base_b = np.zeros((4, 32, spec.b_dwords), dtype=np.uint32)
        base_c = np.zeros((4, 32, spec.cd_dwords), dtype=np.uint32)

        # Wave 0: targeted ties + tiny inexact + subnormal inputs/outputs + sNaN + Inf*0
        am = [[0] * 16 for _ in range(16)]
        bm = [[0] * 16 for _ in range(16)]
        cm = [[0] * 16 for _ in range(16)]
        if spec.a_fmt == "f16":
            one, neg1 = f16_bits(1.0), f16_bits(-1.0)
            half_u = f16_bits(2.0**-11) if spec.d_fmt == "f16" else f16_bits(2.0**-12)
            tiny = f16_bits(2.0**-12)
            sub_a = 0x0001  # 2^-24
            snan_a = 0x7C01
            inf_a = 0x7C00
        elif spec.a_fmt == "bf16":
            one, neg1 = bf16_bits(1.0), bf16_bits(-1.0)
            half_u = bf16_bits(2.0**-8) if spec.d_fmt == "bf16" else bf16_bits(2.0**-12)
            tiny = bf16_bits(2.0**-12)
            sub_a = 0x0001  # 2^-133
            snan_a = 0x7F81
            inf_a = 0x7F80
        else:
            one = 0x38 if spec.a_fmt == "fp8" else 0x3C
            neg1 = one ^ 0x80
            half_u = 0x04 if spec.a_fmt == "fp8" else 0x0C  # small
            tiny = 0x01
            sub_a = 0x01
            snan_a = 0x7F
            inf_a = 0x7C if spec.a_fmt == "bf8" else 0x7E
        if spec.b_fmt == "f16":
            one_b = f16_bits(1.0)
            sub_b = 0x0001
            tiny_b = f16_bits(2.0**-12)
        elif spec.b_fmt == "bf16":
            one_b = bf16_bits(1.0)
            sub_b = 0x0001
            tiny_b = bf16_bits(2.0**-12)
        else:
            one_b = 0x38 if spec.b_fmt == "fp8" else 0x3C
            sub_b = 0x01
            tiny_b = 0x01
        c_one = f32_bits(1.0) if spec.d_fmt == "f32" else (f16_bits(1.0) if spec.d_fmt == "f16" else bf16_bits(1.0))
        c_neg1 = c_one ^ (0x8000_0000 if spec.d_fmt == "f32" else 0x8000)
        c_sub = 0x0000_0001 if spec.d_fmt == "f32" else 0x0001
        c_nsub = c_sub ^ (0x8000_0000 if spec.d_fmt == "f32" else 0x8000)
        c_snan = 0x7F80_0123 if spec.d_fmt == "f32" else (0x7C05 if spec.d_fmt == "f16" else 0x7F85)

        # Populate rows 0..15 with diverse targeted cases (all cols get distinct B)
        for r in range(16):
            am[r][0] = one if r % 2 == 0 else neg1
            am[r][1] = tiny if r < 8 else (tiny ^ (0x80 if spec.a_bits == 8 else 0x8000))
            am[r][2] = sub_a
            if r == 12:
                am[r][0] = snan_a
            if r == 13:
                am[r][0] = inf_a
            if r == 14:
                am[r][0] = sub_a
                am[r][1] = 0
                am[r][2] = 0
            if r == 15:
                am[r][0] = 0
                am[r][1] = 0
                am[r][2] = 0
        for c in range(16):
            bm[0][c] = one_b if c < 12 else (sub_b if c < 14 else 0)
            bm[1][c] = tiny_b
            bm[2][c] = sub_b
        for r in range(16):
            for c in range(16):
                cm[r][c] = [c_one, c_neg1, c_sub, c_nsub, 0, c_snan, c_one ^ 1, c_neg1 ^ 1][(r + c) % 8]
        base_a[0] = np.array(pack_a(spec, am), dtype=np.uint32).reshape(32, spec.a_dwords)
        base_b[0] = np.array(pack_b(spec, bm), dtype=np.uint32).reshape(32, spec.b_dwords)
        base_c[0] = np.array(pack_cd(spec, cm), dtype=np.uint32).reshape(32, spec.cd_dwords)

        # Waves 1..3: random raw + finite + subnormal-heavy
        base_a[1:] = rng.integers(0, 1 << 32, size=(3, 32, spec.a_dwords), dtype=np.uint32)
        base_b[1:] = rng.integers(0, 1 << 32, size=(3, 32, spec.b_dwords), dtype=np.uint32)
        base_c[1:] = rng.integers(0, 1 << 32, size=(3, 32, spec.cd_dwords), dtype=np.uint32)
        # Make wave 2 subnormal-heavy in C (exp=0)
        if spec.d_fmt == "f32":
            base_c[2] &= 0x807F_FFFF
        elif spec.d_fmt == "f16":
            base_c[2] &= 0x83FF_83FF
        else:
            base_c[2] &= 0x807F_807F

        # Tile across all wmma_modes
        a_all = np.concatenate([base_a for _ in wmma_modes], axis=0)
        b_all = np.concatenate([base_b for _ in wmma_modes], axis=0)
        c_all = np.concatenate([base_c for _ in wmma_modes], axis=0)
        idx_all = np.zeros((len(wmma_modes) * 4, 32), dtype=np.uint32)
        for mi, mv in enumerate(wmma_modes):
            idx_all[4 * mi : 4 * (mi + 1), :] = mv
        mode_inputs[name] = (a_all, b_all, c_all, idx_all)

    for name, (a_all, b_all, c_all, idx_all) in mode_inputs.items():
        jobs.append(
            {
                "id": f"mode_{name}",
                "spec": name,
                "kernel": f"mode_{name}",
                "a": a_all,
                "b": b_all,
                "c": c_all,
                "idx": idx_all,
            }
        )

    # --- 3. Raw CLAMP=1, Sentinel, and OPSEL probes ---
    clamp_specs = [
        "v_wmma_f32_16x16x16_f16",
        "v_wmma_f32_16x16x16_bf16",
        "v_wmma_f16_16x16x16_f16",
        "v_wmma_bf16_16x16x16_bf16",
        "v_wmma_f32_16x16x16_fp8_fp8",
        "v_wmma_f32_16x16x16_bf8_bf8",
    ]
    clamp_vals = [-5.0, -0.5, -0.0, 0.0, 0.25, 0.5, 1.0, 2.0, 100.0, float("inf"), float("-inf"), float("nan")]
    for name in clamp_specs:
        spec = SPECS[name]
        am = [[0] * 16 for _ in range(16)]
        bm = [[0] * 16 for _ in range(16)]
        cm = [[0] * 16 for _ in range(16)]
        for r in range(16):
            for c in range(16):
                v = clamp_vals[(r * 16 + c) % len(clamp_vals)]
                if spec.d_fmt == "f32":
                    cm[r][c] = 0x7F80_0123 if np.isnan(v) else f32_bits(v)
                elif spec.d_fmt == "f16":
                    cm[r][c] = 0x7C05 if np.isnan(v) else f16_bits(v)
                else:
                    cm[r][c] = 0x7F85 if np.isnan(v) else bf16_bits(v)
        # Also set A[0,0]=1, B[0,0]=2 so A*B is nonzero too
        one_a = f16_bits(1.0) if spec.a_fmt == "f16" else (bf16_bits(1.0) if spec.a_fmt == "bf16" else (0x38 if spec.a_fmt == "fp8" else 0x3C))
        two_b = f16_bits(2.0) if spec.b_fmt == "f16" else (bf16_bits(2.0) if spec.b_fmt == "bf16" else (0x40 if spec.b_fmt == "fp8" else 0x40))
        am[0][0] = one_a
        bm[0][0] = two_b
        a_arr = np.array(pack_a(spec, am), dtype=np.uint32).reshape(1, 32, spec.a_dwords)
        b_arr = np.array(pack_b(spec, bm), dtype=np.uint32).reshape(1, 32, spec.b_dwords)
        c_arr = np.array(pack_cd(spec, cm), dtype=np.uint32).reshape(1, 32, spec.cd_dwords)

        jobs.append({"id": f"rawclamp_{name}", "spec": name, "kernel": f"rawclamp_{name}", "a": a_arr, "b": b_arr, "c": c_arr})
        if spec.d_fmt in ("f16", "bf16"):
            # Sentinel kernel loads 4 dwords/lane from c_ptr (128 dwords total) and stores 8 dwords/lane to d_ptr (256 dwords).
            # Pack c_arr tightly into the first 128 dwords of an 8-dword-per-lane buffer so hw_run allocates 256 dwords for D.
            c_pad = np.zeros(1 * 32 * 8, dtype=np.uint32)
            c_pad[: 32 * 4] = c_arr.reshape(-1)
            c_pad = c_pad.reshape(1, 32, 8)
            jobs.append({"id": f"sentinel_{name}", "spec": "v_wmma_f32_16x16x16_f16", "kernel": f"sentinel_{name}", "a": a_arr, "b": b_arr, "c": c_pad})
            jobs.append({"id": f"rawopsel_7_{name}", "spec": name, "kernel": f"rawopsel_7_{name}", "a": a_arr, "b": b_arr, "c": c_arr})
            jobs.append({"id": f"rawopsel_hi0_{name}", "spec": name, "kernel": f"rawopsel_hi0_{name}", "a": a_arr, "b": b_arr, "c": c_arr})

    # --- 4. EXEC masking probes ---
    exec_masks = [0xFFFF_FFFF, 0x0000_FFFF, 0xFFFF_0000, 0x5555_5555, 0x0000_0001, 0x0000_0000]
    for name in (
        "v_wmma_f32_16x16x16_f16",
        "v_wmma_f16_16x16x16_f16",
        "v_wmma_f32_16x16x16_fp8_fp8",
        "v_wmma_i32_16x16x16_iu8",
        "v_swmmac_f32_16x16x32_f16",
    ):
        spec = SPECS[name]
        one_a = f16_bits(1.0) if spec.a_fmt == "f16" else (0x38 if spec.a_fmt == "fp8" else 1)
        one_b = f16_bits(1.0) if spec.b_fmt == "f16" else (0x38 if spec.b_fmt == "fp8" else 1)
        # All A = 1, all B = 1, all C = 100 (or 0x11111111 sentinel for swmmac C so we can see if C was preserved!)
        c_val = f32_bits(100.0) if spec.d_fmt == "f32" else (f16_bits(100.0) if spec.d_fmt == "f16" else 100)
        am = [[one_a] * spec.k_a for _ in range(16)]
        bm = [[one_b] * 16 for _ in range(spec.k_b)]
        cm = [[c_val] * 16 for _ in range(16)]
        a1 = np.array(pack_a(spec, am), dtype=np.uint32).reshape(1, 32, spec.a_dwords)
        b1 = np.array(pack_b(spec, bm), dtype=np.uint32).reshape(1, 32, spec.b_dwords)
        c1 = np.array(pack_cd(spec, cm), dtype=np.uint32).reshape(1, 32, spec.cd_dwords)
        a_e = np.repeat(a1, len(exec_masks), axis=0)
        b_e = np.repeat(b1, len(exec_masks), axis=0)
        c_e = np.repeat(c1, len(exec_masks), axis=0)
        idx_e = np.zeros((len(exec_masks), 32), dtype=np.uint32)
        for wi, em in enumerate(exec_masks):
            idx_e[wi, :] = em
        jobs.append({"id": f"exec_{name}", "spec": name, "kernel": f"exec_{name}", "a": a_e, "b": b_e, "c": c_e, "idx": idx_e})

    res = run_jobs(jobs, co_path=co)

    # Print results cleanly!
    print("\n=== 1. MODE CONTROL (v_add_f32 / v_add_f16 positive control + HW_REG_WAVE_MODE readback) ===")
    d_ctrl = res["ctrl"]
    for wi, mv in enumerate(ctrl_modes):
        w = d_ctrl[wi, 0]  # lane 0
        print(
            f"  write={mv:#06x} init={w[0]:#010x} read={w[1]:#010x} "
            f"f32(1+2^-24)={w[2]:#010x} f32(sub+sub)={w[3]:#010x} "
            f"f16(1+2^-11)={w[4]:#06x} f16(sub+sub)={w[5]:#06x} "
            f"after_rndF={w[6]:#010x} after_denF={w[7]:#010x}"
        )

    print("\n=== 2. MODE EFFECT ON ALL 8 FLOAT WMMA VARIANTS (diff vs baseline 0x00F0 across 1024 elems/mode) ===")
    for name in float_dense:
        d = res[f"mode_{name}"]  # shape (len(wmma_modes)*4, 32, cd_dwords)
        base = d[0:4]
        diffs = []
        for mi, mv in enumerate(wmma_modes):
            cur = d[4 * mi : 4 * (mi + 1)]
            n_diff = int(np.count_nonzero(cur != base))
            if n_diff:
                diffs.append(f"{mv:#06x}:{n_diff}dw")
        print(f"  {name:34s}: " + ("IDENTICAL across all 14 MODE settings!" if not diffs else ", ".join(diffs)))

    print("\n=== 3. RAW VOP3P CLAMP=1 / SENTINEL / OPSEL PROBES ===")
    from amdgpu_matrix_model.model import wmma_wave
    for name in clamp_specs:
        spec = SPECS[name]
        # Compare rawclamp against default wmma_wave (unclamped)
        job = next(j for j in jobs if j["id"] == f"rawclamp_{name}")
        d_rc = res[f"rawclamp_{name}"][0]
        pred = np.array(
            wmma_wave(name, job["a"][0].reshape(-1).tolist(), job["b"][0].reshape(-1).tolist(), job["c"][0].reshape(-1).tolist()),
            dtype=np.uint32,
        ).reshape(32, spec.cd_dwords)
        n_diff = int(np.count_nonzero(d_rc != pred))
        print(f"  rawclamp {name:32s}: diff_vs_unclamped={n_diff} dwords")
        if n_diff:
            d_mat = unpack_cd(spec, d_rc.reshape(-1).tolist())
            p_mat = unpack_cd(spec, pred.reshape(-1).tolist())
            for r in range(2):
                for c in range(12):
                    if d_mat[r][c] != p_mat[r][c]:
                        print(f"    ({r},{c}) unclamped={p_mat[r][c]:#x} rawclamp={d_mat[r][c]:#x}")

        if spec.d_fmt in ("f16", "bf16"):
            d_sen = res[f"sentinel_{name}"][0]  # (32, 8)
            lo4_ok = np.array_equal(d_sen[:, :4], pred)
            hi4_preserved = bool((d_sen[:, 4:] == 0xDEAD_BEEF).all())
            print(f"  sentinel {name:32s}: v[28:31]==pred:{lo4_ok}, v[32:35]==0xDEADBEEF:{hi4_preserved}")
            for otag in ("rawopsel_7", "rawopsel_hi0"):
                d_op = res[f"{otag}_{name}"][0]
                nd = int(np.count_nonzero(d_op != pred))
                print(f"  {otag:12s} {name:28s}: diff_vs_default={nd} dwords")

    print("\n=== 4. EXEC MASKING PROBES ===")
    for name in (
        "v_wmma_f32_16x16x16_f16",
        "v_wmma_f16_16x16x16_f16",
        "v_wmma_f32_16x16x16_fp8_fp8",
        "v_wmma_i32_16x16x16_iu8",
        "v_swmmac_f32_16x16x32_f16",
    ):
        spec = SPECS[name]
        d_ex = res[f"exec_{name}"]  # (len(exec_masks), 32, cd_dwords)
        ref = d_ex[0]  # exec_lo = 0xFFFFFFFF -> every element is 100 + 16 = 116!
        sentinel_w = (f32_bits(100.0) if spec.sparse else 0xDEAD_BEEF)
        print(f"  {name}: ref[lane0,dw0]={ref[0,0]:#010x}")
        for wi, em in enumerate(exec_masks):
            cur = d_ex[wi]
            active_eq_ref = True
            inactive_eq_sent = True
            inactive_eq_ref = True
            other_vals = set()
            for lane in range(32):
                is_act = bool((em >> lane) & 1)
                if is_act:
                    if not np.array_equal(cur[lane], ref[lane]):
                        active_eq_ref = False
                        other_vals.add(("act", lane, hex(int(cur[lane, 0]))))
                else:
                    if not (cur[lane] == sentinel_w).all():
                        inactive_eq_sent = False
                    if not np.array_equal(cur[lane], ref[lane]):
                        inactive_eq_ref = False
                        if not (cur[lane] == sentinel_w).all():
                            other_vals.add(("inact", lane, hex(int(cur[lane, 0]))))
            print(
                f"    exec={em:#010x}: active==full116:{active_eq_ref}, "
                f"inactive==sentinel:{inactive_eq_sent}, inactive==full116:{inactive_eq_ref}"
                + (f" other={list(other_vals)[:4]}" if other_vals else "")
            )

    if args.capture:
        import zlib
        import json

        state_doc: dict = {
            "target": "gfx1201",
            "mode_ctrl": [
                {
                    "write": mv,
                    "init": int(d_ctrl[wi, 0, 0]),
                    "read": int(d_ctrl[wi, 0, 1]),
                    "f32_tie": int(d_ctrl[wi, 0, 2]),
                    "f32_sub": int(d_ctrl[wi, 0, 3]),
                    "f16_tie": int(d_ctrl[wi, 0, 4]) & 0xFFFF,
                    "f16_sub": int(d_ctrl[wi, 0, 5]) & 0xFFFF,
                }
                for wi, mv in enumerate(ctrl_modes)
            ],
            "mode_wmma": [],
            "raw_clamp": [],
            "sentinel_opsel": [],
            "exec_mask": [],
        }
        for name in float_dense:
            a_all, b_all, c_all, _ = mode_inputs[name]
            d_all = res[f"mode_{name}"]
            state_doc["mode_wmma"].append(
                {
                    "spec": name,
                    "modes": wmma_modes,
                    "waves_per_mode": 4,
                    "a": a_all[:4].reshape(-1).tolist(),
                    "b": b_all[:4].reshape(-1).tolist(),
                    "c": c_all[:4].reshape(-1).tolist(),
                    "d_by_mode": [d_all[4 * mi : 4 * (mi + 1)].reshape(-1).tolist() for mi in range(len(wmma_modes))],
                }
            )
        for name in clamp_specs:
            job = next(j for j in jobs if j["id"] == f"rawclamp_{name}")
            state_doc["raw_clamp"].append(
                {
                    "spec": name,
                    "a": job["a"][0].reshape(-1).tolist(),
                    "b": job["b"][0].reshape(-1).tolist(),
                    "c": job["c"][0].reshape(-1).tolist(),
                    "d_hw": res[f"rawclamp_{name}"][0].reshape(-1).tolist(),
                }
            )
            if SPECS[name].d_fmt in ("f16", "bf16"):
                state_doc["sentinel_opsel"].append(
                    {
                        "spec": name,
                        "a": job["a"][0].reshape(-1).tolist(),
                        "b": job["b"][0].reshape(-1).tolist(),
                        "c": job["c"][0].reshape(-1).tolist(),
                        "d_sentinel_8dw": res[f"sentinel_{name}"][0].reshape(-1).tolist(),
                        "d_rawopsel_7": res[f"rawopsel_7_{name}"][0].reshape(-1).tolist(),
                        "d_rawopsel_hi0": res[f"rawopsel_hi0_{name}"][0].reshape(-1).tolist(),
                    }
                )
        for name in (
            "v_wmma_f32_16x16x16_f16",
            "v_wmma_f16_16x16x16_f16",
            "v_wmma_f32_16x16x16_fp8_fp8",
            "v_wmma_i32_16x16x16_iu8",
            "v_swmmac_f32_16x16x32_f16",
        ):
            job = next(j for j in jobs if j["id"] == f"exec_{name}")
            state_doc["exec_mask"].append(
                {
                    "spec": name,
                    "masks": exec_masks,
                    "a": job["a"][0].reshape(-1).tolist(),
                    "b": job["b"][0].reshape(-1).tolist(),
                    "c": job["c"][0].reshape(-1).tolist(),
                    "idx": ([0x4444_4444] * 32 if SPECS[name].sparse else None),
                    "d_by_mask": [res[f"exec_{name}"][wi].reshape(-1).tolist() for wi in range(len(exec_masks))],
                }
            )
        out_path = ROOT / "evidence" / "state_gfx1201.mmstate"
        raw_bytes = zlib.compress(json.dumps(state_doc, separators=(",", ":")).encode("utf-8"), level=6)
        out_path.write_bytes(b"AMDST001" + raw_bytes)
        print(f"\nCaptured frozen state evidence to {out_path} ({len(raw_bytes) / 1024:.1f} KiB)")


if __name__ == "__main__":
    run_probes()
