"""Generate and assemble gfx1201 probe kernels for all WMMA and SWMMAC variants."""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from amdgpu_matrix_model.specs import SPECS, WmmaSpec
from tools.abi import wrap_kernels

_LOG2_WAVE_BYTES = {1: 7, 2: 8, 4: 9, 8: 10, 16: 11}
_LANE_OFF_VGPR = {1: "v1", 2: "v2", 4: "v3", 8: "v4"}


def _vreg(base: int, count: int) -> str:
    return f"v{base}" if count == 1 else f"v[{base}:{base + count - 1}]"


def _emit_loads(base_v: int, dwords: int, voff: str, sbase: str) -> list[str]:
    lines: list[str] = []
    at = 0
    while at < dwords:
        w = min(4, dwords - at)
        op = {1: "global_load_b32", 2: "global_load_b64", 4: "global_load_b128"}[w]
        off = f" offset:{at * 4}" if at else ""
        lines.append(f"\t{op} {_vreg(base_v + at, w)}, {voff}, {sbase}{off}")
        at += w
    return lines


def _emit_stores(base_v: int, dwords: int, voff: str, sbase: str) -> list[str]:
    lines: list[str] = []
    at = 0
    while at < dwords:
        w = min(4, dwords - at)
        op = {1: "global_store_b32", 2: "global_store_b64", 4: "global_store_b128"}[w]
        off = f" offset:{at * 4}" if at else ""
        lines.append(f"\t{op} {voff}, {_vreg(base_v + at, w)}, {sbase}{off}")
        at += w
    return lines


def _emit_ptr_add(slo: int, dwords: int) -> list[str]:
    sh = _LOG2_WAVE_BYTES[dwords]
    return [
        f"\ts_lshl_b64 s[16:17], s[14:15], {sh}",
        f"\ts_add_co_u32 s{slo}, s{slo}, s16",
        f"\ts_add_co_ci_u32 s{slo + 1}, s{slo + 1}, s17",
    ]


def kernel_body(spec: WmmaSpec, modifiers: str = "") -> str:
    """Emit one batched wave32 probe kernel.

    kernarg layout (48 bytes; see tools/hw_run.py):
      0x00: a_ptr   (u64)  [G, 32, a_dwords] u32
      0x08: b_ptr   (u64)  [G, 32, b_dwords] u32
      0x10: c_ptr   (u64)  [G, 32, cd_dwords] u32
      0x18: d_ptr   (u64)  [G, 32, cd_dwords] u32
      0x20: idx_ptr (u64)  [G, 32, 1] u32 (SWMMAC only)
      0x28: param_ptr (u64) [G] u32 (unused here)
    """
    lines = [
        "\ts_load_b256 s[4:11], s[0:1], 0x0",
    ]
    if spec.sparse:
        lines.append("\ts_load_b64 s[12:13], s[0:1], 0x20")
    lines += [
        "\ts_mov_b32 s14, ttmp9",
        "\ts_mov_b32 s15, 0",
        "\tv_lshlrev_b32 v1, 2, v0",
        "\tv_lshlrev_b32 v2, 3, v0",
        "\tv_lshlrev_b32 v3, 4, v0",
        "\tv_lshlrev_b32 v4, 5, v0",
        "\ts_wait_kmcnt 0x0",
    ]
    lines += _emit_ptr_add(4, spec.a_dwords)
    lines += _emit_ptr_add(6, spec.b_dwords)
    lines += _emit_ptr_add(8, spec.cd_dwords)
    lines += _emit_ptr_add(10, spec.cd_dwords)
    if spec.sparse:
        lines += _emit_ptr_add(12, 1)

    lines += _emit_loads(8, spec.a_dwords, _LANE_OFF_VGPR[spec.a_dwords], "s[4:5]")
    lines += _emit_loads(12, spec.b_dwords, _LANE_OFF_VGPR[spec.b_dwords], "s[6:7]")
    lines += _emit_loads(20, spec.cd_dwords, _LANE_OFF_VGPR[spec.cd_dwords], "s[8:9]")
    if spec.sparse:
        lines += _emit_loads(36, 1, _LANE_OFF_VGPR[1], "s[12:13]")
    lines.append("\ts_wait_loadcnt 0x0")

    a_reg = _vreg(8, spec.a_dwords)
    b_reg = _vreg(12, spec.b_dwords)
    c_reg = _vreg(20, spec.cd_dwords)
    mod = f" {modifiers}" if modifiers else ""

    if spec.sparse:
        lines.append(f"\t{spec.name} {c_reg}, {a_reg}, {b_reg}, v36{mod}")
        lines.append("\tv_nop")
        lines += _emit_stores(20, spec.cd_dwords, _LANE_OFF_VGPR[spec.cd_dwords], "s[10:11]")
    else:
        d_reg = _vreg(28, spec.cd_dwords)
        lines.append(f"\t{spec.name} {d_reg}, {a_reg}, {b_reg}, {c_reg}{mod}")
        lines.append("\tv_nop")
        lines += _emit_stores(28, spec.cd_dwords, _LANE_OFF_VGPR[spec.cd_dwords], "s[10:11]")

    lines.append("\ts_wait_storecnt 0x0")
    lines.append("\ts_endpgm")
    return "\n".join(lines) + "\n"


def kernel_name(
    spec: WmmaSpec,
    index_key: int = 0,
    neg_lo: tuple[int, ...] = (0, 0, 0),
    neg_hi: tuple[int, ...] = (0, 0, 0),
    clamp: bool = False,
) -> str:
    """Canonical probe-kernel symbol for one instruction and modifier set."""
    name = spec.name + ("_ik1" if index_key else "")
    if any(neg_lo):
        name += "_lo" + "".join(map(str, neg_lo))
    if any(neg_hi):
        name += "_hi" + "".join(map(str, neg_hi))
    return name + ("_clamp" if clamp else "")


def modifier_text(
    index_key: int = 0,
    neg_lo: tuple[int, ...] = (0, 0, 0),
    neg_hi: tuple[int, ...] = (0, 0, 0),
    clamp: bool = False,
) -> str:
    mods = ["index_key:1"] if index_key else []
    if any(neg_lo):
        mods.append("neg_lo:[{},{},{}]".format(*neg_lo))
    if any(neg_hi):
        mods.append("neg_hi:[{},{},{}]".format(*neg_hi))
    if clamp:
        mods.append("clamp")
    return " ".join(mods)


# Float A/B modifier sets probed on f16/bf16 forms (dense and sparse) and C sets on dense forms.
AB_MODS = (
    ((1, 0, 0), (1, 0, 0)),
    ((1, 0, 0), (0, 0, 0)),
    ((0, 0, 0), (0, 1, 0)),
)
SPARSE_AB_MODS = (
    ((1, 0, 0), (0, 0, 0)),
    ((0, 0, 0), (1, 0, 0)),
    ((0, 1, 0), (0, 0, 0)),
    ((0, 0, 0), (0, 1, 0)),
    ((1, 1, 0), (0, 1, 0)),
)
C_MODS = (((0, 0, 1), (0, 0, 0)), ((0, 0, 0), (0, 0, 1)), ((0, 0, 1), (0, 0, 1)))


def variant_fields(spec: WmmaSpec) -> list[tuple[int, tuple[int, ...], tuple[int, ...], bool]]:
    """All (index_key, neg_lo, neg_hi, clamp) field sets that have a probe kernel for `spec`."""
    keys = (0, 1) if spec.has_index_key else (0,)
    out = []
    for ik in keys:
        out.append((ik, (0, 0, 0), (0, 0, 0), False))
        if spec.is_int:
            for sa in (0, 1):
                for sb in (0, 1):
                    for cl in (False, True):
                        if sa or sb or cl:
                            out.append((ik, (sa, sb, 0), (0, 0, 0), cl))
        elif not spec.sparse:
            mods = (AB_MODS if not spec.is_fp8_family else ()) + C_MODS
            out.extend((ik, lo, hi, False) for lo, hi in mods)
        elif not spec.is_fp8_family:
            out.extend((ik, lo, hi, False) for lo, hi in SPARSE_AB_MODS)
    return out


def variant_kernels() -> list[tuple[str, WmmaSpec, str]]:
    """All (symbol, spec, modifiers) probe kernels to assemble."""
    return [
        (kernel_name(spec, *f), spec, modifier_text(*f)) for spec in SPECS.values() for f in variant_fields(spec)
    ]


def build_co(out_path: Path | None = None) -> Path:
    """Assemble and link all probe kernels into one gfx1201 code object."""
    out_path = out_path or (ROOT / "build" / "probes_gfx1201.co")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    kernels = [(sym, kernel_body(spec, mods)) for sym, spec, mods in variant_kernels()]
    asm = wrap_kernels(kernels, kernarg=48, vgprs=40, sgprs=24)

    with tempfile.TemporaryDirectory() as tmp:
        s_path = Path(tmp) / "probes.s"
        o_path = Path(tmp) / "probes.o"
        s_path.write_text(asm)
        r = subprocess.run(
            ["llvm-mc", "-triple=amdgcn-amd-amdhsa", "-mcpu=gfx1201", "-filetype=obj", str(s_path), "-o", str(o_path)],
            capture_output=True,
            text=True,
        )
        if r.returncode:
            raise RuntimeError(f"llvm-mc failed:\n{r.stderr}")
        r = subprocess.run(
            ["ld.lld", "-shared", str(o_path), "-o", str(out_path)],
            capture_output=True,
            text=True,
        )
        if r.returncode:
            raise RuntimeError(f"ld.lld failed:\n{r.stderr}")
    return out_path


if __name__ == "__main__":
    p = build_co()
    print(f"Built {p} ({len(variant_kernels())} kernels, {p.stat().st_size} bytes)")
