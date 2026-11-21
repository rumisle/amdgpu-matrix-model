"""Instruction and format tables for gfx1201 (RDNA 4) wave32 WMMA / SWMMAC."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FloatFormat:
    name: str
    bits: int
    exp_bits: int
    frac_bits: int
    bias: int
    has_inf: bool
    # Canonical / special bit patterns (for destination formats)
    qnan_default: int = 0
    qnan_sig: int = 0
    quiet_bit: int = 0

    @property
    def mask(self) -> int:
        return (1 << self.bits) - 1

    @property
    def sign_bit(self) -> int:
        return 1 << (self.bits - 1)

    @property
    def exp_max(self) -> int:
        return (1 << self.exp_bits) - 1

    @property
    def frac_mask(self) -> int:
        return (1 << self.frac_bits) - 1

    @property
    def emin(self) -> int:
        return 1 - self.bias

    @property
    def emax(self) -> int:
        return self.exp_max - 1 - self.bias if self.has_inf else self.exp_max - self.bias


@dataclass(frozen=True)
class IntFormat:
    name: str
    bits: int

    @property
    def mask(self) -> int:
        return (1 << self.bits) - 1


@dataclass(frozen=True)
class WmmaSpec:
    name: str
    a_fmt: str
    b_fmt: str
    d_fmt: str
    k: int
    sparse: bool
    a_dwords: int
    b_dwords: int
    cd_dwords: int
    opcode: int  # VOP3P OP field (RDNA4 ISA section 15.3.7)
    has_index_key: bool = False

    @property
    def is_int(self) -> bool:
        return self.d_fmt == "i32"

    @property
    def is_fp8_family(self) -> bool:
        return self.a_fmt in ("fp8", "bf8")

    @property
    def k_a(self) -> int:
        return self.k // 2 if self.sparse else self.k

    @property
    def k_b(self) -> int:
        return self.k

    @property
    def a_bits(self) -> int:
        return elem_bits(self.a_fmt)

    @property
    def b_bits(self) -> int:
        return elem_bits(self.b_fmt)

    @property
    def d_bits(self) -> int:
        return elem_bits(self.d_fmt)


FLOAT_FORMATS: dict[str, FloatFormat] = {
    "f32": FloatFormat("f32", 32, 8, 23, 127, True, 0xFFC0_0000, 0xFFC0_0A3D, 0x0040_0000),
    "f16": FloatFormat("f16", 16, 5, 10, 15, True, 0xFE00, 0xFE3D, 0x0200),
    "bf16": FloatFormat("bf16", 16, 8, 7, 127, True, 0xFFC0, 0xFFFD, 0x0040),
    "fp8": FloatFormat("fp8", 8, 4, 3, 7, False),  # OCP E4M3FN: 0x7f / 0xff are NaN, no Inf
    "bf8": FloatFormat("bf8", 8, 5, 2, 15, True),  # OCP E5M2: IEEE-like Inf and NaN
}

INT_FORMATS: dict[str, IntFormat] = {
    "iu8": IntFormat("iu8", 8),
    "iu4": IntFormat("iu4", 4),
    "i32": IntFormat("i32", 32),
}


def elem_bits(fmt: str) -> int:
    if fmt in FLOAT_FORMATS:
        return FLOAT_FORMATS[fmt].bits
    return INT_FORMATS[fmt].bits


_SPEC_LIST: tuple[WmmaSpec, ...] = (
    # Dense WMMA (11 instructions)
    WmmaSpec("v_wmma_f32_16x16x16_f16", "f16", "f16", "f32", 16, False, 4, 4, 8, 64),
    WmmaSpec("v_wmma_f32_16x16x16_bf16", "bf16", "bf16", "f32", 16, False, 4, 4, 8, 65),
    WmmaSpec("v_wmma_f16_16x16x16_f16", "f16", "f16", "f16", 16, False, 4, 4, 4, 66),
    WmmaSpec("v_wmma_bf16_16x16x16_bf16", "bf16", "bf16", "bf16", 16, False, 4, 4, 4, 67),
    WmmaSpec("v_wmma_f32_16x16x16_fp8_fp8", "fp8", "fp8", "f32", 16, False, 2, 2, 8, 70),
    WmmaSpec("v_wmma_f32_16x16x16_fp8_bf8", "fp8", "bf8", "f32", 16, False, 2, 2, 8, 71),
    WmmaSpec("v_wmma_f32_16x16x16_bf8_fp8", "bf8", "fp8", "f32", 16, False, 2, 2, 8, 72),
    WmmaSpec("v_wmma_f32_16x16x16_bf8_bf8", "bf8", "bf8", "f32", 16, False, 2, 2, 8, 73),
    WmmaSpec("v_wmma_i32_16x16x16_iu8", "iu8", "iu8", "i32", 16, False, 2, 2, 8, 68),
    WmmaSpec("v_wmma_i32_16x16x16_iu4", "iu4", "iu4", "i32", 16, False, 1, 1, 8, 69),
    WmmaSpec("v_wmma_i32_16x16x32_iu4", "iu4", "iu4", "i32", 32, False, 2, 2, 8, 74),
    # Sparse SWMMAC (11 instructions)
    WmmaSpec("v_swmmac_f32_16x16x32_f16", "f16", "f16", "f32", 32, True, 4, 8, 8, 80, True),
    WmmaSpec("v_swmmac_f32_16x16x32_bf16", "bf16", "bf16", "f32", 32, True, 4, 8, 8, 81, True),
    WmmaSpec("v_swmmac_f16_16x16x32_f16", "f16", "f16", "f16", 32, True, 4, 8, 4, 82, True),
    WmmaSpec("v_swmmac_bf16_16x16x32_bf16", "bf16", "bf16", "bf16", 32, True, 4, 8, 4, 83, True),
    WmmaSpec("v_swmmac_f32_16x16x32_fp8_fp8", "fp8", "fp8", "f32", 32, True, 2, 4, 8, 87, True),
    WmmaSpec("v_swmmac_f32_16x16x32_fp8_bf8", "fp8", "bf8", "f32", 32, True, 2, 4, 8, 88, True),
    WmmaSpec("v_swmmac_f32_16x16x32_bf8_fp8", "bf8", "fp8", "f32", 32, True, 2, 4, 8, 89, True),
    WmmaSpec("v_swmmac_f32_16x16x32_bf8_bf8", "bf8", "bf8", "f32", 32, True, 2, 4, 8, 90, True),
    WmmaSpec("v_swmmac_i32_16x16x32_iu8", "iu8", "iu8", "i32", 32, True, 2, 4, 8, 84, True),
    WmmaSpec("v_swmmac_i32_16x16x32_iu4", "iu4", "iu4", "i32", 32, True, 1, 2, 8, 85, True),
    WmmaSpec("v_swmmac_i32_16x16x64_iu4", "iu4", "iu4", "i32", 64, True, 2, 4, 8, 86, False),
)

SPECS: dict[str, WmmaSpec] = {s.name: s for s in _SPEC_LIST}
