"""Bit-exact, dependency-free software model of gfx1201 (RDNA 4) WMMA / SWMMAC arithmetic."""

from .layout import (
    pack_a,
    pack_b,
    pack_cd,
    pack_idx,
    unpack_a,
    unpack_b,
    unpack_cd,
    unpack_idx,
)
from .model import (
    bits_to_f32,
    decode_float,
    decode_int,
    dot_float_step,
    f32_to_bits,
    pack_float,
    select_sparse_b,
    wmma_dot,
    wmma_matrix,
    wmma_wave,
)
from .specs import FLOAT_FORMATS, INT_FORMATS, SPECS, FloatFormat, IntFormat, WmmaSpec

__all__ = [
    "FLOAT_FORMATS",
    "INT_FORMATS",
    "SPECS",
    "FloatFormat",
    "IntFormat",
    "WmmaSpec",
    "bits_to_f32",
    "decode_float",
    "decode_int",
    "dot_float_step",
    "f32_to_bits",
    "pack_a",
    "pack_b",
    "pack_cd",
    "pack_float",
    "pack_idx",
    "select_sparse_b",
    "unpack_a",
    "unpack_b",
    "unpack_cd",
    "unpack_idx",
    "wmma_dot",
    "wmma_matrix",
    "wmma_wave",
]
