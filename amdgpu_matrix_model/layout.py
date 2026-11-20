"""Wave32 lane/VGPR operand packing and unpacking for gfx1201 WMMA / SWMMAC.

All functions operate on raw integer element bit patterns and raw u32 VGPR words.
"""

from __future__ import annotations

from collections.abc import Sequence

from .specs import WmmaSpec


def _pack_lane(elems: Sequence[int], bits: int, dwords: int) -> list[int]:
    mask = (1 << bits) - 1
    acc = 0
    for i, v in enumerate(elems):
        acc |= (int(v) & mask) << (i * bits)
    return [(acc >> (32 * d)) & 0xFFFFFFFF for d in range(dwords)]


def _unpack_lane(words: Sequence[int], bits: int, count: int) -> list[int]:
    mask = (1 << bits) - 1
    acc = 0
    for d, w in enumerate(words):
        acc |= (int(w) & 0xFFFFFFFF) << (32 * d)
    return [(acc >> (i * bits)) & mask for i in range(count)]


def pack_a(spec: WmmaSpec, a: Sequence[Sequence[int]]) -> list[int]:
    """Pack logical A (16 x k_a) into 32 * a_dwords u32 words (lane-major)."""
    half = spec.k_a // 2
    out: list[int] = []
    for lane in range(32):
        row, h = lane & 15, lane >> 4
        out.extend(_pack_lane(a[row][h * half : (h + 1) * half], spec.a_bits, spec.a_dwords))
    return out


def unpack_a(spec: WmmaSpec, words: Sequence[int]) -> list[list[int]]:
    """Unpack 32 * a_dwords u32 words into logical A (16 x k_a)."""
    half = spec.k_a // 2
    a = [[0] * spec.k_a for _ in range(16)]
    for lane in range(32):
        row, h = lane & 15, lane >> 4
        base = lane * spec.a_dwords
        vals = _unpack_lane(words[base : base + spec.a_dwords], spec.a_bits, half)
        a[row][h * half : (h + 1) * half] = vals
    return a


def pack_b(spec: WmmaSpec, b: Sequence[Sequence[int]]) -> list[int]:
    """Pack logical B (k_b x 16) into 32 * b_dwords u32 words (lane-major)."""
    half = spec.k_b // 2
    out: list[int] = []
    for lane in range(32):
        col, h = lane & 15, lane >> 4
        col_vals = [b[k][col] for k in range(h * half, (h + 1) * half)]
        out.extend(_pack_lane(col_vals, spec.b_bits, spec.b_dwords))
    return out


def unpack_b(spec: WmmaSpec, words: Sequence[int]) -> list[list[int]]:
    """Unpack 32 * b_dwords u32 words into logical B (k_b x 16)."""
    half = spec.k_b // 2
    b = [[0] * 16 for _ in range(spec.k_b)]
    for lane in range(32):
        col, h = lane & 15, lane >> 4
        base = lane * spec.b_dwords
        vals = _unpack_lane(words[base : base + spec.b_dwords], spec.b_bits, half)
        for i, v in enumerate(vals):
            b[h * half + i][col] = v
    return b


def pack_cd(spec: WmmaSpec, mat: Sequence[Sequence[int]]) -> list[int]:
    """Pack logical C/D (16 x 16) into 32 * cd_dwords u32 words (lane-major)."""
    out: list[int] = []
    for lane in range(32):
        col, h = lane & 15, lane >> 4
        vals = [mat[8 * h + e][col] for e in range(8)]
        out.extend(_pack_lane(vals, spec.d_bits, spec.cd_dwords))
    return out


def unpack_cd(spec: WmmaSpec, words: Sequence[int]) -> list[list[int]]:
    """Unpack 32 * cd_dwords u32 words into logical C/D (16 x 16)."""
    mat = [[0] * 16 for _ in range(16)]
    for lane in range(32):
        col, h = lane & 15, lane >> 4
        base = lane * spec.cd_dwords
        vals = _unpack_lane(words[base : base + spec.cd_dwords], spec.d_bits, 8)
        for e, v in enumerate(vals):
            mat[8 * h + e][col] = v
    return mat


def pack_idx(spec: WmmaSpec, idx: Sequence[Sequence[int]], index_key: int = 0) -> list[int]:
    """Pack logical sparse 2-bit index table (16 x k_a) into 32 u32 words (1 per lane).

    Each entry idx[row][ka] is in 0..3 (offset within the 4-element K group ka // 2).
    For K=32 SWMMAC (has_index_key=True), index_key=0 writes bits [15:0], index_key=1 writes bits [31:16].
    For K=64 SWMMAC (has_index_key=False), all 32 bits [31:0] are used.
    """
    half = spec.k_a // 2
    shift = 16 if (spec.has_index_key and index_key == 1) else 0
    out = [0] * 32
    for lane in range(32):
        row, h = lane & 15, lane >> 4
        w = 0
        for i, sel in enumerate(idx[row][h * half : (h + 1) * half]):
            w |= (int(sel) & 0x3) << (2 * i)
        out[lane] = (w << shift) & 0xFFFFFFFF
    return out


def unpack_idx(spec: WmmaSpec, words: Sequence[int], index_key: int = 0) -> list[list[int]]:
    """Unpack 32 u32 index words into logical sparse 2-bit index table (16 x k_a)."""
    half = spec.k_a // 2
    shift = 16 if (spec.has_index_key and index_key == 1) else 0
    out = [[0] * spec.k_a for _ in range(16)]
    for lane in range(32):
        row, h = lane & 15, lane >> 4
        w = (int(words[lane]) >> shift) & 0xFFFFFFFF
        for i in range(half):
            out[row][h * half + i] = (w >> (2 * i)) & 0x3
    return out

