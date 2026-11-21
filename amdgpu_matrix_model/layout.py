"""Wave32 lane/VGPR operand packing for gfx1201 WMMA / SWMMAC.

Every operand element sits in one *slot* of the wave: lane half ``h`` (lanes ``16 h .. 16 h + 15``),
position ``p`` (the ``p``-th element of the lane's words, LSB-first). The row (A, IDX) or column (B, C, D)
selects the lane inside the half. ``slot_order`` gives the K label of every slot, written as the
lane-order index ``q = h * (K / 2) + p``; for every layout except dense 16-bit A/B that label is ``q``
itself.

Dense 16-bit A/B follow RDNA4 ISA section 7.12.2 (wave32, 16-bit A 16x16): lane ``{k[2], row[3:0]}``,
VGPR ``{k[3], k[1]}``, half ``k[0]`` (B: the same with ``col``). In lane order that is ``k = q`` with
bits 2 and 3 exchanged.

All functions operate on raw element bit patterns and raw u32 VGPR words and check their arguments.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache

from .specs import WmmaSpec

_U32 = 0xFFFF_FFFF


def _swap_bits_2_3(q: int) -> int:
    return (q & ~0b1100) | ((q >> 1) & 0b0100) | ((q << 1) & 0b1000)


@lru_cache(maxsize=None)
def slot_order(spec: WmmaSpec, operand: str) -> tuple[int, ...]:
    """K label of each lane-order slot ``q`` of operand ``"a"`` (length k_a) or ``"b"`` (length k_b)."""
    if operand not in ("a", "b"):
        raise ValueError(f"operand must be 'a' or 'b', not {operand!r}")
    n = spec.k_a if operand == "a" else spec.k_b
    bits = spec.a_bits if operand == "a" else spec.b_bits
    if bits == 16 and not spec.sparse:
        return tuple(_swap_bits_2_3(q) for q in range(n))
    return tuple(range(n))


def _check_matrix(name: str, mat: Sequence[Sequence[int]], rows: int, cols: int, bits: int) -> None:
    if len(mat) != rows or any(len(r) != cols for r in mat):
        raise ValueError(f"{name} must be {rows} x {cols}")
    lim = 1 << bits
    for r in mat:
        for v in r:
            if type(v) is not int or not 0 <= v < lim:
                raise ValueError(f"{name} element {v!r} is not a {bits}-bit raw value")


def check_words(name: str, words: Sequence[int], count: int) -> None:
    """Require exactly ``count`` u32 words."""
    if len(words) != count:
        raise ValueError(f"{name} needs {count} u32 words, got {len(words)}")
    for w in words:
        if type(w) is not int or not 0 <= w <= _U32:
            raise ValueError(f"{name} word {w!r} is not a u32")


def _pack_lane(elems: Sequence[int], bits: int, dwords: int) -> list[int]:
    acc = 0
    for i, v in enumerate(elems):
        acc |= v << (i * bits)
    return [(acc >> (32 * d)) & _U32 for d in range(dwords)]


def _unpack_lane(words: Sequence[int], bits: int, count: int) -> list[int]:
    mask = (1 << bits) - 1
    acc = 0
    for d, w in enumerate(words):
        acc |= w << (32 * d)
    return [(acc >> (i * bits)) & mask for i in range(count)]


def pack_a(spec: WmmaSpec, a: Sequence[Sequence[int]]) -> list[int]:
    """Pack logical A (16 x k_a raw elements) into 32 * a_dwords u32 words (lane-major)."""
    _check_matrix("A", a, 16, spec.k_a, spec.a_bits)
    order, half = slot_order(spec, "a"), spec.k_a // 2
    out: list[int] = []
    for lane in range(32):
        row, h = lane & 15, lane >> 4
        out.extend(_pack_lane([a[row][order[h * half + p]] for p in range(half)], spec.a_bits, spec.a_dwords))
    return out


def unpack_a(spec: WmmaSpec, words: Sequence[int]) -> list[list[int]]:
    """Unpack 32 * a_dwords u32 words into logical A (16 x k_a)."""
    check_words("A", words, 32 * spec.a_dwords)
    order, half = slot_order(spec, "a"), spec.k_a // 2
    a = [[0] * spec.k_a for _ in range(16)]
    for lane in range(32):
        row, h = lane & 15, lane >> 4
        base = lane * spec.a_dwords
        for p, v in enumerate(_unpack_lane(words[base : base + spec.a_dwords], spec.a_bits, half)):
            a[row][order[h * half + p]] = v
    return a


def pack_b(spec: WmmaSpec, b: Sequence[Sequence[int]]) -> list[int]:
    """Pack logical B (k_b x 16 raw elements) into 32 * b_dwords u32 words (lane-major)."""
    _check_matrix("B", b, spec.k_b, 16, spec.b_bits)
    order, half = slot_order(spec, "b"), spec.k_b // 2
    out: list[int] = []
    for lane in range(32):
        col, h = lane & 15, lane >> 4
        out.extend(_pack_lane([b[order[h * half + p]][col] for p in range(half)], spec.b_bits, spec.b_dwords))
    return out


def unpack_b(spec: WmmaSpec, words: Sequence[int]) -> list[list[int]]:
    """Unpack 32 * b_dwords u32 words into logical B (k_b x 16)."""
    check_words("B", words, 32 * spec.b_dwords)
    order, half = slot_order(spec, "b"), spec.k_b // 2
    b = [[0] * 16 for _ in range(spec.k_b)]
    for lane in range(32):
        col, h = lane & 15, lane >> 4
        base = lane * spec.b_dwords
        for p, v in enumerate(_unpack_lane(words[base : base + spec.b_dwords], spec.b_bits, half)):
            b[order[h * half + p]][col] = v
    return b


def pack_cd(spec: WmmaSpec, mat: Sequence[Sequence[int]]) -> list[int]:
    """Pack logical C/D (16 x 16 raw elements) into 32 * cd_dwords u32 words (lane-major)."""
    _check_matrix("C/D", mat, 16, 16, spec.d_bits)
    out: list[int] = []
    for lane in range(32):
        col, h = lane & 15, lane >> 4
        out.extend(_pack_lane([mat[8 * h + e][col] for e in range(8)], spec.d_bits, spec.cd_dwords))
    return out


def unpack_cd(spec: WmmaSpec, words: Sequence[int]) -> list[list[int]]:
    """Unpack 32 * cd_dwords u32 words into logical C/D (16 x 16)."""
    check_words("C/D", words, 32 * spec.cd_dwords)
    mat = [[0] * 16 for _ in range(16)]
    for lane in range(32):
        col, h = lane & 15, lane >> 4
        base = lane * spec.cd_dwords
        for e, v in enumerate(_unpack_lane(words[base : base + spec.cd_dwords], spec.d_bits, 8)):
            mat[8 * h + e][col] = v
    return mat


def _index_shift(spec: WmmaSpec, index_key: int) -> int:
    if not spec.sparse:
        raise ValueError(f"{spec.name} has no sparse index operand")
    allowed = (0, 1) if spec.has_index_key else (0,)
    if type(index_key) is not int or index_key not in allowed:
        raise ValueError(f"{spec.name}: index_key must be one of {allowed}, got {index_key!r}")
    return 16 * index_key


def pack_idx(spec: WmmaSpec, idx: Sequence[Sequence[int]], index_key: int = 0) -> list[int]:
    """Pack the logical sparse index table (16 x k_a entries, each 0..3) into 32 u32 words (1 per lane).

    K=32 forms: ``index_key`` 0 writes bits [15:0], 1 writes bits [31:16]; the other half is zero.
    K=64 form: all 32 bits; ``index_key`` must be 0.
    """
    shift = _index_shift(spec, index_key)
    _check_matrix("IDX", idx, 16, spec.k_a, 2)
    half = spec.k_a // 2
    out = [0] * 32
    for lane in range(32):
        row, h = lane & 15, lane >> 4
        w = 0
        for i, sel in enumerate(idx[row][h * half : (h + 1) * half]):
            w |= sel << (2 * i)
        out[lane] = w << shift
    return out


def unpack_idx(spec: WmmaSpec, words: Sequence[int], index_key: int = 0) -> list[list[int]]:
    """Unpack 32 u32 index words into the logical sparse index table (16 x k_a); unselected bits are ignored."""
    shift = _index_shift(spec, index_key)
    check_words("IDX", words, 32)
    half = spec.k_a // 2
    out = [[0] * spec.k_a for _ in range(16)]
    for lane in range(32):
        row, h = lane & 15, lane >> 4
        w = words[lane] >> shift
        for i in range(half):
            out[row][h * half + i] = (w >> (2 * i)) & 0x3
    return out
