"""Pure-integer, table-driven arithmetic model for gfx1201 (RDNA 4) WMMA / SWMMAC.

No host floating-point arithmetic is used in any modeled result. Public functions check their
arguments and raise ``ValueError`` for anything outside the domain documented in MODEL.md.
"""

from __future__ import annotations

import struct
from collections.abc import Sequence
from functools import lru_cache

from .layout import check_words, pack_cd, slot_order, unpack_a, unpack_b, unpack_cd, unpack_idx
from .specs import FLOAT_FORMATS, INT_FORMATS, SPECS, FloatFormat, WmmaSpec

_EMPTY_EXP = -1024  # grid of a zero product: below every reachable grid, never selected by max()
_I32_MIN = -(1 << 31)
_I32_MAX = (1 << 31) - 1


def f32_to_bits(x: float) -> int:
    """Convert a Python float to IEEE binary32 raw bits (boundary helper only)."""
    return struct.unpack("<I", struct.pack("<f", x))[0]


def bits_to_f32(bits: int) -> float:
    """Convert IEEE binary32 raw bits to a Python float (boundary helper only)."""
    return struct.unpack("<f", struct.pack("<I", bits & 0xFFFFFFFF))[0]


def decode_float(bits: int, fmt: FloatFormat) -> tuple[bool, bool, int, int, int, int]:
    """Decode raw float bits into (is_nan, is_inf, sign, sig, val_exp, align_exp).

    For finite values: mathematical value = (-1)**sign * sig * 2**val_exp.
    `align_exp` is the unbiased encoded exponent, clamped to fmt.emin for subnormals/zero.
    """
    bits &= fmt.mask
    sign = (bits >> (fmt.bits - 1)) & 1
    exp_f = (bits >> fmt.frac_bits) & fmt.exp_max
    frac = bits & fmt.frac_mask
    if exp_f == fmt.exp_max:
        if fmt.has_inf:
            if frac != 0:
                return True, False, sign, 0, 0, 0
            return False, True, sign, 0, 0, 0
        # OCP E4M3FN: only all-ones exponent+fraction is NaN; no Inf
        if frac == fmt.frac_mask:
            return True, False, sign, 0, 0, 0
    if exp_f == 0:
        return False, False, sign, frac, fmt.emin - fmt.frac_bits, fmt.emin
    sig = (1 << fmt.frac_bits) | frac
    ae = exp_f - fmt.bias
    return False, False, sign, sig, ae - fmt.frac_bits, ae


def decode_int(bits: int, width: int, signed: bool) -> int:
    """Decode raw integer bits (width 4, 8, or 32) as unsigned or two's-complement signed."""
    v = bits & ((1 << width) - 1)
    if signed and (v >> (width - 1)):
        return v - (1 << width)
    return v


def _shift_rne(mag: int, shift: int) -> int:
    """Round mag * 2**(-shift) to nearest integer, ties to even."""
    if shift <= 0:
        return mag << (-shift)
    whole = mag >> shift
    half = 1 << (shift - 1)
    rem = mag & ((1 << shift) - 1)
    if rem > half or (rem == half and (whole & 1)):
        whole += 1
    return whole


def pack_float(units: int, grid: int, fmt: FloatFormat) -> int:
    """Round the exact value ``units * 2**grid`` once (RNE) to ``fmt`` bits.

    Overflow gives the signed infinity. A result that is zero, or rounds to zero, encodes as +0.
    """
    if units == 0:
        return 0
    sign_bit = fmt.sign_bit if units < 0 else 0
    mag = -units if units < 0 else units
    top = mag.bit_length() - 1
    e = top + grid
    if e < fmt.emin:
        f = _shift_rne(mag, (fmt.emin - fmt.frac_bits) - grid)
        if f == 0:
            return 0
        return sign_bit | f  # f == 1 << frac_bits is the smallest normal, encoded by the same bits
    s = _shift_rne(mag, top - fmt.frac_bits)
    if s == (1 << (fmt.frac_bits + 1)):
        s >>= 1
        e += 1
    if e > fmt.emax:
        if fmt.has_inf:
            return sign_bit | (fmt.exp_max << fmt.frac_bits)
        return sign_bit | ((fmt.exp_max << fmt.frac_bits) | (fmt.frac_mask - 1))
    return sign_bit | ((e + fmt.bias) << fmt.frac_bits) | (s & fmt.frac_mask)


def _align_floor(sig: int, exp: int, neg: int, grid: int, is_acc: bool) -> int:
    """Align (-1)**neg * sig * 2**exp onto 2**grid by floor.

    When `is_acc` is True, an accumulator shifted completely below the grid (mag == 0)
    contributes 0 even when negative.
    """
    if sig == 0:
        return 0
    shift = exp - grid
    if shift >= 0:
        mag = sig << shift
        return -mag if neg else mag
    rsh = -shift
    mag = sig >> rsh
    if neg:
        if is_acc and mag == 0:
            return 0
        tail = (sig & ((1 << rsh) - 1)) != 0
        return -mag - (1 if tail else 0)
    return mag


def dot_float_step(
    a_words: Sequence[int],
    b_words: Sequence[int],
    acc_bits: int,
    a_fmt: FloatFormat,
    b_fmt: FloatFormat,
    d_fmt: FloatFormat,
    subgroup: int,
    invalid_is_sig: bool,
) -> int:
    """One gfx1201 float dot step: products in consecutive sub-groups of `subgroup`, plus acc, to d_fmt.

    - f32 D (f16/bf16): subgroup=2, invalid_is_sig=False, pack RNE -> f32
    - f16 D (f16):      subgroup=2, invalid_is_sig=False, pack single RNE -> f16
    - bf16 D (bf16):    subgroup=2, invalid_is_sig=False, pack RNE -> f32 then keep the high 16 bits
    - f32 D (fp8/bf8):  subgroup=4, invalid_is_sig=True,  pack RNE -> f32
    """
    invalid_nan = d_fmt.qnan_sig if invalid_is_sig else d_fmt.qnan_default
    pinf = False
    ninf = False
    invalid = False
    prods: list[tuple[int, int, int]] = []
    grids: list[int] = []

    for aw, bw in zip(a_words, b_words):
        an, ai, asg, asig, ave, aae = decode_float(aw, a_fmt)
        bn, bi, bsg, bsig, bve, bae = decode_float(bw, b_fmt)
        if an or bn:
            return d_fmt.qnan_sig
        if (ai and not bi and bsig == 0) or (bi and not ai and asig == 0):
            invalid = True
        if ai or bi:
            if asg ^ bsg:
                ninf = True
            else:
                pinf = True
        psig = asig * bsig
        pve = ave + bve
        pneg = asg ^ bsg
        prods.append((psig, pve, pneg))
        grids.append(aae + bae - 24 if psig != 0 else _EMPTY_EXP)

    if invalid or (pinf and ninf):
        return invalid_nan

    cn, ci, csg, csig, cve, _ = decode_float(acc_bits, d_fmt)
    if cn:
        return (acc_bits & d_fmt.mask) | d_fmt.quiet_bit
    if ci:
        if csg:
            ninf = True
        else:
            pinf = True
    if pinf or ninf:
        if pinf and ninf:
            return invalid_nan
        pos_inf = d_fmt.exp_max << d_fmt.frac_bits
        return (d_fmt.sign_bit | pos_inf) if ninf else pos_inf

    p_grid = max(grids)
    c_lead = (cve + csig.bit_length() - 1) if csig != 0 else _EMPTY_EXP
    c_grid = max(c_lead, d_fmt.emin) - 26
    grid = p_grid if p_grid > c_grid else c_grid

    total = _align_floor(csig, cve, csg, grid, True)
    n = len(prods)
    for base in range(0, n, subgroup):
        sg = max(grids[base : base + subgroup])
        if sg == _EMPTY_EXP:
            continue  # all-zero sub-group contributes 0
        sub_sum = 0
        for idx in range(base, base + subgroup):
            psig, pve, pneg = prods[idx]
            if psig != 0:
                sub_sum += _align_floor(psig, pve, pneg, sg, False)
        if sub_sum != 0:
            total += _align_floor(-sub_sum if sub_sum < 0 else sub_sum, sg, 1 if sub_sum < 0 else 0, grid, False)

    if d_fmt.name == "bf16":
        # Hardware rounds to f32 (RNE) and keeps the high 16 bits of that f32 word
        return (pack_float(total, grid, FLOAT_FORMATS["f32"]) >> 16) & 0xFFFF
    return pack_float(total, grid, d_fmt)


@lru_cache(maxsize=None)
def step_schedule(spec: WmmaSpec) -> tuple[tuple[int, ...], ...]:
    """Ordered steps of the dot product, each a tuple of (compressed) K indices into A and selected B.

    Physical rule (all forms): one step per A VGPR pair and lane half, in the order
    (pair 0, half 0), (pair 0, half 1), (pair 1, half 0), (pair 1, half 1). Each A VGPR of a
    float step is one sub-group. Integer forms have two halves: lane half 0, then lane half 1.
    """
    order = slot_order(spec, "a")
    n_half = spec.k_a // 2
    if spec.is_int:
        return (tuple(order[:n_half]), tuple(order[n_half:]))
    per_pair = 2 * (32 // spec.a_bits)
    steps = []
    for pair in range(spec.a_dwords // 2):
        for h in (0, 1):
            q0 = h * n_half + pair * per_pair
            steps.append(tuple(order[q0 : q0 + per_pair]))
    return tuple(steps)


def _subgroup(spec: WmmaSpec) -> int:
    return 32 // spec.a_bits


def select_sparse_b(b_vec: Sequence[int], idx_vec: Sequence[int]) -> list[int]:
    """Select K/2 active B elements from dense B (length K) using 2-bit per-element indices."""
    return [b_vec[4 * (ka // 2) + (idx_vec[ka] & 0x3)] for ka in range(len(idx_vec))]


def _spec(spec_name: str) -> WmmaSpec:
    spec = SPECS.get(spec_name)
    if spec is None:
        raise ValueError(f"unknown instruction {spec_name!r}")
    return spec


def _bits3(name: str, v: Sequence[int]) -> tuple[int, int, int]:
    t = tuple(v)
    if len(t) != 3 or any(type(x) is not int or x not in (0, 1) for x in t):
        raise ValueError(f"{name} must be three 0/1 integers, got {v!r}")
    return t  # type: ignore[return-value]


def check_modifiers(
    spec: WmmaSpec, neg_lo: Sequence[int], neg_hi: Sequence[int], clamp: bool
) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """Validate the NEG / NEG_HI / CLAMP fields for `spec`; return them as tuples.

    Rejected (ValueError): fields the ISA requires to be zero (integer NEG[2] and NEG_HI, fp8/bf8 A/B NEG),
    and NEG[2]/NEG_HI[2] on SWMMAC, where SRC2 is the sparse index and llvm-mc rejects the field.
    """
    lo, hi = _bits3("neg_lo", neg_lo), _bits3("neg_hi", neg_hi)
    if type(clamp) is not bool:
        raise ValueError(f"clamp must be a bool, got {clamp!r}")
    if spec.is_int:
        if lo[2] or any(hi):
            raise ValueError(f"{spec.name}: integer forms take only neg_lo[0:2] (signedness)")
    elif spec.is_fp8_family and (lo[0] or lo[1] or hi[0] or hi[1]):
        raise ValueError(f"{spec.name}: fp8/bf8 A/B NEG fields must be zero")
    if spec.sparse and (lo[2] or hi[2]):
        raise ValueError(f"{spec.name}: SWMMAC has no C modifier (SRC2 is the sparse index)")
    return lo, hi


def _dot(
    spec: WmmaSpec,
    a_vec: Sequence[int],
    b_vec: Sequence[int],
    c_word: int,
    idx_vec: Sequence[int] | None,
    neg_lo: tuple[int, int, int],
    neg_hi: tuple[int, int, int],
    clamp: bool,
) -> int:
    """One logical D[row, col] from checked operands."""
    b_eff: Sequence[int] = select_sparse_b(b_vec, idx_vec) if spec.sparse else b_vec  # type: ignore[arg-type]
    steps = step_schedule(spec)

    if spec.is_int:
        a_w = INT_FORMATS[spec.a_fmt].bits
        b_w = INT_FORMATS[spec.b_fmt].bits
        sa, sb = bool(neg_lo[0]), bool(neg_lo[1])
        acc = decode_int(c_word, 32, signed=True)
        for step in steps:
            acc += sum(decode_int(a_vec[k], a_w, sa) * decode_int(b_eff[k], b_w, sb) for k in step)
            if clamp:
                acc = _I32_MAX if acc > _I32_MAX else (_I32_MIN if acc < _I32_MIN else acc)
        return acc & 0xFFFFFFFF

    a_fmt = FLOAT_FORMATS[spec.a_fmt]
    b_fmt = FLOAT_FORMATS[spec.b_fmt]
    d_fmt = FLOAT_FORMATS[spec.d_fmt]

    if not spec.is_fp8_family:
        sign_a = a_fmt.sign_bit
        sign_b = b_fmt.sign_bit
        a_mod = [(w ^ sign_a) if (neg_lo[0] if (k & 1) == 0 else neg_hi[0]) else w for k, w in enumerate(a_vec)]
        b_mod = [(w ^ sign_b) if (neg_lo[1] if (k & 1) == 0 else neg_hi[1]) else w for k, w in enumerate(b_eff)]
    else:
        a_mod = list(a_vec)
        b_mod = list(b_eff)

    acc = c_word
    if not spec.is_fp8_family:
        if neg_hi[2]:
            acc &= ~d_fmt.sign_bit & d_fmt.mask
        if neg_lo[2]:
            acc ^= d_fmt.sign_bit

    subgroup = _subgroup(spec)
    for step in steps:
        acc = dot_float_step(
            [a_mod[k] for k in step],
            [b_mod[k] for k in step],
            acc,
            a_fmt,
            b_fmt,
            d_fmt,
            subgroup,
            spec.is_fp8_family,
        )
    return acc


def _check_vec(name: str, v: Sequence[int], count: int, bits: int) -> None:
    if len(v) != count:
        raise ValueError(f"{name} needs {count} elements, got {len(v)}")
    lim = 1 << bits
    for x in v:
        if type(x) is not int or not 0 <= x < lim:
            raise ValueError(f"{name} element {x!r} is not a {bits}-bit raw value")


def wmma_dot(
    spec_name: str,
    a_vec: Sequence[int],
    b_vec: Sequence[int],
    c_word: int,
    *,
    idx_vec: Sequence[int] | None = None,
    neg_lo: Sequence[int] = (0, 0, 0),
    neg_hi: Sequence[int] = (0, 0, 0),
    clamp: bool = False,
) -> int:
    """Compute one logical D[row, col] raw word from logical A[row, :], B[:, col], C[row, col].

    `a_vec` has length k_a (compressed K/2 for sparse forms), `b_vec` length K, and for sparse forms
    `idx_vec` length k_a with entries 0..3 (offset inside each group of 4 B elements).
    """
    spec = _spec(spec_name)
    lo, hi = check_modifiers(spec, neg_lo, neg_hi, clamp)
    _check_vec("a_vec", a_vec, spec.k_a, spec.a_bits)
    _check_vec("b_vec", b_vec, spec.k_b, spec.b_bits)
    _check_vec("c_word", [c_word], 1, spec.d_bits)
    if spec.sparse:
        if idx_vec is None:
            raise ValueError(f"{spec_name} requires idx_vec")
        _check_vec("idx_vec", idx_vec, spec.k_a, 2)
    elif idx_vec is not None:
        raise ValueError(f"{spec_name} takes no idx_vec")
    return _dot(spec, a_vec, b_vec, c_word, idx_vec, lo, hi, clamp)


def wmma_matrix(
    spec_name: str,
    a: Sequence[Sequence[int]],
    b: Sequence[Sequence[int]],
    c: Sequence[Sequence[int]],
    *,
    idx: Sequence[Sequence[int]] | None = None,
    neg_lo: Sequence[int] = (0, 0, 0),
    neg_hi: Sequence[int] = (0, 0, 0),
    clamp: bool = False,
) -> list[list[int]]:
    """Compute a 16x16 logical D matrix of raw words (A 16 x k_a, B K x 16, C 16 x 16, IDX 16 x k_a)."""
    spec = _spec(spec_name)
    lo, hi = check_modifiers(spec, neg_lo, neg_hi, clamp)
    if len(a) != 16 or len(c) != 16 or len(b) != spec.k_b:
        raise ValueError(f"{spec_name}: A must have 16 rows, B {spec.k_b} rows, C 16 rows")
    for i in range(16):
        _check_vec(f"A[{i}]", a[i], spec.k_a, spec.a_bits)
        _check_vec(f"C[{i}]", c[i], 16, spec.d_bits)
    for k in range(spec.k_b):
        _check_vec(f"B[{k}]", b[k], 16, spec.b_bits)
    if spec.sparse:
        if idx is None or len(idx) != 16:
            raise ValueError(f"{spec_name} requires a 16-row idx table")
        for i in range(16):
            _check_vec(f"IDX[{i}]", idx[i], spec.k_a, 2)
    elif idx is not None:
        raise ValueError(f"{spec_name} takes no idx table")
    return _matrix(spec, a, b, c, idx, lo, hi, clamp)


def _matrix(spec, a, b, c, idx, lo, hi, clamp) -> list[list[int]]:
    b_cols = [[b[k][j] for k in range(spec.k_b)] for j in range(16)]
    return [
        [_dot(spec, a[i], b_cols[j], c[i][j], idx[i] if idx is not None else None, lo, hi, clamp) for j in range(16)]
        for i in range(16)
    ]


def _check_state(spec: WmmaSpec, op_sel: int, op_sel_hi: int, mode: int, exec_lo: int) -> None:
    for name, v, lim in (("op_sel", op_sel, 7), ("op_sel_hi", op_sel_hi, 7), ("mode", mode, 0xFFFF), ("exec_lo", exec_lo, 0xFFFF_FFFF)):
        if type(v) is not int or not 0 <= v <= lim:
            raise ValueError(f"{name} must be an integer in 0..{lim:#x}, got {v!r}")
    if (op_sel, op_sel_hi) != (0, 7) and (spec.sparse or spec.is_int):
        raise ValueError(f"{spec.name}: raw OPSEL/OPSEL_HI are modeled only on dense float forms")


def wmma_wave(
    spec_name: str,
    a_words: Sequence[int],
    b_words: Sequence[int],
    c_words: Sequence[int],
    *,
    idx_words: Sequence[int] | None = None,
    index_key: int = 0,
    neg_lo: Sequence[int] = (0, 0, 0),
    neg_hi: Sequence[int] = (0, 0, 0),
    clamp: bool = False,
    op_sel: int = 0,
    op_sel_hi: int = 0b111,
    mode: int = 0x00F0,
    exec_lo: int = 0xFFFF_FFFF,
) -> list[int]:
    """Execute one wave32 WMMA / SWMMAC on lane-major u32 register words; return the 32 * cd_dwords D words.

    State fields: `clamp` is the VOP3P CLAMP bit (integer saturation; no effect on float forms),
    `op_sel` / `op_sel_hi` the raw VOP3P OPSEL fields (no effect on dense float forms), `mode` the value
    written to HW_REG_WAVE_MODE[15:0] before the instruction, and `exec_lo` the EXEC mask at issue.
    MODE and EXEC do not change the result: the instruction reads and writes all 32 lanes.
    """
    spec = _spec(spec_name)
    lo, hi = check_modifiers(spec, neg_lo, neg_hi, clamp)
    _check_state(spec, op_sel, op_sel_hi, mode, exec_lo)
    a_mat = unpack_a(spec, a_words)
    b_mat = unpack_b(spec, b_words)
    c_mat = unpack_cd(spec, c_words)
    if spec.sparse:
        if idx_words is None:
            raise ValueError(f"{spec_name} requires idx_words")
        idx_mat = unpack_idx(spec, idx_words, index_key=index_key)
    else:
        if idx_words is not None or index_key != 0:
            raise ValueError(f"{spec_name} takes no idx_words or index_key")
        idx_mat = None
    return pack_cd(spec, _matrix(spec, a_mat, b_mat, c_mat, idx_mat, lo, hi, clamp))


__all__ = [
    "bits_to_f32",
    "check_modifiers",
    "decode_float",
    "decode_int",
    "dot_float_step",
    "f32_to_bits",
    "pack_float",
    "select_sparse_b",
    "step_schedule",
    "wmma_dot",
    "wmma_matrix",
    "wmma_wave",
]
