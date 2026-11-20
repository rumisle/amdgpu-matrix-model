"""Pure-integer, table-driven arithmetic model for gfx1201 (RDNA 4) WMMA / SWMMAC.

No host floating-point arithmetic is used in any modeled result.
"""

from __future__ import annotations

import struct
from collections.abc import Sequence

from .layout import pack_cd, unpack_a, unpack_b, unpack_cd, unpack_idx
from .specs import FLOAT_FORMATS, INT_FORMATS, SPECS, FloatFormat, WmmaSpec

_EMPTY_EXP = -1024
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


def pack_float(units: int, grid: int, fmt: FloatFormat, neg_zero: bool = False) -> int:
    """Round exact signed integer `units * 2**grid` once (RNE) to `fmt` bits."""
    if units == 0:
        return fmt.sign_bit if neg_zero else 0
    sign_bit = fmt.sign_bit if units < 0 else 0
    mag = -units if units < 0 else units
    top = mag.bit_length() - 1
    e = top + grid
    if e < fmt.emin:
        f = _shift_rne(mag, (fmt.emin - fmt.frac_bits) - grid)
        if f == 0:
            return sign_bit if neg_zero else 0
        if f == (1 << fmt.frac_bits):
            return sign_bit | (1 << fmt.frac_bits)
        return sign_bit | f
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
    """One gfx1201 float dot step (2 sub-groups of `subgroup` products + acc -> d_fmt).

    - f32 D (f16/bf16): subgroup=2, invalid_is_sig=False, pack RNE -> f32
    - f16 D (f16):      subgroup=2, invalid_is_sig=False, pack single RNE -> f16
    - bf16 D (bf16):    subgroup=2, invalid_is_sig=False, pack RNE -> f32 then truncate high 16 bits
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
            continue
        sub_sum = 0
        for idx in range(base, base + subgroup):
            psig, pve, pneg = prods[idx]
            if psig != 0:
                sub_sum += _align_floor(psig, pve, pneg, sg, False)
        if sub_sum != 0:
            total += _align_floor(-sub_sum if sub_sum < 0 else sub_sum, sg, 1 if sub_sum < 0 else 0, grid, False)

    if d_fmt.name == "bf16":
        # Hardware rounds to f32 (RNE) and truncates the low 16 bits to bf16
        f32_word = pack_float(total, grid, FLOAT_FORMATS["f32"], neg_zero=False)
        return (f32_word >> 16) & 0xFFFF
    return pack_float(total, grid, d_fmt, neg_zero=False)


# Float family step schedule: (step_starts, step_size, subgroup_size, invalid_is_sig)
_FLOAT_SCHED: dict[bool, tuple[tuple[int, ...], int, int, bool]] = {
    False: ((0, 8, 4, 12), 4, 2, False),  # f16 / bf16
    True: ((0, 8), 8, 4, True),  # fp8 / bf8
}


def select_sparse_b(b_vec: Sequence[int], idx_vec: Sequence[int]) -> list[int]:
    """Select K/2 active B elements from dense B (length K) using 2-bit per-element indices."""
    return [b_vec[4 * (ka // 2) + (idx_vec[ka] & 0x3)] for ka in range(len(idx_vec))]


def wmma_dot(
    spec_name: str,
    a_vec: Sequence[int],
    b_vec: Sequence[int],
    c_word: int,
    *,
    idx_vec: Sequence[int] | None = None,
    neg_lo: tuple[int, int, int] = (0, 0, 0),
    neg_hi: tuple[int, int, int] = (0, 0, 0),
    clamp: bool = False,
) -> int:
    """Compute one logical D[row, col] raw word from logical A[row, :], B[:, col], C[row, col].

    For sparse `v_swmmac_*` instructions, `a_vec` has compressed length K/2, `b_vec` has
    dense length K, and `idx_vec` has length K/2 with 2-bit offsets (0..3) per group of 4.
    """
    spec = SPECS[spec_name]
    if spec.sparse:
        if idx_vec is None:
            raise ValueError(f"{spec_name} requires idx_vec")
        b_eff: Sequence[int] = select_sparse_b(b_vec, idx_vec)
    else:
        b_eff = b_vec

    if spec.is_int:
        a_w = INT_FORMATS[spec.a_fmt].bits
        b_w = INT_FORMATS[spec.b_fmt].bits
        sa, sb = bool(neg_lo[0]), bool(neg_lo[1])
        acc = decode_int(c_word, 32, signed=True)
        half = len(a_vec) // 2
        for start in (0, half):
            part = 0
            for k in range(start, start + half):
                part += decode_int(a_vec[k], a_w, sa) * decode_int(b_eff[k], b_w, sb)
            acc += part
            if clamp:
                if acc > _I32_MAX:
                    acc = _I32_MAX
                elif acc < _I32_MIN:
                    acc = _I32_MIN
        return acc & 0xFFFFFFFF

    a_fmt = FLOAT_FORMATS[spec.a_fmt]
    b_fmt = FLOAT_FORMATS[spec.b_fmt]
    d_fmt = FLOAT_FORMATS[spec.d_fmt]

    if not spec.is_fp8_family:
        sign_a = a_fmt.sign_bit
        sign_b = b_fmt.sign_bit
        a_mod = [
            (w ^ sign_a) if (neg_lo[0] if (k & 1) == 0 else neg_hi[0]) else w
            for k, w in enumerate(a_vec)
        ]
        b_mod = [
            (w ^ sign_b) if (neg_lo[1] if (k & 1) == 0 else neg_hi[1]) else w
            for k, w in enumerate(b_eff)
        ]
    else:
        a_mod = list(a_vec)
        b_mod = list(b_eff)

    acc = c_word & d_fmt.mask
    if not spec.is_fp8_family:
        if neg_hi[2]:
            acc &= ~d_fmt.sign_bit & d_fmt.mask
        if neg_lo[2]:
            acc ^= d_fmt.sign_bit

    starts, step_size, subgroup, invalid_is_sig = _FLOAT_SCHED[spec.is_fp8_family]
    for s0 in starts:
        acc = dot_float_step(
            a_mod[s0 : s0 + step_size],
            b_mod[s0 : s0 + step_size],
            acc,
            a_fmt,
            b_fmt,
            d_fmt,
            subgroup,
            invalid_is_sig,
        )
    return acc


def wmma_matrix(
    spec_name: str,
    a: Sequence[Sequence[int]],
    b: Sequence[Sequence[int]],
    c: Sequence[Sequence[int]],
    *,
    idx: Sequence[Sequence[int]] | None = None,
    neg_lo: tuple[int, int, int] = (0, 0, 0),
    neg_hi: tuple[int, int, int] = (0, 0, 0),
    clamp: bool = False,
) -> list[list[int]]:
    """Compute a 16x16 logical D matrix of raw words."""
    k_b = len(b)
    b_cols = [[b[k][j] for k in range(k_b)] for j in range(16)]
    return [
        [
            wmma_dot(
                spec_name,
                a[i],
                b_cols[j],
                c[i][j],
                idx_vec=(idx[i] if idx is not None else None),
                neg_lo=neg_lo,
                neg_hi=neg_hi,
                clamp=clamp,
            )
            for j in range(16)
        ]
        for i in range(16)
    ]


def wmma_wave(
    spec_name: str,
    a_words: Sequence[int],
    b_words: Sequence[int],
    c_words: Sequence[int],
    *,
    idx_words: Sequence[int] | None = None,
    index_key: int = 0,
    neg_lo: tuple[int, int, int] = (0, 0, 0),
    neg_hi: tuple[int, int, int] = (0, 0, 0),
    clamp: bool = False,
) -> list[int]:
    """Simulate one wave32 WMMA / SWMMAC instruction directly on lane-major u32 register words."""
    spec = SPECS[spec_name]
    a_mat = unpack_a(spec, a_words)
    b_mat = unpack_b(spec, b_words)
    c_mat = unpack_cd(spec, c_words)
    idx_mat = unpack_idx(spec, idx_words, index_key=index_key) if (spec.sparse and idx_words is not None) else None
    d_mat = wmma_matrix(
        spec_name,
        a_mat,
        b_mat,
        c_mat,
        idx=idx_mat,
        neg_lo=neg_lo,
        neg_hi=neg_hi,
        clamp=clamp,
    )
    return pack_cd(spec, d_mat)

