"""Targeted explicit test cases for the frozen evidence (exact powers of two, built per instruction).

Each case is one logical dot product: (A row, dense B column, sparse index row or None, C word).
`pack_cases` places case t on row t / column t of a wave (16 cases per wave). Off-diagonal
elements combine rows and columns of different cases and are recorded and checked as well.
"""

from __future__ import annotations

import struct

import numpy as np

from amdgpu_matrix_model.layout import pack_a, pack_b, pack_cd, pack_idx
from amdgpu_matrix_model.model import step_schedule
from amdgpu_matrix_model.specs import FLOAT_FORMATS, WmmaSpec

Case = tuple[list[int], list[int], "list[int] | None", int]


def f32_bits(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", x))[0]


def f16_bits(x: float) -> int:
    return int(np.float16(x).view(np.uint16))


def bf16_bits(x: float) -> int:
    return f32_bits(x) >> 16


def exp_range(fmt: str) -> tuple[int, int]:
    """Smallest and largest exponent e for which 2**e is representable (subnormals included)."""
    f = FLOAT_FORMATS[fmt]
    return f.emin - f.frac_bits, f.emax


def pow2(fmt: str, e: int, neg: bool = False) -> int | None:
    """Raw bits of (-1)**neg * 2**e in `fmt`, or None if not representable."""
    f = FLOAT_FORMATS[fmt]
    if f.emin <= e <= f.emax:
        bits = (e + f.bias) << f.frac_bits
    elif f.emin - f.frac_bits <= e < f.emin:
        bits = 1 << (e - (f.emin - f.frac_bits))
    else:
        return None
    return bits | (f.sign_bit if neg else 0)


def split(spec: WmmaSpec, e: int) -> tuple[int, int] | None:
    """Exponents (ea, eb) with ea + eb == e, both representable in the A / B formats (balanced)."""
    (la, ha), (lb, hb) = exp_range(spec.a_fmt), exp_range(spec.b_fmt)
    best = None
    for ea in range(la, ha + 1):
        eb = e - ea
        if lb <= eb <= hb and (best is None or abs(ea - eb) < abs(best[0] - best[1])):
            best = (ea, eb)
    return best


def default_select(ka: int) -> int:
    """Canonical index pair (0, 1) in every group: compressed ka selects dense offset ka % 2."""
    return ka % 2


def case_from_products(spec: WmmaSpec, prods: dict[int, tuple[int, bool]], c_word: int, select=default_select) -> Case | None:
    """Case whose product at (compressed) k is (-1)**neg * 2**e for {k: (e, neg)}; None if not encodable."""
    a = [0] * spec.k_a
    b = [0] * spec.k_b
    idx = [select(ka) for ka in range(spec.k_a)] if spec.sparse else None
    for k, (e, neg) in prods.items():
        s = split(spec, e)
        if s is None:
            return None
        a[k] = pow2(spec.a_fmt, s[0], neg)
        kb = 4 * (k // 2) + idx[k] if spec.sparse else k
        b[kb] = pow2(spec.b_fmt, s[1])
    return a, b, idx, c_word


def c_pow2(spec: WmmaSpec, e: int, neg: bool = False) -> int | None:
    return pow2(spec.d_fmt, e, neg)


def pack_cases(spec: WmmaSpec, cases: list[Case], index_key: int = 0, junk_other_half: bool = False):
    """Pack cases 16 per wave; return (A, B, C, IDX) uint32 arrays of shape (W, 32, dwords)."""
    a_w, b_w, c_w, i_w = [], [], [], []
    for at in range(0, len(cases), 16):
        am = [[0] * spec.k_a for _ in range(16)]
        bm = [[0] * 16 for _ in range(spec.k_b)]
        cm = [[0] * 16 for _ in range(16)]
        im = [[default_select(ka) for ka in range(spec.k_a)] for _ in range(16)]
        for t, (av, bv, iv, cv) in enumerate(cases[at : at + 16]):
            am[t] = list(av)
            for k in range(spec.k_b):
                bm[k][t] = bv[k]
            cm[t][t] = cv
            if spec.sparse:
                im[t] = list(iv)
        a_w.append(pack_a(spec, am))
        b_w.append(pack_b(spec, bm))
        c_w.append(pack_cd(spec, cm))
        if spec.sparse:
            words = pack_idx(spec, im, index_key=index_key)
            if junk_other_half:
                words = [w | (0x4444 << (16 * (1 - index_key))) for w in words]
            i_w.append(words)
    shape = lambda v, dw: np.array(v, dtype=np.uint32).reshape(-1, 32, dw)  # noqa: E731
    return (
        shape(a_w, spec.a_dwords),
        shape(b_w, spec.b_dwords),
        shape(c_w, spec.cd_dwords),
        np.array(i_w, dtype=np.uint32).reshape(-1, 32) if spec.sparse else None,
    )


def _big(spec: WmmaSpec) -> int:
    """Largest product exponent used for cancellation: below the D overflow and encodable."""
    ha, hb = exp_range(spec.a_fmt)[1], exp_range(spec.b_fmt)[1]
    return min(ha + hb, FLOAT_FORMATS[spec.d_fmt].emax - 1, 30)


def k_order_cases(spec: WmmaSpec) -> list[Case]:
    """For every ordered pair (i, j), i != j: +P at slot i, +u at slot j, C = -P, with u = P * 2**-25.

    D == u  iff  slot i is consumed in an earlier step than slot j; D == +0 otherwise
    (same step: u is below the step's alignment grid; j first: u is lost when rounding -P + u).
    """
    e = _big(spec)
    out = []
    for i in range(spec.k_a):
        for j in range(spec.k_a):
            if i != j:
                case = case_from_products(spec, {i: (e, False), j: (e - 25, False)}, c_pow2(spec, e, True))
                assert case is not None
                out.append(case)
    return out


def _grid_levels(spec: WmmaSpec) -> tuple[int, ...]:
    return {"f16": (14, 0, -10), "f32": (30, 0, -20), "bf16": (30, 0, -20)}[spec.d_fmt]


_ARRANGE = (
    ("P", "-P", "s", "-s"),
    ("P", "s", "-P", "-s"),
    ("P", "-s", "-P", "s"),
    ("s", "-s", "P", "-P"),
    ("P", "-P", "s", "0"),
    ("P", "-P", "-s", "0"),
    ("-P", "P", "0", "-s"),
    ("P", "0", "-P", "-s"),
)


def cancel_cases(spec: WmmaSpec) -> list[Case]:
    """Exactly cancelling large products next to small ones, in the first and the last step.

    Lost or kept grid units of the small products show up because the large ones cancel.
    Distinguishes sub-group size 2 from 4 and the floor of negative sub-group sums.
    """
    steps = step_schedule(spec)
    out = []
    for step in (steps[0], steps[-1]):
        for ep in _grid_levels(spec):
            for d in (1, 12, 23, 24, 25, 30):
                for arr in _ARRANGE:
                    prods = {}
                    for slot, what in zip(step, arr):
                        if what != "0":
                            prods[slot] = (ep if "P" in what else ep - d, what.startswith("-"))
                    case = case_from_products(spec, prods, 0)
                    if case is not None:
                        out.append(case)
    return out


def tiny_acc_cases(spec: WmmaSpec) -> list[Case]:
    """C far below the products' alignment grid, with cancelling (+P, -P) or a single +P product.

    A negative C shifted completely below the grid contributes 0 or -1 grid unit; cancellation
    makes the difference visible. Grids are chosen on both sides of the D format's emin.
    """
    steps = step_schedule(spec)
    d_lo, d_hi = exp_range(spec.d_fmt)
    levels = {"f16": (2, 6, 10, 11, 16, 30), "f32": (-40, -10, 0, 30), "bf16": (-120, -100, 0, 30)}[spec.d_fmt]
    if spec.a_fmt == "bf16":  # grids at and just above emin = -126 of f32 / bf16 D
        levels += (-110, -104, -102, -101)
    out = []
    for step in (steps[0], steps[-1]):
        for ep in levels:
            grid = ep - 24
            t_exps = sorted({d_lo, d_lo + 1, FLOAT_FORMATS[spec.d_fmt].emin - 1, FLOAT_FORMATS[spec.d_fmt].emin, grid - 1, grid, grid + 1})
            for te in t_exps:
                if not d_lo <= te <= d_hi:
                    continue
                for neg in (True, False):
                    c = c_pow2(spec, te, neg)
                    for prods in ({step[0]: (ep, False), step[1]: (ep, True)}, {step[0]: (ep, False)}):
                        case = case_from_products(spec, prods, c)
                        if case is not None:
                            out.append(case)
    return out


def signed_zero_cases(spec: WmmaSpec) -> list[Case]:
    """Results that are zero or round/truncate to zero, with both signs, in every step.

    Tiny products in the last step decide the sign of a zero result directly; in an earlier step the
    zero is then consumed by later steps (no product, +1, -1, +Inf).
    """
    steps = step_schedule(spec)
    d = FLOAT_FORMATS[spec.d_fmt]
    tiny = {"f16": (-25, -26, -30), "f32": (-150, -151, -160), "bf16": (-134, -140, -149, -150, -160)}[spec.d_fmt]
    inf_a = (0x7C00 if spec.a_fmt == "f16" else 0x7F80)
    nzero = d.sign_bit
    out: list[Case] = []

    def add(prods, c, extra=None):
        case = case_from_products(spec, prods, c)
        if case is not None:
            if extra is not None:
                k, a_bits = extra
                case[0][k] = a_bits
                kb = 4 * (k // 2) + case[2][k] if spec.sparse else k
                case[1][kb] = pow2(spec.b_fmt, 0)
            out.append(case)

    for neg in (True, False):
        for e in tiny:
            for step in steps:
                for c in (0, nzero):
                    add({step[0]: (e, neg)}, c)
            first, second = steps[0][0], steps[1][0]
            add({first: (e, neg)}, 0)
            add({first: (e, neg), second: (0, False)}, 0)
            add({first: (e, neg), second: (0, True)}, 0)
            add({first: (e, neg)}, 0, extra=(second, inf_a))
            last2, last = steps[-2][0], steps[-1][0]
            add({last2: (e, neg), last: (0, not neg)}, nzero)
        for step in (steps[0], steps[-1]):
            for c in (0, nzero):
                add({step[0]: (0, neg), step[1]: (0, not neg)}, c)
            add({step[0]: (d.emin, neg)}, c_pow2(spec, d.emin, not neg))
            add({step[0]: (exp_range(spec.d_fmt)[0], neg)}, c_pow2(spec, exp_range(spec.d_fmt)[0], not neg))
    return out


def modifier_witness_cases(spec: WmmaSpec) -> list[Case]:
    """SWMMAC: one product per case, at compressed ka with dense B offset `sel` (every ka, every sel).

    A = 1, selected B = 2**(sel + 1), so D shows which B element was used and with which sign; ka and sel
    of different parity separate dense-B parity from compressed parity for the B modifiers.
    """
    out = []
    for ka in range(spec.k_a):
        for sel in range(4):
            idx = [default_select(k) for k in range(spec.k_a)]
            idx[ka] = sel
            a = [0] * spec.k_a
            b = [0] * spec.k_b
            a[ka] = pow2(spec.a_fmt, 0)
            b[4 * (ka // 2) + sel] = pow2(spec.b_fmt, sel + 1)
            out.append((a, b, idx, 0))
    return out
