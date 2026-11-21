# gfx1201 (RDNA 4) WMMA & SWMMAC Arithmetic Specification

This document specifies the arithmetic of all 22 wave32 matrix instructions (11 `v_wmma_*`, 11 `v_swmmac_*`) on AMD `gfx1201`, mapping raw operand bits to raw destination bits. Every rule below either reproduces the frozen hardware evidence ([EVIDENCE.md](EVIDENCE.md)) or is marked as an API decision; section 10 lists what each rule rests on. "Bit-exact" means: the model reproduces every frozen hardware output; it is not a proof for unmeasured inputs.

`amdgpu_matrix_model` implements this specification with Python integer arithmetic; host floating point never participates in a result.

## 1. Instruction Table

Each instruction computes one `16 x 16` tile `D = A * B + C` across a 32-lane wave. `K` is the dense depth, `k_a` the number of stored A elements per row (`K / 2` for sparse forms), `k_b = K`.

| Instruction | A | B | C / D | K | `k_a` | A / B / CD dwords per lane | Steps |
|---|---|---|---|---:|---:|---|---|
| `v_wmma_f32_16x16x16_f16` | `f16` | `f16` | `f32` | 16 | 16 | 4 / 4 / 8 | 4 x 4, RNE to `f32` |
| `v_wmma_f32_16x16x16_bf16` | `bf16` | `bf16` | `f32` | 16 | 16 | 4 / 4 / 8 | 4 x 4, RNE to `f32` |
| `v_wmma_f16_16x16x16_f16` | `f16` | `f16` | `f16` | 16 | 16 | 4 / 4 / 4 | 4 x 4, one RNE to `f16` |
| `v_wmma_bf16_16x16x16_bf16` | `bf16` | `bf16` | `bf16` | 16 | 16 | 4 / 4 / 4 | 4 x 4, RNE to `f32`, high half |
| `v_wmma_f32_16x16x16_{fp8,bf8}_{fp8,bf8}` (4) | `fp8`/`bf8` | `fp8`/`bf8` | `f32` | 16 | 16 | 2 / 2 / 8 | 2 x 8, RNE to `f32` |
| `v_wmma_i32_16x16x16_iu8` | `iu8` | `iu8` | `i32` | 16 | 16 | 2 / 2 / 8 | 2 halves, wrap or `clamp` |
| `v_wmma_i32_16x16x16_iu4` | `iu4` | `iu4` | `i32` | 16 | 16 | 1 / 1 / 8 | 2 halves, wrap or `clamp` |
| `v_wmma_i32_16x16x32_iu4` | `iu4` | `iu4` | `i32` | 32 | 32 | 2 / 2 / 8 | 2 halves, wrap or `clamp` |
| `v_swmmac_f32_16x16x32_{f16,bf16}` | `f16`/`bf16` | same | `f32` | 32 | 16 | 4 / 8 / 8 | sparse 2:4, 4 x 4 |
| `v_swmmac_f16_16x16x32_f16` | `f16` | `f16` | `f16` | 32 | 16 | 4 / 8 / 4 | sparse 2:4, 4 x 4 |
| `v_swmmac_bf16_16x16x32_bf16` | `bf16` | `bf16` | `bf16` | 32 | 16 | 4 / 8 / 4 | sparse 2:4, 4 x 4 |
| `v_swmmac_f32_16x16x32_{fp8,bf8}_{fp8,bf8}` (4) | `fp8`/`bf8` | `fp8`/`bf8` | `f32` | 32 | 16 | 2 / 4 / 8 | sparse 2:4, 2 x 8 |
| `v_swmmac_i32_16x16x32_iu8` | `iu8` | `iu8` | `i32` | 32 | 16 | 2 / 4 / 8 | sparse 2:4, 2 halves |
| `v_swmmac_i32_16x16x32_iu4` | `iu4` | `iu4` | `i32` | 32 | 16 | 1 / 2 / 8 | sparse 2:4, 2 halves |
| `v_swmmac_i32_16x16x64_iu4` | `iu4` | `iu4` | `i32` | 64 | 32 | 2 / 4 / 8 | sparse 2:4 (32-bit index), 2 halves |

K=32 SWMMAC forms take `index_key` 0 or 1; the K=64 form uses the whole index VGPR.

## 2. Format Decoding

Every finite floating-point word decodes to `(-1)**sign * sig * 2**val_exp` plus an alignment exponent `align_exp`, clamped to `emin = 1 - bias` for subnormals and zero:

- Normal (`exp_field != 0`): `sig = (1 << frac_bits) | frac`, `align_exp = exp_field - bias`, `val_exp = align_exp - frac_bits`.
- Subnormal / zero (`exp_field == 0`): `sig = frac`, `align_exp = emin`, `val_exp = emin - frac_bits`.

| Format | Bits | S/E/F | Bias | `emin` | `emax` | Inf | NaN | D NaNs (`default` / `sig` / `quiet_bit`) |
|---|---:|---|---:|---:|---:|---|---|---|
| `f32` | 32 | 1/8/23 | 127 | -126 | 127 | `exp=255, frac=0` | `exp=255, frac!=0` | `0xFFC00000` / `0xFFC00A3D` / `0x00400000` |
| `f16` | 16 | 1/5/10 | 15 | -14 | 15 | `exp=31, frac=0` | `exp=31, frac!=0` | `0xFE00` / `0xFE3D` / `0x0200` |
| `bf16` | 16 | 1/8/7 | 127 | -126 | 127 | `exp=255, frac=0` | `exp=255, frac!=0` | `0xFFC0` / `0xFFFD` / `0x0040` |
| `fp8` (OCP E4M3FN) | 8 | 1/4/3 | 7 | -6 | 8 | none | `0x7F`, `0xFF` | — |
| `bf8` (OCP E5M2) | 8 | 1/5/2 | 15 | -14 | 15 | `0x7C`, `0xFC` | `exp=31, frac!=0` | — |

`align_exp` differs from `floor(log2(|value|))` for subnormal inputs, and the difference is observable: `bf8` subnormal `2^-15` times `2^15` aligns with `align_exp_sum = -14 + 15 = +1`, one bit above normal `1.0 * 1.0`.

## 3. Floating-Point Accumulation

### 3.1 Steps

The dot product runs in steps. Physically, each step consumes one pair of A VGPRs from one lane half, in the order (pair 0, half 0), (pair 0, half 1), (pair 1, half 0), (pair 1, half 1); each A VGPR of a step is one sub-group. B and the sparse index follow A. In the K labels of section 7 this is:

| Family | Steps (K or compressed `ka` indices) | Products per step / sub-group | `invalid_nan` | D packing per step |
|---|---|---|---|---|
| dense `f16`/`bf16` | `0..3`, `4..7`, `8..11`, `12..15` | 4 / 2 | `qnan_default` | see 3.3 |
| sparse `f16`/`bf16` | `0..3`, `8..11`, `4..7`, `12..15` | 4 / 2 | `qnan_default` | see 3.3 |
| `fp8`/`bf8` (dense and sparse) | `0..7`, `8..15` | 8 / 4 | `qnan_sig` of `f32` | RNE to `f32` |

The step order was measured directly (`k_order` chunks: for every ordered pair of slots, a result that is `+u` exactly when the first slot is accumulated earlier; `tools/k_order.py` derives the table above from the frozen outputs).

### 3.2 One finite step

```text
step(products p[0..n-1] with grids g[i], accumulator acc, D format d):
    # p[i] = (-1)^(sa^sb) * sig_a*sig_b * 2^(val_exp_a + val_exp_b), exact
    # g[i] = align_exp_a + align_exp_b - 24 if p[i] != 0, else "empty"
    p_grid = max of the non-empty g[i] (empty if all are empty)
    c_lead = val_exp(acc) + bit_length(sig(acc)) - 1 if acc != 0, else empty
    c_grid = max(c_lead, d.emin) - 26            # c_lead empty: d.emin - 26
    grid   = max(p_grid, c_grid)
    total  = floor_acc(acc, grid)
    for each sub-group S (consecutive, size n/2):
        if all p[i] in S are zero: continue      # contributes 0
        sg  = max(g[i] for nonzero p[i] in S)
        sum = sum(floor(p[i] / 2^sg) for nonzero p[i] in S)
        total += floor(sum * 2^sg / 2^grid)
    return pack(total * 2^grid, d)
```

`floor` is the mathematical floor (toward -inf). `floor_acc` is the floor too, with one exception: a negative accumulator whose magnitude is below one grid unit (`floor(|acc| / 2^grid) == 0`) contributes `0` instead of `-1` **while `grid > d.emin`**; at or below `d.emin` (-14 for `f16` D, -126 for `f32` and `bf16` D) it contributes `-1`. Both sides of the boundary are measured (`tiny_acc` chunks: `f16` D grids -14 and -13; bf16 products at grids -126 and -125). `f16` inputs cannot reach a grid of -126, so for `f32` D with `f16` inputs only the first case occurs.

The `f16` D accumulator grid floor `2^-40` follows from `c_grid` (`2^-25 + 2^-40 -> 0x0001`, `2^-25 + 2^-41 -> 0x0000`). The implementation represents an empty grid by `-1024`, below every reachable grid.

### 3.3 Packing and signed zeros

- `f32` D and `f16` D: one IEEE round-to-nearest-even of `total * 2^grid` directly to `d` (subnormals kept, overflow to `±Inf`). `f16` D does not round through `f32`.
- `bf16` D: one RNE to `f32`, then the high 16 bits of that `f32` word.
- A result that is exactly zero or rounds to zero is `+0` in `f32` and `f16` D, including `-0 + -0` and negative underflow.
- `bf16` D can be `-0` (`0x8000`): a negative `f32` result of magnitude below `2^-133` (the smallest `bf16` subnormal) keeps its sign when truncated. Example: product `-2^-134` (`A = 0x8001`, `B = 0x3F00`) in the last step gives `0x8000`. A later step reads `0x8000` as zero; if that step adds nothing the result is `+0`. Measured in `signed_zero` chunks (final step and earlier steps, both signs).

## 4. Floating-Point Special Values

Evaluated per step in this precedence:

1. Any NaN in `A` or `B` returns `d.qnan_sig` (`0xFFC00A3D` `f32`, `0xFE3D` `f16`, `0xFFFD` `bf16`).
2. An `Inf * 0` product (`Inf * Inf` is a valid `±Inf`), or both `+Inf` and `-Inf` products in the step, returns `invalid_nan`.
3. A NaN accumulator returns `acc | d.quiet_bit` (sign and payload kept).
4. Product infinities combine with an infinite accumulator: opposite signs return `invalid_nan`, otherwise the surviving `±Inf`.

Steps run in order, so a later step's NaN or invalid operation replaces an earlier result, and an earlier overflow enters later steps as `±Inf`.

## 5. Modifiers (`neg_lo`, `neg_hi`, `clamp`)

- Dense `f16`/`bf16`: `neg_lo[0]` flips the sign of A elements with even K, `neg_hi[0]` odd K (K parity is the 16-bit half of the register, section 7); `[1]` does the same for B. `neg_hi[2]` clears C's sign (`abs`), then `neg_lo[2]` flips it (`neg`), for `f32`, `f16` and `bf16` C including NaNs.
- Sparse `f16`/`bf16`: A by compressed `ka` parity (its register half). B modifiers apply to the selected B element by the parity of the compressed `ka` it is used for, after selection, not by the parity of its dense position: `idx = 1` moves dense `B[1]` into `ka = 0`, where `neg_lo[1]` negates it. Measured with both index keys (`ik*_mod_witness_*`, `ik*_mod_*` chunks). This differs from a literal reading of ISA 7.12 ("NEG[1:0] is applied on SRC1 and SRC0's low 16bit").
- `fp8`/`bf8`: C modifiers are ignored on dense forms (measured). A/B `NEG` must be zero (ISA 7.12); the API rejects them.
- `iu8`/`iu4`: `neg_lo[0]`, `neg_lo[1]` select signed (1) or unsigned (0) A and B; `NEG[2]` and `NEG_HI` must be zero (ISA) and are rejected.
- SWMMAC has no C modifier: SRC2 is the sparse index (ISA Table 42) and llvm-mc rejects `neg_lo[2]`/`neg_hi[2]`; the API rejects them.
- `clamp` saturates integer forms (section 6). On float forms the raw VOP3P CLAMP bit has no effect (measured on all 16 float forms; llvm-mc rejects `clamp` there).

## 6. Integer Arithmetic and Two-Half Saturation

A and B decode as unsigned or two's-complement (`4` or `8` bits) per `neg_lo[0:2]`, C as signed `i32`. The exact sum accumulates in two halves over compressed K (lane half 0: `0 .. k_a/2 - 1`, then lane half 1):

- without `clamp`: the result wraps modulo `2^32`;
- with `clamp`: the exact accumulator is clamped to `[-2^31, 2^31 - 1]` after each half, so first-half saturation persists (`INT32_MAX + 100` then `-100` gives `INT32_MAX - 100`).

## 7. Wave32 Layout and Sparse Selection

Lane `l = 16 h + r` (`h` the lane half, `r = l % 16`). Elements are packed LSB-first within each lane's words; `p` is the element position in the lane.

- A (`16 x k_a`): row `r`. B (`k_b x 16`): column `r`.
  - Dense 16-bit A and B follow ISA section 7.12.2 (wave32, 16-bit A 16x16: "lane `{ col[2], row[3:0] }`, vgpr `{ col[3], col[1] }`, startPosn `col[0]`"; B the same with `row` as K): `h = k[2]`, `p = 4 k[3] + 2 k[1] + k[0]`. For example A[0, 4] is lane 16, VGPR 0, low half.
  - All other A/B layouts are contiguous: `h = k / (n/2)`, `p = k % (n/2)` with `n = k_a` (A) or `k_b` (B). For 8-bit and 4-bit dense forms this is what the ISA tables give. For sparse 16-bit forms the ISA has no table; its expansion pseudo-code takes each group's index nibble at bits `col[3:2] * 4` of the index VGPR of the lane holding that A data, which puts groups `4h .. 4h+3` in lane half `h` — the contiguous labels.
  - `slot_order(spec, operand)` returns the K label of each lane-order slot `q = h * n/2 + p`.
- C / D (`16 x 16`): column `r`, rows `8h .. 8h+7`, one element per VGPR (`f32`, `i32`) or two (`f16`, `bf16`, even row in the low half), as ISA 7.12.2 states.
- Sparse index VGPR (one per lane): row `r`, half `h`, `k_a / 2` 2-bit entries LSB-first (K=32: bits `[15:0]` for `index_key:0`, `[31:16]` for `index_key:1`, the other half ignored; K=64: all 32 bits).

Hardware cannot distinguish one permutation of K applied to A and B together: the products are the same. The K labels are therefore the ISA's, and what was measured is which register slots pair (A slot k with B slot k, `tools/prove_layouts.py`) and in which order they are accumulated (section 3.1). The first release of this model used contiguous labels for dense 16-bit A/B together with steps `0, 8, 4, 12`; on raw registers it computed exactly the same function.

Sparse selection: for compressed `ka` (`g = ka // 2`) the selected B element is `B[4g + idx[row][ka], col]` for every 2-bit value, including pairs with `idx0 >= idx1`; the dense pipeline then runs on `(A[row, :], b_sel, C[row, col])`.

## 8. Hardware State (`MODE`, raw VOP3P fields, destination extent, `EXEC`)

From `evidence/state_gfx1201.mmstate` (all 22 forms):

- `HW_REG_WAVE_MODE`: of the tested bits `[15:0]`, writes to `[15:8]` read back 0. Same-wave `v_add_f32`, `v_sub_f32`, `v_add_f16`, `v_sub_f16` follow every round and denorm setting (input and output flushing), and `s_round_mode` / `s_denorm_mode` update `[3:0]` / `[7:4]`. Every WMMA and SWMMAC form gives the same result under all 14 MODE values tested.
- Raw VOP3P CLAMP = 1 has no effect on any float form (16 forms). Raw OPSEL = 7 and OPSEL_HI = 0 have no effect on the 8 dense float forms.
- Destination extent: with 8 destination VGPRs prefilled with `0xDEADBEEF`, the `f16`/`bf16` D forms (dense and sparse) write their 4 D VGPRs and leave the next 4 untouched.
- `EXEC`: with lane-distinct inputs and 8 masks from `0xFFFFFFFF` to `0x00000000`, all 32 lanes read their operands and write D on every form.

The model takes these fields (`clamp`, `op_sel`, `op_sel_hi`, `mode`, `exec_lo`) and, as measured, its result does not depend on them except integer `clamp`.

## 9. Public API

All functions take and return raw bit patterns as Python `int`s and raise `ValueError` for anything outside this domain:

- `wmma_wave(spec_name, a_words, b_words, c_words, *, idx_words=None, index_key=0, neg_lo=(0,0,0), neg_hi=(0,0,0), clamp=False, op_sel=0, op_sel_hi=7, mode=0x00F0, exec_lo=0xFFFFFFFF) -> list[int]`: exactly `32 * a_dwords`, `32 * b_dwords`, `32 * cd_dwords` u32 words (lane-major), and 32 index words for SWMMAC (none otherwise). Returns `32 * cd_dwords` D words.
- `wmma_dot(spec_name, a_vec, b_vec, c_word, *, idx_vec=None, neg_lo, neg_hi, clamp) -> int`: one D element from `k_a` A elements, `k_b` B elements, one C element, and for SWMMAC `k_a` index entries `0..3`.
- `wmma_matrix(spec_name, a, b, c, *, idx=None, neg_lo, neg_hi, clamp) -> list[list[int]]`: `16 x k_a` A, `k_b x 16` B, `16 x 16` C, `16 x k_a` index table.
- Layout: `pack_a/unpack_a`, `pack_b/unpack_b`, `pack_cd/unpack_cd` (logical matrices of raw elements <-> lane-major words), `pack_idx/unpack_idx(spec, ..., index_key=0)`, `slot_order(spec, "a"|"b")`.
- Rules: `step_schedule(spec)`, `check_modifiers(spec, neg_lo, neg_hi, clamp)`, `select_sparse_b(b_vec, idx_vec)`.
- Low-level helpers, unchecked: `decode_float(bits, fmt)`, `decode_int(bits, width, signed)`, `pack_float(units, grid, fmt)`, `dot_float_step(...)` (one step of section 3.2).
- Tables: `SPECS`, `FLOAT_FORMATS`, `INT_FORMATS`, `WmmaSpec`, `FloatFormat`, `IntFormat`; boundary helpers `f32_to_bits`, `bits_to_f32`.

Domain: element values `0 <= v < 2^bits`; modifiers are three 0/1 integers; `index_key` is 0 or 1 on K=32 SWMMAC and 0 elsewhere; `op_sel`/`op_sel_hi` are 3-bit and must keep their defaults on integer and sparse forms; `mode` is 16-bit, `exec_lo` 32-bit. Rejected modifier fields are listed in section 5.

## 10. What Each Rule Rests On

| Rule | Evidence |
|---|---|
| Format decoding, grids, floors, sub-groups, RNE packing, special values | random `finite`/`raw` chunks and `adv` chunks on all float forms |
| Step order (3.1) | `k_order` chunks on the 8 16-bit forms; mutants of the `fp8` order are killed by the random chunks |
| `f16` D sub-group size 2 | `cancel` chunks (exact cancellation exposes lost grid units) |
| Tiny negative accumulator and its `emin` boundary | `tiny_acc` chunks |
| Signed zeros | `signed_zero` chunks; `bf16` `-0` measured |
| Dense modifiers | dense modifier chunks; sparse A/B modifiers: `ik*_mod*` chunks |
| Integer signedness, wrap, two-half clamp | integer chunks incl. `clamp_stress` |
| Sparse selection, `index_key` | random sparse chunks (valid and raw index words), `mod_witness` chunks |
| MODE, CLAMP, OPSEL, destination extent, EXEC | state file |
| Rejected fields (int `NEG[2]`/`NEG_HI`, fp8 A/B `NEG`, SWMMAC C modifiers, OPSEL on int/sparse) | not measured; API decision following the ISA and llvm-mc |

## 11. Limitations

- Only `gfx1201` (Radeon RX 9070 XT) wave32 was measured. Wave64, other `gfx12*` parts, RDNA 3 and CDNA are not covered.
- Hazards, latency, `matrix_a_reuse`/`matrix_b_reuse`, memory ordering, FP exception flags and traps, `FP16_OVFL` and MODE bits above 15, inline literal or SGPR operands, and `VDST` aliasing `SRC0`/`SRC1` were not probed.
- Raw-bit encodings that llvm-mc rejects were probed only for float CLAMP and dense float OPSEL/OPSEL_HI.
