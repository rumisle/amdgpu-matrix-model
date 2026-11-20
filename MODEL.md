# gfx1201 (RDNA 4) WMMA & SWMMAC Arithmetic Specification

This document specifies the bit-exact arithmetic of all 22 wave32 matrix instructions (`11 v_wmma_*` and `11 v_swmmac_*`) on AMD `gfx1201` (RDNA 4), mapping raw operand bits to raw destination bits.

`amdgpu_matrix_model` implements this specification with Python integer arithmetic only; host floating point never participates in any result.

## 1. Instruction Table

Each instruction computes one `16 x 16` tile `D = A * B + C` across a 32-lane wave.

| Instruction | A | B | D / C | K | `k_a` | A / B / CD dwords/lane | Schedule / Packing |
|---|---|---|---|---:|---:|---|---|
| `v_wmma_f32_16x16x16_f16` | `f16` | `f16` | `f32` | 16 | 16 | 4 / 4 / 8 | 4 steps of 4 (2x2), RNE to `f32` |
| `v_wmma_f32_16x16x16_bf16` | `bf16` | `bf16` | `f32` | 16 | 16 | 4 / 4 / 8 | 4 steps of 4 (2x2), RNE to `f32` |
| `v_wmma_f16_16x16x16_f16` | `f16` | `f16` | `f16` | 16 | 16 | 4 / 4 / 4 | 4 steps of 4 (2x2), single RNE to `f16` |
| `v_wmma_bf16_16x16x16_bf16` | `bf16` | `bf16` | `bf16` | 16 | 16 | 4 / 4 / 4 | 4 steps of 4 (2x2), RNE to `f32` then trunc |
| `v_wmma_f32_16x16x16_fp8_fp8` | `fp8` | `fp8` | `f32` | 16 | 16 | 2 / 2 / 8 | 2 steps of 8 (2x4), RNE to `f32` |
| `v_wmma_f32_16x16x16_fp8_bf8` | `fp8` | `bf8` | `f32` | 16 | 16 | 2 / 2 / 8 | 2 steps of 8 (2x4), RNE to `f32` |
| `v_wmma_f32_16x16x16_bf8_fp8` | `bf8` | `fp8` | `f32` | 16 | 16 | 2 / 2 / 8 | 2 steps of 8 (2x4), RNE to `f32` |
| `v_wmma_f32_16x16x16_bf8_bf8` | `bf8` | `bf8` | `f32` | 16 | 16 | 2 / 2 / 8 | 2 steps of 8 (2x4), RNE to `f32` |
| `v_wmma_i32_16x16x16_iu8` | `iu8` | `iu8` | `i32` | 16 | 16 | 2 / 2 / 8 | 2 halves of 8, wrap or per-half `clamp` |
| `v_wmma_i32_16x16x16_iu4` | `iu4` | `iu4` | `i32` | 16 | 16 | 1 / 1 / 8 | 2 halves of 8, wrap or per-half `clamp` |
| `v_wmma_i32_16x16x32_iu4` | `iu4` | `iu4` | `i32` | 32 | 32 | 2 / 2 / 8 | 2 halves of 16, wrap or per-half `clamp` |
| `v_swmmac_f32_16x16x32_f16` | `f16` | `f16` | `f32` | 32 | 16 | 4 / 8 / 8 | Sparse 2:4 (`index_key:0\|1`), dense dot |
| `v_swmmac_f32_16x16x32_bf16` | `bf16` | `bf16` | `f32` | 32 | 16 | 4 / 8 / 8 | Sparse 2:4 (`index_key:0\|1`), dense dot |
| `v_swmmac_f16_16x16x32_f16` | `f16` | `f16` | `f16` | 32 | 16 | 4 / 8 / 4 | Sparse 2:4 (`index_key:0\|1`), dense dot |
| `v_swmmac_bf16_16x16x32_bf16` | `bf16` | `bf16` | `bf16` | 32 | 16 | 4 / 8 / 4 | Sparse 2:4 (`index_key:0\|1`), dense dot |
| `v_swmmac_f32_16x16x32_fp8_fp8` | `fp8` | `fp8` | `f32` | 32 | 16 | 2 / 4 / 8 | Sparse 2:4 (`index_key:0\|1`), dense dot |
| `v_swmmac_f32_16x16x32_fp8_bf8` | `fp8` | `bf8` | `f32` | 32 | 16 | 2 / 4 / 8 | Sparse 2:4 (`index_key:0\|1`), dense dot |
| `v_swmmac_f32_16x16x32_bf8_fp8` | `bf8` | `fp8` | `f32` | 32 | 16 | 2 / 4 / 8 | Sparse 2:4 (`index_key:0\|1`), dense dot |
| `v_swmmac_f32_16x16x32_bf8_bf8` | `bf8` | `bf8` | `f32` | 32 | 16 | 2 / 4 / 8 | Sparse 2:4 (`index_key:0\|1`), dense dot |
| `v_swmmac_i32_16x16x32_iu8` | `iu8` | `iu8` | `i32` | 32 | 16 | 2 / 4 / 8 | Sparse 2:4 (`index_key:0\|1`), dense int |
| `v_swmmac_i32_16x16x32_iu4` | `iu4` | `iu4` | `i32` | 32 | 16 | 1 / 2 / 8 | Sparse 2:4 (`index_key:0\|1`), dense int |
| `v_swmmac_i32_16x16x64_iu4` | `iu4` | `iu4` | `i32` | 64 | 32 | 2 / 4 / 8 | Sparse 2:4 (32-bit index), dense `16x16x32_iu4` |

## 2. Format Decoding

Every finite floating-point word decodes to `(-1)**sign * sig * 2**val_exp` plus its encoded alignment exponent `align_exp`, clamped to `emin = 1 - bias` for subnormals and zero:

- Normal (`exp_field != 0`): `sig = (1 << frac_bits) | frac`, `align_exp = exp_field - bias`, `val_exp = align_exp - frac_bits`.
- Subnormal / zero (`exp_field == 0`): `sig = frac`, `align_exp = emin`, `val_exp = emin - frac_bits`.

| Format | Bits | S/E/F | Bias | `emin` | `emax` | Inf | NaN | Destination NaNs (`default` / `sig` / `quiet_bit`) |
|---|---:|---|---:|---:|---:|---|---|---|
| `f32` | 32 | 1/8/23 | 127 | -126 | 127 | `exp=255, frac=0` | `exp=255, frac!=0` | `0xFFC00000` / `0xFFC00A3D` / `0x00400000` |
| `f16` | 16 | 1/5/10 | 15 | -14 | 15 | `exp=31, frac=0` | `exp=31, frac!=0` | `0xFE00` / `0xFE3D` / `0x0200` |
| `bf16` | 16 | 1/8/7 | 127 | -126 | 127 | `exp=255, frac=0` | `exp=255, frac!=0` | `0xFFC0` / `0xFFFD` / `0x0040` |
| `fp8` (OCP E4M3FN) | 8 | 1/4/3 | 7 | -6 | 8 | none | `0x7F`, `0xFF` | — |
| `bf8` (OCP E5M2) | 8 | 1/5/2 | 15 | -14 | 15 | `0x7C`, `0xFC` | `exp=31, frac!=0` | — |

Keeping `align_exp` separate from `floor(log2(|value|))` is observable on all formats, including subnormal `fp8`/`bf8` inputs (for example, `bf8` subnormal `2^-15 * 2^15 = 1.0` uses `align_exp_sum = -14 + 15 = +1`, shifting the product alignment grid up by 1 bit relative to normal `1.0 * 1.0`).

## 3. Floating-Point Accumulation, Alignment, and Rounding

All 16 floating-point instructions share one multi-step dot kernel (`dot_float_step`). Each step folds `step_size` products (two equal sub-groups of size `subgroup`) into the running accumulator `acc` (in destination format `d_fmt`):

| Family | Step starts (`s0`) | `step_size` | `subgroup` | `invalid_nan` | Accumulator `emin` | Per-step packing |
|---|---|---:|---:|---|---:|---|
| `f16` / `bf16` -> `f32` | `(0, 8, 4, 12)` | 4 | 2 | `0xFFC00000` | `-126` | Single RNE to `f32` |
| `f16` -> `f16` | `(0, 8, 4, 12)` | 4 | 2 | `0xFE00` | `-14` | Single RNE directly to `f16` |
| `bf16` -> `bf16` | `(0, 8, 4, 12)` | 4 | 2 | `0xFFC0` | `-126` | Single RNE to `f32`, then `word >> 16` |
| `fp8` / `bf8` -> `f32` | `(0, 8)` | 8 | 4 | `0xFFC00A3D` | `-126` | Single RNE to `f32` |

Finite step algorithm:

1. Form exact products `(-1)^(sign_a ^ sign_b) * (sig_a * sig_b) * 2^(val_exp_a + val_exp_b)`. Each nonzero product has grid exponent `grids[i] = align_exp(a_i) + align_exp(b_i) - 24` (else `-inf`).
2. Let `p_grid = max(grids)` and `c_lead = val_exp(acc) + bit_length(sig(acc)) - 1` when `acc != 0` (else `-inf`). Set `c_grid = max(c_lead, d_fmt.emin) - 26` and `grid = max(p_grid, c_grid)`. For `f16` D, `d_fmt.emin = -14` sets a minimum grid of `2^-40` (observable at `2^-25 + 2^-40 -> 0x0001` vs `2^-25 + 2^-41 -> 0x0000`).
3. Two-level floor alignment: for products and sub-group sums, `align_floor(x, g) = floor(x / 2^g)`. For the accumulator (`is_acc = True`), if `floor(|acc| / 2^g) == 0`, it contributes `0` even when negative; otherwise `floor(acc / 2^g)`.
   - Within each of the two sub-groups (`subgroup = 2` for `f16`/`bf16`, `4` for `fp8`/`bf8`), set `sg = max(grids[sub])`, sum `align_floor(prod[i], sg)`, and add `align_floor(sub_sum * 2^sg, grid)` to `total`.
   - Add `align_floor(acc, grid, is_acc=True)` to `total`.
4. Pack `total * 2^grid` into `d_fmt`:
   - `f32` D and `f16` D: one IEEE RNE rounding directly to `d_fmt` (subnormals kept, overflow to `±Inf`). `f16` D does not double-round through `f32`.
   - `bf16` D: one IEEE RNE rounding to `f32`, then truncate the low 16 bits (`word >> 16`).
   - Any exact zero result (including `-0.0 + -0.0` and negative underflow to zero) encodes as `+0.0`.

## 4. Floating-Point Special Values

Evaluated per step in this precedence order:

1. Any NaN in `A` or `B` returns `d_fmt.qnan_sig` (`0xFFC00A3D` for `f32`, `0xFE3D` for `f16`, `0xFFFD` for `bf16`).
2. Any `Inf * 0` product (`Inf * Inf` is valid and produces `±Inf`) or both `+Inf` and `-Inf` products in the step returns `invalid_nan` (`d_fmt.qnan_default` for `f16`/`bf16`, `d_fmt.qnan_sig` for `fp8`/`bf8`).
3. If `acc` is NaN, return `acc | d_fmt.quiet_bit` (sign and payload preserved, quiet bit set).
4. Combining product infinities with `acc = ±Inf`: opposite signs return `invalid_nan`; otherwise return the surviving `±Inf`.

Because steps execute sequentially, a later step's factor NaN or invalid operation overrides an earlier step's result, and an earlier step's overflow to `±Inf` enters later steps as `±Inf`.

## 5. Modifiers (`neg_lo`, `neg_hi`)

- `f16` / `bf16` inputs: `neg_lo[0]` flips sign bits of even A elements (`k % 2 == 0`); `neg_hi[0]` flips odd A elements (`k % 2 == 1`); `neg_lo[1]`/`neg_hi[1]` do the same for B. `neg_hi[2]` clears C's sign bit (`abs(C)`) and `neg_lo[2]` flips C's sign bit (`neg(C)`, applied after `abs`) for all rows on `f32`, `f16`, and `bf16` destinations (including NaNs).
- `fp8` / `bf8` inputs: hardware ignores `neg_lo` and `neg_hi` on C (A/B modifiers are disallowed).
- `iu8` / `iu4` inputs: `neg_lo[0]` and `neg_lo[1]` select signed (`1`) vs unsigned (`0`) decoding for A and B independently. C and D are always signed `i32`.

## 6. Integer Arithmetic and Two-Half Saturation (`clamp`)

`v_wmma_i32_*` and `v_swmmac_i32_*` decode A and B as unsigned or two's-complement signed (`4` or `8` bits) per `neg_lo[0:2]`, decode C as signed `i32`, and accumulate in two halves over compressed K (`0 .. k_a/2 - 1`, then `k_a/2 .. k_a - 1`):

- Without `clamp`: wrap modulo `2^32`.
- With `clamp`: clamp the exact integer accumulator to `[-2^31, 2^31 - 1]` after each half. First-half saturation persists into the second half (`INT32_MAX + 100` in half 0 followed by `-100` in half 1 yields `INT32_MAX - 100`).

## 7. Sparse Selection (`v_swmmac_*`) and Wave32 Layout

For lane `l = 16 * h + r` (`r = l % 16`, `h = l // 16`), elements are packed LSB-first:

- A (`16 x k_a`): row `r`, elements `h * k_a/2 .. (h + 1) * k_a/2 - 1`.
- B (`k_b x 16`): column `r`, elements `h * k_b/2 .. (h + 1) * k_b/2 - 1`.
- C / D (`16 x 16`): column `r`, rows `8 * h .. 8 * h + 7` (1 element per VGPR for `f32`/`i32`, 2 per VGPR for `f16`/`bf16`).
- Sparse index VGPR (1 VGPR per lane): row `r`, half `h`, `k_a / 2` 2-bit indices packed LSB-first (`K = 32`: bits `[15:0]` for `index_key:0`, `[31:16]` for `index_key:1`; `K = 64`: all 32 bits).

For each compressed `ka` in `0 .. k_a - 1` (`g = ka // 2`), hardware selects `b_sel[ka] = B[4 * g + idx[row][ka], col]` via direct 4-to-1 multiplexing for every 2-bit value `0..3` (including non-canonical pairs `idx0 >= idx1`), then runs the dense arithmetic pipeline on `(A[row, :], b_sel, C[row, col])`.

## 8. Hardware State (`MODE`, Float `CLAMP`, `OPSEL`, `EXEC`)

Measured on `gfx1201` (`evidence/state_gfx1201.mmstate`):

- `HW_REG_WAVE_MODE`: only bits `[7:0]` (`FP_ROUND_32[1:0]`, `FP_ROUND_16_64[3:2]`, `FP_DENORM_32[5:4]`, `FP_DENORM_16_64[7:6]`) are writable; bits `[15:8]` read back `0`. Same-wave `v_add_f32` and `v_add_f16` obey all rounding and denorm modes, while all 8 dense float WMMA instructions are bit-for-bit invariant across 14 `MODE` settings.
- Float `CLAMP`: rejected by `llvm-mc -mcpu=gfx1201`; forcing VOP3P `CLAMP=1` (bit 15) in raw machine code has no effect on any tested float WMMA variant.
- `f16` / `bf16` D packing and `OPSEL`: `v_wmma_f16_16x16x16_f16` and `v_wmma_bf16_16x16x16_bf16` write both halves of `v[d:d+3]` and leave `v[d+4:d+7]` untouched (`0xDEADBEEF` sentinel preserved). `llvm-mc` rejects `op_sel`, and forcing raw VOP3P `OPSEL = 0b111` or `OPSEL_HI = 0b000` has no effect.
- `EXEC` masking: across `exec_lo` from `0xFFFFFFFF` down to `0x00000000`, all 32 lanes read operands and write destination VGPRs regardless of `EXEC`.

## 9. Public API

- `wmma_dot(spec_name, a_vec, b_vec, c_word, *, idx_vec=None, neg_lo=(0,0,0), neg_hi=(0,0,0), clamp=False) -> int`
- `wmma_matrix(spec_name, a, b, c, *, idx=None, neg_lo=(0,0,0), neg_hi=(0,0,0), clamp=False) -> list[list[int]]`
- `wmma_wave(spec_name, a_words, b_words, c_words, *, idx_words=None, index_key=0, neg_lo=(0,0,0), neg_hi=(0,0,0), clamp=False) -> list[int]`
- `pack_a`, `unpack_a`, `pack_b`, `unpack_b`, `pack_cd`, `unpack_cd`, `pack_idx`, `unpack_idx`
- `SPECS`, `FLOAT_FORMATS`, `INT_FORMATS`, `f32_to_bits`, `bits_to_f32`

## 10. Limitations / Not Covered

- Only `gfx1201` (Radeon RX 9070 XT) wave32 functional arithmetic was tested. Wave64 mode, other `gfx12*` parts or steppings, RDNA 3 (`gfx11*`), and CDNA (`gfx9*` MFMA) are not covered.
- Register hazards, latency, throughput, `matrix_a_reuse`/`matrix_b_reuse` hints, and memory/LDS ordering are out of scope.
- FP exception status flags, trap handlers, inline literal/SGPR C operands, `VDST` aliasing `SRC0`/`SRC1`, and `v_swmmac_*` float `neg_lo`/`neg_hi` modifiers were not probed.
