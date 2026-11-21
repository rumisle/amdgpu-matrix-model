# amdgpu-matrix-model

A dependency-free, pure-integer model of AMD `gfx1201` (RDNA 4) wave32 matrix arithmetic (`v_wmma_*` and `v_swmmac_*`), with a specification ([MODEL.md](MODEL.md)) and frozen hardware outputs from an AMD Radeon RX 9070 XT ([EVIDENCE.md](EVIDENCE.md)).

## Scope

All 22 `gfx1201` wave32 matrix instructions (`16 x 16` tiles):

| Family | Instructions | Modifiers / Features |
|---|---|---|
| Dense `f16`/`bf16` | `v_wmma_{f32,f16}_16x16x16_f16`, `v_wmma_{f32,bf16}_16x16x16_bf16` | `neg_lo`, `neg_hi` on A/B/C |
| Dense `fp8`/`bf8` -> `f32` | `v_wmma_f32_16x16x16_{fp8,bf8}_{fp8,bf8}` (4 forms) | OCP E4M3FN / E5M2; C modifiers ignored |
| Dense integer -> `i32` | `v_wmma_i32_16x16x16_iu8`, `_16x16x16_iu4`, `_16x16x32_iu4` | `neg_lo` signedness, two-half `clamp` |
| Sparse float (`K=32`) | `v_swmmac_{f32,f16,bf16}_16x16x32_*` (8 forms) | 2:4 index, `index_key:0/1`, A/B `neg_lo`/`neg_hi` on `f16`/`bf16` |
| Sparse integer (`K=32,64`) | `v_swmmac_i32_16x16x32_{iu8,iu4}`, `_16x16x64_iu4` | `index_key:0/1` (K=32), `neg_lo`, `clamp` |

Operand layouts follow the RDNA4 ISA (section 7.12.2), including its K labels for dense 16-bit A/B.

## Install and Use

Python 3.10+, no runtime dependencies (`uv pip install .`). Operands and results are raw bit patterns; arguments outside the documented domain raise `ValueError`:

```python
from amdgpu_matrix_model import f32_to_bits, wmma_dot

assert wmma_dot("v_wmma_f32_16x16x16_f16", [0x3C00] * 16, [0x3C00] * 16, 0) == f32_to_bits(16.0)
assert wmma_dot("v_wmma_f32_16x16x16_fp8_bf8", [0x38] * 16, [0x40] * 16, 0) == f32_to_bits(32.0)
```

`wmma_wave` runs one instruction on lane-major register words, `wmma_matrix` on logical matrices; see MODEL.md section 9.

## Validate

No GPU needed. From a repository checkout:

```bash
uv run --no-project python -m amdgpu_matrix_model.validation     # or: amdgpu-matrix-validate
uv run --no-project python -m unittest discover -s tests          # spoofed corpora and rule mutants
```

Full validation passes only if the manifest has the SHA-256 pinned in the validator, the directory holds exactly the 23 listed files with matching hashes, every file passes the strict schema, and every recorded output is reproduced. `--variant NAME` checks single instructions and reports `PARTIAL`.

## Evidence

Frozen in `evidence/` (6.3 MiB), all reproduced with 0 mismatches:

- 2,577,408 D elements in 10,068 waves across all 22 instructions (102,400–152,064 per instruction): random, special-value and modifier chunks from the first capture, plus targeted chunks for the 16-bit forms (K-slot order, cancellation, tiny accumulators, signed zeros, sparse A/B modifiers).
- 108 state records, 262,816 checks: MODE (14 values, all forms) with VALU positive controls, raw VOP3P CLAMP (all float forms) and OPSEL/OPSEL_HI (dense float), destination extent, and EXEC (8 masks, all forms).

Each chunk's capture session, code-object hash (where retained) and input hash are in [manifest.json](evidence/manifest.json). MODEL.md section 10 maps rules to the evidence behind them.

## Re-Capturing Evidence

Tools under `tools/` need a `gfx1201` GPU, ROCm HIP (`libamdhip64.so`) and `numpy`; assembling code objects needs LLVM >= 22 (`llvm-mc`, `ld.lld`), which can run on another machine:

```bash
uv run --no-project --with numpy python tools/probe_asm.py                                   # build/probes_gfx1201.co
uv run --no-project --with numpy python tools/probe_state.py --build-co build/state.co
uv run --no-project --with numpy python tools/probe_state.py --co build/state.co --capture
uv run --no-project --with numpy python tools/capture_evidence.py --update --co build/probes_gfx1201.co
uv run --no-project --with numpy python tools/prove_layouts.py --co build/probes_gfx1201.co   # layout checks
uv run --no-project python tools/k_order.py                                                  # step order, GPU-free
```

`capture_evidence.py --update` keeps stored chunks that still match the corpus plan, captures only new ones and rewrites the manifest; a new release then pins the printed manifest SHA-256 in `RELEASE_MANIFEST_SHA256` (`amdgpu_matrix_model/validation.py`).

## Limitations

`gfx1201` wave32 functional arithmetic only. Wave64, other `gfx12*`/RDNA 3/CDNA parts, hazards, timing, `matrix_{a,b}_reuse`, FP exception flags, `FP16_OVFL`, literal/SGPR operands and `VDST` aliasing `SRC0`/`SRC1` were not tested. Fields the ISA requires to be zero (and SWMMAC C modifiers) are rejected rather than modeled.

## Credits

The float dot step (exact products, pairwise floor alignment, accumulator grid, RNE packing, `f16` D direct rounding and its tiny-accumulator rule, `bf16` D truncation, special-value precedence) is a Python adaptation of `gfx12_dot.h` and `dot_packed16.h` from AMD's `rocjitsu` (MIT; [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)). Added and measured here: the `fp8`/`bf8` forms (eight-product steps, sub-groups of four, subnormal `align_exp`), integer and sparse forms, operand layouts and step order, the tiny-accumulator rule for `f32` and `bf16` D, `bf16` D `-0`, sparse modifiers, and the state behaviour. The repository layout follows SF Tensor's `tcgen05` model; no code was taken from it.

## License

MIT. See [LICENSE](LICENSE) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
