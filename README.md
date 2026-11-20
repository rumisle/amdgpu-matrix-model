# amdgpu-matrix-model

A dependency-free, pure-integer software model of AMD `gfx1201` (RDNA 4) wave32 matrix arithmetic (`v_wmma_*` and `v_swmmac_*`), with a specification ([MODEL.md](MODEL.md)) and frozen hardware outputs ([EVIDENCE.md](EVIDENCE.md)) captured on an AMD Radeon RX 9070 XT.

## Scope

All 22 `gfx1201` wave32 matrix instructions (`16 x 16` tiles):

| Family | Instructions | Modifiers / Features |
|---|---|---|
| Dense `f16`/`bf16` | `v_wmma_{f32,f16}_16x16x16_f16`, `v_wmma_{f32,bf16}_16x16x16_bf16` | `neg_lo`, `neg_hi` on A/B/C |
| Dense `fp8`/`bf8` -> `f32` | `v_wmma_f32_16x16x16_{fp8,bf8}_{fp8,bf8}` (4 forms) | OCP E4M3FN / E5M2; C mods ignored |
| Dense integer -> `i32` | `v_wmma_i32_16x16x16_iu8`, `_16x16x16_iu4`, `_16x16x32_iu4` | `neg_lo` signedness, two-half `clamp` |
| Sparse float (`K=32`) | `v_swmmac_{f32,f16,bf16}_16x16x32_*` (8 forms) | 2:4 index set, `index_key:0/1` |
| Sparse integer (`K=32,64`) | `v_swmmac_i32_16x16x32_{iu8,iu4}`, `_16x16x64_iu4` | `index_key:0/1` (K=32), `neg_lo`, `clamp` |

## Install and Use

Requires Python 3.10+ with no runtime dependencies (`uv pip install .`). Operands and outputs are raw bit patterns:

```python
from amdgpu_matrix_model import f32_to_bits, wmma_dot

assert wmma_dot("v_wmma_f32_16x16x16_f16", [0x3C00] * 16, [0x3C00] * 16, 0) == f32_to_bits(16.0)
assert wmma_dot("v_wmma_f32_16x16x16_fp8_bf8", [0x38] * 16, [0x40] * 16, 0) == f32_to_bits(32.0)
```

## Validate

No GPU required. Run from a repository checkout, or pass `--evidence DIR`:

```bash
amdgpu-matrix-validate
# or without installing:
uv run --no-project python -m amdgpu_matrix_model.validation --evidence ./evidence
```

## What Was Checked

Measured on an AMD Radeon RX 9070 XT (`gfx1201`, HIP 7.1.52802, Linux `7.2.11-200.fc44.x86_64`), 0 mismatches:

- Frozen in `evidence/` (`5.8 MiB` total): 2,325,504 elements across all 22 instructions (`102,400`–`115,456` per instruction; [manifest.json](evidence/manifest.json)) plus 125,445 hardware state checks (`MODE`, raw float `CLAMP`, `f16`/`bf16` D sentinel and `OPSEL`, `EXEC` masking).
- Compared on hardware in total: 23,790,592 elements (`>= 1.04M` per instruction across finite random, raw random, adversarial, modifiers, `clamp` stress, and sparse `index_key:0/1`), plus 655,360 elements proving wave32 layouts across 32 kernels.

## Re-Capturing Evidence

Local tools under `tools/` require a `gfx1201` GPU, ROCm HIP (`libamdhip64.so`), LLVM >= 22 (`llvm-mc`, `ld.lld`), and `numpy`. Pass `--co PATH` to use a pre-assembled code object from `tools/probe_asm.py`:

```bash
uv run --no-project --with numpy python tools/prove_layouts.py
uv run --no-project --with numpy python tools/capture_evidence.py
uv run --no-project --with numpy python tools/probe_state.py --capture
```

## Limitations

Covers `gfx1201` wave32 functional arithmetic only. Wave64, other `gfx12*`/RDNA 3/CDNA parts, hazards, timing, `matrix_{a,b}_reuse`, FP exception flags/traps, literal/SGPR C, `VDST` aliasing `SRC0`/`SRC1`, and `v_swmmac_*` float `neg_lo`/`neg_hi` were not tested.

## Credits

The `f16`/`bf16 -> f32` dot step starts from AMD's `rocjitsu` (`ROCm/rocm-systems`, `experimental/rocjitsu`, MIT). The package shape follows SF Tensor's `tcgen05` model. Measured here: `fp8`/`bf8` quads and denormal `align_exp`, `f16` D single-step RNE and `2^-40` floor, `bf16` D truncation, two-half integer `clamp`, sparse selection, and ignored `MODE`/`clamp`/`op_sel`/`EXEC`/`fp8` C modifiers.

## License

MIT. See [LICENSE](LICENSE).
