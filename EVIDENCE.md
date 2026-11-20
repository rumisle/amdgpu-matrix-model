# Frozen Hardware Evidence Format

The `evidence/` directory stores frozen `gfx1201` (wave32) hardware outputs so any model or emulator can be checked without a GPU.

- `evidence/manifest.json`: hardware metadata (`gpu`, `gfx_target`, `hip_version`, `kernel_driver`, `capture_date`), per-file element/wave counts, sizes, and SHA-256 hashes.
- `evidence/<instruction>.mmbin`: compact binary corpus for each of the 22 instructions (`>= 102,400` elements per file, `2,325,504` total). Random chunks store only their PRNG seed and hardware output `D` words; hand-built adversarial chunks store explicit `A, B, C` inputs plus `D`.
- `evidence/state_gfx1201.mmstate`: hardware state probe corpus (`MODE`, raw float `CLAMP`, `f16`/`bf16` D sentinel and `OPSEL`, and `EXEC` masking; `125,445` checks, explicit inputs and outputs).

## `.mmbin` Binary Layout (`AMDMAT02`)

All integers are little-endian.

```text
bytes  0..8        magic: ASCII "AMDMAT02" (8 bytes)
bytes  8..12       hdr_len: uint32
bytes 12..12+H     UTF-8 JSON header (H = hdr_len bytes)
bytes 12+H..end    zlib-compressed stream of uint32 words
```

### Header JSON fields

```json
{
  "target": "gfx1201",
  "spec": "v_wmma_f32_16x16x16_f16",
  "chunks": [
    {"tag": "finite", "gen": "finite", "waves": 200, "seed": 202601},
    {"tag": "raw", "gen": "raw", "waves": 200, "seed": 202602},
    {"tag": "adv", "gen": "explicit", "waves": 3}
  ]
}
```

Optional chunk fields (`index_key`, `neg_lo`, `neg_hi`, `clamp`) are omitted when equal to their defaults (`0`, `[0, 0, 0]`, `[0, 0, 0]`, `false`).

### Decompressed payload order

Chunks appear sequentially in header order. For each chunk with `W = waves` and per-lane VGPR counts `a_dwords`, `b_dwords`, `cd_dwords` (from [MODEL.md](MODEL.md)):

- When `gen == "explicit"`:
  1. `A`: `W * 32 * a_dwords` words, shaped `(W, 32, a_dwords)`
  2. `B`: `W * 32 * b_dwords` words, shaped `(W, 32, b_dwords)`
  3. `C`: `W * 32 * cd_dwords` words, shaped `(W, 32, cd_dwords)`
  4. `IDX` (`v_swmmac_*` only): `W * 32` words, shaped `(W, 32)`
  5. `D_hw`: `W * 32 * cd_dwords` words, shaped `(W, 32, cd_dwords)`
- When `gen` is `"raw"`, `"finite"`, or `"clamp_stress"`:
  1. `D_hw` only: `W * 32 * cd_dwords` words, shaped `(W, 32, cd_dwords)`. Inputs `(A, B, C, [IDX])` are regenerated deterministically from `seed` and `gen` as specified below.

Within each wave `w` and lane `l = 16 * h + r` (`r = l % 16`, `h = l // 16`), words are packed LSB-first:

- `A[w, l, :]`: row `r`, elements `k = h * (k_a / 2) .. (h + 1) * (k_a / 2) - 1`
- `B[w, l, :]`: column `r`, elements `k = h * (k_b / 2) .. (h + 1) * (k_b / 2) - 1`
- `C[w, l, :]` and `D_hw[w, l, :]`: column `r`, rows `8 * h .. 8 * h + 7` (1 element per word for `f32`/`i32`; 2 per word for `f16`/`bf16`)
- `IDX[w, l]`: row `r`, half `h`, `k_a / 2` 2-bit offsets packed LSB-first (shifted right by `16` when `index_key == 1` on `K = 32` forms)

## Deterministic Input Generator (SplitMix64)

Random chunks use standard 64-bit SplitMix64 initialized with `state = seed & 0xFFFFFFFFFFFFFFFF`:

```python
M64 = 0xFFFFFFFFFFFFFFFF
def next_u64(state: int) -> tuple[int, int]:
    state = (state + 0x9E3779B97F4A7C15) & M64
    z = ((state ^ (state >> 30)) * 0xBF58476D1CE4E5B9) & M64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & M64
    return state, (z ^ (z >> 31)) & M64
```

`next_u32()` returns `next_u64() & 0xFFFFFFFF`. Arrays are generated in exact order `A` (`na = W * 32 * a_dwords`), `B` (`nb = W * 32 * b_dwords`), `C` (`nc = W * 32 * cd_dwords`), then `IDX` (`W * 32`, sparse only):

- `gen == "raw"`: draw raw `next_u32()` for every word of `A`, `B`, `C`.
- `gen == "finite"`: draw one raw `u = next_u32()` per word `i` of `count` and map it by format:
  - `f16` (each 16-bit half `h`): if `(h & 0x7C00) == 0x7C00`, `h &= 0x83FF` (maps NaN/Inf exponent to subnormal/zero).
  - `bf16` (each 16-bit half `h`): if `(h & 0x0F) == 0`, `h &= 0x807F`; else if `i < count // 2`, set exponent to `96 + (((h >> 7) & 0xFF) & 0x3F)`; else if `(h & 0x7F80) == 0x7F80`, `h &= 0x807F`.
  - `fp8` (each byte `b`): if `(b & 0x7F) == 0x7F`, `b &= 0xFE`.
  - `bf8` (each byte `b`): if `(b & 0x7C) == 0x7C`, `b &= 0x83`.
  - `f32`: if `(u & 0x0F) == 0`, `u &= 0x807FFFFF`; else if `i < (count * 3) // 4`, set exponent to `96 + (((u >> 23) & 0xFF) & 0x3F)`; else if `(u & 0x7F800000) == 0x7F800000`, `u &= 0x807FFFFF`.
- `gen == "clamp_stress"`: raw `next_u32()` for `A` and `B`; for word `i` of `C`, draw `u = next_u32()` and compute `((base + ((u & 0xFFFF) - 32768)) & 0xFFFFFFFF)` with `base = 0x7FFFFFFF` for `i < nc // 2` and `0x80000000` otherwise.
- Sparse `IDX` (`W * 32` words): for the first `(W // 2) * 32` words, draw `u = next_u64()` and pack 8 nibbles `VALID[((u >> (8 * n)) & 0xFF) % 6] << (4 * n)` with `VALID = (0x4, 0x8, 0xC, 0x9, 0xD, 0xE)`. For the remaining words, draw raw `next_u32()`.

### Minimal standalone `.mmbin` reader

```python
import array, json, struct, sys, zlib
from pathlib import Path

def read_mmbin(path: Path, a_dw: int, b_dw: int, cd_dw: int, sparse: bool):
    raw = path.read_bytes()
    assert raw[:8] == b"AMDMAT02"
    h_len = struct.unpack_from("<I", raw, 8)[0]
    hdr = json.loads(raw[12 : 12 + h_len].decode("utf-8"))
    words = array.array("I")
    words.frombytes(zlib.decompress(raw[12 + h_len :]))
    if sys.byteorder != "little":
        words.byteswap()
    pos = 0
    for ch in hdr["chunks"]:
        w = ch["waves"]
        nc = w * 32 * cd_dw
        if ch.get("gen", "explicit") == "explicit":
            na, nb, ni = w * 32 * a_dw, w * 32 * b_dw, (w * 32 if sparse else 0)
            a, pos = words[pos : pos + na], pos + na
            b, pos = words[pos : pos + nb], pos + nb
            c, pos = words[pos : pos + nc], pos + nc
            idx, pos = (words[pos : pos + ni] if sparse else None), pos + ni
        else:
            a = b = c = idx = None  # regenerate from (ch["seed"], ch["gen"]) via SplitMix64
        d_hw, pos = words[pos : pos + nc], pos + nc
        yield ch, a, b, c, idx, d_hw
```

## `.mmstate` Binary Layout (`AMDST001`)

```text
bytes 0..8     magic: ASCII "AMDST001" (8 bytes)
bytes 8..end   zlib-compressed UTF-8 JSON object
```

Top-level keys: `target`, `mode_ctrl`, `mode_wmma`, `raw_clamp`, `sentinel_opsel`, and `exec_mask` (all explicit inputs and raw hardware outputs).
