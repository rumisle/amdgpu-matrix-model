# Frozen Hardware Evidence Format

`evidence/` holds frozen `gfx1201` (wave32) hardware outputs, so that any model or emulator can be checked without a GPU. This document is enough to read and check every file without the package.

| File | Content |
|---|---|
| `manifest.json` | capture sessions, the exact file inventory with sizes and SHA-256, per-chunk input hashes, totals |
| `<instruction>.mmbin` (22) | matrix corpus per instruction (`AMDMAT02`) |
| `state_gfx1201.mmstate` | hardware state probes (`AMDST002`) |

All integers are little-endian. JSON objects never repeat a key.

## Manifest (`manifest.json`)

```json
{
  "format": "amdgpu-matrix-model evidence manifest 2",
  "target": "gfx1201",
  "sessions": [{"id": "...", "date": "...", "hardware": {...}, "code_object_sha256": "..." | null, "tool_revision": "..." | null}],
  "files": [
    {"file": "v_wmma_f32_16x16x16_f16.mmbin", "kind": "matrix", "spec": "v_wmma_f32_16x16x16_f16",
     "bytes": 0, "sha256": "...", "waves": 0, "elements": 0,
     "chunks": [{"tag": "finite", "waves": 200, "inputs_sha256": "...", "session": "..."}]},
    {"file": "state_gfx1201.mmstate", "kind": "state", "bytes": 0, "sha256": "...", "session": "...",
     "records": 0, "checks": 0}
  ],
  "totals": {"matrix_files": 22, "matrix_waves": 0, "matrix_elements": 0, "state_records": 0, "state_checks": 0, "bytes": 0}
}
```

`files` lists exactly the 22 instruction files in `SPECS` order (`amdgpu_matrix_model/specs.py`), then the state file; the directory holds no other `.mmbin`/`.mmstate` file. A matrix entry's `chunks` equal the file header's chunks in order; `inputs_sha256` is the SHA-256 of the chunk's input words `A || B || C || IDX` (IDX only for SWMMAC) as u32 little-endian, regenerated for random chunks, stored for explicit ones; `elements = 256 * waves`. `session` names the capture run; the first session (2026-11-20) predates code-object and log retention, later sessions record the SHA-256 of the code object run and the tool commit. Input hashes of first-session chunks were computed when manifest 2 was written, from the same generator code the capture used; they pin the generator from then on, but no hash was recorded at that capture.

No other field is allowed; `kind` is `matrix` or `state`, hashes are 64 lowercase hex digits, and every count and total must equal what the files give (the state entry's `records` and `checks` as defined at the end of this document). The validator trusts a manifest only by its SHA-256, which is pinned in `amdgpu_matrix_model/validation.py` (`RELEASE_MANIFEST_SHA256`); the manifest then pins every file. Hashes establish identity with this release, not authenticity against someone who can also replace the validator.

## `.mmbin` (`AMDMAT02`)

```text
bytes  0..8        magic: ASCII "AMDMAT02"
bytes  8..12       hdr_len: uint32
bytes 12..12+H     UTF-8 JSON header (H = hdr_len)
bytes 12+H..end    one complete zlib stream of uint32 words, nothing after it
```

Header: exactly `{"target": "gfx1201", "spec": <instruction>, "chunks": [...]}`, `spec` equal to the file name stem, at least one chunk. Chunk fields:

| Field | Type | Rule |
|---|---|---|
| `tag` | string `[a-z0-9_]{1,64}` | unique in the file |
| `gen` | `"explicit"`, `"raw"`, `"finite"` (float forms only), `"clamp_stress"` (integer forms only) | required |
| `waves` | integer | `1 <= waves <= 4096` |
| `seed` | integer `0 <= seed < 2^64` | key present exactly when `gen != "explicit"` (`null` is not allowed) |
| `index_key` | `1` | only on K=32 SWMMAC; omitted when 0 |
| `neg_lo`, `neg_hi` | three 0/1 integers, not all 0 | omitted when all 0; legal per MODEL.md section 5 |
| `clamp` | `true` | omitted when false |

No other field is allowed. Payload, chunk after chunk in header order, with `W = waves` and the per-lane dword counts of MODEL.md section 1:

- `gen == "explicit"`: `A` (`W*32*a_dwords` words, shaped `(W, 32, a_dwords)`), `B` (`W*32*b_dwords`), `C` (`W*32*cd_dwords`), `IDX` (SWMMAC only, `W*32`), then `D_hw` (`W*32*cd_dwords`).
- otherwise: `D_hw` only; `A, B, C, IDX` are regenerated from `seed` and `gen` (below).

The payload holds exactly the sum of these counts. Words of wave `w`, lane `l` are the lane's VGPR contents (layout: MODEL.md section 7). Each wave is checked with that chunk's `index_key`, `neg_lo`, `neg_hi`, `clamp`.

## Deterministic Input Generator

Standard SplitMix64, state threaded through one generator object:

```python
M64 = 0xFFFFFFFFFFFFFFFF

class SplitMix64:
    def __init__(self, seed): self.state = seed & M64
    def next_u64(self):
        self.state = (self.state + 0x9E3779B97F4A7C15) & M64
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & M64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & M64
        return z ^ (z >> 31)
    def next_u32(self): return self.next_u64() & 0xFFFFFFFF
```

Reference outputs: seed 0 gives `0xE220A8397B1DCDAF, 0x6E789E6AA1B965F4, 0x06C45D188009454F`; seed 1234567 gives `6457827717110365317, 3203168211198807973`.

One generator per chunk, seeded with `seed`. Arrays in this order: `A` (`na = W*32*a_dwords` words), `B` (`nb`), `C` (`nc = W*32*cd_dwords`), then `IDX` (`W*32`, SWMMAC only). Word `i` of an array of `count` words:

- `raw`: `next_u32()` for every word of `A`, `B`, `C`.
- `finite`: `u = next_u32()`, mapped by the array's format (A: `a_fmt`, B: `b_fmt`, C: `d_fmt`):
  - `f16`, each 16-bit half `h`: if `(h & 0x7C00) == 0x7C00`: `h &= 0x83FF`.
  - `bf16`, each half `h`: if `(h & 0x0F) == 0`: `h &= 0x807F`; elif `i < count // 2`: exponent field `= 96 + ((h >> 7) & 0x3F)`; elif `(h & 0x7F80) == 0x7F80`: `h &= 0x807F`.
  - `fp8`, each byte `b`: if `(b & 0x7F) == 0x7F`: `b &= 0xFE`. `bf8`, each byte: if `(b & 0x7C) == 0x7C`: `b &= 0x83`.
  - `f32`: if `(u & 0x0F) == 0`: `u &= 0x807FFFFF`; elif `i < (count * 3) // 4`: exponent field `= 96 + ((u >> 23) & 0x3F)`; elif `(u & 0x7F800000) == 0x7F800000`: `u &= 0x807FFFFF`.
- `clamp_stress`: `A`, `B` raw; `C[i] = (base + (next_u32() & 0xFFFF) - 32768) mod 2^32` with `base = 0x7FFFFFFF` for `i < nc // 2`, else `0x80000000`.
- `IDX`: the first `(W // 2) * 32` words take `u = next_u64()` and pack 8 nibbles `VALID[((u >> 8n) & 0xFF) % 6] << 4n`, `VALID = (0x4, 0x8, 0xC, 0x9, 0xD, 0xE)`; the remaining words are `next_u32()`.

Golden vectors (2 waves): `v_wmma_f32_16x16x16_f16` seed 202601 `finite`: `A[0:2] = 0xBEEF724A, 0x24AB6025`, inputs SHA-256 `8d16a0f7b928bac4...`; `v_wmma_bf16_16x16x16_bf16` seed 7 `finite`: `0x49323DD7, 0xC43C361C`, `f0c1b3827a5c6d9e...`; `v_swmmac_f32_16x16x32_fp8_bf8` seed 202650 `raw`: `0x06630A18, 0xE88CD028`, `c82b8f6e46144064...`; `v_wmma_i32_16x16x16_iu8` seed 3 `clamp_stress`: `0xDB018FED, 0x7B81A989`, `a3a2e53bc91f5326...`. Every chunk's input hash is in the manifest.

## Minimal checked `.mmbin` reader

```python
import array, json, struct, sys, zlib

def read_mmbin(path, a_dw, b_dw, cd_dw, sparse):
    raw = open(path, "rb").read()
    assert raw[:8] == b"AMDMAT02"
    n = struct.unpack_from("<I", raw, 8)[0]
    hdr = json.loads(raw[12:12 + n])
    z = zlib.decompressobj()
    data = z.decompress(raw[12 + n:])
    assert z.eof and not z.unused_data, "truncated or trailing bytes"
    words = array.array("I", data)
    if sys.byteorder != "little":
        words.byteswap()
    sizes = []
    for ch in hdr["chunks"]:
        w = ch["waves"]
        assert type(w) is int and 1 <= w <= 4096
        nd = w * 32 * cd_dw
        explicit = ch["gen"] == "explicit"
        sizes.append((w * 32 * a_dw, w * 32 * b_dw, nd, w * 32 if sparse else 0, nd) if explicit else (0, 0, 0, 0, nd))
    assert len(words) == sum(map(sum, sizes)), "payload size does not match the header"
    pos = 0
    for ch, (na, nb, nc, ni, nd) in zip(hdr["chunks"], sizes):
        a, b, c, idx = (words[pos:pos + na], words[pos + na:pos + na + nb],
                        words[pos + na + nb:pos + na + nb + nc], words[pos + na + nb + nc:pos + na + nb + nc + ni])
        pos += na + nb + nc + ni
        d_hw = words[pos:pos + nd]
        pos += nd
        yield ch, a, b, c, idx, d_hw   # empty a/b/c/idx: regenerate from (seed, gen)
```

## Chunk catalogue

First capture (all 22 files): random `finite`/`raw` chunks, dense float `adv` (special values, alignment), dense modifier chunks (`neg_a`, `neg_lo_a`, `neg_hi_b`, `neg_c`, `abs_c`, `negabs_c`), integer signedness and `clamp_stress` chunks, sparse chunks on both index keys. Second capture (the 8 16-bit-input files, explicit inputs built by `tools/cases.py`):

| Tag | Content |
|---|---|
| `k_order` | for every ordered slot pair (i, j): `+P` at i, `+P*2^-25` at j, `C = -P`; D is nonzero exactly when i is accumulated first |
| `cancel` | `±P` cancelling next to small `±s` in the first and last step |
| `tiny_acc` | C below the grid with cancelling or single products, grids on both sides of D `emin` |
| `signed_zero` | zero or underflowing results of both signs in every step, and consumed by later steps |
| `ik{0,1}_mod_witness_<neg_lo>_<neg_hi>` | SWMMAC: one product per case for every compressed `ka` and selected offset, with A/B modifiers |
| `ik{0,1}_mod_<neg_lo>_<neg_hi>` | SWMMAC: random `finite` chunks with A/B modifiers |

In explicit chunks case `t` of a wave sits in row `t` of A and C and column `t` of B; off-diagonal elements are recorded and checked too.

## `.mmstate` (`AMDST002`)

```text
bytes 0..8     magic: ASCII "AMDST002"
bytes 8..end   one complete zlib stream of UTF-8 JSON, nothing after it
```

Document: exactly `{"target": "gfx1201", "records": [...]}`. Two record kinds.

**`mode_control`**: `{"kind", "write", "operands", "results"}`. One wave writes `write` (16 bits) to `HW_REG_WAVE_MODE[15:0]` with `s_setreg_b32`, then runs scalar VALU controls with these fixed `operands` (`[a, b]`; `f16` values 16-bit):

| Name | Operation | Operands |
|---|---|---|
| `f32_add_rne_tie` | `v_add_f32` | `0x3F800000, 0x33800000` (1 + 2^-24) |
| `f32_add_denorm_in` | `v_add_f32` | `0x00400000, 0x00400000` |
| `f32_sub_denorm_out` | `v_sub_f32` | `0x00C00000, 0x00800000` |
| `f16_add_rne_tie` | `v_add_f16` | `0x3C00, 0x1000` (1 + 2^-11) |
| `f16_add_denorm_in` | `v_add_f16` | `0x0200, 0x0200` |
| `f16_sub_denorm_out` | `v_sub_f16` | `0x0600, 0x0400` |

`results` holds those six results (f16 zero-extended) and `init` (MODE at wave start), `read` (after the write), `read_after_s_round_mode_15`, `read_after_s_denorm_mode_15` (after `s_round_mode 0xf`, then `s_denorm_mode 0xf`). All lanes agreed; lane 0 is stored. The validator compares every value with the MODE semantics in `validation.expected_control`: writable bits `[7:0]`, round field 1 = toward +inf, denorm field bit 0 keeps input and bit 1 output denormals. 16 rows, `write` in this order: `0x00F0 0x00F1 0x00F2 0x00F3 0x00F4 0x00F8 0x00FC 0x00C0 0x00D0 0x00E0 0x0030 0x0070 0x00B0 0x01F0 0x02F0 0xFFF0`.

**`wmma`**: one instruction encoding, one input set, several runs.

| Field | Meaning |
|---|---|
| `probe` | `mode`, `exec` (all forms); `baseline`, `clamp` (float forms); `op_sel`, `op_sel_hi` (dense float forms) |
| `spec` | instruction |
| `encoding` | the two VOP3P dwords executed: dw0 `VDST[7:0] NEG_HI[10:8] OPSEL[13:11] OPSEL_HI[2]:14 CLAMP:15 OP[22:16] 0xCC[31:24]`, dw1 `SRC0[8:0] SRC1[17:9] SRC2[26:18] OPSEL_HI[1:0]:[28:27] NEG[31:29]` |
| `index_key`, `neg_lo`, `neg_hi`, `clamp`, `op_sel`, `op_sel_hi` | the fields of `encoding` (on SWMMAC `index_key` is OPSEL[0] and `op_sel` the rest) |
| `store_dwords`, `prefill` | `8`, `0xDEADBEEF` |
| `waves`, `a`, `b`, `c`, `idx` | input words as in `.mmbin` explicit chunks; `idx` null on dense forms |
| `runs` | list of `{"mode", "exec_lo", "d"}` |

Each run sets `HW_REG_WAVE_MODE[15:0] = mode` before the instruction (`mode` probe; otherwise MODE stays `0x00F0`) or `EXEC_LO = exec_lo` around it (`exec` probe; otherwise `0xFFFFFFFF`). The 8 destination VGPRs of each lane (dense: `D`; SWMMAC: `C`, which is also D) that are not inputs are prefilled with `prefill`, and all 8 are stored: `d` has `waves * 32 * 8` words, lane-major. Expected per lane: the instruction's `cd_dwords` D words (computed with all fields and the run's state), then `prefill` for the remaining `8 - cd_dwords`. Runs: `mode` 14 values `0x00F0 0x00FF 0x00F5 0x00FA 0x0000 0x0055 0x00AA 0x000F 0x01F0 0x02F0 0x03F0 0x0FF0 0x0F00 0xFFF0`; `exec` 8 masks `0xFFFFFFFF 0x0000FFFF 0xFFFF0000 0x55555555 0xA5A5A5A5 0x80000000 0x00000001 0x00000000`; one run otherwise. Inputs: lane-distinct random words (`finite` for float, `raw` for integer forms, valid 2:4 index words), plus targeted waves (ties, tiny terms, subnormal inputs and outputs, NaN, Inf; and C and A*B on both sides of `[0, 1]` for the CLAMP and OPSEL probes).

Records appear in this order: the 16 `mode_control` rows, then for each instruction in `SPECS` order its probes in the order `mode, exec, baseline, clamp, op_sel, op_sel_hi` (as applicable). A check is one D element (two per word for 16-bit D), one preserved word, or one control value.
