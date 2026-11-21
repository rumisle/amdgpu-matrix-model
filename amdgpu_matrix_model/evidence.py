"""Readers and writers for the frozen evidence formats (EVIDENCE.md): `.mmbin`, `.mmstate`, manifest.

Every reader is strict: unknown fields, wrong types, impossible counts, short or long payloads and
trailing bytes raise `EvidenceError`. Standard library only.
"""

from __future__ import annotations

import array
import hashlib
import json
import re
import struct
import sys
import zlib
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

from .model import check_modifiers
from .specs import SPECS, WmmaSpec

TARGET = "gfx1201"
MATRIX_MAGIC = b"AMDMAT02"
STATE_MAGIC = b"AMDST002"
MANIFEST_FORMAT = "amdgpu-matrix-model evidence manifest 2"
STATE_FILE = "state_gfx1201.mmstate"
MAX_WAVES = 4096
GENS = ("explicit", "raw", "finite", "clamp_stress")
_TAG = re.compile(r"[a-z0-9_]{1,64}")
_MASK64 = 0xFFFF_FFFF_FFFF_FFFF
_VALID_24_NIBBLES = (0x4, 0x8, 0xC, 0x9, 0xD, 0xE)


class EvidenceError(ValueError):
    """Malformed, inconsistent or unexpected evidence."""


def _strict_json(data: bytes, what: str):
    def no_dupes(pairs):
        keys = [k for k, _ in pairs]
        if len(keys) != len(set(keys)):
            raise EvidenceError(f"{what}: duplicate JSON key")
        return dict(pairs)

    try:
        return json.loads(data.decode("utf-8"), object_pairs_hook=no_dupes)
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise EvidenceError(f"{what}: invalid JSON ({e})") from None


def _inflate_exact(blob: bytes, what: str) -> bytes:
    d = zlib.decompressobj()
    try:
        data = d.decompress(blob)
    except zlib.error as e:
        raise EvidenceError(f"{what}: bad zlib stream ({e})") from None
    if not d.eof or d.unused_data or d.unconsumed_tail:
        raise EvidenceError(f"{what}: zlib stream truncated or followed by trailing bytes")
    return data


def is_u(v, bits: int) -> bool:
    return type(v) is int and 0 <= v < (1 << bits)


def _words_le(values: Sequence[int]) -> bytes:
    arr = array.array("I", values)
    if sys.byteorder != "little":
        arr.byteswap()
    return arr.tobytes()


# --- deterministic input generator -------------------------------------------------------------


class SplitMix64:
    """Standard 64-bit SplitMix64 PRNG."""

    __slots__ = ("state",)

    def __init__(self, seed: int) -> None:
        self.state = seed & _MASK64

    def next_u64(self) -> int:
        s = (self.state + 0x9E37_79B9_7F4A_7C15) & _MASK64
        self.state = s
        z = ((s ^ (s >> 30)) * 0xBF58_476D_1CE4_E5B9) & _MASK64
        z = ((z ^ (z >> 27)) * 0x94D0_49BB_1331_11EB) & _MASK64
        return (z ^ (z >> 31)) & _MASK64

    def next_u32(self) -> int:
        return self.next_u64() & 0xFFFF_FFFF


def finite_word(u: int, fmt: str, i: int, count: int) -> int:
    """Map raw word `u` (index `i` of `count`) to a finite word of float format `fmt` (EVIDENCE.md)."""
    if fmt == "f16":
        out = 0
        for sh in (0, 16):
            h = (u >> sh) & 0xFFFF
            if (h & 0x7C00) == 0x7C00:
                h &= 0x83FF
            out |= h << sh
        return out
    if fmt == "bf16":
        out = 0
        for sh in (0, 16):
            h = (u >> sh) & 0xFFFF
            if (h & 0x0F) == 0:
                h &= 0x807F
            elif i < count // 2:
                h = (h & 0x807F) | ((96 + (((h >> 7) & 0xFF) & 0x3F)) << 7)
            elif (h & 0x7F80) == 0x7F80:
                h &= 0x807F
            out |= h << sh
        return out
    if fmt in ("fp8", "bf8"):
        out = 0
        for sh in (0, 8, 16, 24):
            b = (u >> sh) & 0xFF
            if fmt == "fp8" and (b & 0x7F) == 0x7F:
                b &= 0xFE
            elif fmt == "bf8" and (b & 0x7C) == 0x7C:
                b &= 0x83
            out |= b << sh
        return out
    if fmt == "f32":
        if (u & 0x0F) == 0:
            return u & 0x807F_FFFF
        if i < (count * 3) // 4:
            return (u & 0x807F_FFFF) | ((96 + (((u >> 23) & 0xFF) & 0x3F)) << 23)
        if (u & 0x7F80_0000) == 0x7F80_0000:
            return u & 0x807F_FFFF
        return u
    raise EvidenceError(f"gen 'finite' has no mapping for format {fmt!r}")


def generate_chunk_inputs(
    spec: WmmaSpec, waves: int, seed: int, gen: str
) -> tuple[list[int], list[int], list[int], list[int] | None]:
    """Regenerate (A, B, C, IDX) u32 word lists of a random chunk from (seed, gen) with SplitMix64."""
    rng = SplitMix64(seed)
    na, nb, nc = waves * 32 * spec.a_dwords, waves * 32 * spec.b_dwords, waves * 32 * spec.cd_dwords
    if gen == "raw":
        a = [rng.next_u32() for _ in range(na)]
        b = [rng.next_u32() for _ in range(nb)]
        c = [rng.next_u32() for _ in range(nc)]
    elif gen == "finite":
        a = [finite_word(rng.next_u32(), spec.a_fmt, i, na) for i in range(na)]
        b = [finite_word(rng.next_u32(), spec.b_fmt, i, nb) for i in range(nb)]
        c = [finite_word(rng.next_u32(), spec.d_fmt, i, nc) for i in range(nc)]
    elif gen == "clamp_stress":
        a = [rng.next_u32() for _ in range(na)]
        b = [rng.next_u32() for _ in range(nb)]
        c = [
            ((0x7FFF_FFFF if i < nc // 2 else 0x8000_0000) + ((rng.next_u32() & 0xFFFF) - 32768)) & 0xFFFF_FFFF
            for i in range(nc)
        ]
    else:
        raise EvidenceError(f"unknown chunk generator {gen!r}")
    idx = None
    if spec.sparse:
        idx = []
        for i in range(waves * 32):
            if i < (waves // 2) * 32:
                u = rng.next_u64()
                idx.append(sum(_VALID_24_NIBBLES[((u >> (8 * n)) & 0xFF) % 6] << (4 * n) for n in range(8)))
            else:
                idx.append(rng.next_u32())
    return a, b, c, idx


def inputs_sha256(a: Sequence[int], b: Sequence[int], c: Sequence[int], idx: Sequence[int] | None) -> str:
    """SHA-256 of the little-endian u32 concatenation A || B || C || IDX (IDX only for SWMMAC)."""
    h = hashlib.sha256()
    for part in (a, b, c) + ((idx,) if idx is not None else ()):
        h.update(_words_le(part))
    return h.hexdigest()


# --- .mmbin ------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Chunk:
    tag: str
    gen: str
    waves: int
    seed: int | None = None
    index_key: int = 0
    neg_lo: tuple[int, int, int] = (0, 0, 0)
    neg_hi: tuple[int, int, int] = (0, 0, 0)
    clamp: bool = False

    def header(self) -> dict:
        """Canonical header entry: optional fields are present only when they differ from the default."""
        h: dict = {"tag": self.tag, "gen": self.gen, "waves": self.waves}
        if self.seed is not None:
            h["seed"] = self.seed
        if self.index_key:
            h["index_key"] = self.index_key
        if any(self.neg_lo):
            h["neg_lo"] = list(self.neg_lo)
        if any(self.neg_hi):
            h["neg_hi"] = list(self.neg_hi)
        if self.clamp:
            h["clamp"] = True
        return h

    def word_counts(self, spec: WmmaSpec) -> tuple[int, int, int, int, int]:
        """Stored payload words (A, B, C, IDX, D) of this chunk."""
        w32 = self.waves * 32
        nd = w32 * spec.cd_dwords
        if self.gen != "explicit":
            return 0, 0, 0, 0, nd
        return w32 * spec.a_dwords, w32 * spec.b_dwords, nd, (w32 if spec.sparse else 0), nd


def parse_chunk(spec: WmmaSpec, raw) -> Chunk:
    """Validate one header chunk entry against the schema and the instruction."""
    if not isinstance(raw, dict):
        raise EvidenceError("chunk entry must be an object")
    allowed = {"tag", "gen", "waves", "seed", "index_key", "neg_lo", "neg_hi", "clamp"}
    if not {"tag", "gen", "waves"} <= raw.keys() <= allowed:
        raise EvidenceError(f"chunk fields {sorted(raw)} do not match the schema")
    tag, gen, waves = raw["tag"], raw["gen"], raw["waves"]
    if not isinstance(tag, str) or not _TAG.fullmatch(tag):
        raise EvidenceError(f"bad chunk tag {tag!r}")
    if gen not in GENS:
        raise EvidenceError(f"{tag}: unknown gen {gen!r}")
    if gen == "finite" and spec.is_int or gen == "clamp_stress" and not spec.is_int:
        raise EvidenceError(f"{tag}: gen {gen!r} does not apply to {spec.name}")
    if not is_u(waves, 32) or not 1 <= waves <= MAX_WAVES:
        raise EvidenceError(f"{tag}: waves must be an integer in 1..{MAX_WAVES}")
    seed = raw.get("seed")
    if (gen == "explicit") != (seed is None) or (seed is not None and not is_u(seed, 64)):
        raise EvidenceError(f"{tag}: random chunks need a u64 seed, explicit chunks none")
    ik = raw.get("index_key", 0)
    if "index_key" in raw and (ik != 1 or type(ik) is not int or not spec.has_index_key):
        raise EvidenceError(f"{tag}: index_key is written only as 1, on K=32 SWMMAC forms")
    mods = []
    for key in ("neg_lo", "neg_hi"):
        v = raw.get(key, [0, 0, 0])
        if key in raw and (not isinstance(v, list) or not any(v)):
            raise EvidenceError(f"{tag}: {key} is written only when nonzero, as a list")
        mods.append(v)
    cl = raw.get("clamp", False)
    if "clamp" in raw and cl is not True:
        raise EvidenceError(f"{tag}: clamp is written only as true")
    try:
        lo, hi = check_modifiers(spec, mods[0], mods[1], cl)
    except ValueError as e:
        raise EvidenceError(f"{tag}: {e}") from None
    return Chunk(tag, gen, waves, seed, ik, lo, hi, cl)


@dataclass
class MatrixFile:
    spec: WmmaSpec
    chunks: list[Chunk]
    words: array.array


def read_matrix_file(path: Path, expected_spec: str | None = None) -> MatrixFile:
    """Read and fully check an `.mmbin` file (magic, header schema, target, spec, payload size)."""
    raw = path.read_bytes()
    if len(raw) < 12 or raw[:8] != MATRIX_MAGIC:
        raise EvidenceError(f"{path.name}: not an {MATRIX_MAGIC.decode()} file")
    hdr_len = struct.unpack_from("<I", raw, 8)[0]
    if 12 + hdr_len > len(raw):
        raise EvidenceError(f"{path.name}: header length past end of file")
    header = _strict_json(raw[12 : 12 + hdr_len], path.name)
    if not isinstance(header, dict) or set(header) != {"target", "spec", "chunks"}:
        raise EvidenceError(f"{path.name}: header must have exactly target, spec, chunks")
    if header["target"] != TARGET:
        raise EvidenceError(f"{path.name}: target {header['target']!r} is not {TARGET}")
    name = header["spec"]
    if name not in SPECS:
        raise EvidenceError(f"{path.name}: unknown instruction {name!r}")
    if expected_spec is not None and name != expected_spec:
        raise EvidenceError(f"{path.name}: header names {name}, expected {expected_spec}")
    spec = SPECS[name]
    if not isinstance(header["chunks"], list) or not header["chunks"]:
        raise EvidenceError(f"{path.name}: chunks must be a non-empty list")
    chunks = [parse_chunk(spec, c) for c in header["chunks"]]
    if len({c.tag for c in chunks}) != len(chunks):
        raise EvidenceError(f"{path.name}: duplicate chunk tags")
    payload = _inflate_exact(raw[12 + hdr_len :], path.name)
    expect = sum(sum(c.word_counts(spec)) for c in chunks)
    if len(payload) != 4 * expect:
        raise EvidenceError(f"{path.name}: payload has {len(payload)} bytes, header implies {4 * expect}")
    words = array.array("I")
    words.frombytes(payload)
    if sys.byteorder != "little":
        words.byteswap()
    return MatrixFile(spec, chunks, words)


def iter_chunk_data(mf: MatrixFile) -> Iterator[tuple[Chunk, list[int], list[int], list[int], list[int] | None, list[int]]]:
    """Yield (chunk, A, B, C, IDX, D_hw) word lists; random chunk inputs are regenerated."""
    pos = 0
    for ch in mf.chunks:
        na, nb, nc, ni, nd = ch.word_counts(mf.spec)
        if ch.gen == "explicit":
            seg = mf.words[pos : pos + na + nb + nc + ni].tolist()
            a, b, c = seg[:na], seg[na : na + nb], seg[na + nb : na + nb + nc]
            idx = seg[na + nb + nc :] if mf.spec.sparse else None
            pos += na + nb + nc + ni
        else:
            a, b, c, idx = generate_chunk_inputs(mf.spec, ch.waves, ch.seed, ch.gen)
        d = mf.words[pos : pos + nd].tolist()
        pos += nd
        yield ch, a, b, c, idx, d


def write_matrix_file(path: Path, spec: WmmaSpec, chunks: Sequence[Chunk], words: Sequence[int]) -> None:
    """Write an `.mmbin` file (magic, u32 header length, canonical JSON header, zlib u32 payload)."""
    header = {"target": TARGET, "spec": spec.name, "chunks": [c.header() for c in chunks]}
    hdr = json.dumps(header, separators=(",", ":")).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(MATRIX_MAGIC + struct.pack("<I", len(hdr)) + hdr + zlib.compress(_words_le(words), level=9))


# --- .mmstate ----------------------------------------------------------------------------------


def read_state_file(path: Path) -> dict:
    """Read an `.mmstate` file into its JSON document (schema checked by the validator)."""
    raw = path.read_bytes()
    if len(raw) < 8 or raw[:8] != STATE_MAGIC:
        raise EvidenceError(f"{path.name}: not an {STATE_MAGIC.decode()} file")
    doc = _strict_json(_inflate_exact(raw[8:], path.name), path.name)
    if not isinstance(doc, dict) or set(doc) != {"target", "records"} or doc["target"] != TARGET:
        raise EvidenceError(f"{path.name}: document must be {{target: {TARGET}, records: [...]}}")
    if not isinstance(doc["records"], list) or not doc["records"]:
        raise EvidenceError(f"{path.name}: records must be a non-empty list")
    return doc


def write_state_file(path: Path, records: list[dict]) -> None:
    doc = json.dumps({"target": TARGET, "records": records}, separators=(",", ":")).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(STATE_MAGIC + zlib.compress(doc, level=9))


# --- manifest ----------------------------------------------------------------------------------


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def expected_files() -> list[str]:
    """The complete release inventory: one `.mmbin` per instruction plus the state file."""
    return [f"{name}.mmbin" for name in SPECS] + [STATE_FILE]


def read_manifest(path: Path) -> dict:
    return _strict_json(path.read_bytes(), path.name)


# --- state inventory of this release (tools/probe_state.py captures exactly this) --------------

CONTROL_WRITES = (0x00F0, 0x00F1, 0x00F2, 0x00F3, 0x00F4, 0x00F8, 0x00FC, 0x00C0, 0x00D0, 0x00E0, 0x0030, 0x0070, 0x00B0, 0x01F0, 0x02F0, 0xFFF0)
WMMA_MODES = (0x00F0, 0x00FF, 0x00F5, 0x00FA, 0x0000, 0x0055, 0x00AA, 0x000F, 0x01F0, 0x02F0, 0x03F0, 0x0FF0, 0x0F00, 0xFFF0)
EXEC_MASKS = (0xFFFF_FFFF, 0x0000_FFFF, 0xFFFF_0000, 0x5555_5555, 0xA5A5_A5A5, 0x8000_0000, 0x0000_0001, 0x0000_0000)
DEFAULT_MODE = 0x00F0  # kernel descriptor: round to nearest even, f32 and f16/f64 denormals kept
STATE_PREFILL = 0xDEAD_BEEF
STATE_STORE_DWORDS = 8

# Scalar operands of the MODE control rows: (a, b); f16 operands are 16-bit values.
CONTROL_OPERANDS = {
    "f32_add_rne_tie": (0x3F80_0000, 0x3380_0000),  # 1 + 2^-24
    "f32_add_denorm_in": (0x0040_0000, 0x0040_0000),  # 2^-127 + 2^-127 = 2^-126
    "f32_sub_denorm_out": (0x00C0_0000, 0x0080_0000),  # 1.5 * 2^-126 - 2^-126 = 2^-127
    "f16_add_rne_tie": (0x3C00, 0x1000),  # 1 + 2^-11
    "f16_add_denorm_in": (0x0200, 0x0200),  # 2^-15 + 2^-15 = 2^-14
    "f16_sub_denorm_out": (0x0600, 0x0400),  # 1.5 * 2^-14 - 2^-14 = 2^-15
}
CONTROL_RESULTS = ("init", "read", "read_after_s_round_mode_15", "read_after_s_denorm_mode_15", *CONTROL_OPERANDS)


def state_probes(spec: WmmaSpec) -> tuple[str, ...]:
    """WMMA state probes captured for `spec`."""
    probes = ("mode", "exec")
    if not spec.is_int:
        probes += ("baseline", "clamp")
        if not spec.sparse:
            probes += ("op_sel", "op_sel_hi")
    return probes


def probe_runs(probe: str) -> list[tuple[int, int]]:
    """(mode, exec_lo) of every run of a probe."""
    if probe == "mode":
        return [(m, 0xFFFF_FFFF) for m in WMMA_MODES]
    if probe == "exec":
        return [(DEFAULT_MODE, e) for e in EXEC_MASKS]
    return [(DEFAULT_MODE, 0xFFFF_FFFF)]


def probe_fields(spec: WmmaSpec, probe: str) -> dict:
    """VOP3P fields of a probe's encoding (everything else as llvm-mc encodes the plain instruction)."""
    return {
        "index_key": 0,
        "neg_lo": [0, 0, 0],
        "neg_hi": [0, 0, 0],
        "clamp": probe == "clamp",
        "op_sel": 7 if probe == "op_sel" else 0,
        "op_sel_hi": 0 if probe == "op_sel_hi" else 7,
    }


def decode_vop3p(dw0: int, dw1: int) -> dict:
    """VOP3P fields (RDNA4 ISA section 15.3.7): dw0 VDST[7:0] NEG_HI[10:8] OPSEL[13:11] OPSEL_HI[2]:14
    CLAMP:15 OP[22:16] ENCODING[31:24]=0xCC; dw1 SRC0[8:0] SRC1[17:9] SRC2[26:18] OPSEL_HI[1:0]:[28:27] NEG[31:29]."""
    return {
        "encoding": dw0 >> 24,
        "op": (dw0 >> 16) & 0x7F,
        "neg_hi": [(dw0 >> (8 + i)) & 1 for i in range(3)],
        "op_sel": (dw0 >> 11) & 7,
        "op_sel_hi": (((dw0 >> 14) & 1) << 2) | ((dw1 >> 27) & 3),
        "clamp": bool((dw0 >> 15) & 1),
        "neg_lo": [(dw1 >> (29 + i)) & 1 for i in range(3)],
    }
