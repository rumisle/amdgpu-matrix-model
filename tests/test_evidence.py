"""GPU-free tests of the evidence formats, the validator's integrity gate, and rule-specific mutants.

Every spoof must be rejected (exit 2) and every mutant must fail (exit 1) on exactly the instruction
files whose rule it changes. Run: uv run --no-project python -m unittest discover -s tests
"""

from __future__ import annotations

import hashlib
import io
import json
import shutil
import struct
import tempfile
import unittest
import zlib
from pathlib import Path
from unittest import mock

import amdgpu_matrix_model.model as M
from amdgpu_matrix_model.evidence import STATE_MAGIC, SplitMix64, generate_chunk_inputs, inputs_sha256
from amdgpu_matrix_model.specs import SPECS
from amdgpu_matrix_model.validation import RELEASE_MANIFEST_SHA256, validate

ROOT = Path(__file__).resolve().parent.parent
EVIDENCE = ROOT / "evidence"
F16_FORMS = [n for n, s in SPECS.items() if s.a_fmt in ("f16", "bf16")]


def run(ev=EVIDENCE, variants=None, sha=RELEASE_MANIFEST_SHA256, model=M.wmma_wave) -> tuple[int, str]:
    out = io.StringIO()
    rc = validate(ev, variants, 1, sha, model=model, out=out)
    return rc, out.getvalue()


def failing(text: str) -> set[str]:
    return {line.split()[1] for line in text.splitlines() if line.startswith("[FAIL]")}


class Generator(unittest.TestCase):
    def test_splitmix64_reference_vectors(self):
        r = SplitMix64(0)
        self.assertEqual([r.next_u64() for _ in range(3)], [0xE220A8397B1DCDAF, 0x6E789E6AA1B965F4, 0x06C45D188009454F])
        r = SplitMix64(1234567)
        self.assertEqual([r.next_u64() for _ in range(2)], [6457827717110365317, 3203168211198807973])

    def test_generator_golden_vectors(self):
        """Published in EVIDENCE.md."""
        golden = [
            ("v_wmma_f32_16x16x16_f16", 202601, "finite", (0xBEEF724A, 0x24AB6025), "8d16a0f7b928bac4"),
            ("v_wmma_bf16_16x16x16_bf16", 7, "finite", (0x49323DD7, 0xC43C361C), "f0c1b3827a5c6d9e"),
            ("v_swmmac_f32_16x16x32_fp8_bf8", 202650, "raw", (0x06630A18, 0xE88CD028), "c82b8f6e46144064"),
            ("v_wmma_i32_16x16x16_iu8", 3, "clamp_stress", (0xDB018FED, 0x7B81A989), "a3a2e53bc91f5326"),
        ]
        for name, seed, gen, first, digest in golden:
            a, b, c, idx = generate_chunk_inputs(SPECS[name], 2, seed, gen)
            self.assertEqual(tuple(a[:2]), first)
            self.assertTrue(inputs_sha256(a, b, c, idx).startswith(digest))


class Release(unittest.TestCase):
    def test_full_release_passes(self):
        rc, text = run()
        self.assertEqual(rc, 0, text)
        self.assertIn("FULL: 22/22", text)

    def test_partial_is_labelled(self):
        rc, text = run(variants=["v_wmma_i32_16x16x16_iu4"])
        self.assertEqual(rc, 0, text)
        self.assertIn("PARTIAL", text)

    def test_unknown_variant(self):
        self.assertEqual(run(variants=["v_wmma_i32_16x16x16_iu4", "not-an-instruction"])[0], 2)


class Spoofs(unittest.TestCase):
    """Malformed, partial and mislabeled corpora. A spoof that also rewrites the manifest is run with
    that manifest's hash trusted explicitly, so the later checks have to catch it."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.ev = self.tmp / "evidence"
        shutil.copytree(EVIDENCE, self.ev)

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def manifest(self) -> dict:
        return json.loads((self.ev / "manifest.json").read_text())

    def save_manifest(self, m: dict) -> str:
        path = self.ev / "manifest.json"
        path.write_text(json.dumps(m, indent=1) + "\n")
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def refresh(self, name: str) -> str:
        """Update the manifest size/hash of one file and return the new manifest hash."""
        m = self.manifest()
        for f in m["files"]:
            if f["file"] == name:
                p = self.ev / name
                f["bytes"], f["sha256"] = p.stat().st_size, hashlib.sha256(p.read_bytes()).hexdigest()
        m["totals"]["bytes"] = sum(f["bytes"] for f in m["files"])
        return self.save_manifest(m)

    def rewrite_header(self, name: str, edit) -> str:
        p = self.ev / name
        raw = p.read_bytes()
        n = struct.unpack_from("<I", raw, 8)[0]
        hdr = json.loads(raw[12 : 12 + n])
        payload = raw[12 + n :]
        hdr, payload = edit(hdr, payload)
        h = json.dumps(hdr, separators=(",", ":")).encode()
        p.write_bytes(raw[:8] + struct.pack("<I", len(h)) + h + payload)
        return self.refresh(name)

    def assertRejected(self, reason: str, sha=RELEASE_MANIFEST_SHA256, variants=("v_wmma_i32_16x16x16_iu4",)):
        """Exit 2 with `reason` in the report. Only the spoofed file is parsed; the inventory is always checked."""
        rc, text = run(self.ev, list(variants), sha)
        self.assertEqual(rc, 2, text)
        self.assertIn(reason, text)

    def test_missing_file(self):
        (self.ev / "v_wmma_i32_16x16x16_iu4.mmbin").unlink()
        self.assertRejected("evidence directory holds")

    def test_missing_state(self):
        (self.ev / "state_gfx1201.mmstate").unlink()
        self.assertRejected("evidence directory holds")

    def test_extra_file(self):
        shutil.copy(self.ev / "v_wmma_i32_16x16x16_iu4.mmbin", self.ev / "extra.mmbin")
        self.assertRejected("evidence directory holds")

    def test_untrusted_manifest(self):
        m = self.manifest()
        m["target"] = "gfx0000"
        self.save_manifest(m)
        self.assertRejected("is not the trusted")

    def test_manifest_drops_an_entry(self):
        m = self.manifest()
        m["files"] = m["files"][1:]
        self.assertRejected("does not list exactly the release inventory", self.save_manifest(m))

    def test_wrong_target(self):
        def edit(h, p):
            h["target"] = "gfx0000"
            return h, p

        self.assertRejected("is not gfx1201", self.rewrite_header("v_wmma_i32_16x16x16_iu4.mmbin", edit))

    def test_mislabeled_instruction(self):
        """Genuine iu4 bytes stored under the iu8 name (manifest updated to match)."""
        shutil.copy(self.ev / "v_wmma_i32_16x16x16_iu4.mmbin", self.ev / "v_wmma_i32_16x16x16_iu8.mmbin")
        sha = self.refresh("v_wmma_i32_16x16x16_iu8.mmbin")
        self.assertRejected("header names v_wmma_i32_16x16x16_iu4, expected v_wmma_i32_16x16x16_iu8", sha, ["v_wmma_i32_16x16x16_iu8"])

    def test_unknown_header_field(self):
        def edit(h, p):
            h["kind"] = "not-hardware"
            return h, p

        self.assertRejected("header must have exactly", self.rewrite_header("v_wmma_i32_16x16x16_iu4.mmbin", edit))

    def test_zero_negative_and_empty_chunks(self):
        spoofs = [
            ([], "chunks must be a non-empty list"),
            ([{"tag": "x", "gen": "raw", "waves": 0, "seed": 0}], "waves must be"),
            ([{"tag": "missing", "gen": "raw", "waves": 1, "seed": 0}, {"tag": "negative", "gen": "raw", "waves": -1, "seed": 0}], "waves must be"),
        ]
        for chunks, reason in spoofs:
            def edit(h, p, chunks=chunks):
                h["chunks"] = chunks
                return h, zlib.compress(b"")

            self.assertRejected(reason, self.rewrite_header("v_wmma_i32_16x16x16_iu4.mmbin", edit))

    def test_trailing_bytes(self):
        self.assertRejected("trailing bytes", self.rewrite_header("v_wmma_i32_16x16x16_iu4.mmbin", lambda h, p: (h, p + b"junk")))

    def test_short_payload(self):
        def edit(h, p):
            return h, zlib.compress(zlib.decompress(p)[:-4])

        self.assertRejected("header implies", self.rewrite_header("v_wmma_i32_16x16x16_iu4.mmbin", edit))

    def test_state_record_removed_or_relabelled(self):
        p = self.ev / "state_gfx1201.mmstate"
        doc = json.loads(zlib.decompress(p.read_bytes()[8:]))
        for change, reason in (("drop", "differ from the release inventory"), ("relabel", "unknown probe"), ("mode", "runs differ")):
            d = json.loads(json.dumps(doc))
            if change == "drop":
                d["records"].pop()
            elif change == "relabel":
                d["records"][-1]["probe"] = "FAKE"
            else:
                rec = next(r for r in d["records"] if r["kind"] == "wmma" and r["probe"] == "mode")
                rec["runs"][1]["mode"] = 0x1234
            p.write_bytes(STATE_MAGIC + zlib.compress(json.dumps(d).encode()))
            self.assertRejected(reason, self.refresh("state_gfx1201.mmstate"), ["state"])

    def test_short_or_empty_prediction(self):
        for bad in (lambda *a, **k: [], lambda *a, **k: M.wmma_wave(*a, **k)[:-1]):
            rc, text = run(self.ev, ["v_swmmac_i32_16x16x64_iu4"], model=bad)
            self.assertEqual(rc, 2, text)
            self.assertIn("words, hardware has", text)

    def test_input_hash_binds_regeneration(self):
        """A generator that flips the sign of every float A/B word would pass on outputs alone."""
        import amdgpu_matrix_model.evidence as E

        orig = E.generate_chunk_inputs

        def flipped(spec, waves, seed, gen):
            a, b, c, idx = orig(spec, waves, seed, gen)
            if spec.a_fmt in ("f16", "bf16"):
                a = [w ^ 0x8000_8000 for w in a]
                b = [w ^ 0x8000_8000 for w in b]
            return a, b, c, idx

        with mock.patch.object(E, "generate_chunk_inputs", flipped):
            rc, text = run(self.ev, ["v_wmma_f32_16x16x16_f16"])
        self.assertEqual(rc, 2, text)
        self.assertIn("input SHA-256 differ", text)


def _patched_step(change):
    orig = M.dot_float_step

    def step(a, b, acc, af, bf, df, sg, inv):
        return change(orig, a, b, acc, af, bf, df, sg, inv)

    return mock.patch.object(M, "dot_float_step", step)


class Mutants(unittest.TestCase):
    """Each alternative rule must be rejected by the evidence of the instructions it changes."""

    def kills(self, patch, variants, expected=None):
        with patch:
            rc, text = run(variants=variants)
        self.assertEqual(rc, 1, text)
        self.assertEqual(failing(text), set(expected if expected is not None else [v for v in variants if v != "state"]), text)
        return text

    def test_f16_d_subgroup_4(self):
        def change(orig, a, b, acc, af, bf, df, sg, inv):
            return orig(a, b, acc, af, bf, df, 4 if df.name == "f16" else sg, inv)

        self.kills(_patched_step(change), ["v_wmma_f16_16x16x16_f16", "v_swmmac_f16_16x16x32_f16"])

    def test_accumulator_rules(self):
        bf16_in = [n for n in F16_FORMS if SPECS[n].a_fmt == "bf16"]
        f16_d = [n for n in F16_FORMS if SPECS[n].d_fmt == "f16"]
        alternatives = {
            "always zero": (lambda grid, d: True, f16_d + bf16_in),
            "always floor": (lambda grid, d: False, F16_FORMS),
            "f16 D only": (lambda grid, d: grid > d.emin if d.name == "f16" else True, bf16_in),
            "boundary +1": (lambda grid, d: grid > d.emin + 1, f16_d + bf16_in),
            "boundary -1": (lambda grid, d: grid > d.emin - 1, f16_d + bf16_in),
        }
        for label, (rule, expected) in alternatives.items():
            with self.subTest(label):
                self.kills(mock.patch.object(M, "_acc_drops_tiny_negative", rule), F16_FORMS, expected)

    def test_bf16_canonical_zero(self):
        def change(orig, a, b, acc, af, bf, df, sg, inv):
            r = orig(a, b, acc, af, bf, df, sg, inv)
            return 0 if df.name == "bf16" and r == 0x8000 else r

        self.kills(_patched_step(change), ["v_wmma_bf16_16x16x16_bf16", "v_swmmac_bf16_16x16x32_bf16"])

    def test_sparse_b_modifier_before_selection(self):
        orig_dot = M._dot

        def dot(spec, a, b, c, idx, lo, hi, clamp):
            if spec.sparse and not spec.is_int and (lo[1] or hi[1]):
                b = [(w ^ 0x8000) if (lo[1] if k % 2 == 0 else hi[1]) else w for k, w in enumerate(b)]
                lo, hi = (lo[0], 0, lo[2]), (hi[0], 0, hi[2])
            return orig_dot(spec, a, b, c, idx, lo, hi, clamp)

        sparse16 = [n for n in F16_FORMS if SPECS[n].sparse]
        self.kills(mock.patch.object(M, "_dot", dot), sparse16)

    def test_dense_16bit_old_step_order(self):
        orig = M.step_schedule

        def sched(spec):
            s = orig(spec)
            return (s[0], s[2], s[1], s[3]) if not spec.sparse and spec.a_bits == 16 else s

        dense16 = [n for n in F16_FORMS if not SPECS[n].sparse]
        self.kills(mock.patch.object(M, "step_schedule", sched), dense16)

    def test_state_aware_mutants(self):
        """Float CLAMP, MODE, EXEC and OPSEL honoured instead of ignored: killed by the state file."""
        mutants = {
            "clamp": lambda r, spec, k: [0] * len(r) if k["clamp"] and not spec.is_int else r,
            "mode": lambda r, spec, k: [0] * len(r) if k["mode"] != 0x00F0 else r,
            "exec": lambda r, spec, k: [0] * len(r) if k["exec_lo"] != 0xFFFF_FFFF else r,
            "op_sel": lambda r, spec, k: [w ^ 1 for w in r] if k["op_sel"] else r,
        }
        for label, change in mutants.items():
            def model(name, *a, change=change, **k):
                full = {"clamp": False, "mode": 0x00F0, "exec_lo": 0xFFFF_FFFF, "op_sel": 0, **k}
                return change(M.wmma_wave(name, *a, **k), SPECS[name], full)

            with self.subTest(label):
                rc, text = run(variants=["state"], model=model)
                self.assertEqual(rc, 1, f"{label}\n{text}")


if __name__ == "__main__":
    unittest.main()
