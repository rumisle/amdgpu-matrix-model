"""GPU-free unit tests of the model API: documented examples, layout, argument checks."""

from __future__ import annotations

import unittest

from amdgpu_matrix_model import (
    SPECS,
    f32_to_bits,
    pack_a,
    pack_b,
    select_sparse_b,
    slot_order,
    step_schedule,
    unpack_a,
    wmma_dot,
    wmma_matrix,
    wmma_wave,
)


class Examples(unittest.TestCase):
    def test_readme(self):
        self.assertEqual(wmma_dot("v_wmma_f32_16x16x16_f16", [0x3C00] * 16, [0x3C00] * 16, 0), f32_to_bits(16.0))
        self.assertEqual(wmma_dot("v_wmma_f32_16x16x16_fp8_bf8", [0x38] * 16, [0x40] * 16, 0), f32_to_bits(32.0))

    def test_isa_16bit_layout(self):
        """RDNA4 ISA 7.12.2: 16-bit A[0][4] sits in lane 16, VGPR 0, low half."""
        spec = SPECS["v_wmma_f32_16x16x16_f16"]
        a = [[0] * 16 for _ in range(16)]
        a[0][4] = 0x1234
        words = pack_a(spec, a)
        self.assertEqual(words[16 * 4 + 0], 0x1234)
        self.assertEqual(unpack_a(spec, words), a)
        self.assertEqual(slot_order(spec, "a")[:12], (0, 1, 2, 3, 8, 9, 10, 11, 4, 5, 6, 7))

    def test_step_schedules(self):
        self.assertEqual(step_schedule(SPECS["v_wmma_f16_16x16x16_f16"]), tuple(tuple(range(s, s + 4)) for s in (0, 4, 8, 12)))
        self.assertEqual(step_schedule(SPECS["v_swmmac_f16_16x16x32_f16"]), tuple(tuple(range(s, s + 4)) for s in (0, 8, 4, 12)))
        self.assertEqual(step_schedule(SPECS["v_wmma_f32_16x16x16_bf8_bf8"]), (tuple(range(8)), tuple(range(8, 16))))

    def test_logical_and_register_agree(self):
        """The reviewer's witness: the same logical matrices through wmma_dot and through ISA-packed registers."""
        name = "v_wmma_f32_16x16x16_f16"
        spec = SPECS[name]
        a = [[0] * 16 for _ in range(16)]
        b = [[0] * 16 for _ in range(16)]
        a[0][0], a[0][4], a[0][8] = 0x7800, 0xF800, 0x3C00
        b[0][0], b[4][0], b[8][0] = 0x7800, 0x7800, 0x3C00
        d = wmma_wave(name, pack_a(spec, a), pack_b(spec, b), [0] * 256)
        self.assertEqual(d[0], wmma_dot(name, a[0], [b[k][0] for k in range(16)], 0))
        self.assertEqual(d[0], f32_to_bits(1.0))

    def test_bf16_negative_zero(self):
        a, b = [0] * 16, [0] * 16
        a[12], b[12] = 0x8001, 0x3F00  # -2^-134 in the last step
        self.assertEqual(wmma_dot("v_wmma_bf16_16x16x16_bf16", a, b, 0), 0x8000)
        a[12], a[0], b[0] = 0, 0x8001, 0x3F00  # same in the first step: later steps turn it into +0
        self.assertEqual(wmma_dot("v_wmma_bf16_16x16x16_bf16", a, b, 0), 0x0000)

    def test_tiny_accumulator(self):
        a, b = [0] * 16, [0] * 16
        a[0], a[1], b[0], b[1] = 0x4000, 0xC000, 0x4000, 0x4000  # +4, -4: grid 2 - 24 = -22 <= -14
        self.assertEqual(wmma_dot("v_wmma_f16_16x16x16_f16", a, b, 0x8001), 0x8004)
        a[0], a[1], b[0], b[1] = 0x7800, 0xF800, 0x7800, 0x7800  # +-2^30: grid 6 > -14
        self.assertEqual(wmma_dot("v_wmma_f16_16x16x16_f16", a, b, 0x8001), 0x0000)


class Arguments(unittest.TestCase):
    def test_lengths_and_ranges(self):
        f = "v_wmma_f32_16x16x16_f16"
        for bad in ([0] * 15, [0] * 17, [0x10000] + [0] * 15, [-1] + [0] * 15, [True] + [0] * 15):
            with self.assertRaises(ValueError):
                wmma_dot(f, bad, [0] * 16, 0)
        with self.assertRaises(ValueError):
            wmma_dot("v_wmma_i32_16x16x16_iu8", [0] * 15, [0] * 16, 0)
        with self.assertRaises(ValueError):
            wmma_wave(f, [0] * 127, [0] * 128, [0] * 256)
        with self.assertRaises(ValueError):
            wmma_wave(f, [0] * 128, [0] * 128, [0] * 256 + [0])
        with self.assertRaises(ValueError):
            wmma_dot("not_an_instruction", [0] * 16, [0] * 16, 0)

    def test_modifier_domain(self):
        cases = [
            ("v_wmma_i32_16x16x16_iu8", (0, 0, 1), (0, 0, 0)),
            ("v_wmma_i32_16x16x16_iu8", (0, 0, 0), (1, 0, 0)),
            ("v_wmma_f32_16x16x16_fp8_fp8", (1, 0, 0), (0, 0, 0)),
            ("v_swmmac_f32_16x16x32_f16", (0, 0, 1), (0, 0, 0)),
            ("v_swmmac_f32_16x16x32_fp8_fp8", (0, 0, 0), (0, 0, 1)),
            ("v_wmma_f32_16x16x16_f16", (2, 0, 0), (0, 0, 0)),
            ("v_wmma_f32_16x16x16_f16", (0, 0), (0, 0, 0)),
        ]
        for name, lo, hi in cases:
            spec = SPECS[name]
            idx = [0] * spec.k_a if spec.sparse else None
            with self.assertRaises(ValueError, msg=(name, lo, hi)):
                wmma_dot(name, [0] * spec.k_a, [0] * spec.k_b, 0, idx_vec=idx, neg_lo=lo, neg_hi=hi)
        wmma_dot("v_wmma_f32_16x16x16_fp8_fp8", [0] * 16, [0] * 16, 0, neg_lo=(0, 0, 1), neg_hi=(0, 0, 1))

    def test_index_key_and_state(self):
        with self.assertRaises(ValueError):
            wmma_wave("v_swmmac_i32_16x16x64_iu4", [0] * 64, [0] * 128, [0] * 256, idx_words=[0] * 32, index_key=1)
        with self.assertRaises(ValueError):
            wmma_wave("v_swmmac_f32_16x16x32_f16", [0] * 128, [0] * 256, [0] * 256, idx_words=[0] * 32, index_key=2)
        with self.assertRaises(ValueError):
            wmma_wave("v_wmma_f32_16x16x16_f16", [0] * 128, [0] * 128, [0] * 256, idx_words=[0] * 32)
        with self.assertRaises(ValueError):
            wmma_wave("v_wmma_i32_16x16x16_iu8", [0] * 64, [0] * 64, [0] * 256, op_sel=1)
        with self.assertRaises(ValueError):
            wmma_wave("v_wmma_f32_16x16x16_f16", [0] * 128, [0] * 128, [0] * 256, mode=0x10000)

    def test_select_sparse_b(self):
        self.assertEqual(select_sparse_b([10, 20, 30, 40], [1, 3]), [20, 40])
        for b, idx in (([10, 20, 30, 40], [4, 0]), ([10, 20, 30, 40], [1]), ([10, 20, 30], [0, 1]), ([10, 20, 30, 40], [True, 0])):
            with self.assertRaises(ValueError):
                select_sparse_b(b, idx)
        with self.assertRaises(ValueError):
            wmma_wave("v_wmma_f32_16x16x16_f16", [0] * 128, [0] * 128, [0] * 256, index_key=False)

    def test_matrix_shape(self):
        with self.assertRaises(ValueError):
            wmma_matrix("v_wmma_f32_16x16x16_f16", [[0] * 16] * 15, [[0] * 16] * 16, [[0] * 16] * 16)


if __name__ == "__main__":
    unittest.main()
