# Third-Party Notices

## rocjitsu (AMD)

`amdgpu_matrix_model/model.py` adapts, translated to Python and generalized over the format tables, the
float dot-step algorithm of these files of AMD's rocjitsu:

- `emulation/rocjitsu/lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/gfx12_dot.h`
- `emulation/rocjitsu/lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/dot_packed16.h`

from https://github.com/ROCm/rocm-systems at commit `4270d0f2bbcca1cb57c3ef4d8a441744767c9f20`. The adapted parts are `_shift_rne`, `pack_float`,
`_align_floor`, `_acc_drops_tiny_negative` and `dot_float_step`: exact products, pairwise floor alignment,
the accumulator grid, RNE packing, `f16` D direct rounding, `bf16` D truncation, the `f16` D
tiny-accumulator rule, and the special-value precedence. Their license:

```text
MIT License

Copyright (c) 2025-2026 Advanced Micro Devices, Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
associated documentation files (the "Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the
following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial
portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT
LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO
EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
USE OR OTHER DEALINGS IN THE SOFTWARE.
```
