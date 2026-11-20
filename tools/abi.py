"""AMDHSA v6 kernel descriptor and metadata wrapper for gfx1201 wave32 kernels."""

from __future__ import annotations

_HEAD = """\
\t.amdgcn_target "amdgcn-amd-amdhsa--gfx1201"
\t.amdhsa_code_object_version 6
"""

_KERNEL = """\
\t.text
\t.globl {sym}
\t.p2align 8
\t.type {sym},@function
{sym}:
{body}
\t.section .rodata,"a",@progbits
\t.p2align 6
\t.amdhsa_kernel {sym}
\t\t.amdhsa_group_segment_fixed_size 0
\t\t.amdhsa_private_segment_fixed_size 0
\t\t.amdhsa_kernarg_size {kernarg}
\t\t.amdhsa_user_sgpr_count 2
\t\t.amdhsa_user_sgpr_kernarg_segment_ptr 1
\t\t.amdhsa_wavefront_size32 1
\t\t.amdhsa_enable_private_segment 0
\t\t.amdhsa_system_sgpr_workgroup_id_x 1
\t\t.amdhsa_system_sgpr_workgroup_id_y 0
\t\t.amdhsa_system_sgpr_workgroup_id_z 0
\t\t.amdhsa_system_vgpr_workitem_id 0
\t\t.amdhsa_next_free_vgpr {vgprs}
\t\t.amdhsa_next_free_sgpr {sgprs}
\t\t.amdhsa_reserve_vcc 1
\t\t.amdhsa_float_round_mode_32 0
\t\t.amdhsa_float_denorm_mode_32 3
\t\t.amdhsa_float_denorm_mode_16_64 3
\t\t.amdhsa_workgroup_processor_mode 1
\t\t.amdhsa_memory_ordered 1
\t\t.amdhsa_forward_progress 1
\t.end_amdhsa_kernel
"""

_META_HEAD = """\
\t.amdgpu_metadata
---
amdhsa.version: [ 1, 2 ]
amdhsa.target: amdgcn-amd-amdhsa--gfx1201
amdhsa.kernels:
"""

_META_KERNEL = """\
  - .name: {sym}
    .symbol: {sym}.kd
    .kernarg_segment_size: {kernarg}
    .kernarg_segment_align: 8
    .group_segment_fixed_size: 0
    .private_segment_fixed_size: 0
    .wavefront_size: 32
    .sgpr_count: {sgprs}
    .vgpr_count: {vgprs}
    .max_flat_workgroup_size: 32
    .uniform_work_group_size: 1
    .workgroup_processor_mode: 1
    .args:
      - {{ .offset: 0, .size: {kernarg}, .value_kind: by_value }}
"""

_META_TAIL = """\
...
\t.end_amdgpu_metadata
"""


def wrap_kernels(
    kernels: list[tuple[str, str]],
    kernarg: int = 40,
    vgprs: int = 40,
    sgprs: int = 24,
) -> str:
    """Wrap one or more (symbol, body) kernels into a gfx1201 assembly module."""
    code = "".join(
        _KERNEL.format(sym=sym, body=body.rstrip(), kernarg=kernarg, vgprs=vgprs, sgprs=sgprs)
        for sym, body in kernels
    )
    meta = "".join(
        _META_KERNEL.format(sym=sym, kernarg=kernarg, vgprs=vgprs, sgprs=sgprs)
        for sym, _ in kernels
    )
    return _HEAD + code + _META_HEAD + meta + _META_TAIL
