"""Local HIP ctypes runner for gfx1201 WMMA / SWMMAC probe batches.

Requires ROCm HIP (`libamdhip64.so`) and `numpy` on the machine with the gfx1201 GPU.
"""

from __future__ import annotations

import ctypes as C
import os
from pathlib import Path

import numpy as np

from amdgpu_matrix_model.specs import SPECS


class Hip:
    def __init__(self, device: int = 0):
        self.lib = C.CDLL(os.environ.get("HIP_LIB", "/opt/rocm/lib/libamdhip64.so"))
        self._ok(self.lib.hipInit(0), "hipInit")
        self._ok(self.lib.hipSetDevice(device), "hipSetDevice")
        self._mod = C.c_void_p()
        self._co_buf = None
        self._fns: dict[str, C.c_void_p] = {}

    def _ok(self, err: int, what: str) -> None:
        if err:
            raise RuntimeError(f"{what}: hip error {err}")

    def load_module(self, co: bytes) -> None:
        self._co_buf = C.create_string_buffer(co, len(co))
        self._ok(self.lib.hipModuleLoadData(C.byref(self._mod), self._co_buf), "hipModuleLoadData")

    def get_fn(self, name: str) -> C.c_void_p:
        if name not in self._fns:
            fn = C.c_void_p()
            self._ok(self.lib.hipModuleGetFunction(C.byref(fn), self._mod, name.encode()), f"get_fn({name})")
            self._fns[name] = fn
        return self._fns[name]

    def malloc(self, nbytes: int) -> int:
        p = C.c_void_p()
        self._ok(self.lib.hipMalloc(C.byref(p), C.c_size_t(max(nbytes, 4))), "hipMalloc")
        return int(p.value)

    def free(self, ptr: int) -> None:
        self._ok(self.lib.hipFree(C.c_void_p(ptr)), "hipFree")

    def h2d(self, dst: int, arr: np.ndarray) -> None:
        arr = np.ascontiguousarray(arr)
        self._ok(
            self.lib.hipMemcpyHtoD(C.c_void_p(dst), arr.ctypes.data_as(C.c_void_p), C.c_size_t(arr.nbytes)),
            "hipMemcpyHtoD",
        )

    def d2h(self, arr: np.ndarray, src: int) -> None:
        self._ok(
            self.lib.hipMemcpyDtoH(arr.ctypes.data_as(C.c_void_p), C.c_void_p(src), C.c_size_t(arr.nbytes)),
            "hipMemcpyDtoH",
        )

    def launch(self, fn: C.c_void_p, grid_x: int, block_x: int, kernargs: bytes) -> None:
        buf = C.create_string_buffer(kernargs, len(kernargs))
        size = C.c_size_t(len(kernargs))
        extra = (C.c_void_p * 5)(
            1,
            C.cast(buf, C.c_void_p),
            2,
            C.cast(C.pointer(size), C.c_void_p),
            3,
        )
        self._ok(
            self.lib.hipModuleLaunchKernel(fn, grid_x, 1, 1, block_x, 1, 1, 0, None, None, extra),
            "hipModuleLaunchKernel",
        )

    def sync(self) -> None:
        self._ok(self.lib.hipDeviceSynchronize(), "hipDeviceSynchronize")


def run_jobs(jobs: list[dict], co_path: Path, device: int | None = None) -> dict[str, np.ndarray]:
    """Run a batch of WMMA/SWMMAC probe jobs on the local gfx1201 GPU.

    Each job dict contains:
      id: str
      spec: str (key in SPECS)
      kernel: str (optional, defaults to spec)
      a, b, c: uint32 arrays of shape (G, 32, dwords) or flat
      idx: uint32 array of shape (G, 32) or flat (required for sparse or mode/exec probes)
    """
    dev_idx = int(os.environ.get("HIP_DEVICE", "0")) if device is None else device
    hip = Hip(dev_idx)
    hip.load_module(Path(co_path).read_bytes())

    results: dict[str, np.ndarray] = {}
    for job in jobs:
        jid = job["id"]
        spec = SPECS[job["spec"]]
        kernel = job.get("kernel", spec.name)
        a = np.ascontiguousarray(job["a"], dtype=np.uint32).reshape(-1)
        b = np.ascontiguousarray(job["b"], dtype=np.uint32).reshape(-1)
        c = np.ascontiguousarray(job["c"], dtype=np.uint32).reshape(-1)
        grid = c.size // (32 * spec.cd_dwords)
        assert a.size == grid * 32 * spec.a_dwords, f"{jid}: A size mismatch"
        assert b.size == grid * 32 * spec.b_dwords, f"{jid}: B size mismatch"
        assert c.size == grid * 32 * spec.cd_dwords, f"{jid}: C size mismatch"

        has_idx = spec.sparse or ("idx" in job and job["idx"] is not None)
        idx = np.ascontiguousarray(job["idx"], dtype=np.uint32).reshape(-1) if has_idx else None
        if idx is not None:
            assert idx.size == grid * 32, f"{jid}: idx size mismatch"

        da = hip.malloc(a.nbytes)
        db = hip.malloc(b.nbytes)
        dc = hip.malloc(c.nbytes)
        dd = hip.malloc(c.nbytes)
        didx = hip.malloc(idx.nbytes) if idx is not None else 0
        try:
            hip.h2d(da, a)
            hip.h2d(db, b)
            hip.h2d(dc, c)
            if idx is not None:
                hip.h2d(didx, idx)

            args = b"".join(p.to_bytes(8, "little") for p in (da, db, dc, dd, didx))
            hip.launch(hip.get_fn(kernel), grid, 32, args)
            hip.sync()

            d_out = np.empty_like(c)
            hip.d2h(d_out, dd)
            results[jid] = d_out.reshape(grid, 32, spec.cd_dwords)
        finally:
            hip.free(da)
            hip.free(db)
            hip.free(dc)
            hip.free(dd)
            if didx:
                hip.free(didx)

    return results
