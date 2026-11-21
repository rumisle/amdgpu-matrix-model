"""Derive the accumulation order of K slots from the frozen `k_order` chunks (no GPU needed).

Each case of a `k_order` chunk (tools/cases.py:k_order_cases) puts +P on slot i, +u on slot j and
C = -P; the hardware returns u exactly when slot i is consumed in an earlier step than slot j. From
all ordered pairs this tool rebuilds the steps, prints them in three labelings (the RDNA4 ISA K label
used by the model, the lane-order slot index q, and register coordinates) and checks them against
`step_schedule`.

    uv run --no-project python tools/k_order.py [--evidence DIR]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from amdgpu_matrix_model.evidence import iter_chunk_data, read_matrix_file  # noqa: E402
from amdgpu_matrix_model.layout import slot_order, unpack_cd  # noqa: E402
from amdgpu_matrix_model.model import step_schedule  # noqa: E402
from amdgpu_matrix_model.specs import SPECS  # noqa: E402


def derive_steps(spec, d_words: list[int]) -> list[list[int]]:
    """Ordered steps (lists of K labels) from the D words of a k_order chunk."""
    n = spec.k_a
    pairs = [(i, j) for i in range(n) for j in range(n) if i != j]
    cd = 32 * spec.cd_dwords
    before = {}
    for t, (i, j) in enumerate(pairs):
        wave, row = divmod(t, 16)
        before[i, j] = unpack_cd(spec, d_words[wave * cd : (wave + 1) * cd])[row][row]
    if len(set(before.values()) - {0}) != 1:
        raise ValueError("k_order results must be +0 or one common value u")
    before = {key: d != 0 for key, d in before.items()}
    steps: list[list[int]] = []
    for k in sorted(range(n), key=lambda k: sum(before[x, k] for x in range(n) if x != k)):
        same = [s for s in steps if not before[s[0], k] and not before[k, s[0]]]
        if same:
            same[0].append(k)
        else:
            steps.append([k])
    for a in range(len(steps)):
        for b in range(len(steps)):
            for i in steps[a]:
                for j in steps[b]:
                    if i != j and before[i, j] != (a < b):
                        raise ValueError("order relation is not a sequence of steps")
    return steps


def slot_coordinates(spec, k: int) -> str:
    order = slot_order(spec, "a")
    q = order.index(k)
    half, p = divmod(q, spec.k_a // 2)
    per = 32 // spec.a_bits
    return f"h{half}v{p // per}{'lo' if p % per == 0 else 'hi'}" if per == 2 else f"h{half}v{p // per}b{p % per}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--evidence", type=Path, default=ROOT / "evidence")
    args = ap.parse_args()
    ok = True
    for spec in SPECS.values():
        path = args.evidence / f"{spec.name}.mmbin"
        mf = read_matrix_file(path, spec.name)
        chunk = next((x for x in iter_chunk_data(mf) if x[0].tag == "k_order"), None)
        if chunk is None:
            continue
        steps = derive_steps(spec, chunk[5])
        order = slot_order(spec, "a")
        agrees = [tuple(sorted(s)) for s in steps] == [tuple(sorted(s)) for s in step_schedule(spec)]
        ok &= agrees
        print(f"{spec.name}: {'matches' if agrees else 'DIFFERS FROM'} step_schedule")
        for n, s in enumerate(steps):
            s = sorted(s)
            print(
                f"  step {n}: K {s}  lane-order q {[order.index(k) for k in s]}  "
                f"registers {[slot_coordinates(spec, k) for k in s]}"
            )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
