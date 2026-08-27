"""Fail if the CI Python matrix drifts from what pyproject claims to support.

Three places state the supported versions -- ``requires-python``, the
``Programming Language :: Python`` classifiers, and the CI matrix -- and
nothing else keeps them in step.  A floor bump that misses one leaves the
package either advertising a version it never tests or testing one it
tells installers it does not support.

Only the *endpoints* have to be tested.  The package is pure Python, so
the failures a version matrix can catch cluster at the floor (syntax or
stdlib that is too new) and the ceiling (deprecation removals, a
dependency without a wheel yet); the versions between mostly re-run the
same code on the same semantics.  So a classified version that sits
strictly inside the tested range is fine, while an untested floor or an
untested highest-classified version is not.

Note what this does *not* check: that each job's interpreter satisfies
``requires-python``.  Both pip and uv already refuse to install the
package on an interpreter below the floor, and they say so more clearly
than a bolt-on assertion would.
"""

from __future__ import annotations

import re
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github" / "workflows" / "test_and_lint.yml"


def _fmt(versions: list[tuple[int, int]]) -> str:
    """Render version tuples the way a human writes them."""
    return ", ".join(f"{a}.{b}" for a, b in versions)


def main() -> int:
    """Compare the three declarations and report every disagreement."""
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]

    spec = project["requires-python"].strip()
    floor_match = re.fullmatch(r">=\s*(\d+)\.(\d+)", spec)
    if floor_match is None:
        print(f"cannot parse requires-python {spec!r}; expected '>=X.Y'")
        return 1
    floor = (int(floor_match[1]), int(floor_match[2]))

    classified = sorted(
        tuple(int(p) for p in c.rsplit(" ", 1)[1].split("."))
        for c in project["classifiers"]
        if re.fullmatch(r"Programming Language :: Python :: 3\.\d+", c)
    )

    matrix_match = re.search(
        r"python-version:\s*\[([^\]]*)\]", WORKFLOW.read_text()
    )
    if matrix_match is None:
        print(f"no python-version matrix found in {WORKFLOW.name}")
        return 1
    matrix = sorted(
        tuple(int(p) for p in v.strip().strip("'\"").split("."))
        for v in matrix_match[1].split(",")
        if v.strip()
    )

    problems = []
    if not matrix:
        problems.append("the CI matrix is empty")
        classified = []
    elif matrix[0] != floor:
        problems.append(
            f"CI tests down to {_fmt(matrix[:1])} but requires-python says "
            f">={_fmt([floor])}"
        )
    if classified and matrix and classified[-1] != matrix[-1]:
        problems.append(
            f"classifiers advertise up to {_fmt(classified[-1:])} but CI "
            f"tests up to {_fmt(matrix[-1:])}"
        )
    unclaimed = [v for v in matrix if classified and v not in classified]
    if unclaimed:
        problems.append(
            f"CI tests {_fmt(unclaimed)}, which the classifiers do not "
            f"advertise"
        )
    below = [v for v in classified if v < floor]
    if below:
        problems.append(
            f"classifiers advertise {_fmt(below)}, below requires-python "
            f">={_fmt([floor])}"
        )

    if problems:
        print("Python version declarations disagree:")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print(
        f"ok: requires-python >={_fmt([floor])}, "
        f"classified {_fmt(classified)}, tested {_fmt(matrix)}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
