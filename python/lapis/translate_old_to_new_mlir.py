"""
mlir_old_to_new.py

Usage:
  python3 mlir_old_to_new.py <input_old.mlir> [output_new.mlir]

If output filename is omitted, writes transformed MLIR to stdout.

Transforms:
1) scf.parallel: delete only the final top-level `scf.yield` (with no operands)
   in each scf.parallel body.
2) scf.reduce typing:
   Old: scf.reduce(%5)  : f64 {
   New: scf.reduce(%5 : f64) {
   (also handles trailing attributes/loc on the line)

Note: code written by ChatGPT 5.2
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from typing import Iterable, List, Tuple


# ----------------------------- Transform framework -----------------------------

@dataclass
class TransformResult:
    text: str
    changed: bool
    count: int = 0


class Transform:
    name: str = "unnamed-transform"

    def apply(self, text: str) -> TransformResult:
        raise NotImplementedError


def apply_transforms(text: str, transforms: Iterable[Transform]) -> Tuple[str, List[Tuple[str, int]]]:
    stats: List[Tuple[str, int]] = []
    for t in transforms:
        res = t.apply(text)
        text = res.text
        stats.append((t.name, res.count))
    return text, stats


# ----------------------------- Transform #1 ------------------------------------

class DeleteFinalParallelYieldTransform(Transform):
    name = "scf.parallel: delete final top-level scf.yield()"

    _parallel_start = re.compile(r"\bscf\.parallel\b.*\{")

    _yield_no_operands = re.compile(
        r"""^\s*scf\.yield(\s*(?:\{.*\})?\s*(?:loc\([^)]+\))?\s*(?::.*)?\s*)$"""
    )

    def apply(self, text: str) -> TransformResult:
        lines = text.splitlines(keepends=True)

        brace_level = 0
        parallel_stack: List[dict] = []

        changed = False
        count = 0

        for i, line in enumerate(lines):
            starts_parallel = bool(self._parallel_start.search(line))
            opens = line.count("{")
            closes = line.count("}")

            if starts_parallel:
                body_level = brace_level + 1
                parallel_stack.append({"body_level": body_level, "last_yield_line": None})

            brace_level += opens

            if parallel_stack:
                ctx = parallel_stack[-1]
                if brace_level == ctx["body_level"] and self._yield_no_operands.match(line):
                    ctx["last_yield_line"] = i

            for _ in range(closes):
                brace_level -= 1
                if parallel_stack and brace_level < parallel_stack[-1]["body_level"]:
                    ctx = parallel_stack.pop()
                    j = ctx["last_yield_line"]
                    if j is not None:
                        lines[j] = ""
                        changed = True
                        count += 1

        return TransformResult("".join(lines), changed=changed, count=count)


# ----------------------------- Transform #2 ------------------------------------
# The earlier regex missed cases where the scf.reduce line ends with "{".
# This version explicitly allows an optional "{" plus optional trailing text.

class ReduceTypePlacementTransform(Transform):
    name = "scf.reduce: move result type into operand list"

    # Matches, e.g.:
    #   scf.reduce(%5)  : f64 {
    #   scf.reduce(%5) : memref<?xf64> loc(#loc)
    #   %x = scf.reduce(%5) : f64 {   (assignment allowed)
    #
    # Captures:
    #   pre   => up to "scf.reduce"
    #   args  => inside (...)
    #   type  => type after ':'
    #   tail  => remaining suffix on the line (including "{" / attrs / loc), if any
    _rx = re.compile(
        r"""
        ^(?P<pre>\s*(?:[%@][\w.$-]+\s*=\s*)?\s*scf\.reduce)\(
            (?P<args>[^)]*)
        \)
        \s*:\s*
        (?P<type>[^{}\n]+?)          # type token(s), stop before '{' or newline
        (?P<tail>\s*(?:\{)?\s*(?:\{.*\})?\s*(?:loc\([^)]+\))?\s*)$
        """,
        flags=re.VERBOSE | re.MULTILINE,
    )

    def apply(self, text: str) -> TransformResult:
        def repl(m: re.Match) -> str:
            args = m.group("args")
            # Already-new syntax if ":" appears inside parentheses.
            if re.search(r"\S\s*:\s*\S", args):
                return m.group(0)

            pre = m.group("pre")
            ty = m.group("type").strip()
            tail = m.group("tail") or ""

            if args.strip() == "":
                return m.group(0)

            return f"{pre}({args} : {ty}){tail}"

        new_text, n = self._rx.subn(repl, text)
        return TransformResult(new_text, changed=(n != 0), count=n)


# ----------------------------- Main --------------------------------------------

DEFAULT_TRANSFORMS: List[Transform] = [
    DeleteFinalParallelYieldTransform(),
    ReduceTypePlacementTransform(),
]

def translate_old_to_new_mlir(module):
    out, _ = apply_transforms(module, DEFAULT_TRANSFORMS)
    return out

