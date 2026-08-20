#!/usr/bin/env python3
"""Run the exact remaining P0 patch after removing stale controller edits.

`p0_patch_head.py` was generated before the sustained-pressure controller fix
was committed directly. Its C++/CUDA/TierMachine/pipeline transformations are
still exact and desired, but its controller replacements are intentionally
obsolete. This wrapper removes only that obsolete block in memory, then
executes the remaining fail-closed script. Neither source script survives the
verified patch commit.
"""

from __future__ import annotations

from pathlib import Path

SOURCE = Path("tools/p0_patch_head.py")
text = SOURCE.read_text()

start_marker = 'controller = Path("elastic/elastic-core/src/controller.rs")\n'
end_marker = 'pipeline = Path("slhav2-vram/src/pipeline.rs")\n'

start = text.find(start_marker)
end = text.find(end_marker)
if start < 0 or end < 0 or end <= start:
    raise SystemExit("stale-controller block markers not found exactly")

sanitized = text[:start] + text[end:]
if 'Rejection::CostTooHigh' in sanitized:
    raise SystemExit("obsolete controller transformation leaked into sanitized patch")

code = compile(sanitized, str(SOURCE), "exec")
namespace = {"__name__": "__main__", "__file__": str(SOURCE)}
exec(code, namespace, namespace)
