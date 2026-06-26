"""Lightweight worker entry for the dedP process pool. Imports ONLY clair.csp (no torch / no
run_general) so the spawned workers start fast and never touch CUDA."""
from __future__ import annotations

import hashlib

from .. import csp as C


def canonical_key(item):
    """Stable 16-byte content hash of (constraints, domain-size, current per-cell domains), used as
    the disk-cache key for an instance's dedP target. Canonicalised (sorted relations / domains) so
    the key is reproducible across runs and processes (ints only -> no hash-seed sensitivity)."""
    csp, dom = item
    cons = tuple((tuple(sc), tuple(sorted(al))) for sc, al in csp.cons)
    domc = tuple(tuple(sorted(c)) for c in dom)
    return hashlib.blake2b(repr((cons, csp.d, domc)).encode(), digest_size=16).digest()


def dedP_one(item):
    """Exact per-cell dedP for one (csp, dom): the set of values used by SOME full solution
    consistent with `dom`. Identical to run_general.Exact.dedP (union over C.solutions), so targets
    are bitwise-identical to the serial path. Top-level + torch-free => cheap to pickle/spawn."""
    csp, dom = item
    sols = C.solutions(csp, dom)
    if not sols:
        return tuple(frozenset() for _ in range(csp.n))
    return tuple(frozenset(s[i] for s in sols) for i in range(csp.n))
