//! clair_fast — Rust+PyO3 port of the CPU hot path in `clair.csp`.
//!
//! Ports, with IDENTICAL semantics to the pure-Python originals:
//!   * `solutions`   — smallest-domain-first backtracking with forward constraint checking
//!   * `exact_dedp`  — per-cell survivor union over all solutions, with the witness early-stop
//!   * `ac_step`     — one generalized-arc-consistency pass
//!
//! Representation marshalled from Python (built in `clair.csp`):
//!   n: usize, d: usize,
//!   scopes:   Vec<Vec<usize>>          (one per constraint; cell indices)
//!   alloweds: Vec<Vec<Vec<i64>>>       (per constraint: list of allowed value-tuples)
//!   dom:      Vec<Vec<u32>>            (per cell: SORTED alive values)
//!
//! Supported domain: d <= 64 (bitmask) and arity <= 8 (8-bit packing). The Python wrapper
//! checks these bounds and falls back to pure Python otherwise; we also assert here.

use pyo3::prelude::*;
use rustc_hash::FxHashSet;

/// 8-bit-per-slot packing of a value tuple (arity <= 8, values < 256).
#[inline(always)]
fn pack(assign: &[i64], scope: &[usize]) -> u64 {
    let mut key = 0u64;
    for (p, &c) in scope.iter().enumerate() {
        key |= (assign[c] as u64) << (8 * p);
    }
    key
}

struct Prob {
    n: usize,                     // number of CELLS (not constraints)
    scopes: Vec<Vec<usize>>,
    tuples: Vec<Vec<Vec<i64>>>,   // allowed tuples per constraint (for ac_step iteration)
    sets: Vec<FxHashSet<u64>>,    // packed allowed tuples per constraint (for O(1) membership)
}

impl Prob {
    fn build(n: usize, scopes: Vec<Vec<usize>>, alloweds: Vec<Vec<Vec<i64>>>) -> Self {
        let mut sets = Vec::with_capacity(scopes.len());
        for al in &alloweds {
            let mut s = FxHashSet::default();
            for t in al {
                // pack tuple t in scope order (scope order == tuple order, as in Python `al`)
                let mut key = 0u64;
                for (p, &v) in t.iter().enumerate() {
                    key |= (v as u64) << (8 * p);
                }
                s.insert(key);
            }
            sets.push(s);
        }
        Prob { n, scopes, tuples: alloweds, sets }
    }
}

/// Precomputed backtracking schedule: visit order (smallest-domain-first, ties by cell index)
/// and, per order-position, the constraint indices whose last cell lands at that position.
struct Schedule {
    order: Vec<usize>,
    checks: Vec<Vec<usize>>,
}

fn schedule(n: usize, scopes: &[Vec<usize>], dom: &[Vec<u32>]) -> Schedule {
    let mut order: Vec<usize> = (0..n).collect();
    // stable smallest-domain-first == sort by (len, index)
    order.sort_by(|&a, &b| (dom[a].len(), a).cmp(&(dom[b].len(), b)));
    let mut rank = vec![0usize; n];
    for (k, &c) in order.iter().enumerate() {
        rank[c] = k;
    }
    let mut checks: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (ci, sc) in scopes.iter().enumerate() {
        let pos = sc.iter().map(|&c| rank[c]).max().unwrap();
        checks[pos].push(ci);
    }
    Schedule { order, checks }
}

/// DFS with forward constraint-checking. `on_sol` returns true to halt (mirrors `_Stop`).
///
/// `nodes`/`budget`/`aborted`: a node-visit cap. When the visit count exceeds `budget` the search
/// halts early and sets `aborted` — used by dedP to bail out of an enumeration whose witness
/// early-stop cannot fire (e.g. a pinned cell caps its witnessable count below its domain size, so a
/// huge solution set would be walked in full) and hand off to the exact per-value reachability path.
/// Callers that want the full search (`solutions`) pass `budget = u64::MAX` so it never trips.
fn backtrack(
    prob: &Prob,
    dom: &[Vec<u32>],
    sch: &Schedule,
    assign: &mut [i64],
    k: usize,
    on_sol: &mut dyn FnMut(&[i64]) -> bool,
    nodes: &mut u64,
    budget: u64,
    aborted: &mut bool,
) -> bool {
    *nodes += 1;
    if *nodes > budget {
        *aborted = true;
        return true; // halt the whole recursion; caller inspects `aborted`
    }
    // terminal at the schedule length: == prob.n for the full-instance schedule (`solutions`), or the
    // size of the constrained core for the free-cell-stripped dedP schedule.
    if k == sch.order.len() {
        return on_sol(assign);
    }
    let cell = sch.order[k];
    let ck = &sch.checks[k];
    let dlen = dom[cell].len();
    for idx in 0..dlen {
        let v = dom[cell][idx];
        assign[cell] = v as i64;
        let ok = ck.iter().all(|&ci| {
            prob.sets[ci].contains(&pack(assign, &prob.scopes[ci]))
        });
        if ok && backtrack(prob, dom, sch, assign, k + 1, on_sol, nodes, budget, aborted) {
            return true;
        }
    }
    false
}

/// Is there ANY solution that fixes cell `pin_cell` = `pin_val`, within the (GAC-reduced) box `dmask`?
/// EXACT membership test used by per-value reachability: GAC after pinning prunes, then a limit-1
/// search over the constrained core decides. The per-value union of the SAT cells IS the dedP set.
fn core_sat(prob: &Prob, dmask: &[u64], inscope: &[bool], pin_cell: usize, pin_val: u32) -> bool {
    let mut dm = dmask.to_vec();
    dm[pin_cell] = 1u64 << pin_val;
    if !ac_fixpoint(prob, &mut dm) {
        return false;
    }
    let domv: Vec<Vec<u32>> = (0..prob.n).map(|i| bits_to_sorted(dm[i])).collect();
    let core: Vec<usize> = (0..prob.n).filter(|&i| inscope[i]).collect();
    if core.is_empty() {
        return true; // no constraints -> trivially satisfiable
    }
    let sch = schedule_cells(&core, &prob.scopes, &domv);
    let mut assign = vec![0i64; prob.n];
    let mut found = false;
    let mut nodes = 0u64;
    let mut aborted = false;
    {
        let found_ref = &mut found;
        let mut on_sol = |_a: &[i64]| -> bool {
            *found_ref = true;
            true // first solution suffices
        };
        backtrack(prob, &domv, &sch, &mut assign, 0, &mut on_sol, &mut nodes, u64::MAX, &mut aborted);
    }
    found
}

/// EXACT dedP by per-(cell,value) reachability over the constrained core (free cells already carry
/// their domain). The fallback when the primary witness-early-stop enumeration blows its node budget:
/// dedP_i = { v in domv[i] : a solution exists with x_i = v }, decided one value at a time by a cheap
/// GAC-pruned limit-1 search. Each query terminates at the FIRST witness (or a GAC conflict), so this
/// is fast exactly when the enumeration is slow (a vast solution set the early-stop can't summarize).
fn dedp_per_value(prob: &Prob, domv: &[Vec<u32>], dmask: &[u64], inscope: &[bool]) -> Vec<Vec<u32>> {
    let n = prob.n;
    let mut surv = vec![0u64; n];
    for i in 0..n {
        if !inscope[i] {
            continue;
        }
        for &v in &domv[i] {
            if core_sat(prob, dmask, inscope, i, v) {
                surv[i] |= 1u64 << v;
            }
        }
        if surv[i] == 0 {
            return vec![Vec::new(); n]; // a constrained cell with no reachable value -> ⊥
        }
    }
    (0..n)
        .map(|i| if inscope[i] { bits_to_sorted(surv[i]) } else { domv[i].clone() })
        .collect()
}

fn check_bounds(d: usize, scopes: &[Vec<usize>]) -> PyResult<()> {
    if d > 64 {
        return Err(pyo3::exceptions::PyValueError::new_err("clair_fast: d>64 unsupported"));
    }
    if scopes.iter().any(|s| s.len() > 8) {
        return Err(pyo3::exceptions::PyValueError::new_err("clair_fast: arity>8 unsupported"));
    }
    Ok(())
}

fn bits_to_sorted(mask: u64) -> Vec<u32> {
    let mut out = Vec::new();
    let mut m = mask;
    while m != 0 {
        let b = m.trailing_zeros();
        out.push(b);
        m &= m - 1;
    }
    out
}

/// Generalized-arc-consistency to FIXPOINT over bitmask domains. EXACT-PRESERVING for dedP: GAC only
/// removes a value that has no constraint-support in the current domains, so it NEVER drops a value
/// used by any solution — the solution set (hence the per-cell dedP) is unchanged. Returns false (and
/// leaves `dmask` partially reduced) if some cell empties (⊥ / unsatisfiable). Iterating in place to a
/// fixpoint is order-independent (the GAC fixpoint is unique), so the result matches the pure-Python
/// `to_fixpoint(ac_step, ...)` domains exactly. This is the pre-pass that collapses the loose / affine
/// blow-up: it prunes the dead-but-alive values that make the witness early-stop fail.
fn ac_fixpoint(prob: &Prob, dmask: &mut [u64]) -> bool {
    loop {
        let mut changed = false;
        for i in 0..prob.n {
            let mut keep = 0u64;
            let mut m = dmask[i];
            while m != 0 {
                let v = m.trailing_zeros();
                m &= m - 1;
                let mut ok = true;
                for ci in 0..prob.scopes.len() {
                    let sc = &prob.scopes[ci];
                    let pos = match sc.iter().position(|&c| c == i) {
                        Some(p) => p,
                        None => continue,
                    };
                    // support: some allowed tuple places v at i and is consistent with current domains
                    let mut support = false;
                    for t in &prob.tuples[ci] {
                        if t[pos] != v as i64 {
                            continue;
                        }
                        let mut all_in = true;
                        for (kk, &c) in sc.iter().enumerate() {
                            if (dmask[c] >> (t[kk] as u64)) & 1 == 0 {
                                all_in = false;
                                break;
                            }
                        }
                        if all_in {
                            support = true;
                            break;
                        }
                    }
                    if !support {
                        ok = false;
                        break;
                    }
                }
                if ok {
                    keep |= 1u64 << v;
                }
            }
            if keep != dmask[i] {
                dmask[i] = keep;
                changed = true;
            }
            if keep == 0 {
                return false;
            }
        }
        if !changed {
            return true;
        }
    }
}

/// Backtracking schedule restricted to an explicit `cells` list (smallest-domain-first, ties by cell
/// index), with each constraint's checks placed at the order-position of the LAST of its cells. Used by
/// dedP after the free (unconstrained) cells are stripped: constraints only ever touch in-scope cells,
/// so the search ranges over just those. Identical visit semantics to `schedule`, over a cell subset.
fn schedule_cells(cells: &[usize], scopes: &[Vec<usize>], dom: &[Vec<u32>]) -> Schedule {
    let mut order: Vec<usize> = cells.to_vec();
    order.sort_by(|&a, &b| (dom[a].len(), a).cmp(&(dom[b].len(), b)));
    let mut rank = vec![usize::MAX; dom.len()];
    for (k, &c) in order.iter().enumerate() {
        rank[c] = k;
    }
    let mut checks: Vec<Vec<usize>> = vec![Vec::new(); order.len()];
    for (ci, sc) in scopes.iter().enumerate() {
        // every cell of every constraint is in-scope (hence ranked); place at the max rank
        let pos = sc.iter().map(|&c| rank[c]).max().unwrap();
        checks[pos].push(ci);
    }
    Schedule { order, checks }
}

/// Core exact-dedP (no Python types) so it can run inside a rayon batch.
///
/// Two EXACT-PRESERVING accelerators wrap the witness-early-stop backtracking, so the per-cell
/// survivor SETS are bitwise-identical to the naive enumeration while killing the 15-160s affine /
/// high-treewidth stalls (which arise when a narrowed domain holds a dead-but-alive value, so the
/// early-stop can't fire and the search enumerates a `d^free`-sized tree):
///   1. GAC to fixpoint (sound) prunes locally-dead values up front;
///   2. cells in NO constraint scope are STRIPPED from the search — each is independently free, so its
///      reachable set is exactly its (GAC-reduced) domain when the constrained "core" is satisfiable,
///      and ∅ when the core is ⊥. Removing them deletes the `d^free` multiplier.
fn dedp_core(n: usize, scopes: Vec<Vec<usize>>, alloweds: Vec<Vec<Vec<i64>>>, dom: Vec<Vec<u32>>) -> Vec<Vec<u32>> {
    let prob = Prob::build(n, scopes, alloweds);
    // (1) GAC fixpoint over bitmask domains — sound, so reachable values are untouched.
    let mut dmask = vec![0u64; n];
    for i in 0..n {
        for &v in &dom[i] {
            dmask[i] |= 1u64 << v;
        }
    }
    if !ac_fixpoint(&prob, &mut dmask) {
        return vec![Vec::new(); n]; // ⊥: some cell emptied -> dedP is all-∅
    }
    let domv: Vec<Vec<u32>> = (0..n).map(|i| bits_to_sorted(dmask[i])).collect();
    // (2) classify cells: in some constraint scope (searched) vs free (reachable set == its domain).
    let mut inscope = vec![false; n];
    for sc in &prob.scopes {
        for &c in sc {
            inscope[c] = true;
        }
    }
    let core: Vec<usize> = (0..n).filter(|&i| inscope[i]).collect();
    if core.is_empty() {
        // no constraints -> trivially satisfiable; every value of every cell is reachable.
        return domv;
    }
    // dedP over the core via the witness-early-stop backtracking (free cells excluded from need/search),
    // capped by a node budget. The early-stop fires once every alive core value is witnessed; it can
    // FAIL to fire when a cell's witnessable count is below its domain size (e.g. a pin), at which point
    // a vast solution set would be enumerated in full — so on budget exhaustion we bail to the EXACT
    // per-value reachability path instead (same dedP set, decided value-by-value).
    const NODE_BUDGET: u64 = 2_000_000;
    let sch = schedule_cells(&core, &prob.scopes, &domv);
    let need: usize = core.iter().map(|&i| domv[i].len()).sum();
    let mut surv = vec![0u64; n];
    let mut seen = 0usize;
    let mut assign = vec![0i64; n];
    let mut nodes = 0u64;
    let mut aborted = false;
    {
        let surv_ref = &mut surv;
        let seen_ref = &mut seen;
        let core_ref = &core;
        let mut on_sol = |a: &[i64]| -> bool {
            for &i in core_ref {
                let bit = 1u64 << (a[i] as u32);
                if surv_ref[i] & bit == 0 {
                    surv_ref[i] |= bit;
                    *seen_ref += 1;
                }
            }
            *seen_ref >= need
        };
        backtrack(&prob, &domv, &sch, &mut assign, 0, &mut on_sol, &mut nodes, NODE_BUDGET, &mut aborted);
    }
    if aborted {
        // enumeration could not summarize the solution set within budget -> exact per-value fallback.
        return dedp_per_value(&prob, &domv, &dmask, &inscope);
    }
    if seen == 0 {
        return vec![Vec::new(); n]; // core unsatisfiable -> whole instance ⊥
    }
    // core cells -> witnessed survivors; free cells -> their full (GAC-reduced) domain.
    (0..n)
        .map(|i| if inscope[i] { bits_to_sorted(surv[i]) } else { domv[i].clone() })
        .collect()
}

/// exact_dedP for one instance — returns per-cell sorted survivor lists (empty-all = unsat/⊥).
#[pyfunction]
fn exact_dedp(
    n: usize,
    d: usize,
    scopes: Vec<Vec<usize>>,
    alloweds: Vec<Vec<Vec<i64>>>,
    dom: Vec<Vec<u32>>,
) -> PyResult<Vec<Vec<u32>>> {
    check_bounds(d, &scopes)?;
    Ok(dedp_core(n, scopes, alloweds, dom))
}

/// All solutions (assignments by cell index), DFS order; optional `limit` to stop early.
#[pyfunction]
#[pyo3(signature = (n, d, scopes, alloweds, dom, limit=None))]
fn solutions(
    n: usize,
    d: usize,
    scopes: Vec<Vec<usize>>,
    alloweds: Vec<Vec<Vec<i64>>>,
    dom: Vec<Vec<u32>>,
    limit: Option<usize>,
) -> PyResult<Vec<Vec<u32>>> {
    check_bounds(d, &scopes)?;
    let prob = Prob::build(n, scopes, alloweds);
    let sch = schedule(n, &prob.scopes, &dom);
    let mut assign = vec![0i64; n];
    let mut out: Vec<Vec<u32>> = Vec::new();
    {
        let out_ref = &mut out;
        let mut on_sol = |a: &[i64]| -> bool {
            out_ref.push(a.iter().map(|&x| x as u32).collect());
            matches!(limit, Some(l) if out_ref.len() >= l)
        };
        // full enumeration: no node cap (budget = MAX), so the visit semantics are unchanged.
        let mut nodes = 0u64;
        let mut aborted = false;
        backtrack(&prob, &dom, &sch, &mut assign, 0, &mut on_sol, &mut nodes, u64::MAX, &mut aborted);
    }
    Ok(out)
}

/// One generalized-arc-consistency pass — returns per-cell sorted kept-value lists.
#[pyfunction]
fn ac_step(
    n: usize,
    d: usize,
    scopes: Vec<Vec<usize>>,
    alloweds: Vec<Vec<Vec<i64>>>,
    dom: Vec<Vec<u32>>,
) -> PyResult<Vec<Vec<u32>>> {
    check_bounds(d, &scopes)?;
    let prob = Prob::build(n, scopes, alloweds);
    // current-domain bitmasks for O(1) membership
    let mut dmask = vec![0u64; n];
    for i in 0..n {
        for &v in &dom[i] {
            dmask[i] |= 1u64 << v;
        }
    }
    let mut out: Vec<Vec<u32>> = Vec::with_capacity(n);
    for i in 0..n {
        let mut keep = 0u64;
        for &v in &dom[i] {
            let mut ok = true;
            'cons: for ci in 0..prob.scopes.len() {
                let sc = &prob.scopes[ci];
                let pos = match sc.iter().position(|&c| c == i) {
                    Some(p) => p,
                    None => continue,
                };
                // support: some allowed tuple places v at i and is consistent with current domains
                let mut support = false;
                for t in &prob.tuples[ci] {
                    if t[pos] != v as i64 {
                        continue;
                    }
                    let mut all_in = true;
                    for (kk, &c) in sc.iter().enumerate() {
                        if (dmask[c] >> (t[kk] as u64)) & 1 == 0 {
                            all_in = false;
                            break;
                        }
                    }
                    if all_in {
                        support = true;
                        break;
                    }
                }
                if !support {
                    ok = false;
                    break 'cons;
                }
            }
            if ok {
                keep |= 1u64 << v;
            }
        }
        out.push(bits_to_sorted(keep));
    }
    Ok(out)
}

/// Batch exact-dedP over many instances, parallelised across cores in-process (GIL released).
/// `items`: list of (n, d, scopes, alloweds, dom). Returns one per-cell survivor list per item.
#[pyfunction]
fn dedp_batch(
    py: Python<'_>,
    items: Vec<(usize, usize, Vec<Vec<usize>>, Vec<Vec<Vec<i64>>>, Vec<Vec<u32>>)>,
) -> PyResult<Vec<Vec<Vec<u32>>>> {
    for (_, d, scopes, _, _) in &items {
        check_bounds(*d, scopes)?;
    }
    let out = py.allow_threads(|| {
        use rayon::prelude::*;
        items
            .into_par_iter()
            .map(|(n, _d, scopes, alloweds, dom)| dedp_core(n, scopes, alloweds, dom))
            .collect::<Vec<_>>()
    });
    Ok(out)
}

#[pymodule]
fn clair_fast(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(exact_dedp, m)?)?;
    m.add_function(wrap_pyfunction!(solutions, m)?)?;
    m.add_function(wrap_pyfunction!(ac_step, m)?)?;
    m.add_function(wrap_pyfunction!(dedp_batch, m)?)?;
    Ok(())
}
