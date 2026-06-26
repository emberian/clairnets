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
fn backtrack(
    prob: &Prob,
    dom: &[Vec<u32>],
    sch: &Schedule,
    assign: &mut [i64],
    k: usize,
    on_sol: &mut dyn FnMut(&[i64]) -> bool,
) -> bool {
    if k == prob.n {
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
        if ok && backtrack(prob, dom, sch, assign, k + 1, on_sol) {
            return true;
        }
    }
    false
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

/// Core exact-dedP (no Python types) so it can run inside a rayon batch.
fn dedp_core(n: usize, scopes: Vec<Vec<usize>>, alloweds: Vec<Vec<Vec<i64>>>, dom: Vec<Vec<u32>>) -> Vec<Vec<u32>> {
    let prob = Prob::build(n, scopes, alloweds);
    let sch = schedule(n, &prob.scopes, &dom);
    let need: usize = dom.iter().map(|c| c.len()).sum();
    let mut surv = vec![0u64; n];
    let mut seen = 0usize;
    let mut assign = vec![0i64; n];
    {
        let surv_ref = &mut surv;
        let seen_ref = &mut seen;
        let mut on_sol = |a: &[i64]| -> bool {
            for i in 0..n {
                let bit = 1u64 << (a[i] as u32);
                if surv_ref[i] & bit == 0 {
                    surv_ref[i] |= bit;
                    *seen_ref += 1;
                }
            }
            *seen_ref >= need
        };
        backtrack(&prob, &dom, &sch, &mut assign, 0, &mut on_sol);
    }
    if seen == 0 {
        return vec![Vec::new(); n];
    }
    surv.iter().map(|&m| bits_to_sorted(m)).collect()
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
        backtrack(&prob, &dom, &sch, &mut assign, 0, &mut on_sol);
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
