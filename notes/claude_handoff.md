# Claude Handoff: Clifford/LDT Direction

Please read `notes/geometric_product_options.md` first. The main goal is not to argue for "wedge
usefulness" in isolation. The better framing is:

> Use full geometric-product interactions as learned proposers/mixers/compilers, while checked
> lattice projection, exact alpha targets, verifiers, or certificates remain the authority for
> reliable deduction.

The local project context:

- `clair/ldt.py`, `clair/glados.py`, `clair/cliffordnet.py`, `clair/organs.py` are the core Python
  files to inspect.
- `pdfs/2601.06793-cliffordnet-geometric-product.pdf` is the geometric-product architecture source.
- `pdfs/2605.08605-lattice-deduction-transformer.pdf` is the reliable deduction architecture source.
- `pdfs/2210.10749-transformers-shortcuts-automata.pdf` is important for performance shortcuts.
- `~/dev/graphplay/Graphplay/Integrations/LDT*.lean` contains Lean proofs clarifying soundness,
  completeness, abstraction hierarchy, and XOR/affine walls.

Important conclusions so far:

1. Wedge alone is probably the wrong thesis.
   The CliffordNet-style primitive is the full geometric product:

   ```text
   uv = u . v + u ^ v
   F(H, C) = P(H . C concat H ^ C)
   ```

   Inner-only and wedge-only are both meaningful ablations, but the full product is the actual bet.

2. The neural part should not be trusted as a proof system.
   It should propose. Reliability should come from meet/projection, output checking, exact alpha
   targets, or checkable certificates.

3. The current multi-domain gauntlet is underdetermined.
   SAT/coloring/maze instances are not really visible to the model if it only sees `alive,given`.
   The model needs the instance program: clauses, graph edges, walls/topology, factors, etc.
   Otherwise the experiment does not test general constraint reasoning.

4. Planted-solution supervision is not soundness for multi-solution domains.
   For SAT/coloring/maze, the alpha target should keep any value used by some surviving solution.
   Use exact enumeration on tiny CSPs or sampled multi-solution alpha for larger domains.

5. The Lean files matter.
   They say soundness is cheap if we check outputs/steps; completeness or useful non-abstention is
   the real research variable. They also show why pair/triple/factor lattices matter: per-cell
   abstractions forget correlations, especially around XOR/parity/affine structure.

6. The automata shortcuts paper suggests a concrete performance route.
   A recurrent deduction operator on a finite lattice can be accelerated by macro operators:

   ```text
   M_k(problem, a) ~= F_problem^{2^k}(a)
   ```

   But the more interesting version may represent and compose transformation maps, not just states.
   Any shortcut must be checked and tested on length/depth/distribution shifts because learned
   shortcuts can be brittle.

Suggested next work:

1. Build an exact finite-CSP harness.
   Include chain, equality, XOR/parity, 2-SAT, tiny 3-SAT, tiny coloring. Enumerate satisfying
   assignments and compute exact alpha targets. Measure false elimination against exact alpha,
   completeness, abstention, and fixed-point drift.

2. Add a CliffordNet-faithful LDT proposer.
   Keep separate switches for:

   ```text
   mode:      inner | wedge | full
   self:      absolute | differential C-H
   topology:  shared shifts | separate dot/wedge shifts | learned sparse shifts
   gating:    off | scalar gate | candidate-wise gate
   grade:     vector-vector | vector-bivector | grade-3 probe
   context:   attention | factor graph | global | hybrid
   ```

3. Fix the multi-domain gauntlet interface.
   Feed the problem program into the organ. A factor-graph representation is probably the cleanest
   shared interface.

4. Then try automata-style macro deduction.
   Distill `F_2`, `F_4`, `F_8`; use speculative fallback from larger macro to smaller macro to the
   base step. Track lifting error and checked soundness.

Please do not collapse this back into "replace dot products with wedges everywhere." The question is
where a full geometric-product block improves proposal quality, compression, or multimodal/factor
fusion while the lattice/checker boundary protects reliability.
