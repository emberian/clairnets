"""clair.organ — THE assembled GLaDOS organ + its canonical pipeline.

ONE coherent, trainable, LLM-ready package. The organ itself (bank + spine + composer) and the single
pretrain → weave → rlvr flow with its graft and eval entries all live here:

  protocol  : the typed interface (Reduction / CSPState / Certificate + the structured-γ readout)
  bank      : the registry of validated beasts as reductions (certified ops + neural guidance)
  compose   : verifier-gated certified-reduction composition (the sound reduced product)
  graft     : the MODEL-AGNOSTIC neural-graft — graft_organ(host, organ, config) onto any of 7 bases
  train     : the canonical staged pipeline — pretrain_organ → weave → rlvr
  eval      : the arbiter (re-exports clair.eval_suite.run_eval_suite)
  smoke     : the one-command end-to-end smoke (needs a GPU)
  selftest  : the consolidated CPU smoke (run it: python -m clair.organ.selftest)

See GLADOS.md (repo root) for the single front-door map; README.md for the per-beast status table.
"""
from .protocol import Certificate, CSPState, Reduction, exact_oracle, false_elim
from .bank import build_bank, certified_csp_reductions, load_core_organ, load_blade_organ
from .compose import reduced_product, Trace

__all__ = [
    "Certificate", "CSPState", "Reduction", "exact_oracle", "false_elim",
    "build_bank", "certified_csp_reductions", "load_core_organ", "load_blade_organ",
    "reduced_product", "Trace",
]
