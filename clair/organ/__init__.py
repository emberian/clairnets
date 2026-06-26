"""clair.organ — THE assembled GLaDOS organ.

ONE coherent, trainable, LLM-ready package that wires every validated beast into a typed spine:

  protocol  : the typed interface (Reduction / CSPState / Certificate + the structured-gamma readout)
  bank      : the registry of validated beasts as reductions (certified ops + neural guidance)
  compose   : verifier-gated certified-reduction composition (the sound reduced product)
  train     : the single training + LLM-weaving entry point (dominate-dedP + the staged woven recipe)
  selftest  : the consolidated smoke (run it: python -m clair.organ.selftest)

See README.md for the per-beast status table and how to train / weave it.
"""
from .protocol import Certificate, CSPState, Reduction, exact_oracle, false_elim
from .bank import build_bank, certified_csp_reductions, load_core_organ, load_blade_organ
from .compose import reduced_product, Trace

__all__ = [
    "Certificate", "CSPState", "Reduction", "exact_oracle", "false_elim",
    "build_bank", "certified_csp_reductions", "load_core_organ", "load_blade_organ",
    "reduced_product", "Trace",
]
