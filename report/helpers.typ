// Shared helpers, imported by glados.typ and each section.
#let term(x) = text(style: "italic", weight: "medium", x)
#let arxiv(id) = box(text(size: 8.5pt, fill: rgb("#444"))[arXiv:#id])
#let banner(body) = align(center)[#block(width: 94%, inset: 8pt, stroke: 0.5pt + gray, radius: 3pt)[#text(9pt)[#body]]]
#let keynote(body) = block(width: 100%, inset: 8pt, fill: luma(245), radius: 3pt)[#body]
