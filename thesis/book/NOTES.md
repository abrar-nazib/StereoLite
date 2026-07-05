# Thesis book: decisions log + working notes

Read `thesis/WRITING_PLAN.md` FIRST in every writing session. This file
records decisions and open items only.

## Decisions (settled)

| Date | Decision |
|---|---|
| 2026-07-05 | Title FIXED: "AI-Enhanced Stereo Matching for High-Accuracy Depth Mapping and 3D Reconstruction" |
| 2026-07-05 | Objectives FIXED verbatim (see ch1.tex section 1.3; do not reword) |
| 2026-07-05 | Citations: pure IEEE numeric (`IEEEtran.bst` + `Reference.bib`, 72 PDF-verified entries copied from review_paper) |
| 2026-07-05 | Similarity + AI report: obtained AFTER the book is complete (Phase 4 end), placeholders in Appendix C |
| 2026-07-05 | Architecture figures: draw.io pipeline (WRITING_PLAN section 9 grammar contract); matplotlib for data charts only |
| 2026-07-05 | Model public name "StereoLite" thesis-wide; variant slug only in Appendix A |
| 2026-07-05 | Two title pages skipped: single merged title page (Book Template style) implemented in frontmatter/titlepage.tex |

## Figure ledger (updated 2026-07-05, second pass)

ALL figures placed. Architecture diagrams are .drawio (editable; a
viewer-render preview PNG is embedded until the user exports cropped
PDFs from draw.io): 1.1 geometry, 2.1 classical pipeline, 2.3 timeline,
2.5 taxonomy, 3.1 overview, 3.2-3.8 per-block. Reproduced-with-citation
paper figures: 2.4 four-paradigm panel (PSMNet/RAFT/HITNet/
LiteAnyStereo), 2.7 KD stages (LiteAnyStereo). Data charts (matplotlib,
final PDFs): 1.3 edge gap, 2.6a/b Pareto pair, 3.10 input protocol,
3.11 param budget, 4.1 curves, 4.2 convergence, 4.3 qualitative,
4.4 MB14 bars, 4.5 MB14 qualitative, 4.6 camera panels, 4.7 ablation
summary, 4.8 final Pareto.

ALL tables filled with verified numbers (sources: meta.json,
mb14_zero_shot.json, EXPERIMENTS.md, comparison.md,
verified_performance.md): 2.1 datasets, 2.2 method comparison,
3.2 training config, 3.3 efficiency findings, 4.1 SF results,
4.2 SF comparison, 4.3 MB14 ladder, 4.5-4.7 ablations, 4.8 latency,
A.1 architecture config, A.2 MB14 per-scene. ONLY Table 4.4
(rectification sweep) awaits its experiment.

Citations wired: figure captions, tables, and section stubs carry
\cite keys from the 72 PDF-verified Reference.bib entries; References
section renders in IEEE style.

## Open items

- [ ] Head of MTE name (certificate countersign block)
- [ ] Submission month/year (titlepage + acknowledgments; currently July 2026)
- [ ] Appendix E: review-paper entry vs "none" (supervisor)
- [ ] MB14 per-scene table: Ch4 body vs appendix (layout-time call)
- [ ] Rectification-robustness sweep (WRITING_PLAN section 11a): BLOCKS Ch4 section 4.5
- [ ] Export fig_3_1_architecture.drawio to cropped PDF once user finishes hand-adjustments
- [ ] Jetson Orin Nano board: swap asterisked projections for measurements on arrival

## Build

```
cd thesis/book
pdflatex -interaction=nonstopmode Thesis.tex
bibtex Thesis
makeindex Thesis.nlo -s nomencl.ist -o Thesis.nls
pdflatex -interaction=nonstopmode Thesis.tex
pdflatex -interaction=nonstopmode Thesis.tex
```

Submission gates: `grep -rn "\\\\todo" mainmatter frontmatter appendices`
must return ZERO hits; no `??` in the PDF; roman i starts at Declaration,
arabic 1 at Chapter 1; every table caption above, figure caption below.

## PO tracking

Append one line per demonstrated PO to `po_tracker_notes.md` WHILE
drafting (format: `section | PO | one-sentence evidence`). Appendix D is
assembled from that file at the end.
