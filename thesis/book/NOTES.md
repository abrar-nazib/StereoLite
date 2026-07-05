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

## Figure ledger (2026-07-05)

PLACED (embedded, final captions): fig 1.1 geometry, fig 1.3 edge gap,
fig 2.2 timeline, fig 2.5 taxonomy, fig 2.6a/b Pareto pair, fig 3.1
architecture (INTERIM preview PNG; swap for the drawio PDF export),
fig 4.1 training curves, fig 4.2 convergence filmstrip, fig 4.3
qualitative grid, fig 4.4 MB14 per-scene bars.

STILL TODO (stubs with instructions in the tex): fig 2.1 classical
pipeline, fig 2.4 four-paradigm panel, figs 3.2 to 3.8 per-block drawio,
fig 3.10 input protocol, fig 4.5 MB14 qualitative (needs --save_viz
Modal rerun), fig 4.6 camera panels (regenerate with best.pth), fig 4.7
ablation bars, fig 4.8 final Pareto. Architecture figures follow the
drawio grammar contract (WRITING_PLAN section 9).

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
