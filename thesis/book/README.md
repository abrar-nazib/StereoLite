# StereoLite Thesis Book (RUET MTE B.Sc.)

LaTeX source for the undergraduate thesis *"AI-Enhanced Stereo Matching for
High-Accuracy Depth Mapping and 3D Reconstruction"* (Department of
Mechatronics Engineering, Rajshahi University of Engineering & Technology).

Authors: Md. Raihanul Haque Rahi (Roll 2008011), Nazib Abrar (Roll 2008026).
Supervisor: Md Zunaid Hossen, Lecturer, MTE, RUET.

## Compile the PDF

Run from inside this `thesis/book/` directory (the master file is `Thesis.tex`):

```bash
cd thesis/book
pdflatex -interaction=nonstopmode Thesis.tex
bibtex Thesis
pdflatex -interaction=nonstopmode Thesis.tex
pdflatex -interaction=nonstopmode Thesis.tex
```

The three `pdflatex` passes plus the `bibtex` pass resolve the table of
contents, the figure and table lists, the numeric citations, and all
cross-references. The output is `Thesis.pdf` (about 77 pages, A4).

One-liner:

```bash
cd thesis/book && \
  pdflatex -interaction=nonstopmode Thesis.tex >/tmp/lt.log 2>&1 && \
  bibtex Thesis >/dev/null 2>&1 && \
  pdflatex -interaction=nonstopmode Thesis.tex >/tmp/lt.log 2>&1 && \
  pdflatex -interaction=nonstopmode Thesis.tex >/tmp/lt.log 2>&1
```

## Requirements

A TeX Live distribution with `pdflatex` and `bibtex`, plus the packages the
document loads: `txfonts`, `titlesec`, `placeins`, `geometry`, `graphicx`,
`caption`, `subcaption`, `booktabs`, `amsmath`, `setspace`, `hyperref`,
`nomencl`. On Debian/Ubuntu, `texlive-full` covers all of them.

## Layout

```
Thesis.tex              master document (geometry, headings, page numbering)
frontmatter/            title page, certificate, declaration,
                        acknowledgments, abstract, nomenclature
mainmatter/             ch1.tex .. ch5.tex (Introduction .. Conclusion)
appendices/             app_a.tex .. app_e.tex (specs, project mgmt,
                        ethics/AI disclosure, PO/KPA tracker, publications)
Reference.bib           verified BibTeX entries
figures/                embedded figure PDFs/PNGs
figures/_src/           Python scripts that GENERATE the figures
```

## Regenerating figures

The figure sources live in `figures/_src/`. Activate the project venv first
(`source ../../venv/bin/activate`). Notable scripts:

- `make_fig46_camera.py` and `make_fig4x_reconstruction.py` run the trained
  checkpoint on the real-rig scenes for the qualitative and 3D-reconstruction
  figures.
- `make_drawio_batch.py` builds the `.drawio` architecture diagrams;
  `render_drawio_hires.py` renders them to crisp PNG previews with headless
  chromium.
- `eval_realcam_rectification.py` computes the real-camera rectification
  numbers cited in Chapter 4.

The reproduced per-paradigm panels (PSMNet, RAFT-Stereo, HITNet,
LiteAnyStereo) are cropped from the original papers under `papers/raw/` and
cited in place.

## Notes

- No em dash or en dash in body prose (project-wide rule).
- Every reported number is cited to a repository source (a run's
  `meta.json`, `papers/verified_performance.md`, or a benchmark table).
- Figures use captions below, tables use captions above; front matter is
  roman-numbered, the main matter is arabic from Chapter 1.
