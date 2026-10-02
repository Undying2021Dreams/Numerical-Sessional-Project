# Final report

`final_report.pdf` is the compiled report. The rest of this folder is its
LaTeX source, in the ACM `sigconf` format (`acmart.cls`).
All figures are in `figures/` and the bibliography is in `references.bib`.

Build from this directory with TeX Live or MiKTeX:

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

Every figure is also produced by a script under `experiments/` and written to
`results/`; the copies here are the versions used in the report.
