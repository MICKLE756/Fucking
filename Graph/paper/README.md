# Paper draft (LaTeX)

A complete first draft of the MECPE paper, formatted with the Elsevier
`elsarticle` class (suitable for **Neurocomputing / Knowledge-Based Systems /
Expert Systems with Applications**, the CCF-C international journals recommended
as the fastest path to the minimum graduation requirement).

## Files
- `main.tex` — the full paper (Abstract, Introduction, Related Work, Method,
  Experiments + Ablation + Analysis, Conclusion).
- `refs.bib` — bibliography.

## Build
```bash
cd Graph/paper
latexmk -pdf main.tex     # or: pdflatex main && bibtex main && pdflatex main && pdflatex main
```
Output: `main.pdf`.

## IMPORTANT — this is a DRAFT
- **Every number in the Experiments section is a PLACEHOLDER.** Search the source
  for `TODO` to find each value/figure that must be replaced with the real
  experimental results once the runs finish (use
  `scripts/aggregate_results.py` to produce the main + ablation tables, and the
  `valid` evaluation protocol described in `../REPRODUCE.md`).
- Replace the placeholder author names/affiliation in the front matter.
- The Method section mirrors `../英_改Method定.md` (§3); keep the two in sync if
  you edit the method.
- Add the real architecture figure (`Fig. 1` is currently a placeholder box).
- Do not claim "state of the art" until the final numbers support it.
