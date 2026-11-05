# Road-Conditioned Diffusion for Traffic Trajectory Synthesis

English thesis research draft v2. The chapter on interventions is a validation-only follow-up carried out after the locked test. Three frozen model seeds and sixteen compatible bus/car routes provide a controlled class comparison. Main file: main.tex. Compile with pdfLaTeX.

## Open in Overleaf

Upload the ZIP as a new project, select `main.tex` as the main document and pdfLaTeX as the compiler. Use a recent TeX Live release. All text, tables, images and bibliography are included. No external data connection or shell escape is required. For a local build, run `latexmk -pdf main.tex` or run `pdflatex main.tex` three times. The packaged PDF was built and checked with TeX Live 2026.

Edit the administrative fields in `metadata.tex` before distribution. The working title and subtitle are editable. The candidate name, programme, supervisor, academic year and graduation session still require confirmation. `cover.tex` is an English adaptation of the referenced PoliTo layout. See `TEMPLATE.md` for attribution and the length comparison.

## Evidence and scope

The main text comprises nine chapters. Three appendices document the frozen run and schema, five fixed synthesis cases, and detailed class and downstream results. The empirical claims use the 6031e3f experiment snapshot. The group classifier weights were unavailable. The inherited encoding was reused and a new classifier was trained. The formal test covers 53 eligible requests out of 270 participants. Performance on the TS export, a new city or a joint traffic simulation was not evaluated. Draft v2 additionally reports newly run sampling interventions on development participants with frozen model weights; see docs/results/g8-development in this research branch.

The canonical figures are packaged already. Optionally, `tools/build_result_assets.py <experiment docs/results directory> <final_test_synthetic_trajectories.zip>` rebuilds the appendix tables, case plots and confusion plot from the archived metrics and synthetic export. It uses NumPy and ReportLab and is separate from the LaTeX build.

### Figure provenance

- The map overlay uses the original map-alignment audit (every thousandth input row), unchanged coordinates and public MoST lane geometry. The cover mark comes from the referenced template.
- The framework, fixed map examples, paired motion intervals and augmentation effects come from the project's locked final report. The case curves use the identical five seed-42 requests as the archived map examples.
- The encoding example comes from saved diffusion candidate 4, seed 42, for the first bus request. The original adapter ran with the frozen coordinate and channel extrema and the archived 512-point horizon. It illustrates the exact three-channel classifier input.
- All appendix aggregates use saved predictions or metrics. The cases were fixed by rule; no additional outcome-dependent filtering was performed.
