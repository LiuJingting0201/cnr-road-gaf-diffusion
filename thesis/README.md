# Road-Conditioned Diffusion for Traffic Trajectory Synthesis

English thesis research draft v11. The chapter on interventions is a validation-only follow-up carried out after the locked test. Three frozen model seeds and sixteen compatible bus/car routes provide a controlled class comparison. Main file: main.tex. Compile with pdfLaTeX.

## Open in Overleaf

Upload the ZIP as a new project, select `main.tex` as the main document and pdfLaTeX as the compiler. Use a recent TeX Live release. All text, tables, images and bibliography are included. No external data connection or shell escape is required. For a local build, run `latexmk -pdf main.tex` or run `pdflatex main.tex` three times. The packaged PDF was built and checked with TeX Live 2026.

Edit the administrative fields in `metadata.tex` before distribution. The working title and subtitle are editable. The candidate name, programme, supervisor, academic year and graduation session still require confirmation. `cover.tex` is an English adaptation of the referenced PoliTo layout. See `TEMPLATE.md` for attribution and the length comparison.

## Evidence and scope

The main text comprises eleven chapters. Four appendices document the frozen run and schema, five fixed synthesis cases, detailed class and downstream results, and the fixed-time guidance experiments with their independent replication. The empirical claims use the 6031e3f experiment snapshot. The group classifier weights were unavailable. The inherited encoding was reused and a new classifier was trained. The formal test covers 53 eligible requests out of 270 participants. Performance on the TS export, a new city or a joint traffic simulation was not evaluated. Draft v5 additionally reports newly run sampling interventions on development participants with frozen model weights; see docs/results/g8-development in this research branch.

The canonical figures are packaged already. Optionally, `tools/build_result_assets.py <experiment docs/results directory> <final_test_synthetic_trajectories.zip>` rebuilds the appendix tables, case plots and confusion plot from the archived metrics and synthetic export. It uses NumPy and ReportLab and is separate from the LaTeX build.

### Figure provenance

- The map overlay uses the original map-alignment audit (every thousandth input row), unchanged coordinates and public MoST lane geometry. The cover mark comes from the referenced template.
- The framework, fixed map examples, paired motion intervals and augmentation effects come from the project's locked final report. The case curves use the identical five seed-42 requests as the archived map examples.
- The encoding example comes from saved diffusion candidate 4, seed 42, for the first bus request. The original adapter ran with the frozen coordinate and channel extrema and the archived 512-point horizon. It illustrates the exact three-channel classifier input.
- All appendix aggregates use saved predictions or metrics. The cases were fixed by rule; no additional outcome-dependent filtering was performed.

G9 timing and motion-quality results are archived in docs/results/g9-timing.

Independent SUMO replication: docs/results/g10-independent and docs/G10_PROTOCOL.md. The original test was not reopened.

G11 state-trigger experiment: docs/G11_PROTOCOL.md and docs/results/g11-state. Fixed-time analyses and the first independent SUMO replication are now in the appendix.

The scoring-network audit in docs/results/g12-scorer separates training-internal model selection, confidence calibration and ranking utility.

G13 replicates shared-road class control on fresh episodes. See docs/results/g13-shared-road. First paired example, all-route effect plot and simplified framework figure are included in the source package.

G14--G17 add the separate balanced SUMO corpus and matched data-volume experiments. See docs/G16_PROTOCOL.md, docs/G17_PROTOCOL.md and the respective result directories.

Draft 9 adds the G18 motion-pattern audit and G19 road-conditioned diffusion / eight-class original-encoding feedback experiment. See docs/G19_PROTOCOL.md and docs/results/g19-motion-diffusion.

Draft 10 leads with the complete-trip generator. G21 compares short-window CNNs and G22 fits a variable-duration OD model and complete-journey scorers. See docs/G22_FULL_TRIPS.md and docs/results/g22-full-trips.

Draft 11 separates position-Hilbert trajectory images for class recognition from whole-speed angular images for mode recognition. The first results chapter compares their terminal feedback on eight new simulator episodes.
