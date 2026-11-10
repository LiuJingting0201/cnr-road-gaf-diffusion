# CNR Road / GAF Trajectory Diffusion

Map-, OD- and participant-class-conditioned synthesis of complete timestamped traffic journeys. The main learned motion component is temporal diffusion. Output follows `time,id,x,y,type`.

## Current work — 2026-11-10

G40 scientific work is complete: 16 motion fits, 12 learned code priors, eight prospective SUMO reference episodes, 128 generation/evaluation cells, 50,176 generated candidates and 30 independent downstream fits. Thesis v26 presents these results as a standalone study: 132 PDF pages, 82 main-text pages, 45 figures, 29 tables and 54 cited entries. All pages were rendered and reviewed; the clean Overleaf ZIP rebuild matches the delivered PDF page text.

- [Complete G40 interpretation and evidence](docs/results/g40-generative-conditioning/REPORT.md)
- [Thesis v26 PDF](deliverables/thesis-v26/Road_Conditioned_Diffusion_Thesis_v26.pdf), [Overleaf ZIP](deliverables/thesis-v26/PoliTo_Thesis_v26_Overleaf.zip), and [source/build instructions](thesis/README.md)
- [Execution state](docs/G40_RUN_STATE.md), [taskbook](docs/THESIS_TASKBOOK.md), [experiment log](docs/EXPERIMENT_LOG.md)
- [Fixed execution protocol](docs/G40_EXECUTION_PROTOCOL.md) and [generative-conditioning design](docs/G40_GAF_GENERATIVE_CONDITIONING_DESIGN.md)

The system learns preferences over feasible route catalogs, samples duration, generates motion on a 0.25-second clock, and decodes to road coordinates. G40 tests learned spatial and speed GAF conditions with frozen, joint, raw-series, wrong-trip and class-centroid controls. Image encoders are trained in this project; no group CNN weights were supplied. The inherited ordered Hilbert–GASF/GADF/MTF spatial transform is retained. Speed images are a separate view derived from XY/time. New generation uses a training-only code prior, not an image of the unknown future or image inversion.

The full evidence does **not** establish an overall physical-quality gain from GAF. Spatial-only augmentation gives a small pooled downstream gain (96.95% macro F1 versus 96.26% for baseline augmentation), but effects have opposite signs across the two evaluation OD cohorts. The dual-view method worsens native acceleration distance by 2.44%. All classes, controls, failures, coverage limits and adverse endpoints are retained. No further model search is automatically queued.

## Data and evaluation boundaries

The supplied principal CNR export supplies the source-domain audit. Expanded native MoST demand supplies 29,895 training legs from 24,073 source parents, across bus, motorcycle, private car, bicycle and taxi/ride-hailing. Legs, parents and independent simulation scenes are different units. The second supplied export lacks a confirmed Monaco map match and is not silently pooled. Original coordinates are local planar XY, not GPS longitude/latitude.

The final comparison uses four new non-bus OD cohorts with paired time windows. Public bus services and the road map remain shared. Physical quality, route-use distributions, diversity and independent synthetic-data augmentation are separate endpoints. Observed references are SUMO data, not field observations. Topology support comes from the route interface/decoder; it is not a GAF result or an interactive traffic-safety guarantee.

## Earlier closed evidence

- [G38 route-preference comparison](docs/results/g38-route-preferences/STATUS.md) and [protocol](docs/G38_PROTOCOL.md).
- [G39 complete evidence](docs/results/g39-native-alignment/completed-batch/) and [performance study](docs/results/g39-performance/REPORT.md).
- [G33 physical-clock pilot](docs/results/g33-generation/REPORT.md), [G34 duration control](docs/results/g34-fixed-duration/REPORT.md), and [G35/G36 generation](docs/results/g36-generation/STATUS.md).
- [CNR/random-demand diagnosis](docs/results/g29-source-audit/REPORT.md) and [native-demand recovery](docs/results/g30-native-demand/REPORT.md).
- [Earlier exposed E7 test](docs/results/final-test-e7/REPORT.md), retained without reuse for current method selection.
- [Historical repository overview](docs/archive/README-before-v20.md) and [early working notes](docs/archive/early-local-notes/).

## Publication and archival

This source repository is public. Code, protocols, small metrics, figures and manuscripts are published here. Raw datasets, model weights and secrets are excluded.

The [private November 9 archive](https://github.com/LiuJingting0201/cnr-road-gaf-artifacts/releases/tag/archive-2026-11-09) contains 138 verified attachments (54.40 GiB), including the then-completed G38 snapshot and earlier assets. [Index and restore instructions](docs/GITHUB_ARCHIVE.md). That historical archive does not imply that the subsequently trained G39/G40 weights have already been added; their immutable server snapshots remain preserved.

Resume work by reading `AGENTS.md`, the current taskbook entry and the latest experiment log. Never restart a closed queue merely because an older plan describes it as pending. The earlier server worktree has uncommitted legacy work and remains protected.
