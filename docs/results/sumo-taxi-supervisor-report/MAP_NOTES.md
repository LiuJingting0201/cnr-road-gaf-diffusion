# Map comparison

Every figure shows the first, middle and last request in the original scoring order and includes both training seeds. Each panel compares a single saved, unsnapped generation with the observed coordinates associated with its request. Origin and destination markers and their Euclidean separation in metres are drawn before any grid assignment. The requested endpoints are grid-cell IDs. Alternative choices of position within the same cell are possible.

| Figure | Evaluation population | One-based requests |
|---|---|---|
| [Supplied vehicle observations](figures/maps-supplied.png) | 177 reserved vehicles | 1, 89 and 177 |
| [Simulated taxis, 10k training journeys](figures/maps-taxi-10k.png) | 1,000 development journeys | 1, 501 and 1,000 |
| [Simulated taxis, 20k training journeys](figures/maps-taxi-20k.png) | The same 1,000 development journeys | 1, 501 and 1,000 |

For the supplied table, the three examples correspond to bus, motorcycle and private car using the class column of the original CSV. The baseline does not receive those labels. For the two taxi datasets, the plotting script checks vehicle identities and every observed coordinate in all six paired panels. Each corresponding request and seed in the 10k and 20k figures uses the same axis limits. The third lower panel covers a larger area because the 20k seed-43 draw extends away from the observed route.

The map geometry comes from the public Monaco MoST scenario (network SHA-256 a06b5c0765fb2d49d4ad33cf54048f15f2ff1592b2b3d0180dbeec4feb73bfc0). The new SUMO journeys were simulated on this network. For the supplied CSV the network serves as a geographical reference. The previous audit tested every 1,000th input row with its original coordinates, finding all 4,841 checked points within 5 m of a lane centreline. The exact network version and demand configuration used to produce the supplied CSV were not supplied. [Map alignment audit](../../map_alignment.json) · [MoST scenario](https://github.com/lcodeca/MoSTScenario)

The example metadata stores per-request start and end offsets, observed native polyline length and generated length along the model's original 200 positions. The four-metric evaluation restores the output to the observed point count before measuring length. These two lengths can differ slightly. The full comparison uses all 177 or all 1,000 requests; the illustrations provide examples. [Values for all plotted panels](map-examples.json) · [Plotting script](../../../experiments/difftraj/plot_supervisor_sumo_maps.py)

## Geographic distribution over all trips

The occupancy figure follows the geographic comparison of generated and observed point distributions in DiffTraj Figure 2, applied here to three SUMO cohorts. The display uses the same 50 m square cells, map extent and log colour scale in all nine panels. Colour shows positions per million recorded points in each cohort. The denominator includes positions outside the common physical viewport; each panel prints their fraction. All 177 supplied holdout requests and all 1,000 taxi development requests are included. The latter population is identical for the 10k and 20k models. Generated trajectories are sampled back to each corresponding track's native observation count before binning. Longer tracks therefore contribute more positions. The physical extent and fine cell width here apply to the visualisation; the scalar density JS uses its original 16-by-16 training grid. [Overall occupancy](figures/occupancy.png) · [Counts and map bounds](occupancy.json) · [DiffTraj Figure 2](https://arxiv.org/html/2304.11582#S5.F2)
