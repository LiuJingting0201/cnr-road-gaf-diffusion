# Beijing GeoLife conditional generation

Both training seeds used the same 3,000 held-out requests and sampling seed. Each fit completed 200 epochs.

| Training seed | Model | Density JS ↓ | OD JS ↓ | Length JS ↓ | Top-20 cell F1 ↑ |
|---:|---|---:|---:|---:|---:|
| 42 | unconditional | 0.13085 | 0.27785 | 0.11861 | 0.60 |
| 42 | conditional | 0.01594 | 0.06913 | 0.02336 | 0.85 |
| 43 | unconditional | 0.14597 | 0.25671 | 0.07861 | 0.55 |
| 43 | conditional | 0.02081 | 0.08106 | 0.03676 | 0.85 |

Using the original trip attributes improved all four distributional scores at both seeds. Both models restore the requested point count after sampling. The experiment evaluates the combined trip attributes and does not isolate the effect of speed or vehicle class.
