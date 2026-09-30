# Scientific and engineering limitations

The earlier audit found a pandas 3 regression: `DataFrameGroupBy.apply` removed
grouping columns, after which a fallback collapsed every recording into trial 0.
This release uses a Series transform to preserve metadata and adds a regression
test. Pupil baseline correction now happens before standardization.

The following issues remain and should be addressed before interpreting real
CIT model accuracy. Track work in the repository's GitHub Issues.

| Limitation | Proposed acceptance criterion |
|---|---|
| CIT segmentation uses changes in stimulus name; calibration/video stages may be included. | Use stimulus start/end events and a documented phase filter; compare trial counts against the experiment protocol. |
| AOI columns and available fixation/saccade events are unused; fixation count is heuristic. | Extract dwell time, fixation duration, and critical-versus-neutral AOI contrasts from vendor event fields. |
| One fixed participant split and small local data subset give unstable accuracy. | Verify the full dataset manifest, use repeated group cross-validation, and report participant-level uncertainty. |
| Missing labels can silently default to class 0. | Fail on missing/unknown required labels; distinguish optional from mandatory schema fields. |
| Validity selection currently favors the left eye. | Combine left/right validity using a documented rule and test asymmetric eye validity. |
| XLSX loading reads all columns and holds all recordings in memory. | Load only required columns and cache a compact intermediate table with documented provenance. |
| `online_safe` mode only makes smoothing causal; interpolation and normalization may use future samples. | Build and test a fully causal preprocessing path before live inference. |
| Within-subject splitting randomly splits examples and can separate overlapping windows. | Keep entire trials in a split or apply a temporal gap before claiming within-subject generalization. |
| Baseline validation split is reserved but unused; model parameters are mostly fixed. | Tune parameters on training/validation folds without consulting test participants. |
| Deep checkpoints contain weights only. | Bundle architecture/config, feature order, and preprocessing metadata for reproducible inference. |
| Bag-of-Lies adapter is a placeholder. | Add a source-grounded parser and independent fixtures before claiming support. |

Historical locally generated metrics have not been committed: they were obtained
before these preprocessing fixes and are not evidence for the corrected pipeline.

## GitHub tracking

- [Metadata preservation regression and fix (#1)](https://github.com/terfo1/deception-detection-prototype/issues/1)
- [Protocol-based CIT segmentation (#2)](https://github.com/terfo1/deception-detection-prototype/issues/2)
- [Fixation and critical/neutral AOI features (#3)](https://github.com/terfo1/deception-detection-prototype/issues/3)
- [Dataset completeness and repeated participant validation (#4)](https://github.com/terfo1/deception-detection-prototype/issues/4)
