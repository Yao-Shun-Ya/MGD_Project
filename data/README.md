# Data

The full MGD-1K data is **not stored in this git repository** because of size and licence constraints.

Expected layout:

```
data/
  images/          # original infrared images
  gland_masks/     # gland masks, same basename
  eyelid_masks/    # eyelid masks, same basename
  dataset_splits.json
  dataset_metadata.csv
```

- Full data: download from the project site / MGD-1K and place the three folders under `data/`.
- The patient-level split in this project is 694/153/152 (239/50/49 patients), file-list signature `6bdacfb0…`, assignment signature `7a6fd581…`.
- Regenerate the split after replacing the data with `python tools/make_patient_split.py`.
- The downloadable demo packages contain 6 real pairs for smoke testing only; do not report metrics from those 6 pairs.
