# Pokémon data

This lab uses the instructor-supplied copy of Rounak Banik's [The Complete Pokémon Dataset](https://www.kaggle.com/datasets/rounakbanik/pokemon). The local pokemon.csv contains 801 catalog entries, 41 columns, generations 1–7, and the source's legendary-status labels. It is a historical snapshot, not a current inventory or a sample of individual battles. The notebook's compact variable guide explains the columns needed for this lab.

SHA-256 of the exact supplied bytes:

```text
d11af5a43aeb5a3661060f1eddc5c63ec4b4657ffe0a8018ba5aa6b260b4c97a
```

The notebook reads this checked-in file and verifies its checksum. No Kaggle account, API key, or data download is needed. Package installation requires internet access.

## Snapshot details

- There are 70 legendary and 731 non-legendary entries; source labels are retained without reclassification.
- Missing entries: height_m 20, weight_kg 20, percentage_male 98, and type2 384. A blank type2 indicates no recorded secondary type, not a missing physical measurement.
- base_total exactly equals the sum of the six battle statistics for every row.
- The notebook preprocessing excludes entries whose primary type is flying and Minior before either modeling task; the checked-in source CSV remains unchanged.
- Forms and game rules can change across releases. Interpret values as this supplied snapshot rather than silently updating entries from another source.
