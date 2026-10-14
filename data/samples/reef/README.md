# Sample reef photos (dashboard demo only)

Three images from the **validation** split of Coralscapes (Sauder et al., 2025, EPFL-ECEO;
https://huggingface.co/datasets/EPFL-ECEO/coralscapes, Apache-2.0), saved at full resolution for uploading in the
dashboard's "Analyse an Image" page (habitat "Auto-detect coral" or "Coral reef").

| File | Expert labels (share of the coral area) | Use |
|---|---|---|
| `coralscapes_val_141.jpg` | 22% live, 32% bleached, 45% dead | heavily bleached reef |
| `coralscapes_val_060.jpg` | 87% live, 0% bleached, 13% dead | mostly healthy reef |
| `coralscapes_val_010.jpg` | no coral at all (sand / rock / other) | correct answer is 'No coral detected' |

The published Coralscapes model that v2 starts from was most likely tuned on the validation split, so use these
images for visual demos only, never for accuracy figures. Accuracy is reported on the official TEST split.
