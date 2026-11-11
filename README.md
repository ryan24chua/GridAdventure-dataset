# GridAdventure dataset

Labelled grid-cell images used to train the image parser of a Grid Adventure V1
agent. Each sample is one cell of a rendered level, cropped and resized to 32×32 RGB
and labelled with its entity class.

## Download

The data is in the release **v1**, file `GridAdventure-dataset-v1.zip`
(489 MB, 498 files):

https://github.com/ryan24chua/GridAdventure-dataset/releases/tag/v1

```
sha256  f566ba0f12da2f00a84cd273a9a8460702d3a279fb157cef11c2c9de73fb3bb2
```

The zip unpacks to `data/generated/`, the folder the training scripts read.

## How it was made

`make_dataset.py --seed 0` (generator version 3) creates it, and is deterministic:
running it again gives identical files.

1. The sprite variants of each class are split into 5 fold groups and 1 test group,
   with no sprite shared between groups (20% of each class's sprites go to test).
2. For each group, random levels are built from that group's sprites only and drawn
   with the game's own renderer. Every cell is cropped and labelled with the
   renderer's own selection helpers. Each group has 500 single-entity cells per
   class plus about 6,000 overlapping cells.
3. The public levels are cropped as the environment renders them (`public`), and
   are also re-rendered with only the test sprites (`public_test`). They are also
   re-rendered 4 times with each fold's sprites (`public_fold0` .. `public_fold4`),
   which measures whole-level parse rate on sprite designs a model never saw.

Grouping by sprite design means a model validated on fold k, or scored on test, has
never seen those designs. This matches the graded levels, which use unseen sprite
variants.

## Files

| File | Cells | Use |
|---|---|---|
| `cells_fold0.npz` .. `cells_fold4.npz` | ~13,000 each | cross-validation folds |
| `cells_test.npz` | 13,047 | held-out test sprites, scored once |
| `cells_public.npz` | 3,688 | the public levels as the environment renders them |
| `cells_public_test.npz` | 3,688 | public levels, test sprites only |
| `cells_public_fold0.npz` .. `cells_public_fold4.npz` | 14,752 each | public levels, one fold's sprites, 4 renders |
| `meta.json` | | class names, preprocessing, sprite split, counts |
| `assets_fold0/` .. `assets_test/` | | the sprite pack each group was drawn with |
| `preview.png` | | contact sheet from the dataset checker |

### Keys in each `cells_*.npz`

| Key | Shape / type | Meaning |
|---|---|---|
| `x` | (N, 32, 32, 3) uint8 | cell crop, resized with PIL bilinear |
| `cls14` | (N,) int8 | class index in `meta.json["classes14"]`, or -1 for overlapping cells |
| `mask` | (N, 32, 32) uint8 | sprite pixels (255) vs background (0); fold and test files only |
| `bg`, `main`, `under`, `corner` | int | finer labels: background, main entity, entity underneath, corner icons |
| `kind` | (N,) int8 | 0 single entity, 1 overlap |
| `pos`, `grid_wh`, `cell_px` | int | cell position, level size, native cell size in pixels |
| `level_seed` | (N,) int32 | level seed (fold and test files) |
| `level_idx`, `render_idx` | (N,) int16 | public level and re-render index (public files) |
| `xc`, `yc` | uint8 / int8 | corner-icon crops and their labels (fold and test files) |

The 14 classes are floor, wall, human, box, coin, gem, key, locked, opened, lava,
exit, boots, shield and ghost.

```python
import numpy as np
d = np.load("data/generated/cells_fold0.npz")
x, y = d["x"], d["cls14"]
x, y = x[y >= 0], y[y >= 0]        # drop overlapping cells
```
