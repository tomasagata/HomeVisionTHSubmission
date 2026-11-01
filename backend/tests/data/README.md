# Test data

Each subfolder is a *dataset*: a group of images that share a quirk I want to measure the pipeline against.

```
data/
  <dataset>/
    annotations.json    COCO format, exported from Label Studio
    images/             the images referenced by annotations.json
```

| Dataset | Contents |
|---|---|
| `fullres` | The four form pages at their original resolution (up to 2550×4200), extracted from the challenge PDF. |
| `lowres`  | The same four pages at roughly a third of the resolution (837–997 px wide). These are the images I originally labelled. The `fullres` labels are these boxes rescaled. |

## Labels

I labelled the images in Label Studio and exported them as COCO. Each annotation is a `[x, y, width, height]` box in pixels, with one of three categories:

- `checked checkbox`
- `unchecked checkbox`
- `ambiguous checkbox`: boxes where I couldn't honestly say whether they're checked. Detection counts them, but classification ignores them.

The tests match categories by name, not by id, so the id order in an export doesn't matter.

## Adding a dataset

1. Create `data/<name>/images/` and put the images in it.
2. Export the labels from Label Studio as COCO and save them as `data/<name>/annotations.json`.

That's all. The test suite discovers the folder on its next run and reports its metrics as a separate group. `file_name` in the export is resolved relative to the dataset folder. If it doesn't point to an existing file (Label Studio usually writes something like `../images/<hash>-name.png`), the loader looks for the bare filename inside `images/`.
