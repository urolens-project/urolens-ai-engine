# Dataset rebuild: UroLens-3 → UroLens-4-clean

**Status:** complete. Produced by `scripts/build_dataset.py --seed 0`.
**Supersedes:** the Roboflow export "UroLens v3 v3-rebalanced" (22 May 2026) for
all training and evaluation.

---

## 1. Why the rebuild was necessary

Every accuracy figure this project produced before this rebuild was measured, in
part, on images the model had trained on. Two independent contamination
mechanisms were stacked on top of each other, and neither was visible from
Roboflow's own summary — its README states *"No image augmentation techniques were
applied,"* which is true of Roboflow's generate step and irrelevant to the problem.

### 1.1 Duplicate and pre-augmented copies split at random

The export contained 35,796 files but only **15,045 distinct images**. The rest
were brightness/contrast variants and mirrored copies, uploaded as if they were
separate source images. Roboflow's splitter therefore treated them as independent
and scattered them across train/valid/test.

| Measure | Value |
|---|---|
| Files in export | 35,796 |
| Distinct images | 15,045 |
| Boxes in export | 210,212 |
| Boxes after dedup | 88,801 (42%) |
| Images with a same-named sibling in another split | 18,234 (50.9%) |

More than half the annotation volume was the same annotations counted two or three
times, which also silently reweighted the class balance toward whichever images
happened to be duplicated most.

### 1.2 Mirrored copies — invisible to filename matching

Part of the corpus carries flip and rotation augmentation baked in as files.
Deduplicating by filename does not catch these, and worse, would arbitrarily pick
between a field and its mirror.

Verification used two independent tests, because these images are very low
contrast (grayscale σ ≈ 4.4 of 255) and a naive descriptor matches the
microscope's illumination gradient rather than the specimen:

1. **High-pass pixel test** — subtract a σ=6 gaussian at 128 px to isolate
   particle detail, then correlate.
2. **Label mirror test** — require the second image's bounding boxes to land on
   the first's *mirrored* coordinates. Box geometry cannot agree under mirroring
   by coincidence.

Of 490 candidates, **319 passed both**. The clearest case: source
`1580332483256`, present twice with 12 boxes each, related by horizontal flip —
coarse correlation 0.998, upright −0.360, high-pass +0.848, and **all 12 boxes
matching mirrored coordinates**. Transform counts were near-uniform (fliplr 170,
flipud 160, rot180 160), the signature of systematic augmentation rather than
coincidence.

### 1.3 Specimen contamination — partly unmeasurable

Where filenames encode patient and session, several microscope fields belong to
one specimen and must never be split apart. They were:

| Split | Images sharing a specimen with train |
|---|---|
| valid | 75.1% |
| test | 75.5% |

This is measurable for only **24.9% of the corpus**. For the remaining 26,883
images the filename carries no patient identifier at all, so specimen leakage
there is *unmeasurable rather than absent*. See §4.

### 1.4 Measured effect

Same weights, same settings, specimen-disjoint subset versus the test set as
shipped:

| Metric | Test as shipped | Specimen-disjoint | Change |
|---|---|---|---|
| mAP@50 | 0.843 | 0.746 | −11.5% |
| mAP@50-95 | 0.590 | 0.436 | −26.2% |
| Precision | 0.835 | 0.740 | −11.3% |
| Recall | 0.763 | 0.716 | −6.2% |

The effect is strongly non-uniform by class — bacteria −40.0%, sperm-cells −26.8%,
erythrocytes −17.7%, while trichomonas is unchanged at 0.0%. The leak was masking
precisely the classes that fail to generalise.

---

## 2. Source composition

The corpus is a merge of at least six acquired sources, identifiable by filename
convention. **Licences and original provenance still need to be recorded here
before publication.**

| Family | Example | Images | Share | Specimen recoverable |
|---|---|---|---|---|
| `nh…` / `nl…` id | `nh01021` | 14,055 | 39.3% | No |
| Pure number | `1580332483256` | 12,548 | 35.1% | No |
| `WIN_` camera | `WIN_20211006_17_08_50_Pro` | 4,019 | 11.2% | Yes |
| Field/date/time | `00501_200205_02_34_09_-_01_bmp` | 2,811 | 7.9% | Yes |
| Timestamp + field | `20211106002601_01` | 2,083 | 5.8% | Yes |
| Other | `a1582149090150`, `1-39-` | 280 | 0.8% | No |

The verified mirrored pairs all fall in the pure-number family — one specific
acquired source, not the project's own `WIN_` captures or QC slides.

---

## 3. What the rebuild does

`scripts/build_dataset.py`, deterministic under `--seed`:

1. Fingerprint every image: 32×32 z-normalised grayscale descriptor (invariant to
   the brightness variants) plus a 64-bit median-threshold hash for each of four
   orientations, bucketed so mirrored pairs collide.
2. Union same-source names directly; verify cross-name candidates with the
   high-pass and label-mirror tests from §1.2.
3. Collapse each visual group to one image, keeping the copy with the most boxes.
4. Union groups further by specimen where recoverable.
5. Split 70/15/15 **by group**, stratified by source family *and* dominant class.
6. Assert no group, source name, or verified duplicate spans two splits — and fail
   the build if one does.

### Result

| | Images | Boxes |
|---|---|---|
| train | 10,548 | 62,698 |
| valid | 2,250 | 13,063 |
| test | 2,247 | 13,040 |
| **total** | **15,045** | **88,801** |

Independent post-build audit: **zero** source-name overlap and **zero** specimen
overlap across all three split pairs. All ten classes are present in all three
splits — including `mucus-threads` (1,444 / 367 / 348), which had vanished
entirely from an earlier ad-hoc clean subset.

### Artifacts written beside the dataset

| File | Contents |
|---|---|
| `split_manifest.json` | Per-image split, family, specimen, box count; the seed and ratios |
| `duplicates.csv` | Every verified duplicate pair, its transform, and all correlation scores |
| `conflicts.csv` | The 936 groups whose copies disagreed on box count |

---

## 4. Known limitations

**Specimen grouping covers only 24.9% of images.** The `nh`/`nl` and pure-number
families encode no patient identifier, so fields from one patient may still be
split across partitions there. Content-based deduplication mitigates this — near-
identical fields are caught regardless of filename — but two genuinely different
fields from the same specimen are not detectable from the data alone. Recovering
this would mean going back to the original source datasets for provenance.

**936 groups disagree on their own labels.** Where copies of one image carry
different box counts, the richest copy is kept, on the reasoning that a blank copy
of an image annotated elsewhere is a missed annotation rather than a genuinely
empty field. This auto-resolves the harmful annotated-versus-blank cases, but it
is a heuristic. `conflicts.csv` exists so a Medical Technologist can adjudicate.
The disagreement rate is also a useful incidental measurement of annotation noise:
roughly 6–7% of duplicate groups, from what amounts to an accidental
re-annotation experiment.

**Old checkpoints cannot be evaluated on this test set.** 88.5% of
`UroLens-4-clean/test` was in `UroLens-3/train`, so the pre-rebuild `best.pt` has
seen most of it. `test_fair_for_old_model.txt` lists the 258 images that are fair
for the old model. For like-for-like comparison against the rebuilt corpus, use
the specimen-disjoint baseline in §1.4 (mAP@50 **0.746**, mAP@50-95 **0.436**) —
**never the 0.843 figure**, which is the leaked number.

---

## 5. Reproducing

```bash
python scripts/build_dataset.py \
    --src C:/Users/Harley/UroLens/UroLens-3 \
    --dst C:/Users/Harley/UroLens/UroLens-4-clean \
    --seed 0
```

Two runs with the same seed produce a byte-identical `split_manifest.json`;
`tests/unit/test_build_dataset.py` asserts this, along with split disjointness and
— on a synthetic corpus with a planted mirrored pair — that the flip-aware path
actually fires rather than being assumed to work.
