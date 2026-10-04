# UroLens Smart Diagnosis Rule Engine Design

**Document version:** 1.0  
**Status:** DOMAIN EXPERT REVIEWED — Sign-off obtained  
**Author:** AI Engineer  
**Last updated:** Sprint 6  

---

## 1. Purpose

This document describes the design of the UroLens Smart Diagnosis rule engine — the component responsible for mapping confirmed particle classifications to clinical probability indicators (LOW / MODERATE / HIGH) for three target conditions.

---

## 2. Target Conditions

| Condition | Clinical Definition | Diagnostic Confidence from Urinalysis |
|---|---|---|
| **Gout** | Elevated uric acid crystals indicating hyperuricemia | Moderate — crystal elevation is a valid urinalysis indicator |
| **Glomerulonephritis (GN)** | Inflammation of the kidney glomeruli | Low — additional tests (biopsy, serology) required for definitive diagnosis |
| **Nephrolithiasis** | Kidney stone formation | Low — imaging (ultrasound, CT) required for definitive diagnosis |

---

## 3. Particle-Condition Associations

### 3.1 YOLOv8 Model Classes

The current MVP model detects 10 particle classes:

| Class | Description |
|---|---|
| `bacteria` | Bacterial rods or cocci |
| `crystals` | All crystal morphologies (uric acid, calcium oxalate, struvite, phosphate) |
| `epithelial-cells` | Squamous, transitional, or renal tubular epithelial cells |
| `erythrocytes` | Red blood cells (normal and dysmorphic) |
| `leukocytes` | White blood cells |
| `mucus-threads` | Mucus strands |
| `sperm-cells` | Spermatozoa |
| `trichomonas-vaginalis` | Trichomonas vaginalis parasites |
| `urinary-casts` | All cast types (hyaline, granular, cellular, RBC) |
| `yeast` | Yeast cells |

### 3.2 Condition-Particle Mapping

| Condition | Particle | Role | Weight | Rationale |
|---|---|---|---|---|
| Gout | `crystals` | Primary | 1.0 | Uric acid crystals are the hallmark of gout-related urinalysis findings |
| Glomerulonephritis | `urinary_casts` | Primary | 0.4 | Reduced per MedTech review — cast presence noted but limited diagnostic specificity |
| Glomerulonephritis | `erythrocytes` | Supporting | 0.2 | Reduced per MedTech review — hematuria is non-specific for GN |
| Nephrolithiasis | `crystals` | Primary | 0.3 | Reduced per MedTech review — crystal elevation is supportive only |
| Nephrolithiasis | `erythrocytes` | Supporting | 0.1 | Reduced per MedTech review — weak supporting indicator only |

---

## 4. Scoring Logic

### 4.1 Weighted Excess Scoring

```
contribution = max(0, detected_count - normal_range_max) * weight
weighted_score = sum of contributions across all condition particles
```

### 4.2 Probability Level Mapping

```
weighted_score <= low_max_score   -> LOW
weighted_score >= high_min_score  -> HIGH
between                           -> MODERATE
```

### 4.3 Threshold Values

> **`src/urolens_ai/smart_diagnosis/config.yaml` is the single source of truth.**
> This table is a copy for readers and must be updated alongside it. The values
> below were verified against config.yaml as of commit `2926b49`.

| Condition | low_max | high_min | Normal Range Max | Evidence Min Count |
|---|---|---|---|---|
| Gout | 2.0 | 50.0 | crystals: 5 | 2 |
| Glomerulonephritis | 3.0 | 35.0 | urinary_casts: 0, erythrocytes: 3 | 1 |
| Nephrolithiasis | 3.0 | 35.0 | crystals: 5, erythrocytes: 3 | 2 |

An earlier revision of this table listed Gout as 3.0/10.0, Glomerulonephritis as
2.0/10.0 and Nephrolithiasis as 3.0/15.0. Those values predated commit `2926b49`,
which raised every `high_min_score` per MedTech review, and the Gout and
Glomerulonephritis `low_max` values were additionally transposed. Anything derived
from the old table -- including draft paper appendices -- should be regenerated.

### 4.3.1 Known limitation: the HIGH band is effectively unreachable

Replaying the current configuration over 3,592 annotated fields produced:

| Condition | Fields reaching HIGH | Note |
|---|---|---|
| Gout | **0.0%** | needs 55+ crystals in one field; the maximum observed is 40 |
| Glomerulonephritis | 0.3% | |
| Nephrolithiasis | 0.2% | |

Roughly 97% of fields return LOW for every condition. The thresholds were raised
deliberately, and conservatism is correct for a screening aid -- but a band that
*cannot* be reached is not conservatism, it is dead code in the score mapping. It
also inflates any agreement metric computed against these levels, since predicting
LOW unconditionally already scores ~97%.

Re-derivation from observed score percentiles is pending MedTech re-review; see
`scripts/evaluate.py`, which reports Cohen's kappa and the always-LOW baseline
alongside raw agreement so this cannot be misread.

### 4.4 Threshold Rationale

| Condition | Rationale |
|---|---|
| Gout | Standard thresholds retained — crystal elevation is a valid urinalysis indicator per MedTech review |
| Glomerulonephritis | Thresholds raised significantly per MedTech review — GN cannot be reliably diagnosed from urinalysis alone. HIGH is very difficult to reach by design. |
| Nephrolithiasis | Thresholds raised significantly per MedTech review — nephrolithiasis requires imaging for definitive diagnosis. System produces LOW in most cases. |

---

### 4.5 Literature Basis

The weights and normal ranges below were set by MedTech consultation (Section 9).
This section records the published sources that independently support them, so the
justification is not "one reviewer said so" alone. This is literature grounding,
not statistical validation -- no confirmed-diagnosis dataset exists yet, and a
sensitivity/specificity study against one would be the stronger follow-up.

#### Reference ranges

- **Erythrocytes, `normal_range_max = 3`.** Normal urinary RBC is 0-3 per
  high-power field. Matches the configured value exactly.
  AAFP, *Urinalysis: A Comprehensive Review*, Am Fam Physician, 2005.
  <https://www.aafp.org/pubs/afp/issues/2005/0315/p1153.html>
- **Urinary casts, `normal_range_max = 0`.** Casts are absent or very rare in
  normal urine. Same source.
- **Leukocytes are deliberately not scored** by any condition. Pyuria indicates
  infection, not gout, GN, or nephrolithiasis. This is an intentional exclusion,
  not an oversight.

#### Glomerulonephritis — why "supportive only" is the correct framing

RBC and granular casts carry roughly **97% specificity but low sensitivity** for
glomerular disease, appearing in only ~2.7% of biopsy-proven cases, and are not
fully specific -- RBC casts are reported after exercise and in interstitial
nephritis. High specificity means a positive finding is meaningful; low
sensitivity means LOW or MODERATE rules nothing out. That is exactly the framing
already encoded in the reduced weights and raised thresholds.
Kudose et al., *Glomerular Hematuria and the Utility of Urine Microscopy*,
Am J Kidney Dis, 2022.
<https://www.ajkd.org/article/S0272-6386(22)00584-4/fulltext>

#### Nephrolithiasis — why the erythrocyte weight stays low

Hematuria is **~77% sensitive** for kidney stones overall, ranging 55-86% by stone
location, and **up to 15% of confirmed stone patients show no hematuria at all**.
This is a measured diagnostic-accuracy figure supporting `weight = 0.1`.
StatPearls, *Renal Calculi, Nephrolithiasis*.
<https://www.ncbi.nlm.nih.gov/books/NBK442014/>

Crystal morphology can permit presumptive stone-type identification, but
crystalluria without sub-typing is nonspecific -- which is the "crystal sub-typing
not available" limitation in Section 7, now with a citation behind it.
StatPearls, *Urinary Crystals Identification and Analysis*, 2023.
<https://www.ncbi.nlm.nih.gov/books/NBK606103/>

#### Gout

Uric acid crystalluria marks hyperuricosuria and urinary supersaturation and
correlates with, but does not prove, gout. Definitive diagnosis still requires
joint fluid crystal analysis during an attack. Gout is the most clinically
grounded of the three conditions here while still not being standalone-diagnostic.
StatPearls, *Hyperuricemia*. <https://www.ncbi.nlm.nih.gov/books/NBK459218/>

#### Comparable systems, for benchmarking

| System | Classes | Reported | Source |
|---|---|---|---|
| YOLOv8 urine sediment detector | 11 | mAP 91% | Springer, 2023 |
| YOLOv5 + GA hyperparameter search | 6 | mAP 85.8% (YOLOv5l) | ScienceDirect, 2023 |
| Multi-head YOLOv12, self-supervised pretraining | 6 groups | -- | Sci Rep, 2025 |

UroLens detects 10 classes, closest to the 11-class YOLOv8 study. **Any comparison
against that 91% figure must use a leak-free measurement** -- see
`docs/dataset_rebuild.md`. The pre-rebuild 0.843 mAP@50 is not a valid basis for
comparison; the specimen-disjoint figure is 0.746.

---

## 5. Evidence Attribution

For each condition, an evidence list is generated:
- Particles sorted by `contribution_score` descending
- Highest contributor → `contribution_role = "primary"`
- All others → `contribution_role = "supporting"`
- Particles with `detected_count < evidence_min_count` excluded

---

## 6. No Significant Indicators Case

When all particle counts are within normal ranges:
- All three condition scores are LOW
- All three evidence lists are empty
- `SmartDiagnosisOutput.no_significant_indicators = True`

This is a valid, expected output for a normal urinalysis.

---

## 7. MVP Limitations

- **Crystal sub-typing not available** — `crystals` class is a general detector. Cannot distinguish uric acid (Gout) from calcium oxalate, struvite, or phosphate (Nephrolithiasis). Post-MVP retraining planned.
- **Cast sub-typing not available** — `urinary_casts` includes all cast types. RBC casts cannot be distinguished from hyaline casts. Post-MVP retraining planned.
- **Not a diagnostic replacement** — probability indicators support clinical decision-making only. Final judgment belongs to the Laboratory Supervisor or physician.
- **Single-field analysis** — MVP analyzes one image per submission. Multi-field aggregation across high-power fields is a post-MVP feature.
- **GN and Nephrolithiasis are supportive only** — per MedTech review, these conditions require additional clinical tests or imaging for definitive diagnosis. The system intentionally produces LOW/MODERATE in most cases for these two conditions.

---

## 8. Domain Expert Review Checklist

| Item | Status |
|---|---|
| Particle-condition associations are clinically appropriate | ✓ Confirmed |
| Normal range max values are consistent with reference ranges | ✓ Confirmed |
| Particle weights reflect relative clinical significance | ✓ Confirmed with adjustments |
| Probability level thresholds are appropriate for MVP | ✓ Confirmed with adjustments |
| Evidence min count values are appropriate | ✓ Confirmed |
| Crystal sub-typing limitation is acceptable for MVP | ✓ Confirmed |
| Cast sub-typing limitation is acceptable for MVP | ✓ Confirmed |
| Not a diagnostic replacement framing is appropriate | ✓ Confirmed |
| GN weights and thresholds reduced — supportive indicator only | ✓ Recommended by MedTech |
| Nephrolithiasis weights and thresholds reduced — supportive indicator only | ✓ Recommended by MedTech |

---

## 9. Domain Expert Sign-off

| Field | Value |
|---|---|
| Reviewer Name | Wendell Jeffrey G. Bayron |
| Credentials | Medical Technologist |
| Review Date | May 25, 2026 |
| Recommended Changes | Reduced weights and raised thresholds for Glomerulonephritis and Nephrolithiasis — these conditions cannot be reliably indicated from urine sediment alone. Gout thresholds and weights approved as-is. |
| Sign-off Statement | Confirmed that the thresholds, weights, and particle-condition associations are clinically appropriate for a urinalysis decision-support context at the MVP stage, with the noted adjustments to GN and Nephrolithiasis. |
| Reference | In-person consultation with Sir Wendell, May 25, 2026, MedTech Faculty Office |