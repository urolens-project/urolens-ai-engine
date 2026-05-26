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

| Condition | low_max | high_min | Normal Range Max | Evidence Min Count |
|---|---|---|---|---|
| Gout | 3.0 | 10.0 | crystals: 5 | 2 |
| Glomerulonephritis | 2.0 | 10.0 | urinary_casts: 0, erythrocytes: 3 | 1 |
| Nephrolithiasis | 3.0 | 15.0 | crystals: 5, erythrocytes: 3 | 2 |

### 4.4 Threshold Rationale

| Condition | Rationale |
|---|---|
| Gout | Standard thresholds retained — crystal elevation is a valid urinalysis indicator per MedTech review |
| Glomerulonephritis | Thresholds raised significantly per MedTech review — GN cannot be reliably diagnosed from urinalysis alone. HIGH is very difficult to reach by design. |
| Nephrolithiasis | Thresholds raised significantly per MedTech review — nephrolithiasis requires imaging for definitive diagnosis. System produces LOW in most cases. |

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