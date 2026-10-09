# urolens-ai-engine — Public API

The package exports exactly two functions. Everything else (preprocessing, the
YOLO engine, rule files, scoring, evidence) is an implementation detail and must
not be imported by the backend.

```python
from urolens_ai import infer, generate_smart_diagnosis
```

Importing anything deeper than `urolens_ai` couples the backend to internals that
are expected to change — the detector is being retrained and the rule thresholds
recalibrated. These two signatures are the contract; nothing behind them is.

---

## 1. `infer(image_bytes) -> InferenceResult`

Runs particle detection on one urine sediment microscopy image.

```python
result = infer(image_bytes)
result.particles          # {"erythrocytes": 5, "urinary-casts": 2}
result.confidence_scores  # {"erythrocytes": 0.8341, "urinary-casts": 0.7102}
```

**Parameters**

| Name | Type | Notes |
|---|---|---|
| `image_bytes` | `bytes` | Raw JPEG or PNG. Minimum 640 × 480. |

**Returns — `InferenceResult`**

| Field | Type | Meaning |
|---|---|---|
| `model_version` | `str` | From the `MODEL_VERSION` env var, else `"unknown"` |
| `particles` | `dict[str, int]` | Class name → count. Classes with zero detections are absent |
| `confidence_scores` | `dict[str, float]` | Class name → mean confidence, 4 dp |
| `raw_detection_count` | `int` | Detections before aggregation |
| `filtered_detection_count` | `int` | Detections after aggregation |
| `inference_time_ms` | `float` | Wall-clock, 2 dp |
| `image_width`, `image_height` | `int` | Source dimensions, before normalisation |

**Raises**

| Exception | `code` | Cause |
|---|---|---|
| `ImageValidationError` | `CORRUPT_IMAGE` | Bytes are not a decodable image |
| | `FORMAT_UNSUPPORTED` | Not JPEG or PNG; the detected format is in the message |
| | `RESOLUTION_TOO_LOW` | Below 640 × 480; required and actual are in the message |
| | `IMAGE_EXPOSURE` | Near-black or blown-out frame, or more than 75% black (e.g. phone held too far from the eyepiece). Show "check the light, fill the screen with the eyepiece circle, and retake" |
| | `NOT_MICROSCOPY` | Input gate says this is not a urine microscopy image (e.g. a selfie). Show "retake image" — never a result |
| `InferenceError` | `MODEL_NOT_LOADED` | Weights missing or unreadable |
| | `INFERENCE_FAILED` | Any other failure during detection |

All three carry `.code` and `.message`. Branch on `.code`, never on message text.

**Two behaviours worth knowing**

- **EXIF is stripped**, not merely ignored. Orientation is applied, then all
  metadata is discarded. This is a patient-privacy requirement: EXIF can carry GPS
  coordinates and device serial numbers.
- **Particle keys here use dashes** (`urinary-casts`), matching `labels.txt`.
  `generate_smart_diagnosis` expects **underscores** (`urinary_casts`). See §3.

---

## 2. `generate_smart_diagnosis(classification) -> SmartDiagnosisOutput`

Applies the deterministic rule engine to a **confirmed** classification. This is
not a second opinion on the image — it is arithmetic over counts a Medical
Technologist has reviewed.

```python
output = generate_smart_diagnosis({"crystals": 12, "erythrocytes": 4})
output.gout.level            # ProbabilityLevel.MODERATE
output.gout.weighted_score   # 7.0
output.gout.evidence         # [EvidenceItem(...), ...]
```

**Parameters**

| Name | Type | Notes |
|---|---|---|
| `classification` | `dict[str, int]` | Particle name → confirmed count. Unknown keys ignored; missing keys treated as 0 |

**Returns — `SmartDiagnosisOutput`**

| Field | Type |
|---|---|
| `gout`, `glomerulonephritis`, `nephrolithiasis` | `ConditionScore` |
| `no_significant_indicators` | `bool` |
| `engine_version` | `str` |

Each `ConditionScore` carries `condition`, `level` (`LOW` / `MODERATE` / `HIGH`),
`weighted_score`, and an `evidence` list attributing the score to particles.

**Raises**

| Exception | `code` | Cause |
|---|---|---|
| `RuleEngineError` | `INVALID_CLASSIFICATION` | Not a dict, or a value is negative or not an `int` |
| | `CONFIG_ERROR` | `config.yaml` missing or malformed |
| | `RULE_EVALUATION_FAILED` | Failure inside a rule |

**Scoring**, per particle:

```
contribution = max(0, detected_count - normal_range_max) * weight
```

Summed per condition, then mapped: `<= low_max_score` → LOW,
`>= high_min_score` → HIGH, otherwise MODERATE. Boundaries are inclusive on both
ends. Weights and thresholds live in
`src/urolens_ai/smart_diagnosis/config.yaml`, which is the single source of truth.

> **Interpretation limit.** These are screening indicators, not diagnoses.
> Glomerulonephritis and nephrolithiasis in particular cannot be established from
> urine sediment alone — they need biopsy/serology and imaging respectively. Under
> the current thresholds the HIGH band is close to unreachable in practice: across
> 2,247 held-out fields, HIGH occurred 0 times for gout, once for
> glomerulonephritis, and 0 times for nephrolithiasis. Recalibration is pending
> MedTech re-review; see `docs/rule_engine_design.md` §4.3.1. Do not build UI that
> depends on HIGH appearing.

---

## 3. Particle key formats

The two functions use different separators, and the boundary between them is a
common source of silent zeros.

| Context | Format | Example |
|---|---|---|
| `labels.txt`, `infer().particles` | dash | `urinary-casts` |
| `generate_smart_diagnosis()` input, `config.yaml` | underscore | `urinary_casts` |

`map_detections()` performs `.replace("-", "_")` on the live path. If you pass
`infer()` output straight into `generate_smart_diagnosis()`, normalise first —
unrecognised keys are **ignored silently**, so a mismatch does not raise, it just
scores zero:

```python
counts = {k.replace("-", "_"): v for k, v in result.particles.items()}
diagnosis = generate_smart_diagnosis(counts)
```

The ten classes: `bacteria`, `crystals`, `epithelial_cells`, `erythrocytes`,
`leukocytes`, `mucus_threads`, `sperm_cells`, `trichomonas_vaginalis`,
`urinary_casts`, `yeast`.

---

## 4. Configuration

Read from the environment at import time. Changing them afterwards has no effect
until the process restarts (or `reset_engine()` is called).

| Variable | Default | Notes |
|---|---|---|
| `MODEL_WEIGHTS_PATH` | `src/urolens_ai/models/yolov8/best.pt` | |
| `MODEL_VERSION` | `unknown` | Surfaced in `InferenceResult` |
| `INFERENCE_CONF_THRESHOLD` | `0.35` | See note below |
| `CLASS_THRESHOLDS_PATH` | `src/urolens_ai/models/yolov8/thresholds.yaml` | Per-class cut-offs; classes not listed use `INFERENCE_CONF_THRESHOLD`. See note below |
| `INFERENCE_IOU_THRESHOLD` | `0.5` | NMS IoU |
| `MAX_DETECTIONS` | `1000` | Most boxes per image. The Ultralytics default of 300 capped dense fields (~500 particles) at exactly 300 |
| `RULE_ENGINE_CONFIG_PATH` | `src/urolens_ai/smart_diagnosis/config.yaml` | |
| `MIN_IMAGE_WIDTH` / `MIN_IMAGE_HEIGHT` | `640` / `480` | |
| `ACCEPTED_IMAGE_FORMATS` | `JPEG,PNG` | |
| `GATE_WEIGHTS_PATH` | `src/urolens_ai/models/gate/gate.pt` | Slide / not-slide classifier |
| `GATE_MIN_SLIDE_PROB` | `0.43` | Highest cut-off that passes ≥99.9% of real test slides (`scripts/evaluate_gate.py`) |

> **Confidence threshold — set this explicitly.** The code default and
> `.env.example` both use `0.35`, chosen from measured count bias on the
> rule-engine classes (`scripts/evaluate.py --sweep-conf`): it roughly halves
> the bias of the earlier `0.25`. The threshold directly scales every particle
> count and therefore every probability level: a stricter threshold suppresses
> detections and pushes scores toward LOW. Re-run the sweep whenever the model
> weights change, and set the value explicitly rather than relying on the default.

> **Per-class thresholds.** `thresholds.yaml` overrides the global threshold per
> class. It was chosen on the valid split with `scripts/evaluate.py
> --sweep-per-class`, which only moves a class if its count error improves by a
> minimum margin. Measured once on test through `infer()`: mean count error fell
> from 0.204 to 0.181 with identical diagnosis agreement, and the rule-engine
> classes (crystals, erythrocytes, urinary casts) stayed at `0.35`. If the file
> is missing, every class uses `INFERENCE_CONF_THRESHOLD`. The values are tied
> to the weights: regenerate the file with `--sweep-per-class --split valid
> --thresholds-out <path>` whenever the model changes.

---

## 5. Typical backend flow

```python
from urolens_ai import infer, generate_smart_diagnosis
from urolens_ai.utils.exceptions import ImageValidationError, InferenceError

try:
    result = infer(image_bytes)
except ImageValidationError as exc:
    return http_400(code=exc.code, message=exc.message)
except InferenceError as exc:
    return http_500(code=exc.code, message=exc.message)

# MedTech reviews and confirms result.particles here. The rule engine runs on the
# confirmed counts, not the raw ones -- that human step is part of the design.
confirmed = {k.replace("-", "_"): v for k, v in reviewed_counts.items()}
diagnosis = generate_smart_diagnosis(confirmed)
```

The first `infer()` call loads the model and is slow; the engine is a process-wide
singleton, so warm it at startup rather than on the first user request.
