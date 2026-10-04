# Literature Grounding for the Smart Diagnosis Rule Engine — DRAFT

**Status:** SUPERSEDED — merged into `docs/rule_engine_design.md` Section 4.5 on 2026-09-05.
Kept for the fuller reasoning and the follow-up actions in Section 6; the citations
themselves now live in the design doc. Section 2 of this draft is resolved: the
stale Section 4.3 table has been corrected.
**Purpose:** Ground the current rule-engine weights/thresholds in published clinical literature
so the conference submission has citations behind the numbers instead of only an informal
single-reviewer sign-off.

---

## 1. Why this matters for the conference submission

Right now every weight and threshold in `config.yaml` is justified only by an in-person
consultation with one Medical Technologist (see `rule_engine_design.md` Section 9). A reviewer
will likely ask "where do these numbers come from?" This document collects published sources
that back up (or should adjust) the current values, so that justification can move from
"one reviewer said so" to "consistent with published reference ranges and diagnostic-accuracy
literature, refined by MedTech review."

This is **literature grounding, not statistical validation** — it does not require a labeled
patient dataset. If you later get access to confirmed-diagnosis cases, a proper
sensitivity/specificity/AUC evaluation would be the stronger follow-up (see Section 5).

---

## 2. First fix needed: the design doc is out of sync with config.yaml

Independent of the literature work, `docs/rule_engine_design.md` Section 4.3 currently has
**stale and partly swapped values**. This should be fixed regardless of anything else below —
if this doc is used to write a paper appendix, it would misrepresent the actual system.

| Condition | Doc currently says (low_max / high_min) | Actual `config.yaml` value | 
|---|---|---|
| Gout | 3.0 / 10.0 | **2.0 / 50.0** |
| Glomerulonephritis | 2.0 / 10.0 | **3.0 / 35.0** |
| Nephrolithiasis | 3.0 / 15.0 | **3.0 / 35.0** |

Note Gout and GN's `low_max` are swapped in the doc versus the real config, and all three
`high_min` values predate the `2926b49` commit that raised them per MedTech review.

**Proposed fix:** replace the Section 4.3 table with the corrected values, and add a note that
`config.yaml` is the single source of truth going forward.

---

## 3. Literature findings, mapped to your current config values

### 3.1 Reference ranges (validates existing normal_range_max values)

- **Normal RBC in urine: 0–3 per high-power field (HPF).**
  Your `erythrocytes.normal_range_max = 3` (used identically in both Glomerulonephritis and
  Nephrolithiasis) matches this clinical reference range exactly.
  Source: AAFP, *Urinalysis: A Comprehensive Review*, Am Fam Physician, 2005.
  https://www.aafp.org/pubs/afp/issues/2005/0315/p1153.html

- **Normal WBC in urine: <2/HPF (men), <5/HPF (women), generally cited as 2–5/HPF.**
  Leukocytes are not currently scored by any of the three conditions — that's appropriate,
  since pyuria/leukocyturia is a marker for infection, not gout/GN/nephrolithiasis. Worth
  stating explicitly in the design doc as a deliberate exclusion rather than an oversight.
  Source: same AAFP review; also Medscape *Urinalysis: Reference Range*.
  https://emedicine.medscape.com/article/2074001-overview

- **Casts should be absent or very rare in normal urine.**
  Your `urinary_casts.normal_range_max = 0` (Glomerulonephritis) matches this directly.

### 3.2 Glomerulonephritis — why raising the weight/threshold is literature-backed, not just cautious

- **RBC/granular casts have ~97% specificity but low sensitivity for glomerular disease**, and
  appear in only ~2.7% of biopsy-proven cases across diverse kidney disease entities (pauci-immune
  GN, IgA nephropathy, membranous nephropathy, lupus nephritis). They are also not fully specific —
  RBC casts have been reported after exercise and in interstitial nephritis.
  This is a strong citation for *why* GN should be treated as "supportive only": high specificity
  means a positive finding is meaningful, but low sensitivity means a LOW/MODERATE score does not
  rule anything out — exactly the framing already in your `config.yaml` comments, now with a source.
  Source: Kudose et al., *Glomerular Hematuria and the Utility of Urine Microscopy: A Review*,
  Am J Kidney Dis, 2022. https://www.ajkd.org/article/S0272-6386(22)00584-4/fulltext
  (PubMed: https://pubmed.ncbi.nlm.nih.gov/35777984/)

- RBC casts on urine microscopy are also used as a component of the 2012 SLICC lupus nephritis
  classification criteria — evidence that casts are a recognized, if narrow, diagnostic signal
  in nephrology more broadly, not something invented for this project.

### 3.3 Nephrolithiasis — why the erythrocyte weight should stay low

- **Hematuria sensitivity for kidney stones is ~77% overall, but ranges 55–86% depending on
  stone location** (ureteral vs. renal-only), **and up to 15% of confirmed stone patients show
  no hematuria at all.**
  This directly supports keeping `erythrocytes.weight = 0.1` low and Nephrolithiasis
  "supportive only" — a real diagnostic-accuracy number, not a guess.
  Source: StatPearls, *Renal Calculi, Nephrolithiasis*, NCBI Bookshelf.
  https://www.ncbi.nlm.nih.gov/books/NBK442014/
  Also: *Hematuria: Is it useful in predicting renal or ureteral stones...*, PMC, 2023.
  https://pmc.ncbi.nlm.nih.gov/articles/PMC10896326/

- **Crystal morphology can permit presumptive stone-type identification**, but general
  crystalluria (without sub-typing) is nonspecific. This is exactly the "Crystal sub-typing not
  available" limitation already listed in `rule_engine_design.md` Section 7 — now with a citation
  instead of being an unsupported internal note.
  Source: StatPearls, *Urinary Crystals Identification and Analysis*, NCBI Bookshelf, 2023.
  https://www.ncbi.nlm.nih.gov/books/NBK606103/

### 3.4 Gout — supporting the crystals weight/threshold

- Uric acid crystalluria is a recognized marker of hyperuricosuria and urinary supersaturation,
  and correlates with — but is not diagnostic proof of — hyperuricemia/gout. Definitive gout
  diagnosis still requires joint fluid crystal analysis during an attack; urine crystalluria is
  supportive evidence only. This matches the existing framing of Gout as the *most* clinically
  grounded of the three conditions (per MedTech review), while still correctly not treating it
  as standalone-diagnostic.
  Source: StatPearls, *Hyperuricemia*, NCBI Bookshelf.
  https://www.ncbi.nlm.nih.gov/books/NBK459218/

---

## 4. Related work for the conference paper (comparable YOLO detection systems)

These are directly comparable prior systems — useful for the paper's related-work section and
as a benchmark for your own model's mAP:

| System | Classes | Reported performance | Source |
|---|---|---|---|
| YOLOv8 urine sediment detector | 11 particle classes | mAP 91% | Springer, 2023. https://link.springer.com/chapter/10.1007/978-3-031-37129-5_22 |
| YOLOv5s-CBL ("Efficient Particle YOLO Detector") | Urine sediment particles | — | Springer, 2022. https://link.springer.com/chapter/10.1007/978-3-031-20102-8_23 |
| YOLOv5 + genetic-algorithm hyperparameter search | 6 particle classes | YOLOv5l: mAP 85.8% | ScienceDirect, 2023. https://www.sciencedirect.com/science/article/abs/pii/S0010482523013604 |
| Multi-head YOLOv12 with self-supervised pretraining | Cells / Casts / Crystals / Microorganisms & Yeast / Artifact / Other | — | *Scientific Reports*, 2025. https://www.nature.com/articles/s41598-025-25339-z |

Your own model detects 10 classes (`labels.txt`), closely comparable to the 11-class YOLOv8 study
above — worth reporting your own mAP against that 91% figure directly in the paper if you have it,
or running an evaluation pass to get it if you don't yet.

---

## 5. Cheap next step if you *do* get access to any labeled data

Even a small retrospective set of cases with a MedTech- or physician-confirmed diagnosis would let
us compute real sensitivity/specificity/AUC for the current rule thresholds against ground truth —
this is a much stronger conference result than literature grounding alone, but wasn't pursued now
since no such dataset is confirmed to exist yet. Flag it if that changes.

---

## 6. Proposed follow-up actions (not yet applied — for your review)

1. Fix `docs/rule_engine_design.md` Section 4.3 table (values above) — corrects an internal
   inconsistency independent of literature work.
2. Add a "Literature Basis" section to `rule_engine_design.md` (Sections 3–4) with the citations
   above, replacing "MedTech review" as the *sole* justification with "MedTech review, consistent
   with published diagnostic-accuracy literature [citations]".
3. Add a short "Related Work" section citing the comparable YOLO papers, and a note to benchmark
   your own model's mAP against the 91% figure once you have eval numbers.
4. Optionally: add one-line citation references inline in `config.yaml`'s existing rationale
   comments, pointing at this document, so the config stays self-explanatory.

None of this has been written into `rule_engine_design.md` or `config.yaml` yet — this file is
staged for your review first.
