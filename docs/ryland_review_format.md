# Expert review format — what to return

Thanks for reviewing the 855 GOLDEN-tier derm predictions. Here is the
simplest format you can return so ingestion is instant on our side.

## Option A: edit the CSV we send you (preferred)

You will receive a CSV with one row per prediction and pre-filled columns
`prediction_id`, `drug`, `disease`, `source_tier`, `mechanism_notes`. Add
your assessment in two columns we leave empty:

| column | what to put |
|---|---|
| `verdict` | one of `known`, `plausible`, `implausible`, `adverse`, `unsure` |
| `reasoning` | one short sentence — mechanism, contraindication concern, or why you're unsure |

Optional:

| column | what to put |
|---|---|
| `reviewer_confidence` | 1–5 (1 = low, 5 = high); blank fine |

Leave `prediction_id`, `drug`, `disease`, `provenance` exactly as we sent
them. We use `prediction_id` to join back to our rows.

## Verdict meanings

- **`known`** — this is already standard-of-care or a recognised off-label
  use. We will exclude these from novel-precision calculations but count
  them toward recall.
- **`plausible`** — novel indication, biologically reasonable mechanism.
  This is the "hit" bucket for us.
- **`implausible`** — you don't see a mechanism and nothing in the
  literature supports it. This is the "miss" bucket.
- **`adverse`** — the drug is known to cause or worsen this disease.
  These are safety signals we need to ingest into our inverse-indication
  filter. Especially valuable.
- **`unsure`** — genuinely uncertain or outside your specialty. Excluded
  from headline numbers but recorded.

## Skip as many as you like

Any row you leave without a `verdict` is silently dropped from analysis.
If a prediction is outside dermatology or you don't have an opinion,
leaving it blank is the right answer.

## What we do with the result

1. `scripts/import_ryland_review.py` validates and ingests your file into
   `data/reference/expert_labels_ryland.json`. Your labels are never
   mixed into any training data, only used for independent evaluation.
2. `scripts/h907_eval_expert_labels.py` computes parallel precision
   columns on the deliverable — tier-by-tier, with and without your
   expert labels included as ground truth.
3. `adverse` verdicts feed the inverse-indication filter audit (h918).
4. Anything that changes tier calibration will be validated against
   holdout before shipping.

## Format fallback

If editing the CSV is inconvenient, you can return:

- An **XLSX** with the same columns, or
- A **JSON** array following `data/reference/ryland_review_schema.json`, or
- A free-form email listing `drug → disease: verdict | one-sentence reason`
  and we will format it for you. Not your time to spend on formatting.

## Questions before you start

- If you want a confidence scale beyond 1–5, tell us and we will widen it.
- If more than 5% of rows need an "outside my specialty" option beyond
  `unsure`, tell us and we will add a category.
- If the derm slice has important omissions (diseases you expected to
  see), flag them — that is a coverage signal we need.
