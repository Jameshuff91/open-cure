# h1272 — Per-disease audit of SUBSET_D_GLOBAL fusion lift

**Aggregate:** mean Δ per-dis AUPRC = +0.00470  std = 0.05295  n = 1002  (455+ / 339- / 208=0)

## Per-n_gt bucket

| n_gt bucket | n_rows | mean Δ AUPRC | std | frac+ |
|---|---|---|---|---|
| 1 | 58 | -0.00431 | 0.16025 | 3.4% |
| 2-5 | 193 | +0.00855 | 0.05674 | 26.4% |
| 6-20 | 387 | +0.00578 | 0.03836 | 46.3% |
| 21-50 | 186 | +0.00067 | 0.02154 | 55.9% |
| 51+ | 178 | +0.00532 | 0.01334 | 66.9% |

## Per-category (sorted by mean Δ)

| Category | n_rows | mean Δ AUPRC | std | frac+ |
|---|---|---|---|---|
| `endocrine` | 6 | +0.02471 | 0.03327 | 66.7% |
| `dermatological` | 48 | +0.01562 | 0.03395 | 66.7% |
| `neurological` | 50 | +0.01318 | 0.04334 | 52.0% |
| `autoimmune` | 31 | +0.01259 | 0.02805 | 74.2% |
| `infectious` | 149 | +0.01134 | 0.08285 | 49.0% |
| `respiratory` | 21 | +0.00973 | 0.01596 | 52.4% |
| `ophthalmic` | 22 | +0.00940 | 0.02229 | 50.0% |
| `hematological` | 17 | +0.00725 | 0.04948 | 35.3% |
| `cardiovascular` | 63 | +0.00671 | 0.01604 | 60.3% |
| `metabolic` | 54 | +0.00405 | 0.02037 | 29.6% |
| `immunological` | 7 | +0.00347 | 0.00567 | 57.1% |
| `psychiatric` | 18 | +0.00115 | 0.00414 | 50.0% |
| `renal` | 23 | +0.00068 | 0.01372 | 56.5% |
| `gastrointestinal` | 38 | +0.00016 | 0.02305 | 34.2% |
| `other` | 320 | +0.00014 | 0.06560 | 36.2% |
| `musculoskeletal` | 18 | -0.00050 | 0.01319 | 33.3% |
| `cancer` | 116 | -0.00172 | 0.02445 | 46.6% |
| `reproductive` | 1 | -0.00794 | 0.00000 | 0.0% |

## Per-ATC-L3-entropy quartile

| Quartile | n_rows | mean Δ AUPRC | std | frac+ |
|---|---|---|---|---|
| Q1_low_ent | 249 | +0.00034 | 0.08507 | 19.7% |
| Q2 | 241 | +0.00912 | 0.05679 | 43.6% |
| Q3 | 261 | +0.00444 | 0.02497 | 51.3% |
| Q4_high_ent | 251 | +0.00504 | 0.01522 | 66.5% |

## Top 20 best diseases

| Mean Δ | n_seeds | n_gt | ATC ent | Category | Name |
|---|---|---|---|---|---|
| +0.7500 | 1 | 1 | -0.00 | `infectious` | schistosoma japonicum infection |
| +0.2699 | 3 | 2 | 1.00 | `infectious` | legionella infection |
| +0.1984 | 1 | 8 | 0.65 | `hematological` | pernicious anemia |
| +0.1972 | 1 | 12 | 0.59 | `other` | endometrial hyperplasia |
| +0.1738 | 1 | 11 | -0.00 | `other` | primary dysbetalipoproteinemia fredrickson type iii |
| +0.1671 | 3 | 6 | 1.52 | `neurological` | tuberous sclerosis complex |
| +0.1348 | 3 | 6 | 1.50 | `other` | fibrocystic breast disease |
| +0.1250 | 2 | 2 | -0.00 | `other` | septic abortion |
| +0.1250 | 1 | 2 | -0.00 | `cancer` | vulvar cancer |
| +0.1204 | 1 | 6 | 1.50 | `other` | blind loop syndrome |
| +0.1174 | 1 | 67 | 3.33 | `autoimmune` | nonradiographic axial spondyloarthritis |
| +0.1075 | 3 | 7 | 1.92 | `dermatological` | folliculitis |
| +0.1000 | 1 | 5 | 2.32 | `other` | sjogrens syndrome |
| +0.0902 | 1 | 10 | 1.46 | `infectious` | infections caused by acinetobacter |
| +0.0883 | 2 | 6 | 2.32 | `dermatological` | keratosis pilaris |
| +0.0839 | 1 | 16 | 2.27 | `other` | amenorrhea |
| +0.0829 | 2 | 2 | 1.00 | `metabolic` | aldosteroneproducing adrenal adenomas |
| +0.0807 | 1 | 14 | 3.03 | `endocrine` | pituitary hypothalamic injury from trauma |
| +0.0768 | 2 | 10 | 2.06 | `infectious` | sporotrichosis |
| +0.0698 | 1 | 35 | 3.02 | `autoimmune` | lupus nephritis |

## Bottom 20 worst diseases

| Mean Δ | n_seeds | n_gt | ATC ent | Category | Name |
|---|---|---|---|---|---|
| -0.9630 | 1 | 1 | -0.00 | `other` | beriberi |
| -0.2222 | 1 | 3 | 1.00 | `infectious` | aspergillus species infections |
| -0.1306 | 1 | 7 | 1.37 | `other` | head lice infestations |
| -0.1071 | 1 | 4 | -0.00 | `other` | vitamin b 12 deficiency |
| -0.1059 | 2 | 6 | 1.50 | `other` | childhood enuresis |
| -0.1053 | 1 | 5 | 1.00 | `other` | xlinked hypophosphataemia |
| -0.0838 | 1 | 8 | 1.50 | `other` | ventriculitis |
| -0.0775 | 1 | 35 | 2.25 | `infectious` | gonococcal pharyngitis |
| -0.0750 | 2 | 2 | 0.00 | `other` | idiopathic steatorrhea |
| -0.0736 | 1 | 13 | 0.87 | `cancer` | relapsed or refractory diffuse large bcell lymphoma |
| -0.0668 | 2 | 4 | 0.92 | `infectious` | fusariosis |
| -0.0625 | 1 | 2 | 1.00 | `cancer` | gestational trophoblastic neoplasia |
| -0.0575 | 2 | 36 | 3.22 | `gastrointestinal` | reflux esophagitis |
| -0.0551 | 1 | 19 | 2.85 | `infectious` | visceral leishmaniasis caused by leishmania donovani |
| -0.0510 | 1 | 7 | 1.37 | `hematological` | paroxysmal nocturnal hemoglobinuria pnh |
| -0.0508 | 1 | 38 | 0.88 | `cancer` | advanced or recurrent cervical cancer |
| -0.0492 | 1 | 33 | 2.00 | `cardiovascular` | ocular hypertension |
| -0.0490 | 2 | 31 | 2.66 | `infectious` | esophageal candidiasis |
| -0.0469 | 1 | 11 | 2.85 | `other` | recurrent pericarditis |
| -0.0466 | 1 | 6 | 1.25 | `gastrointestinal` | amebic liver abscess |
