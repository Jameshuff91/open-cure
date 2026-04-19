## Attrition Analysis: Preprint vs Post-h961/h952

Every Cure indicationList: 3,996 unique disease names (preprint baseline).

| Stage | Preprint | Post-Fix | Δ |
|---|---:|---:|---:|
| Raw MeSH mappings | 1,453 | 1,453 | - |
| + h961 aliases | - | 1,533 | +80 |
| DRKG-reachable | - | 1,487 | - |
| Evaluable (no-tx) | **368** | **1,279** | **+911** |
| Attrition | 90.8% | 68.0% | -22.8pp |

Drivers of the improvement:
- **h952** (find_disease_id reverse-index fallback): recovered name-resolution cases where the disease_names string was not present in the raw mesh_mappings keys (British spellings, possessive stripping, hyphenation variants).
- **h961** (principled alias generator): added algorithmic aliases at load time (US/UK spelling, hyphen/possessive variants, 114 British + 668 disease_names backfill).