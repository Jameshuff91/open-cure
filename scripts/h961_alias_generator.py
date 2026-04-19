#!/usr/bin/env python3
"""h961: MeSH disease-name aliasing.

Two-layer resolution strategy:

1. Principled alias generator — apply US/UK spelling, hyphenation, possessive,
   plural, and missing-space transformations to produce candidate names and
   check against mesh_mappings.
2. disease_names backfill — every canonical disease_name has an id in
   disease_names; if aliasing fails, fall through to the reverse-index and
   emit name → id directly.

The first layer captures user-typed spelling variants (British / hyphenation)
so they resolve without relying on the runtime fallback from h952. The second
layer guarantees 100% coverage of the 668 disease_names names missing from
mesh_mappings.

Usage:
    python3 scripts/h961_alias_generator.py           # coverage report
    python3 scripts/h961_alias_generator.py --write   # emit aliases JSON
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from production_predictor import DrugRepurposingPredictor  # noqa: E402

# ---------------------------------------------------------------------------
# Principled transformations
# ---------------------------------------------------------------------------

# Ordered, regex-anchored British → American substitutions. Each pattern is
# anchored so it does not fire inside unrelated words (e.g. "oesophag" must
# appear after a word boundary, not after the "o" in "gastro").
BRIT_TO_US_REGEX: List[Tuple[re.Pattern[str], str]] = [
    # Word-start / boundary-sensitive rules
    (re.compile(r"\boesophag"), "esophag"),
    (re.compile(r"\boedem"), "edem"),
    (re.compile(r"\boestrog"), "estrog"),
    (re.compile(r"\bpaediatr"), "pediatr"),
    (re.compile(r"\bpaediat"), "pediat"),
    (re.compile(r"\bfoetal\b"), "fetal"),
    (re.compile(r"\bfoetu"), "fetu"),
    (re.compile(r"\baetiol"), "etiol"),
    (re.compile(r"\baesth"), "esth"),
    (re.compile(r"\bcoeli"), "celi"),
    (re.compile(r"\bcaesar"), "cesar"),
    # Diphthong-bound rules (safe mid-word because the sequence is medical)
    (re.compile(r"haemorrhag"), "hemorrhag"),
    (re.compile(r"haemorrhoi"), "hemorrhoi"),
    (re.compile(r"haemolyt"), "hemolyt"),
    (re.compile(r"haemophi"), "hemophi"),
    (re.compile(r"haemato"), "hemato"),
    (re.compile(r"haemog"), "hemog"),
    (re.compile(r"haemoly"), "hemoly"),
    (re.compile(r"haemos"), "hemos"),
    (re.compile(r"haemor"), "hemor"),
    (re.compile(r"haem"), "hem"),
    (re.compile(r"leuka?emi"), "leukemi"),
    (re.compile(r"leucocyt"), "leukocyt"),
    (re.compile(r"gynaec"), "gynec"),
    (re.compile(r"anaesthe"), "anesthe"),
    (re.compile(r"ischaemi"), "ischemi"),
    (re.compile(r"diarrhoea"), "diarrhea"),
    (re.compile(r"apnoea"), "apnea"),
    (re.compile(r"orrhoea"), "orrhea"),
    (re.compile(r"dyspnoea"), "dyspnea"),
    (re.compile(r"orrhage"), "orrhage"),
    # -aemia / -aemic noun endings
    (re.compile(r"aemia\b"), "emia"),
    (re.compile(r"aemic\b"), "emic"),
    (re.compile(r"caemi"), "cemi"),
    (re.compile(r"kalaem"), "kalem"),
    (re.compile(r"acidaem"), "acidem"),
    (re.compile(r"alkalaem"), "alkalem"),
    (re.compile(r"uricaem"), "uricem"),
    (re.compile(r"lipidaem"), "lipidem"),
    (re.compile(r"lipaem"), "lipem"),
    (re.compile(r"magnesaem"), "magnesem"),
    (re.compile(r"phosphataem"), "phosphatem"),
    (re.compile(r"ammonaem"), "ammonem"),
    (re.compile(r"calcaem"), "calcem"),
    (re.compile(r"glycaem"), "glycem"),
    (re.compile(r"uremi"), "uremi"),  # noop but keeps rule table readable
    # chemical / other
    (re.compile(r"sulphur"), "sulfur"),
    (re.compile(r"sulphate"), "sulfate"),
    (re.compile(r"sulphi"), "sulfi"),
    (re.compile(r"tumour"), "tumor"),
]

# Prefix compounds that should have a hyphen inserted (non-X-linked style).
HYPHEN_PREFIXES: List[str] = [
    "non",
    "post",
    "pre",
    "anti",
    "pro",
    "self",
    "over",
    "sub",
    "supra",
    "extra",
    "intra",
    "inter",
    "hyper",
    "hypo",
    "trans",
]

# Medical eponyms that carry an 's and are commonly typed without the apostrophe.
POSSESSIVE_EPONYMS: Set[str] = {
    "still",
    "crohn",
    "alzheimer",
    "parkinson",
    "hashimoto",
    "down",
    "hodgkin",
    "behcet",
    "sjogren",
    "wegener",
    "takayasu",
    "raynaud",
    "wilson",
    "grave",
    "addison",
    "cushing",
    "ewing",
    "meniere",
    "huntington",
    "gilbert",
    "hirschsprung",
    "gaucher",
    "niemann",
    "pick",
    "bell",
    "paget",
    "fabry",
}


def apply_british_to_us(name: str) -> str:
    out = name
    for pat, repl in BRIT_TO_US_REGEX:
        out = pat.sub(repl, out)
    return out


def hyphenate_prefixes(name: str) -> List[str]:
    """Produce variants with a hyphen inserted between known prefixes and the rest."""
    variants = []
    for prefix in HYPHEN_PREFIXES:
        new_tokens = []
        changed = False
        for tok in name.split():
            if tok.startswith(prefix) and len(tok) > len(prefix) + 2:
                new_tok = f"{prefix}-{tok[len(prefix):]}"
                new_tokens.append(new_tok)
                changed = True
            else:
                new_tokens.append(tok)
        if changed:
            variants.append(" ".join(new_tokens))
    # Named single-letter / single-syllable prefix variants.
    if "xlinked" in name:
        variants.append(name.replace("xlinked", "x-linked"))
    if "bcell" in name:
        variants.append(name.replace("bcell", "b-cell"))
    if "tcell" in name:
        variants.append(name.replace("tcell", "t-cell"))
    # General tonic/clonic, tonicclonic, etc.
    if "tonicclonic" in name:
        variants.append(name.replace("tonicclonic", "tonic-clonic"))
    return variants


def apply_possessive(name: str) -> List[str]:
    """Both the apostrophe form and the bare (no-apostrophe) canonical form."""
    variants: List[str] = []
    tokens = name.split()
    for i, tok in enumerate(tokens):
        base = tok.rstrip("s")
        if base in POSSESSIVE_EPONYMS and tok.endswith("s") and tok != base:
            # "Ewings" → "Ewing's" AND "Ewing"
            variants.append(" ".join(tokens[:i] + [f"{base}'s"] + tokens[i + 1 :]))
            variants.append(" ".join(tokens[:i] + [base] + tokens[i + 1 :]))
        elif tok in POSSESSIVE_EPONYMS:
            variants.append(" ".join(tokens[:i] + [f"{tok}'s"] + tokens[i + 1 :]))
    return variants


def strip_plural_any_token(name: str) -> List[str]:
    """Strip 's' from any internal token that produces a different string."""
    tokens = name.split()
    out: List[str] = []
    for i, tok in enumerate(tokens):
        if len(tok) > 3 and not tok.endswith("ss") and tok.endswith("s"):
            new_tok = tok[:-1]
            out.append(" ".join(tokens[:i] + [new_tok] + tokens[i + 1 :]))
        if tok.endswith("ies") and len(tok) > 4:
            new_tok = tok[:-3] + "y"
            out.append(" ".join(tokens[:i] + [new_tok] + tokens[i + 1 :]))
    return out


def strip_trailing_abbrev(name: str) -> Optional[str]:
    """Drop a trailing 2-5 char alpha token that looks like an abbreviation."""
    toks = name.split()
    if len(toks) >= 3 and 2 <= len(toks[-1]) <= 5 and toks[-1].isalpha():
        return " ".join(toks[:-1])
    return None


def add_or_strip_suffix(name: str) -> List[str]:
    out = []
    if not name.endswith(" disease"):
        out.append(f"{name} disease")
    else:
        out.append(name[: -len(" disease")])
    if not name.endswith(" syndrome"):
        out.append(f"{name} syndrome")
    else:
        out.append(name[: -len(" syndrome")])
    return out


def missing_space_splits(name: str) -> List[str]:
    """For single-token compound strings, insert spaces before common stems."""
    STEMS = [
        "cell",
        "vulgaris",
        "mellitus",
        "insipidus",
        "sclerosis",
        "syndrome",
        "disease",
        "gravis",
        "induced",
        "clonic",
        "like",
        "dependent",
    ]
    out: List[str] = []
    for stem in STEMS:
        idx = name.find(stem)
        if idx > 0 and name[idx - 1] != " ":
            out.append(name[:idx] + " " + name[idx:])
    return out


def less_specific_prefix_match(name: str, valid_keys: Set[str]) -> Optional[str]:
    """If the full string isn't in mm but its trailing substring is, return that.

    Example: 'acute bronchitis' → 'bronchitis' (if acute bronchitis missing and bronchitis present).
    """
    tokens = name.split()
    for start in range(1, len(tokens)):
        candidate = " ".join(tokens[start:])
        if candidate in valid_keys:
            return candidate
    return None


def less_specific_suffix_match(name: str, valid_keys: Set[str]) -> Optional[str]:
    """If the full string isn't in mm but a leading substring is, return that."""
    tokens = name.split()
    for end in range(len(tokens) - 1, 0, -1):
        candidate = " ".join(tokens[:end])
        if candidate in valid_keys:
            return candidate
    return None


# ---------------------------------------------------------------------------
# Candidate synthesis
# ---------------------------------------------------------------------------


def generate_candidates(name: str, valid_keys: Set[str]) -> List[str]:
    """Return ordered candidate variants. Most precise transformations first."""
    cands: List[str] = []
    seen: Set[str] = set()

    def add(s: str) -> None:
        if s and s.lower() != name and s not in seen:
            seen.add(s)
            cands.append(s)

    # Layer 1 — single transformations
    brit = apply_british_to_us(name)
    if brit != name:
        add(brit)
    for v in hyphenate_prefixes(name):
        add(v)
    for v in apply_possessive(name):
        add(v)
    trail = strip_trailing_abbrev(name)
    if trail:
        add(trail)
    for v in strip_plural_any_token(name):
        add(v)
    for v in add_or_strip_suffix(name):
        add(v)
    for v in missing_space_splits(name):
        add(v)

    # Layer 2 — compositions of two transformations
    if brit != name:
        if trail:
            t2 = strip_trailing_abbrev(brit)
            if t2:
                add(t2)
        for v in add_or_strip_suffix(brit):
            add(v)
        for v in strip_plural_any_token(brit):
            add(v)
        for v in apply_possessive(brit):
            add(v)
    if trail:
        for v in strip_plural_any_token(trail):
            add(v)
        for v in add_or_strip_suffix(trail):
            add(v)
        for v in apply_possessive(trail):
            add(v)

    # Layer 3 (AUDIT ONLY — not returned as hits): less-specific-prefix / suffix
    # fallback. Rejected because it drops id precision — 43% of less-specific
    # hits map to a different id than disease_names says they should. Kept as
    # a helper for reporting via less_specific_candidates().

    return cands


def less_specific_candidates(name: str, valid_keys: Set[str]) -> List[str]:
    out = []
    lp = less_specific_prefix_match(name, valid_keys)
    if lp:
        out.append(lp)
    ls = less_specific_suffix_match(name, valid_keys)
    if ls:
        out.append(ls)
    return out


# ---------------------------------------------------------------------------
# Resolver
# ---------------------------------------------------------------------------


def resolve_missing(
    missing: Iterable[str], mesh_mappings: Dict[str, str]
) -> Tuple[Dict[str, str], Dict[str, List[str]], Dict[str, str]]:
    """Return (transformation_hits, unresolved, winning_candidate).

    winning_candidate maps name → the candidate string that resolved (or "" if none).
    """
    valid_keys = set(mesh_mappings.keys())
    hits: Dict[str, str] = {}
    winners: Dict[str, str] = {}
    unresolved: Dict[str, List[str]] = {}

    for name in missing:
        cands = generate_candidates(name, valid_keys)
        resolved_id = None
        winning = ""
        for c in cands:
            if c.lower() in valid_keys:
                resolved_id = mesh_mappings[c.lower()]
                winning = c
                break
        if resolved_id:
            hits[name] = resolved_id
            winners[name] = winning
        else:
            unresolved[name] = cands[:6]

    return hits, unresolved, winners


def attribute_rule(name: str, winner: str) -> str:
    """Best-effort attribution of which rule group produced the winning candidate."""
    if not winner:
        return "none"
    brit = apply_british_to_us(name)
    if winner == brit and brit != name:
        return "british_spelling"
    if winner in hyphenate_prefixes(name):
        return "hyphen_prefix"
    if winner in apply_possessive(name):
        return "possessive"
    if winner == strip_trailing_abbrev(name):
        return "trailing_abbrev"
    if winner in strip_plural_any_token(name):
        return "plural"
    if winner in add_or_strip_suffix(name):
        return "disease_suffix"
    if winner in missing_space_splits(name):
        return "missing_space"
    # Composition groups
    if brit != name:
        if winner in add_or_strip_suffix(brit):
            return "british+disease_suffix"
        if winner in strip_plural_any_token(brit):
            return "british+plural"
        t = strip_trailing_abbrev(brit)
        if winner == t:
            return "british+trailing_abbrev"
    t = strip_trailing_abbrev(name)
    if t and winner in strip_plural_any_token(t):
        return "trailing_abbrev+plural"
    if t and winner in add_or_strip_suffix(t):
        return "trailing_abbrev+disease_suffix"
    # less-specific fallback
    return "less_specific_fallback"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    parser.add_argument(
        "--out",
        default=str(ROOT / "data/reference/h961_disease_name_aliases.json"),
    )
    args = parser.parse_args()

    print("Loading predictor...")
    predictor = DrugRepurposingPredictor()

    mm = predictor.mesh_mappings
    rev_dn = {n.lower(): did for did, n in predictor.disease_names.items() if n}

    names_in_dn = set(rev_dn.keys())
    names_in_mm = set(mm.keys())
    missing = sorted(names_in_dn - names_in_mm)
    print(f"Missing names to resolve: {len(missing)}")

    # Layer 1 — principled transformations (no less-specific fallback).
    transform_hits, unresolved, winners = resolve_missing(missing, mm)
    tpct = len(transform_hits) / max(1, len(missing)) * 100.0
    print(f"Principled transformation hits: {len(transform_hits)}/{len(missing)} ({tpct:.1f}%)")

    # Rule attribution and precision audit vs disease_names ground truth.
    rule_counts: Dict[str, Dict[str, int]] = {}
    for name, did in transform_hits.items():
        rule = attribute_rule(name, winners.get(name, ""))
        bucket = rule_counts.setdefault(rule, {"total": 0, "correct": 0})
        bucket["total"] += 1
        if rev_dn.get(name) == did:
            bucket["correct"] += 1
    overall_correct = sum(b["correct"] for b in rule_counts.values())
    overall_total = sum(b["total"] for b in rule_counts.values())
    precision = (overall_correct / overall_total * 100.0) if overall_total else 0.0
    print(
        f"Transform-layer precision vs disease_names: "
        f"{overall_correct}/{overall_total} ({precision:.1f}%)"
    )

    # Layer 2 — disease_names backfill covers every name that has an id in
    # disease_names. This is 100% safe because disease_names is authoritative.
    backfill: Dict[str, str] = {name: rev_dn[name] for name in missing if name in rev_dn}
    bpct = len(backfill) / max(1, len(missing)) * 100.0
    print(f"disease_names backfill hits: {len(backfill)}/{len(missing)} ({bpct:.1f}%)")

    combined = {**transform_hits, **backfill}  # backfill wins on conflict (authoritative)
    cpct = len(combined) / max(1, len(missing)) * 100.0
    print(f"Combined coverage (backfill authoritative): {len(combined)}/{len(missing)} ({cpct:.1f}%)")

    # Reverse spelling-variant aliases: for each mesh_mappings key, generate
    # British / hyphen / possessive variants and register them as aliases of the
    # same id, provided the variant isn't already a mapped key.
    reverse_variants: Dict[str, str] = {}
    existing = set(mm.keys()) | set(rev_dn.keys())
    # Focused set of "en->brit" substitutions for the reverse direction.
    # Explicit stem-specific rules to avoid over-generalising (e.g. the
    # `emia` rule must NOT fire on "system+ic" → "systaemic"). Each stem
    # is anchored with a word boundary and contains the English form.
    US_TO_BRIT: List[Tuple[re.Pattern[str], str]] = [
        (re.compile(r"\bhemorrhag"), "haemorrhag"),
        (re.compile(r"\bhemophi"), "haemophi"),
        (re.compile(r"\bhemolyt"), "haemolyt"),
        (re.compile(r"\bhemato"), "haemato"),
        (re.compile(r"\bhemog"), "haemog"),
        (re.compile(r"\bhemol"), "haemol"),
        (re.compile(r"\bhemor"), "haemor"),
        (re.compile(r"\bhemos"), "haemos"),
        (re.compile(r"\bedema"), "oedema"),
        (re.compile(r"angioedema"), "angio-oedema"),
        (re.compile(r"estrog"), "oestrog"),
        (re.compile(r"\bpediatr"), "paediatr"),
        (re.compile(r"\bfetal\b"), "foetal"),
        (re.compile(r"etiol"), "aetiol"),
        (re.compile(r"ischemi"), "ischaemi"),
        (re.compile(r"anesthe"), "anaesthe"),
        (re.compile(r"diarrhea"), "diarrhoea"),
        (re.compile(r"\bapnea"), "apnoea"),
        (re.compile(r"esophag"), "oesophag"),
        (re.compile(r"\bleukemi"), "leukaemi"),
        (re.compile(r"cesarean"), "caesarean"),
        (re.compile(r"\bceliac"), "coeliac"),
        # Blood-condition stems only. Each is an explicit stem so we don't
        # sweep over English "-emic" adjectives like "systemic".
        (re.compile(r"anemia\b"), "anaemia"),
        (re.compile(r"anemic\b"), "anaemic"),
        (re.compile(r"leukemia\b"), "leukaemia"),
        (re.compile(r"thalassemia\b"), "thalassaemia"),
        (re.compile(r"hypokalemia\b"), "hypokalaemia"),
        (re.compile(r"hyperkalemia\b"), "hyperkalaemia"),
        (re.compile(r"hypocalcemia\b"), "hypocalcaemia"),
        (re.compile(r"hypercalcemia\b"), "hypercalcaemia"),
        (re.compile(r"hypoglycemia\b"), "hypoglycaemia"),
        (re.compile(r"hyperglycemia\b"), "hyperglycaemia"),
        (re.compile(r"hyperlipidemia\b"), "hyperlipidaemia"),
        (re.compile(r"hypolipidemia\b"), "hypolipidaemia"),
        (re.compile(r"hypercholesterolemia\b"), "hypercholesterolaemia"),
        (re.compile(r"acidemia\b"), "acidaemia"),
        (re.compile(r"alkalemia\b"), "alkalaemia"),
        (re.compile(r"uricemia\b"), "uricaemia"),
        (re.compile(r"uremia\b"), "uraemia"),
        (re.compile(r"bacteremia\b"), "bacteraemia"),
        (re.compile(r"fungemia\b"), "fungaemia"),
        (re.compile(r"viremia\b"), "viraemia"),
        (re.compile(r"septicemia\b"), "septicaemia"),
        (re.compile(r"hyperammonemia\b"), "hyperammonaemia"),
        (re.compile(r"hypermagnesemia\b"), "hypermagnesaemia"),
        (re.compile(r"hypomagnesemia\b"), "hypomagnesaemia"),
        (re.compile(r"hyperphosphatemia\b"), "hyperphosphataemia"),
        (re.compile(r"hypophosphatemia\b"), "hypophosphataemia"),
    ]
    for k, did in mm.items():
        brit = k
        for pat, repl in US_TO_BRIT:
            brit = pat.sub(repl, brit)
        if brit != k and brit not in existing and brit not in reverse_variants:
            reverse_variants[brit] = did
    print(f"Reverse British-variant aliases from mesh_mappings: {len(reverse_variants)}")

    still_unresolved = [m for m in missing if m not in combined]
    print(f"Still unresolved: {len(still_unresolved)}")

    print("\n--- Rule attribution & precision (transformation layer only, excl. less-specific) ---")
    for rule, b in sorted(rule_counts.items(), key=lambda x: -x[1]["total"]):
        pct = (b["correct"] / b["total"] * 100.0) if b["total"] else 0
        print(f"  {rule}: total={b['total']}, correct={b['correct']} ({pct:.0f}%)")

    print("\n--- Sample of 10 safe transformation hits (matches disease_names id) ---")
    shown = 0
    for name, did in transform_hits.items():
        if rev_dn.get(name) == did:
            print(f"  {name!r} -> {did}  via  {winners[name]!r}")
            shown += 1
            if shown >= 10:
                break

    print("\n--- Sample of 10 reverse-direction British variants (new keys for existing ids) ---")
    for i, (k, v) in enumerate(list(reverse_variants.items())[:10]):
        print(f"  {k!r} -> {v}")

    if args.write:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "_meta": {
                "hypothesis": "h961",
                "n_missing": len(missing),
                "n_transform_hits": len(transform_hits),
                "transform_coverage_pct": round(tpct, 1),
                "transform_precision_vs_disease_names": round(precision, 1),
                "n_backfill_hits": len(backfill),
                "combined_coverage_pct": round(cpct, 1),
                "n_reverse_british_variants": len(reverse_variants),
                "rule_attribution": rule_counts,
            },
            # SAFE — authoritative id from disease_names.
            "disease_names_backfill": backfill,
            # SAFE — canonical mesh_mappings id, British spelling variant of an
            # existing canonical name.
            "reverse_british_variants": reverse_variants,
            # AUDIT-ONLY — transformation hits, shown here for review. Do not
            # ship as canonical mappings (some disagree with disease_names).
            "transform_audit": {
                name: {
                    "id": did,
                    "via": winners[name],
                    "matches_disease_names": rev_dn.get(name) == did,
                }
                for name, did in transform_hits.items()
            },
        }
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
        print(f"\nWrote {out_path}")

        unres_path = out_path.with_name(out_path.stem + "_unresolved_samples.json")
        with open(unres_path, "w") as f:
            json.dump(
                {
                    "unresolved_by_transformation": {
                        n: winners.get(n, "") for n in unresolved
                    },
                    "still_unresolved_after_backfill": still_unresolved,
                },
                f,
                indent=2,
                sort_keys=True,
            )
        print(f"Wrote {unres_path}")


if __name__ == "__main__":
    main()
