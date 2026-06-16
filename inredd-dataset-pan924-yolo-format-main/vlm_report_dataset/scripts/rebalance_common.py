"""Rebalance the common train split for the long-tailed dental conditions.

Combines two complementary levers (discussed in the readiness report):

  Method 2 (density)   : oversample ONLY the regional_report crops, never the full reports.
                         A quadrant crop lists ~11 teeth vs ~21 for a full report, so a rare
                         condition is a larger fraction of each answer -> a louder learning
                         signal per token, less drowned by the flood of 'H'.

  Method 3 (frequency) : duplicate the rare-condition crops with a KG-derived multiplier
                         (inverse image-prevalence), so the model SEES the tail more often.

Oversampling only the regional crops applies both levers at once: the copies are both denser
and more frequent.

ANTI-OVERFIT SAFEGUARDS (the whole point of doing this carefully):
  1. CAP on the multiplier (default 8x)         -> no class explodes without bound.
  2. Per-sample multiplier = MAX over its rare conditions, never the product
                                                 -> no multiplicative compounding.
  3. Copies are spread across ALL distinct crops that contain a condition; a single image is
     never duplicated more than CAP times       -> the model cannot memorise one picture.
  4. Stochastic rounding of the fractional part (seeded)
                                                 -> smooth distribution, not rigid integer blow-up.
  5. val/test are copied VERBATIM, never oversampled
                                                 -> evaluation stays honest and is the real
                                                    overfit detector (train up / val down).
  6. Duplicate rows get a unique id suffix + metadata flag so they are fully traceable/removable.

Image pixels are NOT fabricated: there are still only the original rare-condition images. This
helps the model attend to the tail; it is not a substitute for collecting more data.

Outputs:
  - common_balanced/{train,val,test}.jsonl   (train rebalanced; val/test verbatim)
  - common/metadata/rebalance_summary.json
"""
import argparse
import json
import os
import random
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
DS = os.path.abspath(os.path.join(HERE, ".."))

# The clear tail (train tooth-occurrence < ~160), per split_report.json / dental_kg.json.
RARE = ["Dc", "Im", "P", "Rr", "M3f"]
CAP = 8.0
SEED = 924


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(ln) for ln in f if ln.strip()]


def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def gold_conditions(row):
    """Conditions present in a row, parsed from the gold assistant JSON (both schemas)."""
    rep = json.loads(row["messages"][1]["content"])
    out = []
    if "regions" in rep:
        for p in rep["regions"].values():
            out += [t["condition"] for t in p.get("teeth", [])]
    elif "teeth" in rep:
        out += [t["condition"] for t in rep["teeth"]]
    return out


def compute_multipliers(regional_rows, cap):
    """KG-style inverse image-prevalence multiplier per rare condition, capped at `cap`."""
    n = len(regional_rows)
    contains = Counter()
    for r in regional_rows:
        for c in set(gold_conditions(r)):
            if c in RARE:
                contains[c] += 1
    prev = {c: contains[c] / n for c in RARE if contains[c] > 0}
    min_prev = min(prev.values())
    mult = {c: max(1.0, min(cap, (min_prev / prev[c]) * cap)) for c in prev}
    return mult, contains, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", default="common")
    ap.add_argument("--out-dir", default="common_balanced")
    ap.add_argument("--cap", type=float, default=CAP)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    in_dir = os.path.join(DS, args.in_dir)
    out_dir = os.path.join(DS, args.out_dir)
    rng = random.Random(args.seed)

    train = read_jsonl(os.path.join(in_dir, "train.jsonl"))
    regional = [r for r in train if r.get("task") == "regional_report"]
    full = [r for r in train if r.get("task") != "regional_report"]

    mult, contains, n_reg = compute_multipliers(regional, args.cap)

    # Build the rebalanced train split.
    balanced = list(full)  # full reports kept verbatim (Method 2: only boost the denser crops)
    added_per_cond = Counter()
    dup_total = 0

    for r in regional:
        balanced.append(r)  # the original copy, always once
        present = [c for c in set(gold_conditions(r)) if c in mult]
        if not present:
            continue
        m = max(mult[c] for c in present)          # safeguard 2: MAX, not product
        copies = int(m) + (1 if rng.random() < (m - int(m)) else 0)  # safeguard 4
        for k in range(1, copies):                 # copies-1 duplicates (original already added)
            d = json.loads(json.dumps(r))
            d["id"] = f"{r['id']}__dup{k}"          # safeguard 6: unique + traceable
            d.setdefault("metadata", {})["oversampled"] = True
            balanced.append(d)
            dup_total += 1
            for c in present:
                added_per_cond[c] += 1

    rng.shuffle(balanced)

    # Copy val/test verbatim (safeguard 5).
    val = read_jsonl(os.path.join(in_dir, "val.jsonl"))
    test = read_jsonl(os.path.join(in_dir, "test.jsonl"))
    write_jsonl(os.path.join(out_dir, "train.jsonl"), balanced)
    write_jsonl(os.path.join(out_dir, "val.jsonl"), val)
    write_jsonl(os.path.join(out_dir, "test.jsonl"), test)

    # Per-condition tooth-occurrence before vs after (train).
    def tooth_counts(rows):
        c = Counter()
        for r in rows:
            c.update(gold_conditions(r))
        return c

    before = tooth_counts(train)
    after = tooth_counts(balanced)

    summary = {
        "seed": args.seed,
        "cap": args.cap,
        "method": "regional-only oversampling, KG inverse-prevalence multiplier, MAX-combine, stochastic rounding",
        "rare_conditions": RARE,
        "multipliers": {c: round(mult.get(c, 1.0), 2) for c in RARE},
        "regional_samples_containing": dict(contains),
        "train_rows_before": len(train),
        "train_rows_after": len(balanced),
        "duplicates_added": dup_total,
        "growth_pct": round(100 * dup_total / len(train), 1),
        "val_rows": len(val),
        "test_rows": len(test),
        "tooth_occurrences_before": {c: before[c] for c in sorted(before)},
        "tooth_occurrences_after": {c: after[c] for c in sorted(after)},
    }
    write_jsonl(os.path.join(DS, "common", "metadata", "rebalance_summary.json"), [])  # ensure dir
    with open(os.path.join(DS, "common", "metadata", "rebalance_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # ---- console report ----
    print(f"Rebalanced train: {len(train)} -> {len(balanced)} rows (+{dup_total}, +{summary['growth_pct']}%)")
    print(f"val={len(val)} test={len(test)} (copied verbatim, never oversampled)\n")
    print(f"{'cond':<6}{'mult':>7}{'before':>9}{'after':>9}{'x':>7}")
    for c in RARE:
        b, a = before[c], after[c]
        print(f"{c:<6}{mult.get(c,1.0):>6.1f}x{b:>9}{a:>9}{a/b if b else 0:>6.1f}x")
    print("\nHead classes (unchanged-ish, only via co-occurrence in duplicated crops):")
    for c in ["H", "R", "Te", "C", "CpuM", "Di", "M3i"]:
        print(f"  {c:<5} {before[c]:>6} -> {after[c]:>6}")


if __name__ == "__main__":
    main()
