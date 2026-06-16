"""Build a dental Knowledge Graph (KG) from the common train split.

The KG is a lightweight, data-derived prior intended to help with the rare-condition
imbalance (see report discussion). It is built ONLY from `common/train.jsonl` so it never
leaks val/test statistics.

Nodes:
  - Condition  (the 12 condition codes + structural pseudo-labels)
  - Tooth      (FDI position, e.g. "18")
  - Region     (image-space region: image_upper_left, ...)

Relations captured (as adjacency/count tables):
  - condition -> valid FDI positions          (where each condition actually occurs)
  - condition -> image regions                (which regions a condition appears in)
  - tooth(FDI) -> image regions               (empirical FDI -> region mapping)
  - condition <-> condition co-occurrence      (same region; symmetric counts)
  - per-condition frequency + rarity tier

Outputs:
  - common/metadata/dental_kg.json     (machine-readable KG)
  - prints a human-readable summary
"""
import json
import os
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
DS = os.path.abspath(os.path.join(HERE, ".."))          # vlm_report_dataset/
TRAIN = os.path.join(DS, "common", "train.jsonl")
CONDMAP = os.path.join(DS, "config", "condition_map.json")
OUT = os.path.join(DS, "common", "metadata", "dental_kg.json")

REGIONS = ["image_upper_left", "image_upper_right", "image_lower_left", "image_lower_right"]


def load_condition_map():
    with open(CONDMAP, encoding="utf-8") as f:
        return json.load(f)


def iter_full_samples(path):
    """Yield labels.regions only from full_quadrant_report rows.

    The full report already contains every annotated tooth of an image across all four
    regions, so iterating regional rows too would double-count. One full row == one image.
    """
    with open(path, encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            o = json.loads(ln)
            if o.get("task") != "full_quadrant_report":
                continue
            regs = o.get("labels", {}).get("regions", {})
            if isinstance(regs, dict):
                yield regs


def main():
    cond_map = load_condition_map()

    cond_count = Counter()                          # condition -> tooth occurrences
    cond_fdi = defaultdict(Counter)                 # condition -> {fdi: count}
    cond_region = defaultdict(Counter)              # condition -> {region: count}
    fdi_region = defaultdict(Counter)               # fdi -> {region: count}
    cooccur = defaultdict(Counter)                  # condition -> {other_condition: count} (same region)
    images_with_cond = defaultdict(set)             # condition -> set(image index) for image-level prevalence
    n_images = 0

    for img_idx, regs in enumerate(iter_full_samples(TRAIN)):
        n_images += 1
        for region, teeth in regs.items():
            conds_here = []
            for t in teeth:
                fdi = t.get("fdi")
                cond = t.get("condition")
                if not cond:
                    continue
                cond_count[cond] += 1
                if fdi:
                    cond_fdi[cond][fdi] += 1
                    fdi_region[fdi][region] += 1
                cond_region[cond][region] += 1
                conds_here.append(cond)
                images_with_cond[cond].add(img_idx)
            # co-occurrence within the same region (unordered pairs)
            uniq = set(conds_here)
            for a in uniq:
                for b in uniq:
                    if a != b:
                        cooccur[a][b] += 1

    # rarity tiers based on tooth-occurrence count
    def tier(n):
        if n >= 3000:
            return "dominant"
        if n >= 500:
            return "common"
        if n >= 150:
            return "uncommon"
        return "rare"

    conditions = {}
    for cond, n in sorted(cond_count.items(), key=lambda x: -x[1]):
        meta = cond_map.get(cond, {})
        valid_fdi = sorted(cond_fdi[cond], key=lambda k: -cond_fdi[cond][k])
        conditions[cond] = {
            "name": meta.get("name", cond),
            "description": meta.get("description", ""),
            "tooth_occurrences": n,
            "image_prevalence": len(images_with_cond[cond]),
            "image_prevalence_pct": round(100 * len(images_with_cond[cond]) / max(1, n_images), 2),
            "rarity": tier(n),
            "valid_fdi": valid_fdi,
            "n_valid_fdi": len(valid_fdi),
            "regions": dict(cond_region[cond].most_common()),
            "top_cooccur": dict(cooccur[cond].most_common(5)),
        }

    kg = {
        "source": "vlm_report_dataset/common/train.jsonl (full_quadrant_report rows only)",
        "n_images": n_images,
        "regions": REGIONS,
        "conditions": conditions,
        "fdi_to_region": {fdi: dict(rc.most_common()) for fdi, rc in sorted(fdi_region.items())},
    }

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(kg, f, indent=2, ensure_ascii=False)

    # ---- human-readable summary ----
    print(f"KG built from {n_images} train images -> {os.path.relpath(OUT, DS)}\n")
    print(f"{'cond':<6}{'name':<26}{'rarity':<10}{'teeth':>7}{'imgs%':>7}{'#FDI':>6}  valid_fdi")
    for c, d in conditions.items():
        fdi_str = ",".join(d["valid_fdi"][:8]) + ("..." if d["n_valid_fdi"] > 8 else "")
        print(f"{c:<6}{d['name'][:25]:<26}{d['rarity']:<10}{d['tooth_occurrences']:>7}"
              f"{d['image_prevalence_pct']:>7}{d['n_valid_fdi']:>6}  {fdi_str}")

    print("\nRare-condition co-occurrence (what tends to appear alongside them):")
    for c, d in conditions.items():
        if d["rarity"] == "rare":
            co = ", ".join(f"{k}:{v}" for k, v in d["top_cooccur"].items())
            print(f"  {c:<5} ({d['name']}): {co}")


if __name__ == "__main__":
    main()
