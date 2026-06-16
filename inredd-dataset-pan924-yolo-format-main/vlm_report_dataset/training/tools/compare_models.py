"""Aggregate per-model metric JSONs (written by eval_report.py --out-json) into one comparison
table, so you can see which of the 5 models is best — overall and on the rare conditions.

  python compare_models.py /content/drive/MyDrive/pan924_runs/*/metrics_*_test.json
  python compare_models.py path1.json path2.json ...
"""
import glob
import json
import sys

RARE = ["Dc", "Im", "P", "Rr", "M3f"]
HEADLINE = [
    ("macro_f1", "macroF1"), ("micro_f1", "microF1"), ("weighted_f1", "wtF1"),
    ("fdi_detection_f1", "FDI-F1"), ("condition_acc_on_detected", "condAcc"),
    ("hallucination_rate", "halluc"), ("exact_report_match_rate", "exact"),
    ("text_rouge_l", "ROUGE-L"), ("json_valid_pct", "json%"),
]


def expand(paths):
    out = []
    for p in paths:
        out.extend(sorted(glob.glob(p)) if any(c in p for c in "*?[") else [p])
    return out


def main():
    paths = expand(sys.argv[1:])
    if not paths:
        print("usage: python compare_models.py <metrics_*.json> [...]")
        sys.exit(1)

    rows = []
    for p in paths:
        try:
            m = json.load(open(p, encoding="utf-8"))
        except Exception as e:
            print(f"skip {p}: {e}")
            continue
        rows.append((m.get("tag", p), m))

    # headline table
    hdr = f"{'model':<16}" + "".join(f"{lbl:>9}" for _, lbl in HEADLINE)
    print(hdr)
    print("-" * len(hdr))
    for tag, m in sorted(rows, key=lambda r: -(r[1].get("macro_f1") or 0)):
        line = f"{str(tag):<16}"
        for k, _ in HEADLINE:
            v = m.get(k)
            line += "      n/a" if v is None else f"{v:>9.3f}"
        print(line)

    # rare-condition recall table (the payoff of the rebalancing)
    print("\nRare-condition RECALL (higher = better at catching the tail):")
    hdr2 = f"{'model':<16}" + "".join(f"{c:>7}" for c in RARE)
    print(hdr2)
    print("-" * len(hdr2))
    for tag, m in sorted(rows, key=lambda r: -(r[1].get("macro_f1") or 0)):
        pc = m.get("per_condition", {})
        line = f"{str(tag):<16}"
        for c in RARE:
            r = pc.get(c, {}).get("R")
            line += "    n/a" if r is None else f"{r:>7.3f}"
        print(line)


if __name__ == "__main__":
    main()
