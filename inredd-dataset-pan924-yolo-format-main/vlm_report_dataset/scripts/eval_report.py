"""Evaluation harness for the dental report VLM task — the single source of metrics.

Compares predicted JSON reports against gold JSON reports. A tooth instance is the unit:
(sample_id, region, fdi); the condition code is the class. Handles BOTH target schemas:
  - full_quadrant_report : {"regions": {<region>: {"teeth":[...], "comment":..}}, "summary":..}
  - regional_report      : {"region":.., "teeth":[...], "comment":..}

WHAT IT MEASURES (so you know if the VLM is actually *correct*):
  Detection — does it find the right teeth (FDI)?
      fdi_detection P / R / F1, hallucination_rate (teeth invented), miss_rate (teeth missed)
  Classification — given a found tooth, is the disease/condition right?
      condition_acc_on_detected, per-condition P/R/F1, micro/macro/weighted P/R/F1
  Joint — whole-tooth and whole-report correctness:
      tooth_accuracy (detect+classify), exact_report_match_rate
  Generation quality (VLM) — free text + structure:
      json_valid_pct, text_rouge_l (comment/summary), top condition confusions

USAGE
  python scripts/eval_report.py --gold common/val.jsonl --self-test          # must be ~1.000
  python scripts/eval_report.py --gold common/val.jsonl --simulate-baseline  # demo the metrics
  python scripts/eval_report.py --gold converted/qwen/val.jsonl --pred preds.jsonl \
         --out-json metrics_qwen_val.json --tag qwen/val
  # preds.jsonl: one line per sample -> {"id": ..., "prediction": "<json string or object>"}
"""
import argparse
import json
import os
import random
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
DS = os.path.abspath(os.path.join(HERE, ".."))

RARE = {"Dc", "Im", "P", "Rr", "M3f"}  # tooth_occurrences < ~160 in train split


def resolve_in(path):
    """Resolve an input path robustly regardless of where the script is invoked from:
    absolute -> as-is; else prefer cwd-relative if it exists; else fall back to DS-relative."""
    if os.path.isabs(path):
        return path
    if os.path.exists(path):
        return path
    return os.path.join(DS, path)


# --------------------------------------------------------------------------- parsing
def flatten(report):
    """Normalize either schema into {(region, fdi): condition}."""
    out = {}
    if not isinstance(report, dict):
        return out
    if "regions" in report and isinstance(report["regions"], dict):       # full report
        for region, payload in report["regions"].items():
            teeth = (payload or {}).get("teeth", []) if isinstance(payload, dict) else []
            for t in teeth:
                if isinstance(t, dict) and t.get("fdi") and t.get("condition"):
                    out[(region, str(t["fdi"]))] = t["condition"]
    elif "teeth" in report:                                               # regional report
        region = report.get("region", "?")
        for t in report.get("teeth", []):
            if isinstance(t, dict) and t.get("fdi") and t.get("condition"):
                out[(region, str(t["fdi"]))] = t["condition"]
    return out


def text_fields(report):
    """Concatenate the free-text fields (comments + summary) for a generation metric."""
    if not isinstance(report, dict):
        return ""
    parts = []
    if "regions" in report and isinstance(report["regions"], dict):
        for payload in report["regions"].values():
            if isinstance(payload, dict) and payload.get("comment"):
                parts.append(str(payload["comment"]))
        if report.get("summary"):
            parts.append(str(report["summary"]))
    elif "teeth" in report:
        if report.get("comment"):
            parts.append(str(report["comment"]))
    return " ".join(parts)


def rouge_l_f1(ref, hyp):
    """Self-contained ROUGE-L (LCS-based) F1 on whitespace tokens — no external deps."""
    r, h = ref.split(), hyp.split()
    if not r or not h:
        return 1.0 if not r and not h else 0.0
    # LCS length via DP
    dp = [[0] * (len(h) + 1) for _ in range(len(r) + 1)]
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            dp[i][j] = dp[i - 1][j - 1] + 1 if r[i - 1] == h[j - 1] else max(dp[i - 1][j], dp[i][j - 1])
    lcs = dp[len(r)][len(h)]
    prec, rec = lcs / len(h), lcs / len(r)
    return 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0


def parse_json_maybe(s):
    if isinstance(s, dict):
        return s
    if not isinstance(s, str):
        return None
    try:
        return json.loads(s)
    except Exception:
        # tolerate models that wrap JSON in markdown fences or add prose
        a, b = s.find("{"), s.rfind("}")
        if 0 <= a < b:
            try:
                return json.loads(s[a:b + 1])
            except Exception:
                return None
        return None


def gold_target_string(o):
    """Pull the gold assistant target out of a common/* or converted/* row."""
    for m in o.get("messages", []):
        if m.get("role") == "assistant":
            c = m["content"]
            return c if isinstance(c, str) else (c[0].get("text") if isinstance(c, list) and c else None)
    if "conversations" in o:                                              # internvl sharegpt
        for c in o["conversations"]:
            if c.get("from") == "gpt":
                return c.get("value")
    if "response" in o:                                                   # phi
        return str(o["response"]).replace("<|end|>", "")
    if "suffix" in o:                                                     # paligemma
        return o["suffix"]
    return None


# --------------------------------------------------------------------------- degradation (sim)
def degrade(report, drop_prob=0.9, rng=None):
    """Mimic a naive baseline: learns H/R but defaults rare conditions to 'H'."""
    rng = rng or random
    r = json.loads(json.dumps(report))

    def fix(teeth):
        for t in teeth:
            if isinstance(t, dict) and t.get("condition") in RARE and rng.random() < drop_prob:
                t["condition"], t["condition_name"] = "H", "healthy"

    if "regions" in r:
        for payload in r["regions"].values():
            if isinstance(payload, dict):
                fix(payload.get("teeth", []))
    elif "teeth" in r:
        fix(r["teeth"])
    return r


# --------------------------------------------------------------------------- scoring
def prf(t, fp_, fn_):
    p = t / (t + fp_) if (t + fp_) else 0.0
    r = t / (t + fn_) if (t + fn_) else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f


def evaluate(pairs, with_text=True):
    """pairs: list of (gold_dict, pred_dict_or_None) -> metrics dict."""
    tp, fp, fn = defaultdict(int), defaultdict(int), defaultdict(int)
    conds = set()
    fdi_tp = fdi_fp = fdi_fn = 0
    detected = detected_correct = 0          # teeth whose FDI matched (for condition accuracy)
    union_total = union_correct = 0          # joint detect+classify accuracy
    exact_match = 0
    confusions = defaultdict(int)            # (gold_cond -> pred_cond) on matched FDI, wrong label
    rouge_sum = rouge_n = 0.0
    n = valid = 0

    for gold, pred in pairs:
        n += 1
        g = flatten(gold)
        for c in g.values():
            conds.add(c)
        if pred is None:                     # unparseable -> all gold teeth missed
            for c in g.values():
                fn[c] += 1
            fdi_fn += len(g)
            union_total += len(g)
            continue
        valid += 1
        p = flatten(pred)
        for c in p.values():
            conds.add(c)

        for key in set(g) | set(p):
            gc, pc = g.get(key), p.get(key)
            union_total += 1
            if gc == pc and gc is not None:
                tp[gc] += 1
                union_correct += 1
            else:
                if gc is not None:
                    fn[gc] += 1
                if pc is not None:
                    fp[pc] += 1
            if gc is not None and pc is not None:   # FDI detected by both
                detected += 1
                if gc == pc:
                    detected_correct += 1
                else:
                    confusions[(gc, pc)] += 1

        gk, pk = set(g), set(p)
        fdi_tp += len(gk & pk)
        fdi_fp += len(pk - gk)
        fdi_fn += len(gk - pk)
        if g == p:
            exact_match += 1
        if with_text:
            rouge_sum += rouge_l_f1(text_fields(gold), text_fields(pred))
            rouge_n += 1

    # per-condition + aggregates
    per_cond = {}
    macro_p = macro_r = macro_f = 0.0
    w_p = w_r = w_f = 0.0
    micro_tp = micro_fp = micro_fn = 0
    total_support = 0
    for c in sorted(conds):
        p_, r_, f_ = prf(tp[c], fp[c], fn[c])
        sup = tp[c] + fn[c]
        per_cond[c] = {"support": sup, "P": p_, "R": r_, "F1": f_}
        macro_p += p_; macro_r += r_; macro_f += f_
        w_p += p_ * sup; w_r += r_ * sup; w_f += f_ * sup
        total_support += sup
        micro_tp += tp[c]; micro_fp += fp[c]; micro_fn += fn[c]
    k = len(conds) or 1
    micro = prf(micro_tp, micro_fp, micro_fn)
    fdi = prf(fdi_tp, fdi_fp, fdi_fn)

    top_conf = sorted(confusions.items(), key=lambda x: -x[1])[:5]
    pred_teeth_total = fdi_tp + fdi_fp
    gold_teeth_total = fdi_tp + fdi_fn

    return {
        "n_samples": n,
        "json_valid_pct": round(100 * valid / n, 2) if n else 0.0,
        # classification (condition labels)
        "micro_precision": micro[0], "micro_recall": micro[1], "micro_f1": micro[2],
        "macro_precision": macro_p / k, "macro_recall": macro_r / k, "macro_f1": macro_f / k,
        "weighted_precision": w_p / total_support if total_support else 0.0,
        "weighted_recall": w_r / total_support if total_support else 0.0,
        "weighted_f1": w_f / total_support if total_support else 0.0,
        # detection (FDI)
        "fdi_precision": fdi[0], "fdi_recall": fdi[1], "fdi_detection_f1": fdi[2],
        "hallucination_rate": fdi_fp / pred_teeth_total if pred_teeth_total else 0.0,
        "miss_rate": fdi_fn / gold_teeth_total if gold_teeth_total else 0.0,
        # joint correctness
        "condition_acc_on_detected": detected_correct / detected if detected else 0.0,
        "tooth_accuracy": union_correct / union_total if union_total else 0.0,
        "exact_report_match_rate": exact_match / n if n else 0.0,
        # generation
        "text_rouge_l": (rouge_sum / rouge_n) if rouge_n else None,
        "top_confusions": [{"gold": a, "pred": b, "count": c} for (a, b), c in top_conf],
        "per_condition": per_cond,
    }


# --------------------------------------------------------------------------- reporting
def print_report(title, m):
    print(f"\n===== {title} =====")
    print(f"samples={m['n_samples']}  json_valid={m['json_valid_pct']}%")
    print("-- classification (condition labels) --")
    print(f"  micro    P={m['micro_precision']:.3f} R={m['micro_recall']:.3f} F1={m['micro_f1']:.3f}")
    print(f"  MACRO    P={m['macro_precision']:.3f} R={m['macro_recall']:.3f} F1={m['macro_f1']:.3f}   <- rare-class sensitive")
    print(f"  weighted P={m['weighted_precision']:.3f} R={m['weighted_recall']:.3f} F1={m['weighted_f1']:.3f}")
    print("-- detection (FDI teeth) --")
    print(f"  FDI      P={m['fdi_precision']:.3f} R={m['fdi_recall']:.3f} F1={m['fdi_detection_f1']:.3f}"
          f"   hallucination={m['hallucination_rate']:.3f}  miss={m['miss_rate']:.3f}")
    print("-- joint correctness --")
    print(f"  condition_acc_on_detected={m['condition_acc_on_detected']:.3f}  "
          f"tooth_accuracy={m['tooth_accuracy']:.3f}  exact_report={m['exact_report_match_rate']:.3f}")
    rl = m["text_rouge_l"]
    print(f"-- generation --  text_ROUGE-L={'%.3f' % rl if rl is not None else 'n/a'}")
    print(f"{'cond':<6}{'support':>8}{'P':>8}{'R':>8}{'F1':>8}   tier")
    for c, d in sorted(m["per_condition"].items(), key=lambda x: -x[1]["support"]):
        print(f"{c:<6}{d['support']:>8}{d['P']:>8.3f}{d['R']:>8.3f}{d['F1']:>8.3f}   {'RARE' if c in RARE else ''}")
    if m["top_confusions"]:
        print("top condition confusions (gold -> pred, on correctly-detected teeth):")
        for cf in m["top_confusions"]:
            print(f"  {cf['gold']:>5} -> {cf['pred']:<5}  x{cf['count']}")


# --------------------------------------------------------------------------- main
def load_gold(path):
    rows = {}
    with open(path, encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            o = json.loads(ln)
            g = parse_json_maybe(gold_target_string(o))
            if g is not None:
                rows[o["id"]] = g
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", required=True, help="common/* or converted/* jsonl with gold targets")
    ap.add_argument("--pred", help="jsonl with {id, prediction}")
    ap.add_argument("--self-test", action="store_true", help="use gold as predictions (expect ~1.0)")
    ap.add_argument("--simulate-baseline", action="store_true", help="degrade gold (rare->H) demo")
    ap.add_argument("--out-json", help="write the metrics dict to this path (for compare_models.py)")
    ap.add_argument("--tag", default=None, help="label stored in the output json (e.g. qwen/val)")
    ap.add_argument("--no-text", action="store_true", help="skip the ROUGE-L text metric")
    ap.add_argument("--seed", type=int, default=924)
    args = ap.parse_args()

    gold_path = resolve_in(args.gold)
    gold = load_gold(gold_path)
    print(f"Loaded {len(gold)} gold samples from {gold_path}")
    with_text = not args.no_text
    metrics = None

    if args.self_test:
        metrics = evaluate([(g, g) for g in gold.values()], with_text)
        print_report("SELF-TEST (pred = gold) -- sanity check, must be 1.000", metrics)

    if args.simulate_baseline:
        rng = random.Random(args.seed)
        metrics = evaluate([(g, degrade(g, rng=rng)) for g in gold.values()], with_text)
        print_report("SIMULATED NAIVE BASELINE (rare conditions defaulted to H)", metrics)

    if args.pred:
        pred_path = resolve_in(args.pred)
        preds = {}
        with open(pred_path, encoding="utf-8") as f:
            for ln in f:
                if ln.strip():
                    o = json.loads(ln)
                    preds[o["id"]] = parse_json_maybe(o.get("prediction"))
        missing = sum(1 for i in gold if i not in preds)
        if missing:
            print(f"WARNING: {missing} gold ids have no prediction (counted as misses)")
        metrics = evaluate([(g, preds.get(i)) for i, g in gold.items()], with_text)
        print_report(f"EVAL vs predictions ({os.path.basename(pred_path)})", metrics)

    if args.out_json and metrics is not None:
        metrics_out = dict(metrics)
        metrics_out["tag"] = args.tag or os.path.basename(args.gold)
        out = args.out_json  # cwd-relative (or absolute); created if needed
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(metrics_out, f, indent=2, ensure_ascii=False)
        print(f"\nMetrics written to {out}")


if __name__ == "__main__":
    main()
