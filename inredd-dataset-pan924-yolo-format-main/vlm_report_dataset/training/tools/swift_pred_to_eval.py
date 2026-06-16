"""Adapter: turn an `swift infer` result file into the {id, prediction} jsonl that
eval_report.py consumes.

ms-swift's infer output does not reliably carry our `id`, so we recover ids by ZIPPING the
results with the val/test dataset IN ORDER (swift infer keeps order when temperature=0 and the
dataset is not shuffled). If counts mismatch it warns — inspect the swift result schema then.

Usage:
  python swift_pred_to_eval.py --val <data.jsonl> --swift-result <result.jsonl> --out preds.jsonl
"""
import argparse
import json


def read_jsonl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(ln) for ln in f if ln.strip()]


def get_response(obj, field):
    for k in (field, "response", "generated_text", "infer_response", "completion"):
        if isinstance(obj.get(k), str):
            return obj[k]
    msgs = obj.get("messages")
    if isinstance(msgs, list) and msgs and isinstance(msgs[-1].get("content"), str):
        return msgs[-1]["content"]
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", required=True)
    ap.add_argument("--swift-result", required=True)
    ap.add_argument("--pred-field", default="response")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    val = read_jsonl(args.val)
    res = read_jsonl(args.swift_result)
    if len(val) != len(res):
        print(f"WARNING: val={len(val)} vs results={len(res)} — order-zip may be misaligned. "
              f"Inspect the swift result schema and adjust --pred-field.")
    n = min(len(val), len(res))
    with open(args.out, "w", encoding="utf-8") as f:
        for i in range(n):
            f.write(json.dumps({"id": val[i]["id"],
                                "prediction": get_response(res[i], args.pred_field)},
                               ensure_ascii=False) + "\n")
    print(f"Wrote {n} predictions -> {args.out}")


if __name__ == "__main__":
    main()
