"""Test hypothesis: each polygon vertex touches one side of the bbox."""
import json
from pathlib import Path

ROOT = Path(__file__).parent
with open(ROOT / "annotations" / "teeth_fdi_labels.json", "r", encoding="utf-8") as f:
    data = json.load(f)

anns = [a for a in data["annotations"] if a["image_id"] == 392]
print(f"Analyzing {len(anns)} teeth on image 392\n")

EPS = 1.0

def classify(px, py, x0, y0, x1, y1):
    sides = []
    if abs(px - x0) < EPS: sides.append("L")
    if abs(px - x1) < EPS: sides.append("R")
    if abs(py - y0) < EPS: sides.append("T")
    if abs(py - y1) < EPS: sides.append("B")
    return "+".join(sides) if sides else "interior"

n_perfect = 0
patterns = {}
for a in sorted(anns, key=lambda x: x["category_id"]):
    bx, by, bw, bh = a["bbox"]
    x0, y0, x1, y1 = bx, by, bx + bw, by + bh
    seg = a["segmentation"][0]
    pts = list(zip(seg[0::2], seg[1::2]))
    sides = [classify(px, py, x0, y0, x1, y1) for px, py in pts]
    print(f"FDI {a['category_id']:>2}  ann {a['id']:>5}  "
          f"n_pts={len(pts)}  vertex-sides: {sides}")
    key = tuple(sides) if len(pts) == 4 else f"npts={len(pts)}"
    patterns[key] = patterns.get(key, 0) + 1
    if set(sides) == {"L", "R", "T", "B"}:
        n_perfect += 1

print(f"\n{n_perfect}/{len(anns)} have exactly 1 vertex on each side L/R/T/B")
print("\nPattern frequencies (vertex side order):")
for k, v in sorted(patterns.items(), key=lambda x: -x[1]):
    print(f"  {v:>3}x  {k}")
