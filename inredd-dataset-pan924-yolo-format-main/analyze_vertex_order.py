"""Figure out the vertex ordering rule for segmentation polygons."""
import json
from pathlib import Path

ROOT = Path(__file__).parent
with open(ROOT / "annotations" / "teeth_fdi_labels.json", "r", encoding="utf-8") as f:
    data = json.load(f)

anns = [a for a in data["annotations"] if a["image_id"] == 392]

def shoelace(pts):
    n = len(pts)
    s = 0
    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]
        s += x1 * y2 - x2 * y1
    return s / 2  # positive = CCW in math coords (= CW in image coords with y down)

def starting_corner(pts, bbox):
    """Which corner of bbox is P1 closest to?"""
    bx, by, bw, bh = bbox
    corners = {
        "TL": (bx, by),
        "TR": (bx + bw, by),
        "BL": (bx, by + bh),
        "BR": (bx + bw, by + bh),
    }
    p1 = pts[0]
    dists = {k: (p1[0] - cx) ** 2 + (p1[1] - cy) ** 2 for k, (cx, cy) in corners.items()}
    return min(dists, key=dists.get)

print(f"{'FDI':>4} {'ann':>5} {'pattern':<10} {'area_sign':<11} {'orient':<6} {'P1_near':<8} {'P1':>14}")
print("-" * 80)
for a in sorted(anns, key=lambda x: x["category_id"]):
    seg = a["segmentation"][0]
    pts = list(zip(seg[0::2], seg[1::2]))
    bx, by, bw, bh = a["bbox"]
    # Pattern classification (reuse from previous)
    EPS = 1.0
    def cls(px, py):
        s = []
        if abs(px - bx) < EPS: s.append("L")
        if abs(px - (bx + bw)) < EPS: s.append("R")
        if abs(py - by) < EPS: s.append("T")
        if abs(py - (by + bh)) < EPS: s.append("B")
        return "+".join(s)
    sides = [cls(px, py) for px, py in pts]
    if set(sides) == {"L+T", "L+B", "R+B", "R+T"}:
        pat = "A (axis)"
    elif sides == ["T", "L", "B", "R"]:
        pat = "B (TLBR)"
    elif sides == ["L", "B", "R", "T"]:
        pat = "C (LBRT)"
    else:
        pat = "?"
    s = shoelace(pts)
    orient = "CW" if s < 0 else "CCW"  # image coords: y down → negative shoelace = CW visually
    near = starting_corner(pts, a["bbox"])
    print(f"{a['category_id']:>4} {a['id']:>5} {pat:<10} {s:>10.0f}  {orient:<6} {near:<8} {str(pts[0]):>14}")
