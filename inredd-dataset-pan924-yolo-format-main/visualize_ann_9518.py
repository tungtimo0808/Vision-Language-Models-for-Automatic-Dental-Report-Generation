"""Visualize ann 9518 (FDI 14) on image 1366-M-34.jpg.

Output:
  - ann_9518_full.png    : full image with this tooth highlighted + all other teeth faintly outlined
  - ann_9518_zoom.png    : 600x600 crop around the tooth showing bbox vs polygon clearly
"""
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).parent
JSON_PATH = ROOT / "annotations" / "teeth_fdi_labels.json"
IMG_PATH = ROOT / "images" / "1366-M-34.jpg"
OUT_FULL = ROOT / "ann_9518_full.png"
OUT_ZOOM = ROOT / "ann_9518_zoom.png"

TARGET_ANN_ID = 9518
TARGET_IMG_ID = 392

print("Loading JSON...")
with open(JSON_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

anns_this_img = [a for a in data["annotations"] if a["image_id"] == TARGET_IMG_ID]
target = next(a for a in anns_this_img if a["id"] == TARGET_ANN_ID)
print(f"Image 392 has {len(anns_this_img)} tooth annotations")
print(f"Target ann {TARGET_ANN_ID}: cat={target['category_id']}, "
      f"bbox={target['bbox']}, seg pts={len(target['segmentation'][0]) // 2}")

img = Image.open(IMG_PATH).convert("RGBA")
W, H = img.size

try:
    font_big = ImageFont.truetype("arial.ttf", 36)
    font_sm = ImageFont.truetype("arial.ttf", 22)
except OSError:
    font_big = ImageFont.load_default()
    font_sm = ImageFont.load_default()


def poly_pts(seg):
    return list(zip(seg[0::2], seg[1::2]))


# ---------- FULL VIEW ----------
overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
draw = ImageDraw.Draw(overlay)

# Faint outlines for all other teeth
for ann in anns_this_img:
    if ann["id"] == TARGET_ANN_ID:
        continue
    pts = poly_pts(ann["segmentation"][0])
    draw.polygon(pts, outline=(0, 200, 255, 180), width=2)

# Target tooth: bbox (red, dashed-style) + polygon (green filled)
x, y, w, h = target["bbox"]
bbox_corners = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
poly_corners = poly_pts(target["segmentation"][0])

# polygon (segmentation) — solid green
draw.polygon(poly_corners, fill=(0, 255, 0, 90),
             outline=(0, 255, 0, 255), width=5)
# bbox — red outline
draw.rectangle([x, y, x + w, y + h], outline=(255, 50, 50, 255), width=5)
# polygon vertices
for px, py in poly_corners:
    draw.ellipse((px - 9, py - 9, px + 9, py + 9),
                 fill=(255, 255, 0, 255))

# Legend
legend_x, legend_y = 20, 20
draw.rectangle((legend_x, legend_y, legend_x + 540, legend_y + 150),
               fill=(0, 0, 0, 200))
draw.rectangle((legend_x + 15, legend_y + 18, legend_x + 55, legend_y + 38),
               outline=(255, 50, 50, 255), width=4)
draw.text((legend_x + 70, legend_y + 15),
          "bbox (axis-aligned, 4 numbers)", fill=(255, 255, 255), font=font_sm)
draw.polygon([(legend_x + 15, legend_y + 65),
              (legend_x + 55, legend_y + 55),
              (legend_x + 55, legend_y + 90),
              (legend_x + 15, legend_y + 88)],
             fill=(0, 255, 0, 90), outline=(0, 255, 0, 255), width=3)
draw.text((legend_x + 70, legend_y + 65),
          "segmentation polygon (tilted)", fill=(255, 255, 255), font=font_sm)
draw.text((legend_x + 70, legend_y + 110),
          "cyan = other 27 teeth on this image",
          fill=(180, 220, 255), font=font_sm)

# Label arrow near tooth
label = f"FDI 14 (cat={target['category_id']}, id={TARGET_ANN_ID})"
draw.text((x + w + 15, y), label,
          fill=(255, 255, 0), font=font_big,
          stroke_width=3, stroke_fill=(0, 0, 0))

full = Image.alpha_composite(img, overlay).convert("RGB")
full.save(OUT_FULL, "PNG")
print(f"Saved full: {OUT_FULL}")

# ---------- ZOOM VIEW ----------
pad = 120
zx0, zy0 = int(x) - pad, int(y) - pad
zx1, zy1 = int(x + w) + pad, int(y + h) + pad
zx0, zy0 = max(0, zx0), max(0, zy0)
zx1, zy1 = min(W, zx1), min(H, zy1)

zoom = full.crop((zx0, zy0, zx1, zy1))
# Re-draw with thicker strokes for clarity at this zoom level
zdraw = ImageDraw.Draw(zoom, "RGBA")

# Re-draw fresh (the crop already has overlay baked in, but add point labels)
for i, (px, py) in enumerate(poly_corners):
    lx, ly = px - zx0, py - zy0
    zdraw.ellipse((lx - 11, ly - 11, lx + 11, ly + 11),
                  fill=(255, 255, 0, 255), outline=(0, 0, 0, 255), width=2)
    zdraw.text((lx + 14, ly - 10),
               f"P{i+1}({px},{py})", fill=(255, 255, 0),
               font=font_sm, stroke_width=2, stroke_fill=(0, 0, 0))

zoom.save(OUT_ZOOM, "PNG")
print(f"Saved zoom: {OUT_ZOOM}")
