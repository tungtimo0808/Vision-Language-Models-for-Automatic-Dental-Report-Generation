"""Visualize annotation id=20957 (category_id=4, mouth_and_teeth) on image 997-F-21.jpg."""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).parent
IMG_PATH = ROOT / "images" / "997-F-21.jpg"
OUT_PATH = ROOT / "ann_20957_preview.png"

# Hard-coded annotation 20957 (the one selected in the IDE)
SEG = [327, 248, 327, 1401, 2582, 1401, 2582, 248]
ANN_ID = 20957
CAT_ID = 4
CAT_NAME = "mouth_and_teeth"

img = Image.open(IMG_PATH).convert("RGB")
W, H = img.size
print(f"Image size: {W} x {H}")

polygon = list(zip(SEG[0::2], SEG[1::2]))
xs, ys = zip(*polygon)
x_min, y_min, x_max, y_max = min(xs), min(ys), max(xs), max(ys)
print(f"Polygon points: {polygon}")
print(f"Bbox: ({x_min}, {y_min}) -> ({x_max}, {y_max})  "
      f"size {x_max - x_min} x {y_max - y_min}")
print(f"Coverage: {(x_max - x_min) * (y_max - y_min) / (W * H) * 100:.1f}% of image")

overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
draw = ImageDraw.Draw(overlay)
draw.polygon(polygon, fill=(0, 255, 0, 70), outline=(0, 255, 0, 255), width=6)

for x, y in polygon:
    r = 14
    draw.ellipse((x - r, y - r, x + r, y + r), fill=(255, 50, 50, 255))

try:
    font = ImageFont.truetype("arial.ttf", 42)
except OSError:
    font = ImageFont.load_default()

label = f"id={ANN_ID}  cat={CAT_ID} ({CAT_NAME})"
tb = draw.textbbox((0, 0), label, font=font)
tw, th = tb[2] - tb[0], tb[3] - tb[1]
pad = 10
lx, ly = x_min, max(0, y_min - th - 2 * pad)
draw.rectangle((lx, ly, lx + tw + 2 * pad, ly + th + 2 * pad),
               fill=(0, 0, 0, 200))
draw.text((lx + pad, ly + pad), label, fill=(255, 255, 255, 255), font=font)

corners = [(0, 0), (W, 0), (W, H), (0, H)]
for cx, cy in corners:
    draw.ellipse((cx - 18, cy - 18, cx + 18, cy + 18),
                 fill=(255, 255, 0, 255))

out = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
out.save(OUT_PATH, "PNG")
print(f"Saved: {OUT_PATH}")
