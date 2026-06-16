import json
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
ANN_PATH = ROOT / "annotations" / "teeth_fdi_labels.json"
IMG_DIR = ROOT / "images"
OUT_DIR = ROOT / "visualizations" / "teeth_fdi_examples"
COUNT = 5

PALETTE = [
    (230, 25, 75),
    (60, 180, 75),
    (255, 225, 25),
    (0, 130, 200),
    (245, 130, 48),
    (145, 30, 180),
    (70, 240, 240),
    (240, 50, 230),
    (210, 245, 60),
    (250, 190, 190),
    (0, 128, 128),
    (230, 190, 255),
    (170, 110, 40),
    (255, 250, 200),
    (128, 0, 0),
    (170, 255, 195),
]


def load_font(size: int):
    for name in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def polygon_points(segmentation):
    if not segmentation:
        return []
    points = segmentation[0]
    return list(zip(points[0::2], points[1::2]))


def draw_label(draw, xy, text, font, fill):
    x, y = xy
    bbox = draw.textbbox((x, y), text, font=font)
    pad = 3
    draw.rectangle(
        (bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad),
        fill=(0, 0, 0, 170),
    )
    draw.text((x, y), text, font=font, fill=fill)


def render_bbox(image, anns, cat_names, out_path):
    canvas = image.convert("RGBA")
    overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font = load_font(22)

    for idx, ann in enumerate(anns):
        x, y, w, h = ann["bbox"]
        color = PALETTE[idx % len(PALETTE)]
        rgba = (*color, 255)
        draw.rectangle((x, y, x + w, y + h), outline=rgba, width=4)
        draw_label(draw, (x + 4, y + 4), cat_names[ann["category_id"]], font, rgba)

    out = Image.alpha_composite(canvas, overlay).convert("RGB")
    out.save(out_path)


def render_segmentation(image, anns, cat_names, out_path):
    canvas = image.convert("RGBA")
    overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font = load_font(22)

    for idx, ann in enumerate(anns):
        pts = polygon_points(ann.get("segmentation"))
        if not pts:
            continue
        color = PALETTE[idx % len(PALETTE)]
        draw.polygon(pts, fill=(*color, 70), outline=(*color, 255), width=4)
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        draw_label(
            draw,
            (min(xs) + 4, min(ys) + 4),
            cat_names[ann["category_id"]],
            font,
            (*color, 255),
        )

    out = Image.alpha_composite(canvas, overlay).convert("RGB")
    out.save(out_path)


def is_tooth_sized_annotation(ann, image_info):
    x, y, w, h = ann["bbox"]
    image_area = image_info["width"] * image_info["height"]

    # Some source FDI entries are very large panorama-level polygons. They are
    # valid source data, but they obscure tooth-level visualization examples.
    return (
        w > 0
        and h > 0
        and (w * h) / image_area <= 0.08
        and w / image_info["width"] <= 0.35
        and h / image_info["height"] <= 0.45
    )


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for old_file in list(OUT_DIR.glob("*_bbox.png")) + list(OUT_DIR.glob("*_segmentation.png")):
        old_file.unlink()

    data = json.loads(ANN_PATH.read_text(encoding="utf-8"))
    cat_names = {c["id"]: c["name"] for c in data["categories"]}
    images = {img["id"]: img for img in data["images"]}

    raw_counts = defaultdict(int)
    anns_by_image = defaultdict(list)
    for ann in data["annotations"]:
        image_info = images[ann["image_id"]]
        raw_counts[ann["image_id"]] += 1
        if is_tooth_sized_annotation(ann, image_info):
            anns_by_image[ann["image_id"]].append(ann)

    # Prefer near-complete panorama examples, so one image shows many teeth at once.
    selected = sorted(
        anns_by_image.items(),
        key=lambda item: (-len(item[1]), images[item[0]]["file_name"]),
    )[:COUNT]

    index_lines = []
    for n, (image_id, anns) in enumerate(selected, start=1):
        image_info = images[image_id]
        file_name = image_info["file_name"]
        img_path = IMG_DIR / file_name
        image = Image.open(img_path)

        bbox_out = OUT_DIR / f"{n:02d}_{Path(file_name).stem}_bbox.png"
        seg_out = OUT_DIR / f"{n:02d}_{Path(file_name).stem}_segmentation.png"

        render_bbox(image, anns, cat_names, bbox_out)
        render_segmentation(image, anns, cat_names, seg_out)

        index_lines.append(
            f"{n:02d}. {file_name} | image_id={image_id} | "
            f"drawn_teeth_annotations={len(anns)} | raw_annotations={raw_counts[image_id]}"
        )
        print(f"Saved: {bbox_out}")
        print(f"Saved: {seg_out}")

    index_path = OUT_DIR / "index.txt"
    index_path.write_text("\n".join(index_lines) + "\n", encoding="utf-8")
    print(f"Saved: {index_path}")


if __name__ == "__main__":
    main()
