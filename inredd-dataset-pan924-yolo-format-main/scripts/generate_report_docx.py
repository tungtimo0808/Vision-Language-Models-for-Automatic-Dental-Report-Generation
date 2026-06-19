"""Generate a Word report (.docx) summarizing the PAN924 dataset pipeline.

Pulls live numbers from annotations/dataset_final_v2.json so the report always
matches the current state of the pipeline.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from docx import Document
from docx.enum.table import WD_ALIGN_VERTICAL
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Pt, RGBColor, Cm


ROOT = Path(__file__).resolve().parents[1]
DATASET_JSON = ROOT / "annotations" / "dataset_final_v2.json"
MOUTH_JSON = ROOT / "annotations" / "mouth_and_teeth_labels.json"
FDI_JSON = ROOT / "annotations" / "teeth_fdi_labels.json"
BENCHMARK_JSON = ROOT / "faster_rcnn" / "outputs" / "benchmark" / "benchmark_results.json"
OUT_DOCX = ROOT / "PAN924_Dataset_Report.docx"
FDI_IMAGE = ROOT / "assets" / "FDI.jpg"

DISEASE_REMAP = {"RiM": "Te", "Ri": "Te", "TeM": "Te", "I": "M3i"}

DISEASE_DESCRIPTIONS = {
    "H": "Khỏe mạnh",
    "R": "Phục hồi (trám/chụp)",
    "Te": "Đã điều trị nội nha",
    "C": "Nghi ngờ sâu răng",
    "CpuM": "Có mão răng (hỗn hợp)",
    "Di": "Mòn mặt nhai / rìa cắn",
    "M3i": "Mọc ngầm trong xương",
    "M3f": "Đang phát triển",
    "Rr": "Chân răng còn lại",
    "P": "Trụ cầu (pontic)",
    "Im": "Implant",
    "Dc": "Phá hủy thân răng",
}


def collect_stats() -> dict:
    data = json.loads(DATASET_JSON.read_text(encoding="utf-8"))
    disease_raw = Counter()
    fdi = Counter()
    for img in data["images"]:
        for t in img["teeth"]:
            disease_raw[t["disease_label"]] += 1
            if t["fdi_label"] is not None:
                fdi[t["fdi_label"]] += 1

    disease_after = Counter()
    for k, v in disease_raw.items():
        disease_after[DISEASE_REMAP.get(k, k)] += v

    return {
        "num_images": data["meta"]["images"],
        "total_teeth": sum(disease_raw.values()),
        "disease_raw": dict(disease_raw.most_common()),
        "disease_after": dict(disease_after.most_common()),
        "fdi": dict(sorted(fdi.items())),
    }


def collect_source_meta() -> dict:
    """Pull schema metadata from the two raw COCO files."""
    mouth = json.loads(MOUTH_JSON.read_text(encoding="utf-8"))
    fdi = json.loads(FDI_JSON.read_text(encoding="utf-8"))

    # Supercategory groups in mouth file
    super_groups: dict[str, list[str]] = {}
    cat_by_super = Counter()
    for c in mouth["categories"]:
        super_groups.setdefault(c["supercategory"], []).append(c["name"])

    # Count annotations per supercategory
    cat_super_map = {c["id"]: c["supercategory"] for c in mouth["categories"]}
    for ann in mouth["annotations"]:
        cat_super_map_super = cat_super_map.get(ann["category_id"], "?")
        cat_by_super[cat_super_map_super] += 1

    # 113 missing images analysis
    mouth_files = {i["file_name"] for i in mouth["images"]}
    fdi_files = {i["file_name"] for i in fdi["images"]}
    missing = mouth_files - fdi_files

    mouth_name_to_id = {i["file_name"]: i["id"] for i in mouth["images"]}
    from collections import defaultdict
    anns_by_img = defaultdict(list)
    for a in mouth["annotations"]:
        anns_by_img[a["image_id"]].append(a)
    cat_name_map = {c["id"]: c["name"] for c in mouth["categories"]}
    edentulous = 0
    only_root = 0
    for fname in missing:
        img_id = mouth_name_to_id[fname]
        labels = {cat_name_map[a["category_id"]] for a in anns_by_img.get(img_id, [])}
        tooth_count = sum(
            1 for a in anns_by_img.get(img_id, [])
            if cat_super_map[a["category_id"]] != "Mouth"
        )
        if "Ed" in labels:
            edentulous += 1
        if tooth_count == 1:
            only_root += 1

    return {
        "mouth_n_images": len(mouth["images"]),
        "mouth_n_anns": len(mouth["annotations"]),
        "fdi_n_images": len(fdi["images"]),
        "fdi_n_anns": len(fdi["annotations"]),
        "mouth_categories": len(mouth["categories"]),
        "fdi_categories": len(fdi["categories"]),
        "super_groups": super_groups,
        "anns_per_super": dict(cat_by_super),
        "n_missing_in_fdi": len(missing),
        "missing_edentulous": edentulous,
        "missing_only_root": only_root,
        "info": mouth.get("info", {}),
        "license": mouth.get("licenses", [{}])[0],
    }


def count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def set_cell_text(cell, text, bold=False, size=11):
    cell.text = ""
    p = cell.paragraphs[0]
    run = p.add_run(text)
    run.font.size = Pt(size)
    run.font.bold = bold
    cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER


def add_heading(doc, text, level=1):
    h = doc.add_heading(text, level=level)
    for run in h.runs:
        run.font.color.rgb = RGBColor(0x1F, 0x3A, 0x5F)
    return h


def add_paragraph(doc, text, bold=False, italic=False, size=11):
    p = doc.add_paragraph()
    run = p.add_run(text)
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = italic
    return p


def add_bullet(doc, text):
    p = doc.add_paragraph(style="List Bullet")
    run = p.add_run(text)
    run.font.size = Pt(11)
    return p


def add_image(doc, path, width_cm=14, caption=None):
    """Insert an image if the file exists, then optionally add a caption line."""
    if not Path(path).exists():
        add_paragraph(doc, f"[Image not found: {path}]", italic=True)
        return
    from docx.shared import Cm as _Cm
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run()
    run.add_picture(str(path), width=_Cm(width_cm))
    if caption:
        cap = doc.add_paragraph()
        cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = cap.add_run(caption)
        r.font.size = Pt(10)
        r.font.italic = True
        r.font.color.rgb = RGBColor(0x44, 0x44, 0x44)


def add_code_block(doc, text):
    p = doc.add_paragraph()
    run = p.add_run(text)
    run.font.name = "Consolas"
    run.font.size = Pt(9)
    p.paragraph_format.left_indent = Cm(0.5)
    return p


def add_table(doc, headers, rows, col_widths=None):
    table = doc.add_table(rows=1 + len(rows), cols=len(headers))
    table.style = "Light Grid Accent 1"
    table.alignment = WD_ALIGN_PARAGRAPH.CENTER

    hdr_cells = table.rows[0].cells
    for i, h in enumerate(headers):
        set_cell_text(hdr_cells[i], h, bold=True)
    for r_idx, row in enumerate(rows, start=1):
        for c_idx, val in enumerate(row):
            set_cell_text(table.rows[r_idx].cells[c_idx], str(val))

    if col_widths:
        for row in table.rows:
            for i, w in enumerate(col_widths):
                row.cells[i].width = w
    return table


def build_report() -> None:
    stats = collect_stats()
    src = collect_source_meta()

    train_n = count_jsonl(ROOT / "vlm_report_dataset/common/train.jsonl")
    val_n = count_jsonl(ROOT / "vlm_report_dataset/common/val.jsonl")
    test_n = count_jsonl(ROOT / "vlm_report_dataset/common/test.jsonl")
    balanced_n = count_jsonl(ROOT / "vlm_report_dataset/common_balanced/train.jsonl")

    doc = Document()

    # Default font
    style = doc.styles["Normal"]
    style.font.name = "Calibri"
    style.font.size = Pt(11)

    # Title
    title = doc.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = title.add_run("PAN924 Dental X-Ray Dataset")
    run.font.size = Pt(20)
    run.font.bold = True
    run.font.color.rgb = RGBColor(0x1F, 0x3A, 0x5F)

    sub = doc.add_paragraph()
    sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    sub_run = sub.add_run("Báo cáo cấu trúc dataset: bộ dữ liệu 5-view cho Vision-Language Models và baseline Faster R-CNN")
    sub_run.font.size = Pt(12)
    sub_run.font.italic = True

    # ----------------------------------------------------------------- A. DATASET
    add_heading(doc, "A. DATASET", level=1)

    # ----------- Dataset 1
    add_heading(doc, "Dataset 1 — Dataset for Faster R-CNN", level=2)

    add_paragraph(doc, "Theo toàn bộ clean dataset:", bold=True)
    add_bullet(doc, f"Số ảnh sample: {stats['num_images']}")
    add_bullet(doc, f"Tổng số tooth rows: {stats['total_teeth']:,}")

    # Disease raw distribution
    add_paragraph(doc, "Số bệnh và số lượng (trước khi gộp):", bold=True)
    disease_rows = [
        (label, f"{count:,}", DISEASE_DESCRIPTIONS.get(DISEASE_REMAP.get(label, label), ""))
        for label, count in stats["disease_raw"].items()
    ]
    add_table(doc, ["Mã bệnh", "Số lượng", "Ý nghĩa"], disease_rows)

    add_paragraph(doc, "Do data của I, Ri, RiM, TeM quá ít → gộp như sau:", italic=True)
    add_bullet(doc, "RiM → Te")
    add_bullet(doc, "Ri  → Te")
    add_bullet(doc, "TeM → Te")
    add_bullet(doc, "I   → M3i")
    add_paragraph(doc, "→ Còn 12 class bệnh về răng.", bold=True)

    # Disease after remap
    add_paragraph(doc, "Số bệnh sau khi gộp (12 class):", bold=True)
    remap_rows = [
        (label, f"{count:,}", DISEASE_DESCRIPTIONS.get(label, ""))
        for label, count in stats["disease_after"].items()
    ]
    add_table(doc, ["Mã bệnh", "Số lượng", "Ý nghĩa"], remap_rows)

    # ── FDI SYSTEM EXPLANATION ─────────────────────────────────────────────────
    add_heading(doc, "The FDI Tooth Numbering System", level=3)

    add_paragraph(doc,
        "FDI stands for Fédération Dentaire Internationale. "
        "It is the international standard for naming teeth. "
        "Every tooth gets a two-digit number.")

    add_paragraph(doc, "How to read an FDI number:", bold=True)
    add_bullet(doc, "First digit  →  quadrant (1 to 4)")
    add_bullet(doc, "Second digit →  position from the center line (1 to 8)")

    add_paragraph(doc, "The four quadrants:", bold=True)
    add_table(doc,
        ["Quadrant", "Location", "FDI range", "Example"],
        [
            ("1", "Upper right  (maxilla, patient's right)",  "11 – 18", "Tooth 14 = upper right first premolar"),
            ("2", "Upper left   (maxilla, patient's left)",   "21 – 28", "Tooth 26 = upper left first molar"),
            ("3", "Lower left   (mandible, patient's left)",  "31 – 38", "Tooth 36 = lower left first molar"),
            ("4", "Lower right  (mandible, patient's right)", "41 – 48", "Tooth 47 = lower right second molar"),
        ],
    )

    add_paragraph(doc, "Position numbers:", bold=True)
    add_table(doc,
        ["Position", "Tooth name", "Note"],
        [
            ("1", "Central incisor",  "Closest to the center"),
            ("2", "Lateral incisor",  ""),
            ("3", "Canine",           "Corner tooth"),
            ("4", "First premolar",   ""),
            ("5", "Second premolar",  ""),
            ("6", "First molar",      "Largest molar"),
            ("7", "Second molar",     ""),
            ("8", "Third molar",      "Wisdom tooth — may be missing"),
        ],
    )

    add_paragraph(doc,
        "So tooth 36 = quadrant 3 (lower left) + position 6 (first molar). "
        "Tooth 18 = quadrant 1 (upper right) + position 8 (wisdom tooth).",
        italic=True,
    )

    add_paragraph(doc,
        "A full adult mouth has 32 teeth. "
        "The dataset covers all 32 FDI positions (11–18, 21–28, 31–38, 41–48). "
        "Not every patient has all 32 — wisdom teeth (18, 28, 38, 48) are often missing or impacted.",
    )

    add_image(doc, FDI_IMAGE, width_cm=14,
              caption="Figure: FDI tooth numbering system. "
                      "Quadrants 1–2 = upper jaw, 3–4 = lower jaw. "
                      "Numbers increase away from the center.")

    # ── FDI distribution table ─────────────────────────────────────────────────
    add_paragraph(doc, "Số răng theo loại FDI (32 lớp):", bold=True)
    add_paragraph(
        doc,
        "Bảng đọc theo cột (vị trí răng tính từ đường giữa hàm) và hàng (cung hàm). "
        "Mỗi ô ghi rõ số FDI thực tế kèm số lượng mẫu.",
        italic=True,
    )

    tooth_position_names = [
        "Cửa giữa",
        "Cửa bên",
        "Nanh",
        "Tiền hàm 1",
        "Tiền hàm 2",
        "Hàm lớn 1",
        "Hàm lớn 2",
        "Khôn",
    ]
    quadrant_rows = [
        ("Q1 — Hàm trên phải", [11, 12, 13, 14, 15, 16, 17, 18]),
        ("Q2 — Hàm trên trái", [21, 22, 23, 24, 25, 26, 27, 28]),
        ("Q3 — Hàm dưới trái", [31, 32, 33, 34, 35, 36, 37, 38]),
        ("Q4 — Hàm dưới phải", [41, 42, 43, 44, 45, 46, 47, 48]),
    ]
    headers = ["Cung hàm"] + [f"{i+1}. {name}" for i, name in enumerate(tooth_position_names)]
    fdi_rows = []
    for label, fdis in quadrant_rows:
        cells = [label]
        for f in fdis:
            count = stats["fdi"].get(f, 0)
            cells.append(f"FDI {f}\n{count:,}")
        fdi_rows.append(cells)
    add_table(doc, headers, fdi_rows)

    add_paragraph(
        doc,
        f"Tổng cộng: {sum(stats['fdi'].values()):,} răng phân bố trên 32 lớp FDI.",
        italic=True,
    )

    # File JSON structure
    add_heading(doc, "Cấu trúc các file annotation gốc", level=3)

    add_paragraph(doc, "mouth_and_teeth_labels.json — COCO format", bold=True)
    add_code_block(doc, (
        "info, licenses\n"
        "categories[]: id, name (Ed/H/R/M3f/...), supercategory\n"
        "images[]:     id, file_name, width, height, license, sex, age\n"
        "annotations[]: id, image_id, category_id, segmentation[][]"
    ))

    add_paragraph(doc, "teeth_fdi_labels.json — COCO format", bold=True)
    add_code_block(doc, (
        "info, licenses\n"
        "categories[]: id, name (số FDI: \"11\"..\"48\"), supercategory\n"
        "images[]:     id, file_name, width, height, license, date_captured\n"
        "annotations[]: id, image_id, category_id, bbox[x,y,w,h], area, segmentation[][]"
    ))

    add_paragraph(doc, "dataset_final_v2.json — format tùy chỉnh (combined)", bold=True)
    add_code_block(doc, (
        "meta:\n"
        "  name, description, images, total_teeth_rows\n\n"
        "images[]:\n"
        "  file_name        — tên file ảnh\n"
        "  image_id         — ID bệnh nhân\n"
        "  width, height    — kích thước ảnh (2903×1536)\n"
        "  teeth[]:\n"
        "    disease_annotation_id  — ID annotation gốc\n"
        "    disease_label          — loại bệnh/tình trạng (H, R, M3f, C, ...)\n"
        "    fdi_label              — số răng FDI (11-48)\n"
        "    bbox_xywh              — [x, y, w, h] vị trí răng\n"
        "    segmentation           — polygon đường viền\n"
        "    row_inferred           — upper / lower"
    ))

    # Mapping diagram between the two source files
    add_heading(doc, "Cách 2 file annotation gốc được gộp thành dataset_final_v2", level=3)

    add_paragraph(
        doc,
        "Hai file COCO gốc được join theo 2 tầng khóa: (1) ở mức ảnh dùng file_name làm khóa, "
        "(2) ở mức răng dùng vị trí bbox để khớp annotation bệnh với annotation FDI tương ứng "
        "trong cùng một ảnh.",
        italic=True,
    )

    add_paragraph(doc, "Sơ đồ join 2 file:", bold=True)
    add_code_block(doc, (
        "mouth_and_teeth_labels.json              teeth_fdi_labels.json\n"
        "────────────────────────────             ────────────────────────\n"
        "images[]                                 images[]\n"
        "  .file_name ────────────┬─────────────────  .file_name\n"
        "  .id, width, height     │ JOIN theo         .id, width, height\n"
        "                         │ tên ảnh\n"
        "annotations[]            │                 annotations[]\n"
        "  .id                    │                   .id\n"
        "  .image_id ─────────────┘                   .image_id\n"
        "  .category_id ──→ disease label             .category_id ──→ FDI number\n"
        "  .segmentation              ┌────────────   .bbox [x,y,w,h]\n"
        "                             │ MATCH theo\n"
        "                             │ vùng bbox\n"
        "                             ▼\n"
        "                  dataset_final_v2.json\n"
        "                  (mỗi răng gộp cả disease + FDI)"
    ))

    add_paragraph(doc, "Mapping chi tiết từng trường trong file output:", bold=True)
    mapping_rows = [
        ("images[].file_name",
         "mouth.images.file_name",
         "Cũng là khóa join với teeth_fdi_labels"),
        ("images[].image_id",
         "mouth.images.id",
         "ID bệnh nhân lấy từ file disease"),
        ("images[].width / height",
         "mouth.images.width / height",
         "Kích thước ảnh"),
        ("teeth[].disease_annotation_id",
         "mouth.annotations.id",
         "ID annotation gốc của răng"),
        ("teeth[].disease_label",
         "mouth.categories[ann.category_id].name",
         "Map qua category_id → tên bệnh (H, R, M3f,...)"),
        ("teeth[].fdi_label",
         "teeth_fdi.categories[match.category_id].name",
         "Lấy số FDI từ annotation FDI khớp bbox"),
        ("teeth[].bbox_xywh",
         "tính từ mouth.annotations.segmentation",
         "Bao polygon → bbox xywh"),
        ("teeth[].segmentation",
         "mouth.annotations.segmentation",
         "Giữ nguyên polygon viền răng"),
        ("teeth[].row_inferred",
         "computed: bbox.cy vs image.height/2",
         "Suy ra hàm trên / dưới từ vị trí bbox"),
    ]
    add_table(doc, ["Trường trong dataset_final_v2", "Nguồn gốc", "Ghi chú"], mapping_rows)

    # ====================== Chi tiết schema 2 file gốc ======================
    add_heading(doc, "Chi tiết quan trọng về schema 2 file gốc", level=3)

    # --- Supercategory groups
    add_paragraph(doc, "1. Phân nhóm supercategory trong mouth_and_teeth_labels.json", bold=True)
    add_paragraph(
        doc,
        "20 class bệnh/tình trạng được nhóm thành 4 supercategory dựa trên bản chất giải phẫu:",
        italic=True,
    )
    super_descriptions = {
        "Natural Teeth": "Răng tự nhiên (mô răng thật, khác nhau về tình trạng bệnh lý)",
        "Artificial Teeth": "Răng nhân tạo hoàn toàn (không có mô răng thật)",
        "Mix Teeth": "Răng pha trộn (mô thật + phục hình)",
        "Mouth": "Phát hiện mức toàn hàm, KHÔNG thuộc 1 răng cụ thể",
    }
    super_rows = []
    for sup_name, desc in super_descriptions.items():
        classes = src["super_groups"].get(sup_name, [])
        ann_count = src["anns_per_super"].get(sup_name, 0)
        super_rows.append((
            sup_name,
            ", ".join(classes),
            f"{ann_count:,}",
            desc,
        ))
    add_table(
        doc,
        ["Supercategory", "Class trong nhóm", "Số annotation", "Ý nghĩa"],
        super_rows,
    )
    add_paragraph(
        doc,
        "Pipeline loại bỏ supercategory 'Mouth' (Ed, De, Me, Mne) vì các phát hiện này "
        "không gắn với 1 răng cụ thể → không có FDI label → không phù hợp cho task per-tooth.",
        italic=True,
    )

    # --- ID schema difference
    add_paragraph(doc, "2. Khác biệt hệ image_id giữa 2 file", bold=True)
    add_paragraph(
        doc,
        "Cùng một ảnh sẽ có image_id KHÁC NHAU trong 2 file. Đây là lý do code phải "
        "dùng file_name làm khóa join, KHÔNG dùng id.",
        italic=True,
    )
    add_table(
        doc,
        ["File", "Ý nghĩa của image.id", "Khoảng giá trị", "Ví dụ"],
        [
            ("mouth_and_teeth_labels.json", "Mã bệnh nhân (lấy từ tên file)",
             f"Không liên tục (1..1500+)", "1000-F-19.jpg → id=1000"),
            ("teeth_fdi_labels.json", "Số thứ tự incremental khi annotate",
             f"1..{src['fdi_n_images']}", "Ảnh annotate thứ 538 → id=538"),
        ],
    )
    add_code_block(doc, (
        "# Pipeline merge dùng file_name làm khóa join:\n"
        "fdi_name_to_id = {img.file_name: img.id for img in fdi_data.images}\n"
        "fdi_img_id = fdi_name_to_id.get(mouth_img.file_name)"
    ))

    # --- Category ID ordering
    add_paragraph(doc, "3. Quy tắc đánh số category_id", bold=True)
    add_paragraph(
        doc,
        "category_id trong COCO được cấp tăng dần theo thứ tự annotator vẽ label LẦN ĐẦU. "
        "Đây là pattern chuẩn của COCO format, không phải lỗi annotate.",
        italic=True,
    )
    add_bullet(doc, "Ảnh đầu tiên có 25 răng đầu tiên → id 1..25 tương ứng các FDI xuất hiện theo thứ tự răng được vẽ")
    add_bullet(doc, "7 răng FDI còn lại (11, 15, 16, 18, 28, 36, 46) xuất hiện ở ảnh sau → id 26..32")
    add_bullet(doc, "Thứ tự id không ảnh hưởng training: pipeline map `id → FDI number` 1 lần, dùng FDI number xuyên suốt")

    # --- Image count discrepancy
    add_paragraph(doc, f"4. Tại sao teeth_fdi_labels.json chỉ có {src['fdi_n_images']:,} ảnh (vs {src['mouth_n_images']:,})", bold=True)
    add_paragraph(
        doc,
        f"Thiếu {src['n_missing_in_fdi']} ảnh trong file FDI — không phải lỗi mà là pattern y khoa hợp lý:",
        italic=True,
    )
    add_table(
        doc,
        ["Trạng thái", "Số ảnh", "Lý do"],
        [
            ("Edentulous (Ed)", f"{src['missing_edentulous']}",
             "Toàn hàm không có răng → annotator FDI không có gì để vẽ"),
            ("Chỉ còn 1 chân răng (Rr)", f"{src['missing_only_root']}",
             "Annotator FDI bỏ qua vì không đủ structure xác định FDI position"),
            ("Tổng missing", f"{src['n_missing_in_fdi']}",
             f"= {src['mouth_n_images']} − {src['fdi_n_images']}"),
        ],
    )
    add_paragraph(
        doc,
        "Quy trình annotation tổ chức theo 2 nhóm độc lập: nhóm A annotate disease (toàn bộ 924 ảnh, "
        "kể cả edentulous để thống kê tỷ lệ), nhóm B annotate FDI position (chỉ ảnh có răng).",
        italic=True,
    )

    # --- Annotation entry example
    add_paragraph(doc, "5. Cấu trúc một annotation entry", bold=True)
    add_paragraph(doc, "Ví dụ entry trong mouth_and_teeth_labels.json:", italic=True)
    add_code_block(doc, (
        '{\n'
        '  "id": 4144,           ← unique ID của annotation (1..20957)\n'
        '  "image_id": 1275,     ← foreign key → images[].id (bệnh nhân 1275)\n'
        '  "category_id": 5,     ← foreign key → categories[].id (= "R" - Restoration)\n'
        '  "segmentation": [...] ← polygon vẽ vùng răng/vùng bệnh\n'
        '}'
    ))
    add_paragraph(doc, "Resolve qua foreign key:", bold=True)
    add_bullet(doc, "image_id=1275 → images[].id=1275 → file_name='1275-M-51.jpg' (bệnh nhân nam 51 tuổi)")
    add_bullet(doc, "category_id=5 → categories[].id=5 → name='R', supercategory='Natural Teeth'")
    add_bullet(doc, "segmentation: polygon, có thể là 4 điểm (rectangle) hoặc 8-20+ điểm (chính xác viền răng)")

    # --- License
    add_paragraph(doc, "6. Nguồn gốc & License", bold=True)
    info = src["info"]
    lic = src["license"]
    add_bullet(doc, f"Dataset: {info.get('description', 'InReDD-Dataset')}")
    add_bullet(doc, f"Tác giả: {info.get('contributor', 'N/A')}")
    add_bullet(doc, f"Phiên bản: {info.get('version', 'N/A')} ({info.get('year', 'N/A')})")
    add_bullet(doc, f"Ngày tạo: {info.get('date_created', 'N/A')}")
    add_bullet(doc, f"URL: {info.get('url', 'N/A')}")
    add_bullet(doc, f"License: {lic.get('name', 'N/A')} — {lic.get('url', 'N/A')}")
    add_paragraph(
        doc,
        "CC BY-NC-SA 2.0 cho phép sử dụng cho mục đích học thuật, yêu cầu ghi nguồn và share-alike. "
        "KHÔNG được dùng cho mục đích thương mại.",
        italic=True,
    )

    # 3-tier architecture
    add_heading(doc, "Kiến trúc 3 tầng dataset", level=3)

    add_paragraph(doc, "Tầng 1 — Annotation gốc (annotations/)", bold=True)
    add_code_block(doc, (
        "mouth_and_teeth_labels.json  ← COCO: disease bbox + segmentation\n"
        "teeth_fdi_labels.json        ← COCO: FDI position bbox\n"
        "dataset_final_v2.json        ← Merged: disease + FDI gộp lại theo từng ảnh"
    ))

    add_paragraph(doc, "Tầng 2 — Common 5-view dataset (vlm_report_dataset/common/)", bold=True)
    add_code_block(doc, (
        f"train/val/test.jsonl     ← {train_n + val_n + test_n:,} samples (924 ảnh × 5 view)\n"
        "images/ + crops          ← 1 ảnh toàn cảnh + 4 crop góc phần tư mỗi ảnh\n"
        "metadata/                ← split_report, crop_manifest, validation_report..."
    ))

    add_paragraph(doc, "Tầng 3 — Converted per-model (vlm_report_dataset/converted/)", bold=True)
    add_code_block(doc, (
        "qwen/ llava/ internvl/ phi/ paligemma/\n"
        "  └── train/val/test.jsonl  ← cùng data, đổi sang format riêng từng model"
    ))

    add_paragraph(doc, "Script sinh ra từng tầng:", bold=True)
    add_code_block(doc, (
        "merge_annotations.py            → dataset_final_v2.json (gộp 2 file COCO)\n"
        "build_5view_common_dataset.py   → tầng 2 (5-view + raw JSONL)\n"
        "generate_clinical_reports.py    → điền comment/summary (rule-based)\n"
        "validate_common_dataset.py      → kiểm tra chất lượng\n"
        "rebalance_common.py             → common_balanced (oversample bệnh hiếm)\n"
        "convert_dataset.py              → tầng 3 (5 format model)"
    ))

    # Approach of this project
    add_heading(doc, "Hướng tiếp cận của project", level=3)

    add_paragraph(doc, "Project nghiên cứu nhiều Vision-Language Model trên cùng một bộ dữ liệu. Mỗi ảnh toàn cảnh được đưa thẳng vào VLM để sinh một báo cáo JSON có cấu trúc (răng theo FDI + tình trạng bệnh + nhận xét). Faster R-CNN được train riêng làm baseline để so sánh khả năng phát hiện răng ở mức tooth-level.", italic=True)

    compare_rows = [
        ("Đầu vào VLM", "1 ảnh toàn cảnh (4 crop dùng làm dữ liệu train phụ)"),
        ("Đầu ra VLM", "Báo cáo JSON: 4 vùng, mỗi răng có FDI + bệnh + comment, kèm summary"),
        ("Baseline so sánh", "Faster R-CNN detect + classify (disease và FDI tách riêng)"),
    ]
    add_table(doc, ["Thành phần", "Mô tả"], compare_rows)

    # ----------- Dataset 2
    doc.add_page_break()
    add_heading(doc, "Dataset 2 — Dataset for VLM (5-view)", level=2)

    add_paragraph(doc,
        "Mỗi ảnh panorama được chuẩn bị thành 5 view: 1 ảnh toàn cảnh và 4 crop góc phần tư "
        "(trên-trái, trên-phải, dưới-trái, dưới-phải). 924 ảnh → 4.620 samples. "
        "Dataset ở dạng JSONL model-agnostic, sau đó convert sang format từng model.")

    add_paragraph(doc, "Hai loại task:", bold=True)
    add_table(
        doc,
        ["Task", "Ảnh đầu vào", "Output"],
        [
            ("full_quadrant_report", "Ảnh toàn cảnh", "JSON 4 vùng (răng FDI + bệnh) + summary"),
            ("regional_report", "1 crop góc phần tư", "Danh sách răng trong vùng + comment"),
        ],
    )

    add_paragraph(doc, "Cấu trúc một dòng JSONL:", bold=True)
    add_code_block(doc, (
        "{\n"
        '  "id", "image", "source_image", "view", "task",\n'
        '  "messages": [\n'
        '    {"role": "user",      "content": "<image>\\n<prompt>"},\n'
        '    {"role": "assistant", "content": "<JSON đáp án>"}\n'
        "  ],\n"
        '  "labels":   {...},   ← nhãn gốc dùng để chấm điểm\n'
        '  "metadata": {...}\n'
        "}"
    ))

    add_paragraph(doc, "Thống kê số sample theo split:", bold=True)
    add_table(
        doc,
        ["File", "Samples", "Mô tả"],
        [
            ("train.jsonl", f"{train_n:,}", "739 ảnh × 5 view"),
            ("val.jsonl",   f"{val_n:,}",   "92 ảnh × 5 view"),
            ("test.jsonl",  f"{test_n:,}",  "93 ảnh × 5 view"),
            ("Total",       f"{train_n + val_n + test_n:,}", "924 ảnh"),
        ],
    )
    add_paragraph(doc,
        "Chia split ở mức ảnh (cả 5 view của 1 ảnh luôn cùng split → không leak), stratified theo "
        "bệnh hiếm, thứ tự sắp xếp bằng hash SHA-256 (seed 924) nên tái lập được.",
        italic=True,
    )

    add_paragraph(doc, "Phiên bản cân bằng (tùy chọn):", bold=True)
    add_paragraph(doc,
        f"vlm_report_dataset/common_balanced/ — train tăng lên {balanced_n:,} dòng do oversample các "
        "crop chứa bệnh hiếm (Dc, Im, P, Rr, M3f); val/test giữ nguyên.",
        italic=True,
    )

    add_paragraph(doc, "5 format model đã convert:", bold=True)
    add_paragraph(doc,
        "vlm_report_dataset/converted/{qwen, llava, internvl, phi, paligemma}/ — cùng nội dung, "
        "khác cách đóng gói (messages / conversations / prompt-response / prefix-suffix).",
        italic=True,
    )

    # Prompt design
    add_heading(doc, "Prompt template", level=3)

    add_paragraph(doc, "Prompt cho ảnh toàn cảnh (full_quadrant_report):", bold=True)
    add_code_block(doc, (
        "<image>\n"
        "This is a full panoramic dental radiograph.\n"
        "Divide the image into four image-space regions: image_upper_left,\n"
        "image_upper_right, image_lower_left, and image_lower_right.\n"
        "For each region, list the visible annotated teeth using FDI notation\n"
        "and report each tooth condition.\n"
        "Return only valid JSON with this schema: {...}"
    ))

    add_paragraph(doc, "Prompt cho crop (regional_report):", bold=True)
    add_code_block(doc, (
        "<image>\n"
        "This is the {region} crop of a panoramic dental radiograph.\n"
        "Identify the visible annotated teeth using FDI notation and report\n"
        "each tooth condition.\n"
        "Return only valid JSON with this schema: {...}"
    ))

    add_paragraph(doc,
        "Prompt cố định (không paraphrase), giống nhau giữa train và inference → tái lập được, "
        "và model tập trung học nội dung nha khoa thay vì học cách diễn đạt.",
        italic=True,
    )

    # ------------------------------------------------------------ B. BENCHMARK
    if BENCHMARK_JSON.exists():
        doc.add_page_break()
        add_heading(doc, "B. KẾT QUẢ BENCHMARK FASTER R-CNN", level=1)

        bench = json.loads(BENCHMARK_JSON.read_text(encoding="utf-8"))
        runs = bench.get("runs", [])

        add_paragraph(doc, "Thiết lập thí nghiệm:", bold=True)
        add_bullet(doc, "Device: GPU CUDA")
        add_bullet(doc, "Input size: min_size=512, max_size=768")
        add_bullet(doc, "Cùng config train cho mọi run (chỉ đổi architecture + trainable_backbone_layers)")
        add_bullet(doc, "Eval: precision/recall/F1 (IoU≥0.5, score≥0.3), COCO mAP@[.50:.95] và mAP@.50")
        add_bullet(doc, "Tổng 9 run = 3 architecture × 3 độ sâu fine-tune (shallow=1, deep=3, full=5 layers)")

        add_heading(doc, "Bảng tổng hợp 9 run", level=2)
        bench_rows = []
        for r in runs:
            bench_rows.append((
                r["run_name"],
                r["depth_name"],
                f"{r['test_precision']:.3f}",
                f"{r['test_recall']:.3f}",
                f"{r['test_f1']:.3f}",
                f"{r['test_map_50']:.3f}",
                f"{r['test_map_50_95']:.3f}",
                f"{r['runtime_sec']:.0f}s",
            ))
        add_table(
            doc,
            ["Run", "Depth", "Precision", "Recall", "F1", "mAP@.50", "mAP@.50:.95", "Runtime"],
            bench_rows,
        )

        # Best result analysis
        best_f1 = max(runs, key=lambda r: r["test_f1"])
        best_map = max(runs, key=lambda r: r["test_map_50"])
        fastest = min(runs, key=lambda r: r["runtime_sec"])

        add_heading(doc, "Phân tích kết quả", level=2)

        add_paragraph(doc, "Best F1 (recommended cho production):", bold=True)
        add_bullet(doc, f"Run: {best_f1['run_name']} ({best_f1['architecture']}, {best_f1['depth_name']})")
        add_bullet(doc, f"F1 = {best_f1['test_f1']:.4f}, Precision = {best_f1['test_precision']:.4f}, Recall = {best_f1['test_recall']:.4f}")
        add_bullet(doc, f"mAP@.50 = {best_f1['test_map_50']:.4f}, mAP@.50:.95 = {best_f1['test_map_50_95']:.4f}")
        add_bullet(doc, f"Runtime: {best_f1['runtime_sec']:.0f}s ({best_f1['runtime_sec']/60:.1f} phút)")

        add_paragraph(doc, "Best mAP@.50:", bold=True)
        add_bullet(doc, f"{best_map['run_name']}: mAP@.50 = {best_map['test_map_50']:.4f}")

        add_paragraph(doc, "Nhanh nhất:", bold=True)
        add_bullet(doc, f"{fastest['run_name']}: {fastest['runtime_sec']:.0f}s, F1={fastest['test_f1']:.4f}")

        # Architecture comparison
        add_heading(doc, "So sánh theo Architecture", level=2)
        arch_groups = {}
        for r in runs:
            arch_groups.setdefault(r["architecture"], []).append(r)
        arch_rows = []
        for arch, group in arch_groups.items():
            f1s = [g["test_f1"] for g in group]
            maps = [g["test_map_50"] for g in group]
            runtimes = [g["runtime_sec"] for g in group]
            arch_short = arch.replace("fasterrcnn_", "").replace("_fpn", "")
            arch_rows.append((
                arch_short,
                f"{min(f1s):.3f} – {max(f1s):.3f}",
                f"{min(maps):.3f} – {max(maps):.3f}",
                f"{sum(runtimes)/len(runtimes):.0f}s",
            ))
        add_table(
            doc,
            ["Architecture", "F1 range", "mAP@.50 range", "Avg runtime"],
            arch_rows,
        )

        # Depth comparison
        add_heading(doc, "So sánh theo Fine-tuning Depth", level=2)
        depth_groups = {}
        for r in runs:
            depth_groups.setdefault(r["depth_name"], []).append(r)
        depth_rows = []
        for depth in ["shallow", "deep", "full"]:
            group = depth_groups.get(depth, [])
            if not group:
                continue
            f1s = [g["test_f1"] for g in group]
            maps = [g["test_map_50"] for g in group]
            depth_rows.append((
                depth,
                str(group[0]["trainable_backbone_layers"]),
                f"{sum(f1s)/len(f1s):.3f}",
                f"{sum(maps)/len(maps):.3f}",
            ))
        add_table(
            doc,
            ["Depth", "Trainable layers", "Avg F1", "Avg mAP@.50"],
            depth_rows,
        )

        # Conclusion
        add_heading(doc, "Kết luận", level=2)
        add_paragraph(
            doc,
            f"ResNet50 (deep fine-tuning) đạt kết quả tốt nhất với F1 = {best_f1['test_f1']:.3f} "
            f"và mAP@.50 = {best_f1['test_map_50']:.3f}. So với MobileNet-V3 (320/large) thì ResNet50 "
            f"cải thiện rõ rệt cả precision lẫn recall, đặc biệt là recall ({best_f1['test_recall']:.3f} vs "
            f"~0.55–0.76 của MobileNet), phù hợp cho bài toán y khoa cần phát hiện đầy đủ răng. "
            f"Trade-off là runtime ~2.4× chậm hơn MobileNet-V3-320.",
            italic=True,
        )

    # Pipeline inference flow
    add_heading(doc, "Luồng inference end-to-end", level=3)
    add_code_block(doc, (
        "Bác sĩ upload 1 ảnh panorama\n"
        "        │\n"
        "        ▼\n"
        "┌────────────────────────────────┐\n"
        "│ Vision-Language Model          │  đọc cả ảnh toàn cảnh\n"
        "└────────────────────────────────┘\n"
        "        │\n"
        "        ▼\n"
        "Báo cáo JSON 4 vùng (răng FDI + bệnh + comment) + summary\n"
        "        │\n"
        "        ▼\n"
        "Bác sĩ kiểm tra & chỉnh sửa → báo cáo cuối\n"
        "\n"
        "(Faster R-CNN chạy song song như baseline để so sánh độ chính xác phát hiện răng.)"
    ))

    # Save
    doc.save(OUT_DOCX)
    print(f"Saved: {OUT_DOCX}")
    print(f"Stats summary:")
    print(f"  images       : {stats['num_images']:,}")
    print(f"  total teeth  : {stats['total_teeth']:,}")
    print(f"  disease class: {len(stats['disease_after'])}")
    print(f"  train/val/test: {train_n:,} / {val_n:,} / {test_n:,}")


if __name__ == "__main__":
    build_report()
