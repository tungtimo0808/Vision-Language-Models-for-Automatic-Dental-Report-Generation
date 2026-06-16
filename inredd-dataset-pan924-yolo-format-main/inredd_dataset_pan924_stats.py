"""
Stats script for inredd_dataset_v1
"""

import os
import json
from collections import Counter

import math
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Paths
images_dir = "./images"
mouth_labels_path = "./annotations/mouth_and_teeth_labels.json"
teeth_labels_path = "./annotations/teeth_fdi_labels.json"
stats_dir = Path("./stats")
stats_dir.mkdir(exist_ok=True)

# List all images
image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(".jpg")]
print(f"Total images: {len(image_files)}")

# Load annotation files
def load_annotations(json_path):
    with open(json_path, "r") as f:
        return json.load(f)

mouth_ann = load_annotations(mouth_labels_path)
teeth_ann = load_annotations(teeth_labels_path)

# Count images with annotations
mouth_image_ids = {img['file_name'] for img in mouth_ann.get('images', [])}
teeth_image_ids = {img['file_name'] for img in teeth_ann.get('images', [])}

print(f"Images with mouth/teeth annotations: {len(mouth_image_ids)}")
print(f"Images with teeth FDI annotations: {len(teeth_image_ids)}")

# Count categories
mouth_categories = [cat['name'] for cat in mouth_ann.get('categories', [])]
teeth_categories = [cat['name'] for cat in teeth_ann.get('categories', [])]
print(f"Mouth/Teeth categories: {mouth_categories}")
print(f"Teeth FDI categories: {teeth_categories}")

# Count annotations per category (for mouth_and_teeth_labels.json)
mouth_ann_counter = Counter()
for ann in mouth_ann.get('annotations', []):
    cat_id = ann['category_id']
    cat_name = next((cat['name'] for cat in mouth_ann['categories'] if cat['id'] == cat_id), None)
    if cat_name:
        mouth_ann_counter[cat_name] += 1

print("Annotations per category (mouth_and_teeth_labels):")
for cat, count in mouth_ann_counter.items():
    print(f"  {cat}: {count}")

# Count annotations per category (for teeth_fdi_labels.json)
teeth_ann_counter = Counter()
for ann in teeth_ann.get('annotations', []):
    cat_id = ann['category_id']
    cat_name = next((cat['name'] for cat in teeth_ann['categories'] if cat['id'] == cat_id), None)
    if cat_name:
        teeth_ann_counter[cat_name] += 1

print("Annotations per category (teeth_fdi_labels):")
for cat, count in teeth_ann_counter.items():
    print(f"  {cat}: {count}")

# Count total teeth segmentations
total_teeth_segmentations = sum(teeth_ann_counter.values())
print(f"Total teeth segmentations: {total_teeth_segmentations}")

# Create seaborn graphics
sns.set(style="whitegrid")

# Basic summary plot: total images vs images with any annotation
total_images = len(image_files)
images_with_any = len(set(image_files) & (mouth_image_ids | teeth_image_ids))
images_without = total_images - images_with_any

def save_category_bar(counter: Counter, out_path: Path, title: str, top_n=30):
    items = counter.most_common(top_n)
    if not items:
        return
    cats, counts = zip(*items)
    plt.figure(figsize=(10, max(4, math.ceil(len(cats)/3))))
    sns.barplot(x=list(counts), y=list(cats), palette="viridis")
    plt.xlabel("Annotation count")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

save_category_bar(mouth_ann_counter, stats_dir / "mouth_annotations_per_category.png", "Mouth/Teeth annotations per category")
save_category_bar(teeth_ann_counter, stats_dir / "teeth_fdi_annotations_per_category.png", "Teeth FDI annotations per category")

# Annotations per image distribution
def ann_per_image_distribution(coco_ann, coco_imgs, out_path: Path, title: str):
    ann_by_image = Counter()
    for a in coco_ann.get("annotations", []):
        ann_by_image[a["image_id"]] += 1
    # map image_id -> file_name (some sources use int ids, others filenames)
    img_id_to_name = {img.get("id"): img.get("file_name") for img in coco_imgs}
    counts = list(ann_by_image.values()) if ann_by_image else [0]
    plt.figure(figsize=(8,4))
    sns.histplot(counts, bins=30, kde=False, color="#5b8bd6")
    plt.xlabel("Annotations per image")
    plt.ylabel("Number of images")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return counts

mouth_counts = ann_per_image_distribution(mouth_ann, mouth_ann.get("images", []), stats_dir / "mouth_annotations_per_image_hist.png", "Distribution: annotations per image (mouth_and_teeth)")
teeth_counts = ann_per_image_distribution(teeth_ann, teeth_ann.get("images", []), stats_dir / "teeth_fdi_annotations_per_image_hist.png", "Distribution: annotations per image (teeth_fdi)")

# Image metadata analysis: extract sex and age from filenames like 101-M-51.jpg
rows = []
for fname in image_files:
    stem = Path(fname).stem
    parts = stem.split("-")
    if len(parts) >= 3:
        pid, sex, age = parts[0], parts[1], parts[2]
        try:
            age = int(age)
        except:
            age = None
        rows.append({"file_name": fname, "id": pid, "sex": sex, "age": age})
    else:
        rows.append({"file_name": fname, "id": stem, "sex": None, "age": None})

df = pd.DataFrame(rows)

# Sex count
if "sex" in df.columns:
    sex_counts = df["sex"].value_counts(dropna=True)
    plt.figure(figsize=(4,4))
    sns.barplot(x=sex_counts.index.astype(str), y=sex_counts.values, palette="pastel")
    plt.xlabel("Sex")
    plt.ylabel("Number of images")
    plt.title("Images by sex")
    plt.tight_layout()
    plt.savefig(stats_dir / "images_by_sex.png")
    plt.close()

# Age distribution
if "age" in df.columns and df["age"].notna().any():
    plt.figure(figsize=(8,4))
    sns.histplot(df["age"].dropna(), bins=20, kde=True, color="#ff7f0e")
    plt.xlabel("Age")
    plt.ylabel("Number of images")
    plt.title("Age distribution")
    plt.tight_layout()
    plt.savefig(stats_dir / "age_distribution.png")
    plt.close()

# Add a pie chart for teeth segmentation distribution
plt.figure(figsize=(6, 6))
plt.pie(
    [total_teeth_segmentations, len(teeth_image_ids)],
    labels=[f"Teeth Segmentations ({total_teeth_segmentations})", f"Images ({len(teeth_image_ids)})"],
    autopct="%1.1f%%",
    colors=["#ff9999", "#66b3ff"],
)
plt.title("Teeth Segmentations vs Images")
plt.tight_layout()
plt.savefig(stats_dir / "teeth_segmentations_pie.png")
plt.close()

# Count annotations per FDI label (teeth_fdi_labels.json)
fdi_label_counter = Counter()
for ann in teeth_ann.get('annotations', []):
    cat_id = ann['category_id']
    cat_name = next((cat['name'] for cat in teeth_ann['categories'] if cat['id'] == cat_id), None)
    if cat_name:
        fdi_label_counter[cat_name] += 1

# Print the counts for each FDI label
print("Annotations per FDI label (teeth_fdi_labels):")
for label, count in fdi_label_counter.items():
    print(f"  {label}: {count}")

# Save the counts as a bar chart
plt.figure(figsize=(10, 6))
sns.barplot(x=list(fdi_label_counter.values()), y=list(fdi_label_counter.keys()), palette="viridis")
plt.xlabel("Annotation Count")
plt.ylabel("FDI Label")
plt.title("Annotations per FDI Label (teeth_fdi_labels)")
plt.tight_layout()
plt.savefig(stats_dir / "fdi_annotations_per_label.png")
plt.close()