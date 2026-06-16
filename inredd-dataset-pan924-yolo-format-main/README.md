[![Website](https://img.shields.io/badge/Visit-Website-blue)](https://inredd.com.br)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-brightgreen.svg)](https://creativecommons.org/licenses/by/4.0/)
[![PhysioNet](https://img.shields.io/badge/Hosted%20at-PhysioNet-orange.svg)](https://physionet.org/)

# InReDD Open Data

InReDD Open Data is an initiative by the *Interdisciplinary Research Group in Digital Dentistry* (InReDD) at the University of São Paulo (USP-RP) to share high-quality, curated datasets for advancing research in digital dentistry.

The project aims to provide valuable resources collected from patients at the Ribeirão Preto School of Dentistry, structured with a focus on computational and information science standards.

Our primary goal is to make the `InReDD-dataset` a publicly available, multimodal collection with standardized labels, designed to support reproducible research and innovation in the field.

As an initial step, we are releasing the first version of this dataset, `InReDD-dataset-v1`, which we plan to expand and enhance over time.

## Why we built InReDD‑Dataset

Deep‑learning research in oral radiology is often stifled by **scarce, closed, or weakly‑labeled data**. By releasing InReDD‑Dataset, we aim to:

- Provide a dataset that is **more clinically realistic**.
- Lower the entry barrier for students and clinicians interested in **applying AI to dentistry**.
- Contribute to standardizing labels from a **multimodal perspective**.

---

# InReDD-dataset-v1 [![DOI](https://img.shields.io/badge/DOI-10.1016%2Fj.oooo.2023.12.006-blue.svg)](https://doi.org/10.1016/j.oooo.2023.12.006)

- [Dataset at a glance](#dataset-at-a-glance)
- [Structure](#structure)
- [Annotation schema](#annotation-schema)
  - [Images ("images")](#images-images)
  - [Labels ("annotations")](#labels-annotations)
  - [Categories ("categories")](#categories-categories)
    - [Tooth Condition Labels](#tooth-condition-labels)
  - [Example JSON Structure](#example-json-structure)
- [Disclaimer](#disclaimer)
- [Download](#download)
- [Citation](#citation)
- [License](#license)
- [Contact](#contact)

## Dataset at a glance

The **InReDD-dataset-v1** is a collection of **924 anonymized panoramic dental radiographs**, designed to support research in digital dentistry. Below is a summary of the dataset:

| **Category**          | **Count**            |
|------------------------|----------------------|
| Total images           | 924                 |
| Tooth segmentations    | 200                 |
| Mouth bounding boxes   | 924                 |
| Total bounding boxes   | 22,614              |
| Tooth conditions       | 12                  |
| Tooth positions (FDI)  | 32                  |

### Key Features:

- **Image Resolution**: Size of 2903 × 1536 px (95 dpi), stored as JPG files.
- **Annotations**: Provided in a COCO-compatible JSON format, with dental-specific fields (split and combined).
- **Tooth-Level Labels**: Includes bounding boxes and hierarchical condition annotations.
- **Metadata**: Contains patient details (age, sex).

### Dataset Statistics:  

- **Age Distribution**: Patients range from **14 to 81 years old**, with a median age of **35 years**.
- **Gender Distribution**: 
  - Female: **~60%**
  - Male: **~40%**
- **Tooth Conditions**: 
  - Healthy teeth: **~45%**
  - Restored teeth: **~25%**
  - Caries: **~15%**
  - Other conditions (e.g., implants, residual roots): **~15%**

## Structure

The **InReDD-dataset-v1** is a collection of **924 anonymized panoramic dental radiographs**, labeled to support research in digital dentistry. The dataset includes:

- **200 tooth segmentations** with FDI numbering system labels.
- **924 mouth bounding box detections**, categorized into tooth conditions.


```text
inredd-dataset-v1/
├── images/
│   ├── 2-F-70.jpg
│   ├── 4-F-48.jpg
│   └── …
├── annotations/
│   ├── teeth_fdi_labels/
│   │   ├── 2-F-70.json
│   │   ├── 4-F-48.json
│   │   └── …
│   ├── mouth_and_teeth_labels/
│   │   ├── 2-F-70.json
│   │   ├── 4-F-48.json
│   │   └── …
│   ├── mouth_and_teeth_labels.json
│   └── teeth_fdi_labels.json
├── stats/
│   └── …
├── inredd_dataset_v1_fof.py
├── inredd_dataset_v1_stats.py
├── README.md
└── requirements.txt
```

### Annotations

Annotations are distributed as JSON files in a **format compatible with COCO**, while preserving dental-specific fields. 

We provide labels in both combined and split formats:

- **Split format**: The `teeth_fdi_labels` and `mouth_and_teeth_labels` directories contain one JSON file per image, with annotations specific to that image.
- **Combined format**: The `teeth_fdi_labels.json` and `mouth_and_teeth_labels.json` files contain all annotations combined into a single JSON file for the entire dataset.

Each annotation includes:

- **Bounding boxes**: Defined for teeth and mouth regions.
- **Tooth-level labels**: Following the FDI numbering system (00–88).
- **Condition annotations**: Binary flags for 12 common findings (e.g., caries, crown, implant, root canal treatment).

### Stats

The `stats/` directory contains precomputed statistics for the dataset, including:

- **Age distribution**: Histogram of patient ages.
- **Gender distribution**: Breakdown of male and female patients.
- **Condition frequency**: Counts of each tooth condition across the dataset.
- **Bounding box statistics**: Average size and distribution of bounding boxes.

These statistics are useful for understanding the dataset composition and for preprocessing tasks in machine learning workflows.

### FiftyOne

The dataset is compatible with [FiftyOne](https://voxel51.com/fiftyone/), a powerful tool for visualizing and analyzing datasets. A example load file is provided at `inredd_dataset_v1_fof.py`.

## Annotation schema

The annotations are provided in a **COCO-compatible JSON format**, with additional dental-specific fields. Below is the schema for the JSON file:

### Images ("images")
Each image entry contains metadata about the image:

| Field        | Type   | Description                                   |
|--------------|--------|-----------------------------------------------|
| `id`         | int    | Unique identifier for the image.              |
| `license`    | int    | License ID for the image.                     |
| `file_name`  | string | Name of the image file (e.g., `2-F-70.jpg`).  |
| `height`     | int    | Height of the image in pixels.                |
| `width`      | int    | Width of the image in pixels.                 |
| `sex`        | string | Patient's sex (`M` for male, `F` for female). |
| `age`        | string | Patient's age (e.g., `70`).                   |

*`file_name` contains id, sex and age; however, it's create specific fields for better use. IDs are random, provided after anonymization process, they don't follow a logical order because some raw data didn't attendend quality aspects, so was removed from the dataset. 

### Labels ("annotations")
Each annotation entry contains information about a specific object (e.g., tooth or mouth region) in the image:

| Field          | Type       | Description                                   |
|-----------------|------------|-----------------------------------------------|
| `id`           | int        | Unique identifier for the annotation.         |
| `image_id`     | int        | ID of the image this annotation belongs to.   |
| `category_id`  | int        | ID of the category (e.g., tooth condition).   |
| `bbox` | list[list] | Bbox coordinates for the detection.|
| `segmentation` | list[list] | Polygon coordinates for the segmentation mask.|

### Categories ("categories")
The `categories` field defines the possible classes for annotations:

| Field        | Type   | Description                                   |
|--------------|--------|-----------------------------------------------|
| `id`         | int    | Unique identifier for the category.           |
| `name`       | string | Name of the category (e.g., `Ed`).        |
| `supercategory` | string | Higher-level grouping for the category (could be none).     |

#### Tooth Condition Labels

The dataset provides hierarchical labels for tooth conditions, including:

1. **Major Bounding Box (Mouth Labels)**:
   - **Ed**: Edentulous
   - **De**: Dentate
   - **Me**: Maxilla edentulous
   - **Mne**: Mandible edentulous

2. **Minor Bounding Box (Teeth Labels)**:
   - **Artificial Teeth (DA)**:
     - **Im**: Implant
     - **Cp**: Single prosthetic crown
     - **P**: Pontic
   - **Natural Teeth (DN)**:
     - **H**: Healthy
     - **Rr**: Residual root
     - **M3i**: Impacted third molar
     - **M3f**: Developing third molar
     - **Te**: Endodontic treatment
     - **Ri**: Intraradicular post
     - **Dc**: Crown destruction
     - **Di**: Incisal wear
     - **C**: Caries
     - **R**: Restored
     - **I**: Impacted
   - **Mixed Teeth (DM)**:
     - **TeM**: Endodontic treatment
     - **RiM**: Intraradicular post
     - **CpuM**: Single prosthetic crown

##### Observations

- **Tooth-level bounding boxes**: Following the FDI two-digit numbering system (00–88).
- **Condition annotations**: Binary flags for 12 common findings (e.g., caries, crown, implant, root canal treatment, periapical lesion).

### Example JSON Structure
Here’s an example of the JSON structure:

```json
{
  "images": [
    {
      "id": 2,
      "license": 1,
      "file_name": "2-F-70.jpg",
      "height": 1536,
      "width": 2903,
      "sex": "F",
      "age": "70"
    }
  ],
  "annotations": [
    {
      "id": 9763,
      "image_id": 2,
      "category_id": 1,
      "segmentation": [
        [
          204,
          94,
          204,
          1301,
          2630,
          1301,
          2630,
          94
        ]
      ]
    }
  ],
  "categories": [
    {
      "id": 1,
      "name": "Ed",
      "supercategory": "Mouth"
    }
  ]
}
```


# Disclaimer

There are some incosensty with the article description of the dataset such the amout of the image and quality, and teeth segmentation number, this it's because we had made a quality clean after converting all data into COCO standarsd, and removed some not fit data for sharing this


# Download

// Add citation for physionet (waiting for revision and approval)

# Citation

// Add citation for physionet (waiting for revision and approval)

# License

All images and annotations are released under the **Creative Commons Attribution 4.0 International** (CC BY 4.0) license.

You are free to share and adapt the material for any purpose, even commercially, provided that you give appropriate credit.

# Contact

Questions? Reach the InReDD team on [`#inredd`](https://gitter.im/inredd/dataset) or create a GitHub issue.

Or you can send an email to [inredd@usp.br](mailto:inredd@usp.br)

---

*This repository is maintained by the InReDD research group at USP Ribeirão Preto.*
*Last updated: 2025-09-22*
