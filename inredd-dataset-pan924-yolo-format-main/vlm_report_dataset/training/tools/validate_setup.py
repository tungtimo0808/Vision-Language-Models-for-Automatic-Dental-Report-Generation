"""Offline check — runs WITHOUT a GPU or ms-swift installed.

It confirms the things that can be wrong before you pay for a GPU:
  - the train/val/test data files exist and have the expected number of lines
  - a sample image path actually resolves on disk (so training will find the images)
  - the five RTX 3090 training scripts and the five Colab notebooks are present

Usage:
    python vlm_report_dataset/training/tools/validate_setup.py
"""

import os
import json

# This file is in vlm_report_dataset/training/tools/, three levels below the repo root.
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", "..", ".."))

DATA_FILES = {
    "vlm_report_dataset/converted/qwen/train.jsonl": 5817,
    "vlm_report_dataset/converted/qwen/val.jsonl": 460,
    "vlm_report_dataset/converted/qwen/test.jsonl": 465,
}

MODEL_KEYS = ["qwen", "internvl", "llava", "phi", "paligemma"]


def check(label, condition):
    print(("[ OK ] " if condition else "[FAIL] ") + label)
    return condition


def main():
    print("REPO_ROOT =", REPO_ROOT, "\n")
    all_ok = True
    sample_image = None

    # 1) data files exist with the expected line counts
    for relative_path, expected_lines in DATA_FILES.items():
        full_path = os.path.join(REPO_ROOT, relative_path)
        if not os.path.exists(full_path):
            all_ok = check(relative_path + " (missing)", False) and all_ok
            continue
        with open(full_path, encoding="utf-8") as f:
            lines = [line for line in f if line.strip()]
        all_ok = check(relative_path + " : " + str(len(lines)) + " lines",
                       len(lines) == expected_lines) and all_ok
        if relative_path.endswith("train.jsonl"):
            sample_image = json.loads(lines[0])["images"][0]

    # 2) a sample image resolves
    if sample_image is not None:
        image_path = os.path.join(REPO_ROOT, sample_image)
        all_ok = check("sample image resolves: " + sample_image,
                       os.path.exists(image_path)) and all_ok

    # 3) the five RTX 3090 scripts exist
    for key in MODEL_KEYS:
        rel = "vlm_report_dataset/training/rtx3090/train_" + key + ".py"
        all_ok = check(rel, os.path.exists(os.path.join(REPO_ROOT, rel))) and all_ok

    # 4) the five Colab notebooks exist
    for key in MODEL_KEYS:
        rel = "vlm_report_dataset/training/colab_a100/train_" + key + ".ipynb"
        all_ok = check(rel, os.path.exists(os.path.join(REPO_ROOT, rel))) and all_ok

    print("\nRESULT:", "ALL CHECKS PASSED" if all_ok else "SOME CHECKS FAILED")
    raise SystemExit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
