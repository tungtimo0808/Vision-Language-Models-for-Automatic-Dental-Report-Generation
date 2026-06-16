"""Train InternVL3-8B on the PAN924 dental dataset (RTX 3090, QLoRA 4-bit).

Run it from anywhere:
    python vlm_report_dataset/training/rtx3090/train_internvl.py

If training stops for any reason, run the same command again — it continues from the last
checkpoint, it does not start over.
"""

from shared_config import go_to_repo_root, train_model, evaluate_and_save, OUTPUTS_DIR

# The only settings that are specific to this model:
MODEL_KEY = "internvl"
MODEL_NAME = "OpenGVLab/InternVL3-8B"
OUTPUT_DIR = OUTPUTS_DIR + "/" + MODEL_KEY
USE_MAX_PIXELS = True      # InternVL uses dynamic tiling (good for tiny lesions)
ATTN_IMPL = None           # flash-attn works fine for InternVL

go_to_repo_root()
# 1) train (re-run this script to continue from the last checkpoint)
train_model(MODEL_NAME, OUTPUT_DIR, use_max_pixels=USE_MAX_PIXELS, attn_impl=ATTN_IMPL)
# 2) pick the best checkpoint, evaluate on the test set, save results/internvl_results.txt
evaluate_and_save(MODEL_KEY, OUTPUT_DIR, use_max_pixels=USE_MAX_PIXELS)
