"""Train LLaVA-OneVision-7B on the PAN924 dental dataset (RTX 3090, QLoRA 4-bit).

Run it from anywhere:
    python vlm_report_dataset/training/rtx3090/train_llava.py

If training stops for any reason, run the same command again — it continues from the last
checkpoint, it does not start over.
"""

from shared_config import go_to_repo_root, train_model, evaluate_and_save, OUTPUTS_DIR

# The only settings that are specific to this model:
MODEL_KEY = "llava"
MODEL_NAME = "llava-hf/llava-onevision-qwen2-7b-ov-hf"
OUTPUT_DIR = OUTPUTS_DIR + "/" + MODEL_KEY
USE_MAX_PIXELS = True      # LLaVA-OneVision uses anyres
ATTN_IMPL = None           # flash-attn works fine for LLaVA

go_to_repo_root()
# 1) train (re-run this script to continue from the last checkpoint)
train_model(MODEL_NAME, OUTPUT_DIR, use_max_pixels=USE_MAX_PIXELS, attn_impl=ATTN_IMPL)
# 2) pick the best checkpoint, evaluate on the test set, save results/llava_results.txt
evaluate_and_save(MODEL_KEY, OUTPUT_DIR, use_max_pixels=USE_MAX_PIXELS)
