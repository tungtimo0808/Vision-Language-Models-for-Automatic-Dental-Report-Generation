"""Train PaliGemma 2 (3B, 448px) on the PAN924 dental dataset (RTX 3090, QLoRA 4-bit).

Run it from anywhere:
    python vlm_report_dataset/training/rtx3090/train_paligemma.py

If training stops for any reason, run the same command again — it continues from the last
checkpoint, it does not start over.

NOTE: PaliGemma is a GATED model. Before running, accept the license on its Hugging Face page
and log in once with:  huggingface-cli login
"""

from shared_config import go_to_repo_root, train_model, evaluate_and_save, OUTPUTS_DIR

# The only settings that are specific to this model:
MODEL_KEY = "paligemma"
MODEL_NAME = "google/paligemma2-3b-pt-448"
OUTPUT_DIR = OUTPUTS_DIR + "/" + MODEL_KEY
USE_MAX_PIXELS = False     # PaliGemma has a FIXED 448px resolution, so MAX_PIXELS is not used
ATTN_IMPL = None           # flash-attn works fine for PaliGemma

go_to_repo_root()
# 1) train (re-run this script to continue from the last checkpoint)
train_model(MODEL_NAME, OUTPUT_DIR, use_max_pixels=USE_MAX_PIXELS, attn_impl=ATTN_IMPL)
# 2) pick the best checkpoint, evaluate on the test set, save results/paligemma_results.txt
evaluate_and_save(MODEL_KEY, OUTPUT_DIR, use_max_pixels=USE_MAX_PIXELS)
