"""Re-run the evaluation for an already-trained model, WITHOUT training again.

train_<model>.py already does this automatically at the end. Use this script only if you want
to re-evaluate (for example after copying checkpoints to a new machine). It picks the best
checkpoint on the validation set, evaluates it on the test set, and saves the .txt report to
vlm_report_dataset/training/results/<model_key>_results.txt.

Usage:
    python vlm_report_dataset/training/rtx3090/select_best_checkpoint.py qwen
"""

import sys

from shared_config import go_to_repo_root, evaluate_and_save, OUTPUTS_DIR

if len(sys.argv) < 2:
    print("usage: python select_best_checkpoint.py <model_key>")
    print("  model_key is one of: qwen internvl llava phi paligemma")
    sys.exit(1)

MODEL_KEY = sys.argv[1]
OUTPUT_DIR = OUTPUTS_DIR + "/" + MODEL_KEY
USE_MAX_PIXELS = (MODEL_KEY != "paligemma")   # PaliGemma has a fixed resolution

go_to_repo_root()
evaluate_and_save(MODEL_KEY, OUTPUT_DIR, use_max_pixels=USE_MAX_PIXELS)
