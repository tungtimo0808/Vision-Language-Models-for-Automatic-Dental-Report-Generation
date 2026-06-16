"""Quick smoke test — prove the training pipeline runs before starting the real (paid) run.

It trains a tiny 0.5B model on 8 samples for 3 steps at a low resolution. It only checks that
ms-swift is installed, the data loads, the images resolve, and a checkpoint gets written.
It does NOT produce a useful model.

Run it on the rented GPU box first:
    python vlm_report_dataset/training/rtx3090/smoke_test.py

Run it a SECOND time to confirm that resume works (it should continue from the checkpoint).
"""

import os
import glob
import subprocess

from shared_config import go_to_repo_root, find_last_checkpoint, TRAIN_DATA, VAL_DATA

TINY_MODEL = "llava-hf/llava-onevision-qwen2-0.5b-ov-hf"
OUTPUT_DIR = "vlm_report_dataset/training/outputs/_smoke"


def main():
    go_to_repo_root()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    env = os.environ.copy()
    env["MAX_PIXELS"] = "200704"   # 256*28*28, tiny so it fits anywhere

    command = [
        "swift", "sft",
        "--model", TINY_MODEL,
        "--dataset", TRAIN_DATA + "#8",     # only 8 training samples
        "--val_dataset", VAL_DATA + "#4",   # only 4 validation samples
        "--split_dataset_ratio", "0",
        "--train_type", "lora",
        "--lora_rank", "8",
        "--target_modules", "all-linear",
        "--freeze_vit", "true",
        "--torch_dtype", "bfloat16",
        "--max_steps", "3",
        "--per_device_train_batch_size", "1",
        "--gradient_accumulation_steps", "1",
        "--max_length", "2048",
        "--gradient_checkpointing", "true",
        "--eval_steps", "2",
        "--save_steps", "2",
        "--save_total_limit", "2",
        "--logging_steps", "1",
        "--seed", "924",
        "--add_version", "false",
        "--output_dir", OUTPUT_DIR,
    ]

    last_checkpoint = find_last_checkpoint(OUTPUT_DIR)
    if last_checkpoint is not None:
        print("Resume test: continuing from", last_checkpoint)
        command.append("--resume_from_checkpoint")
        command.append(last_checkpoint)

    print("Running:", " ".join(command))
    result = subprocess.run(command, env=env)
    if result.returncode != 0:
        raise SystemExit("Smoke test FAILED with exit code " + str(result.returncode))

    written = glob.glob(os.path.join(OUTPUT_DIR, "checkpoint-*"))
    print("\nSmoke test PASSED. Checkpoints written:", written)
    print("Run this script again to confirm resume, then start the real training.")


if __name__ == "__main__":
    main()
