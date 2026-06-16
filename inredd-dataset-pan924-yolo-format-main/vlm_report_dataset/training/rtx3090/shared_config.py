"""Shared training settings for the RTX 3090 (QLoRA) path.

All five models use exactly these settings, so comparing the models against each other is fair.
The only things that change between models are the model name, the output folder, and whether
to set MAX_PIXELS. Those three live in each train_<model>.py file, not here.

This file also has two small helper functions:
  - go_to_repo_root()  : moves into the folder that contains vlm_report_dataset/
  - train_model(...)   : builds the ms-swift command and runs it, resuming if possible.
"""

import os
import glob
import json
import subprocess


# ----------------------------------------------------------------------------
# Shared hyper-parameters (identical for every model)
# ----------------------------------------------------------------------------
TRAIN_TYPE = "lora"
QUANT_BITS = 4                 # QLoRA 4-bit, so a 7-8B model fits on a 24GB RTX 3090
QUANT_METHOD = "bnb"
TORCH_DTYPE = "bfloat16"

LORA_RANK = 8
LORA_ALPHA = 32
LORA_DROPOUT = 0.1
TARGET_MODULES = "all-linear"
FREEZE_VIT = "true"

NUM_EPOCHS = 2                 # a ceiling, not a target: the best checkpoint is chosen later
LEARNING_RATE = "1e-4"
WEIGHT_DECAY = "0.1"
WARMUP_RATIO = "0.05"
LR_SCHEDULER = "cosine"
PER_DEVICE_BATCH_SIZE = 1
GRAD_ACCUM_STEPS = 16          # effective batch size = 1 * 16 = 16
MAX_LENGTH = 4096
GRAD_CHECKPOINTING = "true"

EVAL_STEPS = 100
SAVE_STEPS = 100
SAVE_TOTAL_LIMIT = 12
LOGGING_STEPS = 5
SEED = 924

MAX_PIXELS = 802816            # 1024*28*28. Safe on 24GB. Lower this first if you hit OOM.

# The ms-swift "universal" data format works for all five models.
TRAIN_DATA = "vlm_report_dataset/converted/qwen/train.jsonl"
VAL_DATA = "vlm_report_dataset/converted/qwen/val.jsonl"
TEST_DATA = "vlm_report_dataset/converted/qwen/test.jsonl"

# Where each model writes its checkpoints (one folder per model).
OUTPUTS_DIR = "vlm_report_dataset/training/outputs"

# Where the final text results are saved (one .txt file per model).
RESULTS_DIR = "vlm_report_dataset/training/results"

# Helper scripts used during evaluation (paths are relative to the repo root).
EVAL_SCRIPT = "vlm_report_dataset/scripts/eval_report.py"
ADAPTER_SCRIPT = "vlm_report_dataset/training/tools/swift_pred_to_eval.py"


def go_to_repo_root():
    """Change directory to the repo root (the folder that contains vlm_report_dataset/).

    The image paths inside the dataset files are written relative to this folder, so we must
    run from here. This file lives in vlm_report_dataset/training/rtx3090/, which is three
    levels below the repo root.
    """
    this_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(this_dir, "..", "..", ".."))
    os.chdir(repo_root)
    print("Working directory:", repo_root)


def find_last_checkpoint(output_dir):
    """Return the path of the newest checkpoint-N folder, or None if there is none yet."""
    last_path = None
    last_step = -1
    for path in glob.glob(os.path.join(output_dir, "checkpoint-*")):
        number_text = os.path.basename(path).replace("checkpoint-", "")
        if number_text.isdigit():
            step = int(number_text)
            if step > last_step:
                last_step = step
                last_path = path
    return last_path


def train_model(model_name, output_dir, use_max_pixels=True, attn_impl=None):
    """Build the ms-swift training command for one model and run it.

    model_name     : Hugging Face id, e.g. "Qwen/Qwen2.5-VL-7B-Instruct"
    output_dir     : where this model's checkpoints go
    use_max_pixels : True for dynamic-resolution models, False for PaliGemma (fixed size)
    attn_impl      : set to "eager" if flash-attn is not installed (for example, Phi)
    """
    os.makedirs(output_dir, exist_ok=True)

    # ms-swift reads the image resolution cap from this environment variable.
    env = os.environ.copy()
    if use_max_pixels:
        env["MAX_PIXELS"] = str(MAX_PIXELS)

    # Build the command as a list, one option per line so it is easy to read and change.
    command = [
        "swift", "sft",
        "--model", model_name,
        "--dataset", TRAIN_DATA,
        "--val_dataset", VAL_DATA,
        "--split_dataset_ratio", "0",
        "--train_type", TRAIN_TYPE,
        "--quant_bits", str(QUANT_BITS),
        "--quant_method", QUANT_METHOD,
        "--torch_dtype", TORCH_DTYPE,
        "--lora_rank", str(LORA_RANK),
        "--lora_alpha", str(LORA_ALPHA),
        "--lora_dropout", str(LORA_DROPOUT),
        "--target_modules", TARGET_MODULES,
        "--freeze_vit", FREEZE_VIT,
        "--num_train_epochs", str(NUM_EPOCHS),
        "--learning_rate", LEARNING_RATE,
        "--weight_decay", WEIGHT_DECAY,
        "--warmup_ratio", WARMUP_RATIO,
        "--lr_scheduler_type", LR_SCHEDULER,
        "--per_device_train_batch_size", str(PER_DEVICE_BATCH_SIZE),
        "--per_device_eval_batch_size", "1",
        "--gradient_accumulation_steps", str(GRAD_ACCUM_STEPS),
        "--max_length", str(MAX_LENGTH),
        "--gradient_checkpointing", GRAD_CHECKPOINTING,
        "--eval_strategy", "steps",
        "--eval_steps", str(EVAL_STEPS),
        "--save_strategy", "steps",
        "--save_steps", str(SAVE_STEPS),
        "--save_total_limit", str(SAVE_TOTAL_LIMIT),
        "--logging_steps", str(LOGGING_STEPS),
        "--seed", str(SEED),
        "--add_version", "false",
        "--output_dir", output_dir,
    ]

    # Some models (Phi) need this when flash-attn is not installed.
    if attn_impl is not None:
        command.append("--attn_impl")
        command.append(attn_impl)

    # Auto-resume: if a checkpoint already exists, continue from it instead of restarting.
    last_checkpoint = find_last_checkpoint(output_dir)
    if last_checkpoint is not None:
        print("Found a checkpoint. Continuing from:", last_checkpoint)
        command.append("--resume_from_checkpoint")
        command.append(last_checkpoint)
    else:
        print("No checkpoint found. Starting from the beginning.")

    print("Running this command:")
    print(" ".join(command))
    result = subprocess.run(command, env=env)
    if result.returncode != 0:
        raise SystemExit("Training failed with exit code " + str(result.returncode))
    print("Training finished. Checkpoints are in:", output_dir)


def list_checkpoints(output_dir):
    """Return all checkpoint folders, sorted from oldest (smallest step) to newest."""
    pairs = []
    for path in glob.glob(os.path.join(output_dir, "checkpoint-*")):
        number_text = os.path.basename(path).replace("checkpoint-", "")
        if number_text.isdigit():
            pairs.append((int(number_text), path))
    pairs.sort()
    return [path for step, path in pairs]


def _predict_and_score(checkpoint_dir, data_file, split_name, tag, env):
    """Run the model on `data_file`, score the predictions, and return (macro_f1, report_text).

    `report_text` is the full human-readable report printed by eval_report.py.
    """
    infer_output = os.path.join(checkpoint_dir, "infer_" + split_name + ".jsonl")
    pred_output = os.path.join(checkpoint_dir, "preds_" + split_name + ".jsonl")
    metrics_output = os.path.join(checkpoint_dir, "metrics_" + split_name + ".json")

    # 1) generate the model's predictions
    subprocess.run(
        ["swift", "infer", "--adapters", checkpoint_dir, "--val_dataset", data_file,
         "--max_new_tokens", "1024", "--temperature", "0", "--result_path", infer_output],
        env=env, check=True,
    )
    # 2) reshape into {id, prediction}
    subprocess.run(
        ["python", ADAPTER_SCRIPT, "--val", data_file, "--swift-result", infer_output, "--out", pred_output],
        check=True,
    )
    # 3) score (write json + capture the printed report)
    completed = subprocess.run(
        ["python", EVAL_SCRIPT, "--gold", data_file, "--pred", pred_output,
         "--out-json", metrics_output, "--tag", tag],
        check=True, capture_output=True, text=True,
    )
    macro_f1 = json.load(open(metrics_output, encoding="utf-8"))["macro_f1"]
    return macro_f1, completed.stdout


def evaluate_and_save(model_key, output_dir, use_max_pixels=True):
    """Pick the best checkpoint on validation, evaluate it on the test set, and save a .txt report.

    Writes:  vlm_report_dataset/training/results/<model_key>_results.txt
             vlm_report_dataset/training/results/metrics_<model_key>_test.json
    """
    env = os.environ.copy()
    if use_max_pixels:
        env["MAX_PIXELS"] = str(MAX_PIXELS)

    checkpoints = list_checkpoints(output_dir)
    if not checkpoints:
        print("No checkpoints found in", output_dir, "- nothing to evaluate.")
        return

    # 1) score every checkpoint on the validation set, keep the best macro-F1
    print("\nScoring", len(checkpoints), "checkpoints on the validation set...")
    val_summary_lines = []
    best_checkpoint = None
    best_val_macro_f1 = -1.0
    for checkpoint in checkpoints:
        macro_f1, _ = _predict_and_score(checkpoint, VAL_DATA, "val", model_key + "/val", env)
        val_summary_lines.append(os.path.basename(checkpoint) + ": val macro-F1 = " + str(round(macro_f1, 4)))
        print("  " + val_summary_lines[-1])
        if macro_f1 > best_val_macro_f1:
            best_val_macro_f1 = macro_f1
            best_checkpoint = checkpoint

    print("Best checkpoint:", best_checkpoint, "(val macro-F1 =", round(best_val_macro_f1, 4), ")")

    # 2) evaluate the best checkpoint on the test set, capturing the full report
    test_macro_f1, test_report_text = _predict_and_score(
        best_checkpoint, TEST_DATA, "test", model_key + "/test", env)

    # 3) save everything to one .txt file in the results folder
    os.makedirs(RESULTS_DIR, exist_ok=True)
    results_path = os.path.join(RESULTS_DIR, model_key + "_results.txt")
    with open(results_path, "w", encoding="utf-8") as f:
        f.write("Model: " + model_key + "\n")
        f.write("Best checkpoint (chosen by validation macro-F1): " + best_checkpoint + "\n\n")
        f.write("Validation macro-F1 of every checkpoint:\n")
        for line in val_summary_lines:
            f.write("  " + line + "\n")
        f.write("\n===== TEST SET RESULTS =====\n")
        f.write(test_report_text)

    # also copy the machine-readable test metrics into the results folder
    source_json = os.path.join(best_checkpoint, "metrics_test.json")
    if os.path.exists(source_json):
        with open(source_json, encoding="utf-8") as src:
            data = src.read()
        with open(os.path.join(RESULTS_DIR, "metrics_" + model_key + "_test.json"), "w", encoding="utf-8") as dst:
            dst.write(data)

    print("\n" + test_report_text)
    print("Saved results to:", results_path)
