"""Generate the 5 Colab/A100 notebooks from one template.

Building them from a single template keeps all five identical except for the per-model lines,
so the comparison is fair and the folder stays tidy. Re-run this after editing the template.

    python generate_notebooks.py
"""

import os
import json

HERE = os.path.dirname(os.path.abspath(__file__))

# key -> (Hugging Face model, uses MAX_PIXELS?, gated?, extra note)
MODELS = {
    "qwen":      ("Qwen/Qwen2.5-VL-7B-Instruct",             True,  False, ""),
    "internvl":  ("OpenGVLab/InternVL3-8B",                  True,  False, ""),
    "llava":     ("llava-hf/llava-onevision-qwen2-7b-ov-hf", True,  False, ""),
    "phi":       ("microsoft/Phi-3.5-vision-instruct",       True,  False,
                  "If you skip flash-attn, set ATTN_IMPL = 'eager' in the config cell."),
    "paligemma": ("google/paligemma2-3b-pt-448",             False, True,
                  "Gated model: accept the license and run notebook_login() in the install cell. "
                  "Fixed 448px resolution, so MAX_PIXELS is not used."),
}


def markdown_cell(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text}


def code_cell(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": text}


# ---- the config cell: the per-model lines change, the shared block does not ----
def config_cell_source(model_key, model_name, use_max_pixels, attn_default):
    return (
        "# ===== This model (the only part that differs between the 5 notebooks) =====\n"
        'MODEL_NAME = "' + model_name + '"\n'
        'MODEL_KEY = "' + model_key + '"\n'
        "USE_MAX_PIXELS = " + str(use_max_pixels) + "\n"
        "ATTN_IMPL = " + attn_default + "      # 'eager' if flash-attn is not installed\n"
        "\n"
        "# ===== Shared A100 settings (identical in all 5 notebooks = fair comparison) =====\n"
        'TRAIN_TYPE = "lora"          # bf16 LoRA (A100 has the VRAM, no 4-bit needed)\n'
        'TORCH_DTYPE = "bfloat16"\n'
        "LORA_RANK = 8\n"
        "LORA_ALPHA = 32\n"
        "LORA_DROPOUT = 0.1\n"
        'TARGET_MODULES = "all-linear"\n'
        'FREEZE_VIT = "true"\n'
        "NUM_EPOCHS = 2\n"
        'LEARNING_RATE = "1e-4"\n'
        'WEIGHT_DECAY = "0.1"\n'
        'WARMUP_RATIO = "0.05"\n'
        'LR_SCHEDULER = "cosine"\n'
        "PER_DEVICE_BATCH_SIZE = 2    # A100. effective batch = 2 * 8 = 16 (same as the 3090 run)\n"
        "GRAD_ACCUM_STEPS = 8\n"
        "MAX_LENGTH = 4096\n"
        'GRAD_CHECKPOINTING = "true"\n'
        "EVAL_STEPS = 100\n"
        "SAVE_STEPS = 100\n"
        "SAVE_TOTAL_LIMIT = 12\n"
        "SEED = 924\n"
        "MAX_PIXELS = 1003520         # 1280*28*28 (higher than the 3090; lower first if OOM)\n"
        "\n"
        "import os\n"
        'OUTPUT_DIR = "/content/drive/MyDrive/pan924_runs/" + MODEL_KEY\n'
        "os.makedirs(OUTPUT_DIR, exist_ok=True)\n"
        'print("Checkpoints will be saved to:", OUTPUT_DIR)\n'
    )


TRAIN_CELL = (
    "import os\n"
    "import glob\n"
    "import subprocess\n"
    "\n"
    "def find_last_checkpoint(folder):\n"
    '    """Return the newest checkpoint-N folder, or None if there is none yet."""\n'
    "    last_path = None\n"
    "    last_step = -1\n"
    '    for path in glob.glob(os.path.join(folder, "checkpoint-*")):\n'
    '        number_text = os.path.basename(path).replace("checkpoint-", "")\n'
    "        if number_text.isdigit() and int(number_text) > last_step:\n"
    "            last_step = int(number_text)\n"
    "            last_path = path\n"
    "    return last_path\n"
    "\n"
    "env = os.environ.copy()\n"
    "if USE_MAX_PIXELS:\n"
    '    env["MAX_PIXELS"] = str(MAX_PIXELS)\n'
    "\n"
    "command = [\n"
    '    "swift", "sft",\n'
    '    "--model", MODEL_NAME,\n'
    '    "--dataset", "vlm_report_dataset/converted/qwen/train.jsonl",\n'
    '    "--val_dataset", "vlm_report_dataset/converted/qwen/val.jsonl",\n'
    '    "--split_dataset_ratio", "0",\n'
    '    "--train_type", TRAIN_TYPE,\n'
    '    "--torch_dtype", TORCH_DTYPE,\n'
    '    "--lora_rank", str(LORA_RANK),\n'
    '    "--lora_alpha", str(LORA_ALPHA),\n'
    '    "--lora_dropout", str(LORA_DROPOUT),\n'
    '    "--target_modules", TARGET_MODULES,\n'
    '    "--freeze_vit", FREEZE_VIT,\n'
    '    "--num_train_epochs", str(NUM_EPOCHS),\n'
    '    "--learning_rate", LEARNING_RATE,\n'
    '    "--weight_decay", WEIGHT_DECAY,\n'
    '    "--warmup_ratio", WARMUP_RATIO,\n'
    '    "--lr_scheduler_type", LR_SCHEDULER,\n'
    '    "--per_device_train_batch_size", str(PER_DEVICE_BATCH_SIZE),\n'
    '    "--per_device_eval_batch_size", "1",\n'
    '    "--gradient_accumulation_steps", str(GRAD_ACCUM_STEPS),\n'
    '    "--max_length", str(MAX_LENGTH),\n'
    '    "--gradient_checkpointing", GRAD_CHECKPOINTING,\n'
    '    "--eval_strategy", "steps",\n'
    '    "--eval_steps", str(EVAL_STEPS),\n'
    '    "--save_strategy", "steps",\n'
    '    "--save_steps", str(SAVE_STEPS),\n'
    '    "--save_total_limit", str(SAVE_TOTAL_LIMIT),\n'
    '    "--logging_steps", "5",\n'
    '    "--seed", str(SEED),\n'
    '    "--add_version", "false",\n'
    '    "--output_dir", OUTPUT_DIR,\n'
    "]\n"
    "\n"
    "if ATTN_IMPL is not None:\n"
    '    command += ["--attn_impl", ATTN_IMPL]\n'
    "\n"
    "# Auto-resume: continue from the last checkpoint on Drive instead of starting over.\n"
    "last_checkpoint = find_last_checkpoint(OUTPUT_DIR)\n"
    "if last_checkpoint is not None:\n"
    '    print("Continuing from checkpoint:", last_checkpoint)\n'
    '    command += ["--resume_from_checkpoint", last_checkpoint]\n'
    "else:\n"
    '    print("Starting from the beginning.")\n'
    "\n"
    'print(" ".join(command))\n'
    "subprocess.run(command, env=env, check=True)\n"
    "# If you get CUDA OOM: lower MAX_PIXELS (1003520 -> 802816 -> 602112), then\n"
    "# PER_DEVICE_BATCH_SIZE 2 -> 1, and re-run this cell (it resumes).\n"
)


SELECT_CELL = (
    "import os\n"
    "import glob\n"
    "import json\n"
    "import subprocess\n"
    "\n"
    "VAL_DATA = 'vlm_report_dataset/converted/qwen/val.jsonl'\n"
    "EVAL_SCRIPT = 'vlm_report_dataset/scripts/eval_report.py'\n"
    "ADAPTER_SCRIPT = 'vlm_report_dataset/training/tools/swift_pred_to_eval.py'\n"
    "\n"
    "def list_checkpoints(folder):\n"
    "    pairs = []\n"
    "    for path in glob.glob(os.path.join(folder, 'checkpoint-*')):\n"
    "        number_text = os.path.basename(path).replace('checkpoint-', '')\n"
    "        if number_text.isdigit():\n"
    "            pairs.append((int(number_text), path))\n"
    "    pairs.sort()\n"
    "    return [path for step, path in pairs]\n"
    "\n"
    "env = os.environ.copy()\n"
    "if USE_MAX_PIXELS:\n"
    "    env['MAX_PIXELS'] = str(MAX_PIXELS)\n"
    "\n"
    "scores = {}\n"
    "for checkpoint in list_checkpoints(OUTPUT_DIR):\n"
    "    infer_out = os.path.join(checkpoint, 'infer_val.jsonl')\n"
    "    pred_out = os.path.join(checkpoint, 'preds_val.jsonl')\n"
    "    metrics_out = os.path.join(checkpoint, 'metrics_val.json')\n"
    "    subprocess.run(['swift', 'infer', '--adapters', checkpoint, '--val_dataset', VAL_DATA,\n"
    "                    '--max_new_tokens', '1024', '--temperature', '0',\n"
    "                    '--result_path', infer_out], env=env, check=True)\n"
    "    subprocess.run(['python', ADAPTER_SCRIPT, '--val', VAL_DATA,\n"
    "                    '--swift-result', infer_out, '--out', pred_out], check=True)\n"
    "    subprocess.run(['python', EVAL_SCRIPT, '--gold', VAL_DATA, '--pred', pred_out,\n"
    "                    '--out-json', metrics_out, '--tag', MODEL_KEY + '/val'], check=True)\n"
    "    macro_f1 = json.load(open(metrics_out, encoding='utf-8'))['macro_f1']\n"
    "    scores[checkpoint] = macro_f1\n"
    "    print(checkpoint, 'macro-F1 =', round(macro_f1, 4))\n"
    "\n"
    "BEST_CHECKPOINT = max(scores, key=scores.get)\n"
    "print('\\nBest checkpoint:', BEST_CHECKPOINT, '-> macro-F1', round(scores[BEST_CHECKPOINT], 4))\n"
)


FINAL_CELL = (
    "import os\n"
    "import json\n"
    "import subprocess\n"
    "import pandas as pd\n"
    "\n"
    "TEST_DATA = 'vlm_report_dataset/converted/qwen/test.jsonl'\n"
    "EVAL_SCRIPT = 'vlm_report_dataset/scripts/eval_report.py'\n"
    "ADAPTER_SCRIPT = 'vlm_report_dataset/training/tools/swift_pred_to_eval.py'\n"
    "\n"
    "env = os.environ.copy()\n"
    "if USE_MAX_PIXELS:\n"
    "    env['MAX_PIXELS'] = str(MAX_PIXELS)\n"
    "\n"
    "infer_out = os.path.join(BEST_CHECKPOINT, 'infer_test.jsonl')\n"
    "pred_out = os.path.join(BEST_CHECKPOINT, 'preds_test.jsonl')\n"
    "metrics_out = os.path.join(OUTPUT_DIR, 'metrics_' + MODEL_KEY + '_test.json')\n"
    "\n"
    "subprocess.run(['swift', 'infer', '--adapters', BEST_CHECKPOINT, '--val_dataset', TEST_DATA,\n"
    "                '--max_new_tokens', '1024', '--temperature', '0',\n"
    "                '--result_path', infer_out], env=env, check=True)\n"
    "subprocess.run(['python', ADAPTER_SCRIPT, '--val', TEST_DATA,\n"
    "                '--swift-result', infer_out, '--out', pred_out], check=True)\n"
    "subprocess.run(['python', EVAL_SCRIPT, '--gold', TEST_DATA, '--pred', pred_out,\n"
    "                '--out-json', metrics_out, '--tag', MODEL_KEY + '/test'], check=True)\n"
    "\n"
    "metrics = json.load(open(metrics_out, encoding='utf-8'))\n"
    "for name, value in metrics.items():\n"
    "    if isinstance(value, (int, float)):\n"
    "        print(name, '=', round(value, 4))\n"
    "\n"
    "# per-condition table (does the model handle the rare conditions?)\n"
    "pd.DataFrame(metrics['per_condition']).T.sort_values('support', ascending=False)\n"
)


def install_cell_source(gated):
    text = (
        "%pip install -q -U ms-swift accelerate\n"
        "%pip install -q -U qwen_vl_utils timm einops av sentencepiece\n"
        "# flash-attn makes the A100 much faster but takes a few minutes to build:\n"
        "%pip install -q flash-attn --no-build-isolation\n"
    )
    if gated:
        text += (
            "# This model is gated. Accept its license on Hugging Face, then log in here:\n"
            "from huggingface_hub import notebook_login\n"
            "notebook_login()\n"
        )
    text += (
        "import swift, torch\n"
        "print('ms-swift', swift.__version__, '| cuda available:', torch.cuda.is_available())\n"
    )
    return text


def build_notebook(model_key, model_name, use_max_pixels, gated, note):
    attn_default = "'eager'" if model_key == "phi" else "None"
    note_line = ("\n> **Note:** " + note + "\n") if note else ""

    cells = [
        markdown_cell(
            "# Train `" + model_key + "` — PAN924 dental report VLM (Colab · A100)\n\n"
            "**Model:** `" + model_name + "`  ·  **Framework:** ms-swift (LoRA, bf16)\n\n"
            "All 5 notebooks share the same settings, so the models are comparable. Checkpoints "
            "are saved on **Google Drive**, so if Colab disconnects you just **re-run the train "
            "cell and it continues** from the last checkpoint." + note_line
        ),
        markdown_cell("## 1. Check the GPU\n`Runtime -> Change runtime type -> A100 GPU`."),
        code_cell("!nvidia-smi --query-gpu=name,memory.total --format=csv"),

        markdown_cell("## 2. Install ms-swift and the per-model dependencies"),
        code_cell(install_cell_source(gated)),

        markdown_cell("## 3. Mount Google Drive (this is what makes training resumable)"),
        code_cell("from google.colab import drive\ndrive.mount('/content/drive')"),

        markdown_cell(
            "## 4. Put the code and dataset on the machine\n"
            "Zip `vlm_report_dataset/` (with the 4620 images), upload it to Drive, and set the path below."
        ),
        code_cell(
            "import os\n"
            "REPO_DIR = '/content/pan924'                              # will contain vlm_report_dataset/\n"
            "DATA_ZIP = '/content/drive/MyDrive/pan924_dataset.zip'    # your uploaded zip\n"
            "\n"
            "os.makedirs(REPO_DIR, exist_ok=True)\n"
            "if not os.path.isdir(os.path.join(REPO_DIR, 'vlm_report_dataset')):\n"
            "    assert os.path.exists(DATA_ZIP), 'Upload your dataset zip to ' + DATA_ZIP\n"
            "    !unzip -q {DATA_ZIP} -d {REPO_DIR}\n"
            "os.chdir(REPO_DIR)   # the image paths in the data are relative to here\n"
            "\n"
            "import json\n"
            "first_line = open('vlm_report_dataset/converted/qwen/train.jsonl', encoding='utf-8').readline()\n"
            "sample_image = json.loads(first_line)['images'][0]\n"
            "assert os.path.exists(sample_image), 'sample image not found: ' + sample_image\n"
            "print('OK - data and images found. Working directory:', os.getcwd())\n"
        ),

        markdown_cell(
            "## 5. Settings\n"
            "Effective batch = `PER_DEVICE_BATCH_SIZE * GRAD_ACCUM_STEPS` = 2 x 8 = 16 (same as the 3090 run)."
        ),
        code_cell(config_cell_source(model_key, model_name, use_max_pixels, attn_default)),

        markdown_cell(
            "## 6. Train (auto-resume)\n"
            "Re-run this cell after a disconnect and it continues from the last checkpoint on Drive."
        ),
        code_cell(TRAIN_CELL),

        markdown_cell(
            "## 7. Pick the best checkpoint by macro-F1 (not by loss)\n"
            "Loss is dominated by the common `H` class, so we score every checkpoint on the "
            "validation set and keep the one with the best per-condition macro-F1."
        ),
        code_cell(SELECT_CELL),

        markdown_cell(
            "## 8. Final metrics on the test set\n"
            "Reports accuracy / precision / recall / F1 (per condition and overall), FDI detection, "
            "hallucination and miss rates, exact-report match, and ROUGE-L on the text."
        ),
        code_cell(FINAL_CELL),

        markdown_cell(
            "## 9. Compare all models\n"
            "After all 5 notebooks finish, every `metrics_<key>_test.json` is on Drive:\n"
            "```bash\n"
            "python vlm_report_dataset/training/tools/compare_models.py "
            "/content/drive/MyDrive/pan924_runs/*/metrics_*_test.json\n"
            "```"
        ),
    ]

    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
            "accelerator": "GPU",
            "colab": {"provenance": []},
        },
        "nbformat": 4,
        "nbformat_minor": 4,
    }


def main():
    for model_key, (model_name, use_max_pixels, gated, note) in MODELS.items():
        notebook = build_notebook(model_key, model_name, use_max_pixels, gated, note)
        path = os.path.join(HERE, "train_" + model_key + ".ipynb")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(notebook, f, indent=1, ensure_ascii=False)
        print("wrote", path, "(" + str(len(notebook["cells"])) + " cells)")


if __name__ == "__main__":
    main()
