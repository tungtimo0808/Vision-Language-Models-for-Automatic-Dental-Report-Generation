#!/usr/bin/env bash
# One-shot environment setup for a FRESH vast.ai Linux GPU box.
# Installs ms-swift + the per-model extras each of the 5 families needs, so the first run
# doesn't die on a missing import. Run once after renting.
#
#   bash vlm_report_dataset/training/setup_env.sh
#   WITH_FLASH_ATTN=1 bash vlm_report_dataset/training/setup_env.sh   # also build flash-attn (slow)
set -euo pipefail

echo ">>> core: ms-swift + QLoRA + training stack"
pip install -U "ms-swift" bitsandbytes accelerate

echo ">>> per-model extras"
# qwen2.5-vl + llava-onevision use qwen_vl_utils; internvl needs timm+einops; av is a common image/video backend
pip install -U qwen_vl_utils timm einops av sentencepiece

if [ "${WITH_FLASH_ATTN:-0}" = "1" ]; then
  echo ">>> flash-attn (speeds up qwen/phi; build can take 10+ min)"
  pip install -U flash-attn --no-build-isolation || \
    echo "WARN: flash-attn failed to build — that's OK, set ATTN_IMPL = 'eager' in train_phi.py"
fi

echo ""
echo ">>> sanity:"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-')"
python -c "import swift; print('ms-swift', swift.__version__)"

cat <<'NOTE'

------------------------------------------------------------------
NEXT:
  * GATED MODEL: paligemma2 needs a Hugging Face license + token:
      - accept the license at https://huggingface.co/google/paligemma2-3b-pt-448
      - then:  huggingface-cli login        (paste your HF token)
    The other 4 (qwen/internvl/llava/phi) are open — no login needed.
  * DISK: each 7-8B model is ~15GB + checkpoints. Pick an instance with >=100GB disk.
  * If flash-attn is NOT installed, set ATTN_IMPL = 'eager' near the top of train_phi.py
------------------------------------------------------------------
NOTE
