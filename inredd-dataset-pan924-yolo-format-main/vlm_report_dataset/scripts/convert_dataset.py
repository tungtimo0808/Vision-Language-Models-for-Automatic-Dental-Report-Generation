"""Convert common VLM JSONL into model-specific training formats.

The common dataset stays model-agnostic. This script writes one folder per
target model family, using the data layout expected by commonly used official
or upstream training/inference tooling for that family.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable


SUPPORTED_MODELS = ["qwen", "llava", "internvl", "phi", "paligemma"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert common PAN924 VLM data")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--input-dir", type=Path, default=Path("vlm_report_dataset/common"))
    parser.add_argument("--output-root", type=Path, default=Path("vlm_report_dataset/converted"))
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--models", nargs="+", default=SUPPORTED_MODELS, choices=SUPPORTED_MODELS)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def prompt_text(row: dict[str, Any]) -> str:
    return row["messages"][0]["content"]


def prompt_without_image_token(row: dict[str, Any]) -> str:
    return prompt_text(row).replace("<image>", "").strip()


def answer_text(row: dict[str, Any]) -> str:
    return row["messages"][1]["content"]


def image_size(row: dict[str, Any]) -> tuple[int, int]:
    metadata = row.get("metadata", {})
    if row["view"] == "full":
        return int(metadata["width"]), int(metadata["height"])
    x1, y1, x2, y2 = metadata["crop_boxes"][row["view"]]
    return int(x2 - x1), int(y2 - y1)


def convert_qwen(row: dict[str, Any]) -> dict[str, Any]:
    # ms-swift Qwen-VL format: messages contain <image>, images is a list.
    # Source: https://github.com/modelscope/ms-swift/blob/main/docs/source_en/Customization/Custom-dataset.md
    return {
        "id": row["id"],
        "messages": [
            {"role": "user", "content": prompt_text(row)},
            {"role": "assistant", "content": answer_text(row)},
        ],
        "images": [row["image"]],
        "metadata": {
            "source_image": row["source_image"],
            "view": row["view"],
            "task": row["task"],
        },
    }


def convert_llava(row: dict[str, Any]) -> dict[str, Any]:
    # Hugging Face multimodal chat-template format uses role/content messages,
    # where content is a list of typed image/text items.
    # Source: https://huggingface.co/docs/transformers/chat_templating_multimodal
    return {
        "id": row["id"],
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "path": row["image"]},
                    {"type": "text", "text": prompt_without_image_token(row)},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": answer_text(row)}],
            },
        ],
        "metadata": {
            "source_image": row["source_image"],
            "view": row["view"],
            "task": row["task"],
        },
    }


def convert_internvl(row: dict[str, Any]) -> dict[str, Any]:
    width, height = image_size(row)
    return {
        "id": row["id"],
        "image": row["image"],
        "width": width,
        "height": height,
        "conversations": [
            {"from": "human", "value": prompt_text(row)},
            {"from": "gpt", "value": answer_text(row)},
        ],
        "metadata": {
            "source_image": row["source_image"],
            "view": row["view"],
            "task": row["task"],
        },
    }


def convert_phi(row: dict[str, Any]) -> dict[str, Any]:
    # Phi vision model cards specify the chat prompt form:
    # <|user|>\n<|image_1|>\n{prompt}<|end|>\n<|assistant|>\n
    # Source: https://huggingface.co/microsoft/Phi-3.5-vision-instruct
    prompt = "<|user|>\n<|image_1|>\n" + prompt_without_image_token(row) + "<|end|>\n<|assistant|>\n"
    return {
        "id": row["id"],
        "image": row["image"],
        "prompt": prompt,
        "response": answer_text(row) + "<|end|>",
        "metadata": {
            "source_image": row["source_image"],
            "view": row["view"],
            "task": row["task"],
        },
    }


def convert_paligemma(row: dict[str, Any]) -> dict[str, Any]:
    # PaliGemma fine-tuning uses processor(images=..., text=prefix, suffix=answer).
    # Source: https://huggingface.co/docs/transformers/model_doc/paligemma
    return {
        "id": row["id"],
        "image": row["image"],
        "prefix": prompt_without_image_token(row),
        "suffix": answer_text(row),
        "metadata": {
            "source_image": row["source_image"],
            "view": row["view"],
            "task": row["task"],
        },
    }


CONVERTERS: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {
    "qwen": convert_qwen,
    "llava": convert_llava,
    "internvl": convert_internvl,
    "phi": convert_phi,
    "paligemma": convert_paligemma,
}


README_BY_MODEL = {
    "qwen": """Qwen/ms-swift JSONL.

Each row has `messages` and `images`. The user message contains `<image>` and `images` contains the image path list.

Source: https://github.com/modelscope/ms-swift/blob/main/docs/source_en/Customization/Custom-dataset.md
""",
    "llava": """LLaVA-OneVision / Hugging Face multimodal chat-template JSONL.

Each row has `messages`; message `content` is a list of typed items such as `{"type": "image", "path": ...}` and `{"type": "text", "text": ...}`.

Sources:
- https://huggingface.co/docs/transformers/chat_templating_multimodal
- https://huggingface.co/docs/transformers/model_doc/llava_onevision
""",
    "internvl": """InternVL single-image JSONL.

Each row has `image`, `width`, `height`, and ShareGPT-style `conversations` with `from: human/gpt`.

Source: https://internvl.readthedocs.io/en/latest/get_started/chat_data_format.html
""",
    "phi": """Phi vision prompt/completion JSONL.

Each row has an `image`, a full Phi prompt string using `<|user|>`, `<|image_1|>`, `<|end|>`, and `<|assistant|>`, plus a `response`.

Source: https://huggingface.co/microsoft/Phi-3.5-vision-instruct
""",
    "paligemma": """PaliGemma prefix/suffix JSONL.

Each row has `image`, `prefix`, and `suffix`, matching `processor(images=image, text=prefix, suffix=suffix, ...)`.

Source: https://huggingface.co/docs/transformers/model_doc/paligemma
""",
}


def write_model_readme(path: Path, model: str) -> None:
    text = f"""# PAN924 {model} Converted Dataset

{README_BY_MODEL[model]}

Generated from `vlm_report_dataset/common/*.jsonl`.
Do not edit these files by hand. Regenerate them from the common dataset.
"""
    (path / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    input_dir = root / args.input_dir
    output_root = root / args.output_root

    summary: dict[str, Any] = {"models": {}, "splits": args.splits}
    for model in args.models:
        converter = CONVERTERS[model]
        model_dir = output_root / model
        model_dir.mkdir(parents=True, exist_ok=True)
        write_model_readme(model_dir, model)
        summary["models"][model] = {}

        for split in args.splits:
            rows = read_jsonl(input_dir / f"{split}.jsonl")
            converted = [converter(row) for row in rows]
            write_jsonl(model_dir / f"{split}.jsonl", converted)
            summary["models"][model][split] = len(converted)

    summary_path = output_root / "conversion_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
