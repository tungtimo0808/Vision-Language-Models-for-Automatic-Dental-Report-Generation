import json
from pathlib import Path

p = Path("annotations/dataset_final_v2.json")
data = json.loads(p.read_text(encoding="utf-8"))

for img in data["images"]:
    img.pop("has_gold_fdi_image", None)

p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
print("Done")
