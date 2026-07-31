import json
from pathlib import Path

src = Path("backend/data/processed/xgb_multilingual_malicious_20260727_v17.jsonl")
out = Path("backend/data/splits/xgb_v17_routes")
out.mkdir(parents=True, exist_ok=True)
langs = {"powershell", "go", "rust"}
rows = {lang: [] for lang in langs}
with src.open(encoding="utf-8") as handle:
    for line in handle:
        record = json.loads(line)
        language = str(record.get("language", "")).lower()
        split = str(record.get("split", "")).lower()
        if language in langs and split in {"train", "validation", "test"}:
            rows[language].append(record)
for language, records in rows.items():
    path = out / f"{language}.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(language, len(records), path)




