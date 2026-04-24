import csv
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "json"

SOURCES = {
    "human": ROOT / "Human" / "human_classified.csv",
    "Openai_improved": ROOT / "Improvement" / "Openai" / "openai_improved_parse.csv",
}

OUTPUTS = {
    "human": OUTPUT_DIR / "human.json",
    "Openai_improved": OUTPUT_DIR / "Openai_improved.json",
}

ATTACK_LEVELS = {
    "none": "None",
    "nan": "None",
    "": "None",
    "semantic": "Semantic",
    "logic": "Logic",
    "logical": "Logic",
    "logicial": "Logic",
    "decision": "Decision",
}


def read_csv_with_fallback(path: Path) -> list[dict[str, str]]:
    for encoding in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            with path.open("r", encoding=encoding, newline="") as file:
                return list(csv.DictReader(file))
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError(f"Unable to decode {path}")


def normalize_bool(value: object) -> bool:
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n", "", "nan", "none"}:
        return False
    raise ValueError(f"Invalid boolean value: {value!r}")


def normalize_attack_level(value: object, is_poisoned: bool) -> str:
    text = str(value).strip().lower()
    level = ATTACK_LEVELS.get(text)
    if level is None:
        raise ValueError(f"Invalid attack_level value: {value!r}")
    return level if is_poisoned else "None"


def normalize_float(value: object) -> float:
    text = str(value).strip()
    if text == "" or text.lower() in {"nan", "none"}:
        return 0.0
    number = float(text)
    if math.isnan(number):
        return 0.0
    return round(number, 6)


def normalize_video_id(value: object) -> str:
    text = str(value).strip()
    if text.endswith(".mp4"):
        stem = text[:-4]
    else:
        stem = text
    if not stem.isdigit():
        raise ValueError(f"Invalid video_id value: {value!r}")
    return f"{int(stem):02d}.mp4"


def sort_key(record: dict[str, object]) -> int:
    return int(str(record["video_id"]).replace(".mp4", ""))


def convert_row(row: dict[str, str]) -> dict[str, object]:
    video_id = normalize_video_id(row.get("video_id", ""))
    is_poisoned = normalize_bool(row.get("is_poisoned", ""))
    attack_level = normalize_attack_level(row.get("attack_level", ""), is_poisoned)

    return {
        "video_id": video_id,
        "is_poisoned": is_poisoned,
        "attack_level": attack_level,
        "scores": {
            "semantic": normalize_float(row.get("semantic", 0)),
            "logical": normalize_float(row.get("logical", 0)),
            "decision": normalize_float(row.get("decision", 0)),
        },
        "final_score": normalize_float(row.get("final_score", 0)),
        "reasoning": str(row.get("reasoning", "")).strip(),
    }


def validate_records(records: list[dict[str, object]], source_name: str) -> None:
    expected = {f"{index:02d}.mp4" for index in range(100)}
    actual = {str(record["video_id"]) for record in records}

    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        raise ValueError(
            f"{source_name} video_id validation failed. "
            f"Missing: {missing or 'none'}; extra: {extra or 'none'}"
        )

    if len(records) != 100:
        raise ValueError(f"{source_name} should contain 100 rows, got {len(records)}")


def convert_file(source_name: str, input_path: Path, output_path: Path) -> None:
    rows = read_csv_with_fallback(input_path)
    records = sorted((convert_row(row) for row in rows), key=sort_key)
    validate_records(records, source_name)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(records, file, ensure_ascii=False, indent=2)
        file.write("\n")

    print(f"Wrote {len(records)} records to {output_path}")


def main() -> None:
    for source_name, input_path in SOURCES.items():
        convert_file(source_name, input_path, OUTPUTS[source_name])


if __name__ == "__main__":
    main()
