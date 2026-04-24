import csv
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import Improvement.Gemini.Gemini_parse_improved as gemini_improved  # noqa: E402


HERE = Path(__file__).resolve().parent
FAILURE_CSV = HERE / "gemini_improved_failures.csv"
OUTPUT_CSV = HERE / "Rerun_gemini_improved_parse.csv"
RERUN_FAILURE_CSV = HERE / "Rerun_gemini_improved_failures.csv"


def load_failed_video_ids(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Failure CSV not found: {path}")

    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        video_ids = [row["video_id"].strip() for row in reader if row.get("video_id", "").strip()]

    seen = set()
    ordered = []
    for vid in video_ids:
        if vid not in seen:
            seen.add(vid)
            ordered.append(vid)
    return ordered


def main():
    if not os.path.exists(gemini_improved.INSTRUCTION_PATH):
        print("FAIL: Annotation Guideline PDF not found.")
        return

    failed_video_ids = load_failed_video_ids(FAILURE_CSV)
    if not failed_video_ids:
        print(f"No failed video IDs found in: {FAILURE_CSV}")
        return

    video_paths = [
        os.path.join(gemini_improved.RAW_DATA_DIR, video_id)
        for video_id in failed_video_ids
    ]
    video_paths = [path for path in video_paths if os.path.exists(path)]

    if not video_paths:
        print("No matching failed video files found in raw_data.")
        return

    print(f"Re-running {len(video_paths)} failed Gemini samples...")
    instruction_file = None

    with open(OUTPUT_CSV, "w", encoding="utf-8", newline="") as f, open(
        RERUN_FAILURE_CSV, "w", encoding="utf-8", newline=""
    ) as failure_f:
        writer = csv.writer(f)
        failure_writer = csv.writer(failure_f)
        writer.writerow([
            "video_id",
            "is_poisoned",
            "attack_level",
            "semantic",
            "logical",
            "decision",
            "final_score",
            "reasoning",
        ])
        failure_writer.writerow(["video_id", "error"])

        try:
            print(f"Uploading Instruction PDF: {gemini_improved.INSTRUCTION_PATH} ...")
            instruction_file = gemini_improved.base.client.files.upload(
                file=gemini_improved.INSTRUCTION_PATH
            )
            instruction_file = gemini_improved.base.wait_until_ready(
                instruction_file, label="instruction PDF", timeout=300, interval=5
            )
            print("Upload complete.")

            success_count = 0
            failure_count = 0

            for video_path in video_paths:
                row, error = gemini_improved.process_video(video_path, instruction_file)
                if row is not None:
                    writer.writerow(row)
                    f.flush()
                    success_count += 1
                else:
                    failure_writer.writerow([os.path.basename(video_path), error or "Unknown error"])
                    failure_f.flush()
                    failure_count += 1

        finally:
            if instruction_file is not None:
                print("Cleaning up uploaded Instruction PDF...")
                try:
                    gemini_improved.base.client.files.delete(name=instruction_file.name)
                    print("Instruction PDF cleanup complete.")
                except Exception as e:
                    print(f"[Warning] Failed to clean up instruction PDF: {e}")

    print(f"\nRe-run done. Success: {success_count}, Failed: {failure_count}")
    print(f"Re-run results saved to: {OUTPUT_CSV}")
    print(f"Re-run failures saved to: {RERUN_FAILURE_CSV}")


if __name__ == "__main__":
    main()
