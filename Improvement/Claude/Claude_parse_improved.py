import base64
import csv
import glob
import os
import subprocess
import time
from io import StringIO
from pathlib import Path
import sys

import cv2
from anthropic import Anthropic


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Improvement.common_prompt import build_claude_prompt  # noqa: E402


api_key = os.environ.get("CLAUDE_API")
if not api_key:
    raise ValueError(
        "Environment variable CLAUDE_API not found."
    )


client = Anthropic(api_key=api_key)

RAW_DATA_DIR = str(ROOT / "raw_data")
RESULT_CSV = str(Path(__file__).resolve().parent / "claude_improved_parse.csv")
FAILURE_CSV = str(Path(__file__).resolve().parent / "claude_improved_failures.csv")
INSTRUCTION_PATH = str(ROOT / "Annotation Guideline.pdf")
MODEL_NAME = "claude-opus-4-6"
EXPECTED_FRAMES = 8


def parse_and_validate_csv_row(text: str, expected_video_id: str):
    text = extract_candidate_csv_row(text)
    if not text:
        raise ValueError("Model returned empty text")

    row = next(csv.reader(StringIO(text)))
    if len(row) != 8:
        raise ValueError(f"Expected 8 columns, got {len(row)}: {row}")

    video_id, is_poisoned, attack_level, semantic, logical, decision, final_score, reasoning = [
        col.strip() for col in row
    ]

    if video_id != expected_video_id:
        raise ValueError(f"video_id mismatch: expected {expected_video_id}, got {video_id}")

    if is_poisoned not in {"True", "False"}:
        raise ValueError(f"Invalid is_poisoned: {is_poisoned}")

    if attack_level not in {"None", "Semantic", "Logic", "Decision"}:
        raise ValueError(f"Invalid attack_level: {attack_level}")

    semantic = float(semantic)
    logical = float(logical)
    decision = float(decision)
    final_score = float(final_score)

    for name, value in {
        "semantic": semantic,
        "logical": logical,
        "decision": decision,
        "final_score": final_score,
    }.items():
        if not (0.0 <= value <= 1.0):
            raise ValueError(f"{name} out of range [0,1]: {value}")

    avg_score = round((semantic + logical + decision) / 3.0, 2)
    if round(final_score, 2) != avg_score:
        raise ValueError(
            f"final_score mismatch: got {final_score}, expected {avg_score}"
        )

    if is_poisoned == "False":
        if attack_level != "None":
            raise ValueError("If is_poisoned is False, attack_level must be None")
    else:
        if attack_level == "None":
            raise ValueError("If is_poisoned is True, attack_level must not be None")

    if attack_level == "Semantic" and semantic < max(logical, decision):
        raise ValueError("attack_level=Semantic but semantic is not the highest score")

    if attack_level == "Logic" and logical < max(semantic, decision):
        raise ValueError("attack_level=Logic but logical is not the highest score")

    if attack_level == "Decision" and decision < max(semantic, logical):
        raise ValueError("attack_level=Decision but decision is not the highest score")

    return [
        video_id,
        is_poisoned,
        attack_level,
        f"{semantic:.2f}",
        f"{logical:.2f}",
        f"{decision:.2f}",
        f"{final_score:.2f}",
        reasoning,
    ]


def extract_candidate_csv_row(text: str) -> str:
    """Recover the most likely CSV row when Claude prepends analysis text."""
    text = (text or "").strip()
    if not text:
        return ""

    # If the model embedded the row after some narration, cut from the first
    # filename-like token to improve CSV recovery.
    for marker in [f"{i:02d}.mp4," for i in range(100)]:
        pos = text.find(marker)
        if pos != -1:
            candidate = text[pos:].strip()
            try:
                row = next(csv.reader(StringIO(candidate)))
                if len(row) == 8:
                    return candidate
            except Exception:
                pass

    lines = [line.strip() for line in text.splitlines() if line.strip()]

    # Prefer the last line that parses as a single 8-column CSV row.
    for line in reversed(lines):
        try:
            row = next(csv.reader(StringIO(line)))
        except Exception:
            continue
        if len(row) == 8:
            return line

    # Fallback: if the whole text is already one valid CSV row, keep it.
    try:
        row = next(csv.reader(StringIO(text)))
        if len(row) == 8:
            return text
    except Exception:
        pass

    return text


def extract_video_frames(video_path: str, expected_frames: int = EXPECTED_FRAMES):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    frames = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
    finally:
        cap.release()

    if not frames:
        raise ValueError(f"No frames extracted from {video_path}")

    if len(frames) == expected_frames:
        selected_frames = frames
    else:
        indices = [
            round(i * (len(frames) - 1) / (expected_frames - 1))
            for i in range(expected_frames)
        ]
        selected_frames = [frames[idx] for idx in indices]

    encoded = []
    for frame in selected_frames:
        ok, jpg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        if not ok:
            raise RuntimeError(f"Failed to encode frame from {video_path}")
        encoded.append(base64.b64encode(jpg.tobytes()).decode("utf-8"))
    return encoded


def load_instruction_text(pdf_path: str) -> str:
    """Extract text from the annotation PDF once for a fairer Claude setup."""
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"Instruction PDF not found: {pdf_path}")

    proc = subprocess.run(
        ["pdftotext", pdf_path, "-"],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"pdftotext failed: {proc.stderr.strip()}")

    text = proc.stdout.strip()
    if not text:
        raise ValueError("Extracted instruction PDF text is empty")
    return text


def build_message_content(video_path: str, instruction_text: str):
    video_id = os.path.basename(video_path)
    frames = extract_video_frames(video_path)

    content = [
        {
            "type": "text",
            "text": (
                "Below is the text extracted from Annotation Guideline.pdf. "
                "Use it as the evaluation guideline for this task.\n\n"
                f"{instruction_text}"
            ),
        }
    ]

    for frame_b64 in frames:
        content.append(
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": frame_b64,
                },
            }
        )

    content.append(
        {
            "type": "text",
            "text": build_claude_prompt(video_id),
        }
    )
    return content


def extract_text_from_response(response) -> str:
    parts = []
    for block in response.content:
        if getattr(block, "type", None) == "text":
            parts.append(block.text)
    return "".join(parts).strip()


def process_video(video_path: str, instruction_text: str):
    print(f"\n[{time.strftime('%H:%M:%S')}] Started processing video: {video_path}")

    try:
        print("  -> Extracting frames locally...")
        content = build_message_content(video_path, instruction_text)

        print(f"  -> Requesting Claude parsing with model {MODEL_NAME}, please wait...")
        response = client.messages.create(
            model=MODEL_NAME,
            temperature=0.0,
            max_tokens=400,
            messages=[
                {
                    "role": "user",
                    "content": content,
                }
            ],
        )

        response_text = extract_text_from_response(response)
        row = parse_and_validate_csv_row(
            response_text,
            expected_video_id=os.path.basename(video_path),
        )

        print("\n" + "=" * 20 + " Parse Result " + "=" * 20)
        print(
            ",".join(
                [
                    row[0],
                    row[1],
                    row[2],
                    row[3],
                    row[4],
                    row[5],
                    row[6],
                    f'"{row[7]}"',
                ]
            )
        )
        print("=" * 50 + "\n")

        return row, None

    except Exception as e:
        print(f"  -> [Error] Failed to process {video_path}: {e}")
        return None, str(e)


def main():
    if not os.path.exists(INSTRUCTION_PATH):
        print("FAIL: Annotation Guideline PDF not found.")
        return

    video_extensions = ("*.mp4", "*.mov", "*.avi", "*.mkv")
    video_files = []

    for ext in video_extensions:
        search_pattern = os.path.join(RAW_DATA_DIR, ext)
        video_files.extend(glob.glob(search_pattern))

    video_files = sorted(video_files)

    if not video_files:
        print(f"No video files found in the {RAW_DATA_DIR} directory.")
        return

    print(f"Scanned {len(video_files)} video files, preparing for batch parsing...")
    print(f"Loading instruction text from: {INSTRUCTION_PATH}")
    instruction_text = load_instruction_text(INSTRUCTION_PATH)

    with open(RESULT_CSV, "w", encoding="utf-8", newline="") as f, open(
        FAILURE_CSV, "w", encoding="utf-8", newline=""
    ) as failure_f:
        writer = csv.writer(f)
        failure_writer = csv.writer(failure_f)
        writer.writerow(
            [
                "video_id",
                "is_poisoned",
                "attack_level",
                "semantic",
                "logical",
                "decision",
                "final_score",
                "reasoning",
            ]
        )
        failure_writer.writerow(["video_id", "error"])

        success_count = 0
        failure_count = 0

        for v_path in video_files:
            row, error = process_video(v_path, instruction_text)
            if row is not None:
                writer.writerow(row)
                f.flush()
                success_count += 1
            else:
                failure_writer.writerow([os.path.basename(v_path), error or "Unknown error"])
                failure_f.flush()
                failure_count += 1

    print(f"\nAll done. Success: {success_count}, Failed: {failure_count}")
    print(f"Results saved to: {RESULT_CSV}")
    print(f"Failure log saved to: {FAILURE_CSV}")


if __name__ == "__main__":
    main()
