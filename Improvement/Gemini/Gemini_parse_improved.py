import csv
import glob
import os
import tempfile
import time
from io import StringIO
from pathlib import Path
import sys

import cv2


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Improvement.common_prompt import build_gemini_prompt  # noqa: E402
import Gemini.Gemini_parse as base  # noqa: E402


RAW_DATA_DIR = str(ROOT / "raw_data")
RESULT_CSV = str(Path(__file__).resolve().parent / "gemini_improved_parse.csv")
FAILURE_CSV = str(Path(__file__).resolve().parent / "gemini_improved_failures.csv")
INSTRUCTION_PATH = str(ROOT / "Annotation Guideline.pdf")
EXPECTED_FRAMES = 8


def extract_video_frames_to_tempfiles(video_path: str, expected_frames: int = EXPECTED_FRAMES):
    """Extract 8 chronological frames and save them as temporary JPEG files."""
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

    temp_paths = []
    for i, frame in enumerate(selected_frames, start=1):
        fd, temp_path = tempfile.mkstemp(prefix=f"gemini_frame_{i:02d}_", suffix=".jpg")
        os.close(fd)
        ok, jpg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        if not ok:
            raise RuntimeError(f"Failed to encode frame {i} from {video_path}")
        with open(temp_path, "wb") as f:
            f.write(jpg.tobytes())
        temp_paths.append(temp_path)

    return temp_paths


def process_video(video_path: str, instruction_file=None):
    """Process one video by uploading 8 extracted frame images instead of the original video."""
    print(f"\n[{time.strftime('%H:%M:%S')}] Started processing video: {video_path}")

    uploaded_frame_files = []
    temp_frame_paths = []

    try:
        print("  -> Extracting 8 frames locally...")
        temp_frame_paths = extract_video_frames_to_tempfiles(video_path)

        print("  -> Uploading extracted frames to Gemini...")
        for i, frame_path in enumerate(temp_frame_paths, start=1):
            frame_file = base.client.files.upload(file=frame_path)
            frame_file = base.wait_until_ready(frame_file, label=f"frame {i}", timeout=300, interval=3)
            uploaded_frame_files.append(frame_file)

        print("  -> Requesting Gemini parsing from extracted frames, please wait...")
        req_contents = [build_gemini_prompt(os.path.basename(video_path))]
        if instruction_file is not None:
            req_contents.insert(0, instruction_file)
        req_contents.extend(uploaded_frame_files)

        response = base.client.models.generate_content(
            model="gemini-3.1-pro-preview",
            contents=req_contents,
            config=base.types.GenerateContentConfig(
                temperature=0.0,
                response_mime_type="text/plain",
            ),
        )

        row = base.parse_and_validate_csv_row(
            response.text,
            expected_video_id=os.path.basename(video_path),
        )

        print("\n" + "=" * 20 + " Parse Result " + "=" * 20)
        print(",".join([
            row[0], row[1], row[2], row[3], row[4], row[5], row[6], f'"{row[7]}"'
        ]))
        print("=" * 50 + "\n")
        return row, None

    except Exception as e:
        print(f"  -> [Error] Failed to process {video_path}: {e}")
        return None, str(e)

    finally:
        for frame_file in uploaded_frame_files:
            print("  -> Cleaning up uploaded frame file...")
            try:
                base.client.files.delete(name=frame_file.name)
            except Exception as e:
                print(f"  -> [Warning] Failed to delete uploaded frame file: {e}")

        for frame_path in temp_frame_paths:
            try:
                os.remove(frame_path)
            except OSError:
                pass


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
    instruction_file = None

    with open(RESULT_CSV, "w", encoding="utf-8", newline="") as f, open(
        FAILURE_CSV, "w", encoding="utf-8", newline=""
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
            print(f"Uploading Instruction PDF: {INSTRUCTION_PATH} ...")
            instruction_file = base.client.files.upload(file=INSTRUCTION_PATH)
            instruction_file = base.wait_until_ready(
                instruction_file, label="instruction PDF", timeout=300, interval=5
            )
            print("Upload complete.")

            success_count = 0
            failure_count = 0

            for v_path in video_files:
                row, error = process_video(v_path, instruction_file)
                if row is not None:
                    writer.writerow(row)
                    f.flush()
                    success_count += 1
                else:
                    failure_writer.writerow([os.path.basename(v_path), error or "Unknown error"])
                    failure_f.flush()
                    failure_count += 1

        finally:
            if instruction_file is not None:
                print("Cleaning up uploaded Instruction PDF...")
                try:
                    base.client.files.delete(name=instruction_file.name)
                    print("Instruction PDF cleanup complete.")
                except Exception as e:
                    print(f"[Warning] Failed to clean up instruction PDF: {e}")

    print(f"\nAll done. Success: {success_count}, Failed: {failure_count}")
    print(f"Results saved to: {RESULT_CSV}")
    print(f"Failure log saved to: {FAILURE_CSV}")


if __name__ == "__main__":
    main()
