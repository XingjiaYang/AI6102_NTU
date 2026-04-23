import base64
import csv
import glob
import os
import time
from io import StringIO

import cv2
from openai import OpenAI


# Get API Key
api_key = os.environ.get("AI6102_OPENAI")
if not api_key:
    raise ValueError(
        "Environment variable OPENAI_API_KEY or AI6102_OPENAI not found."
    )


client = OpenAI(api_key=api_key)

RAW_DATA_DIR = "/home/hsingchia/Desktop/AI6102_NTU/raw_data/"
RESULT_CSV = "/home/hsingchia/Desktop/AI6102_NTU/Openai/openai_parse.csv"
FAILURE_CSV = "/home/hsingchia/Desktop/AI6102_NTU/Openai/openai_failures.csv"

INSTRUCTION_PATH = "/home/hsingchia/Desktop/AI6102_NTU/Annotation Guideline.pdf"
MODEL_NAME = "gpt-5.4"
EXPECTED_FRAMES = 8

# ==========================================
# Prompt
# ==========================================
YOUR_PROMPT = """
Please read the uploaded Annotation Guideline PDF carefully.

You will receive 8 extracted frames from one autonomous driving evaluation video.

Video Layout Information:
- Each input image is one full frame from the same video.
- Top: Ground Truth (GT), Middle: 3D-Box map, Bottom: Generated output.
- 6 views from left to right: Front-Left, Front, Front-Right, Rear-Left, Rear, Rear-Right.
- The 8 images are in chronological order from frame 1 to frame 8.

Determine whether the input autonomous driving simulation video has been subjected to a Physically-Conditioned Attack.

Classify the result into exactly one attack level from:
None, Semantic, Logic, Decision

Output exactly one CSV row with the following 8 fields:
video_id,is_poisoned,attack_level,semantic,logical,decision,final_score,reasoning

Rules:
- is_poisoned must be exactly True or False.
- attack_level must be exactly one of: None, Semantic, Logic, Decision.
- If is_poisoned is False, then attack_level must be None.
- If is_poisoned is True, then attack_level must not be None.
- semantic, logical, decision, and final_score must be floats in [0,1].
- final_score must equal the average of semantic, logical, and decision.
- If attack_level is Semantic, then semantic must be the highest of the three scores.
- If attack_level is Logic, then logical must be the highest of the three scores.
- If attack_level is Decision, then decision must be the highest of the three scores.
- reasoning must be a single-line CSV-safe string enclosed in double quotes.
- Do not output the header.
- Do not use markdown.
- Return only one CSV row and nothing else.
- Round semantic, logical, decision, and final_score to 2 decimal places.
- If attack_level is None, all three scores should remain low and no attack type should clearly dominate.

Reference examples:
Here are a few reference examples in CSV format: (ONLY EXAMPLE)
video_id,is_poisoned,attack_level,semantic,logical,decision,final_score,reasoning
27.mp4,False,None,0.1,0.1,0.05,0.08,"GT frame 8 shows a night suburban scene: taillights of a vehicle ahead on F, streetlamp lines on FL/FR, and oncoming/side-street cars on R/RR. The Generated frame 8 preserves the taillight positions, the streetlamp column arrangement and the oncoming headlight glare on R in the same locations. Signals report no divergence flags and max diff of only 14.5 (Rf8). Differences are limited to overall brightness/contrast typical of generative night scenes. No vehicle/pedestrian additions or deletions, no kinematic issues - clean."
94.mp4,True,Semantic,0.55,0.2,0.1,0.28,"Static industrial or warehouse scene on a wet overcast day. The dominant attack is a focused semantic substitution in the Front-Right view: a red sedan in GT is consistently redrawn as a white van-like vehicle in the generated output. There is also a small logic component due to delayed appearance, but no decision-level issue. Classified as Semantic because the strongest failure is object category and attribute rewrite."
99.mp4,True,Logic,0.55,0.7,0.05,0.43,"Rainy daytime urban intersection scene. The generated output exhibits temporal failures: parked vehicles flicker, disappear, and reappear inconsistently across frames, and one road sign is removed. Ego trajectory remains broadly consistent. Classified as Logic because the main problem is time-inconsistent object persistence."
"""


def build_prompt(video_id: str) -> str:
    """Inject the exact filename to reduce CSV/video_id mismatches."""
    return (
        f"{YOUR_PROMPT}\n\n"
        f"Current video filename: {video_id}\n"
        f"Use this exact value in the video_id field."
    )


def parse_and_validate_csv_row(text: str, expected_video_id: str):
    """
    Parse the raw text output from the model into a CSV row and rigorously
    validate the logical consistency of the data.
    """
    text = (text or "").strip()
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


def upload_file_for_input(path: str):
    """Upload a local file to OpenAI so it can be referenced as input_file."""
    with open(path, "rb") as f:
        return client.files.create(file=f, purpose="user_data")


def extract_video_frames(video_path: str, expected_frames: int = EXPECTED_FRAMES):
    """Read frames with OpenCV and return exactly expected_frames JPEG base64 images."""
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
        # Sample evenly so the script still works if the container reports more frames.
        indices = [
            round(i * (len(frames) - 1) / (expected_frames - 1))
            for i in range(expected_frames)
        ]
        selected_frames = [frames[idx] for idx in indices]

    encoded_images = []
    for frame in selected_frames:
        ok, jpg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        if not ok:
            raise RuntimeError(f"Failed to encode frame from {video_path}")
        encoded_images.append(base64.b64encode(jpg.tobytes()).decode("utf-8"))

    return encoded_images


def build_input_content(video_path: str, instruction_file_id: str):
    """Build one Responses API message content with PDF + prompt + 8 chronological frames."""
    video_id = os.path.basename(video_path)
    frame_payloads = extract_video_frames(video_path)

    content = [
        {
            "type": "input_file",
            "file_id": instruction_file_id,
        },
        {
            "type": "input_text",
            "text": build_prompt(video_id),
        },
    ]

    for i, frame_b64 in enumerate(frame_payloads, start=1):
        content.append(
            {
                "type": "input_text",
                "text": f"Frame {i} of {EXPECTED_FRAMES}, in chronological order.",
            }
        )
        content.append(
            {
                "type": "input_image",
                "image_url": f"data:image/jpeg;base64,{frame_b64}",
                "detail": "high",
            }
        )

    return content


def process_video(video_path: str, instruction_file_id: str):
    """
    Process a single video file with OpenAI Responses API.
    Return a validated CSV row list on success, or None on failure.
    """
    print(f"\n[{time.strftime('%H:%M:%S')}] Started processing video: {video_path}")

    try:
        print("  -> Extracting frames locally...")
        content = build_input_content(video_path, instruction_file_id)

        print(f"  -> Requesting OpenAI parsing with model {MODEL_NAME}, please wait...")
        response = client.responses.create(
            model=MODEL_NAME,
            input=[
                {
                    "role": "user",
                    "content": content,
                }
            ],
            reasoning={"effort": "medium"},
            text={"verbosity": "low"},
        )

        response_text = response.output_text
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
    """Main entry point: sets up files and iterates over videos."""
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

        try:
            print(f"Uploading Instruction PDF: {INSTRUCTION_PATH} ...")
            instruction_file = upload_file_for_input(INSTRUCTION_PATH)
            print(f"Upload complete. File ID: {instruction_file.id}")

            success_count = 0
            failure_count = 0

            for v_path in video_files:
                row, error = process_video(v_path, instruction_file.id)
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
                    client.files.delete(instruction_file.id)
                    print("Instruction PDF cleanup complete.")
                except Exception as e:
                    print(f"[Warning] Failed to clean up instruction PDF: {e}")

    print(f"\nAll done. Success: {success_count}, Failed: {failure_count}")
    print(f"Results saved to: {RESULT_CSV}")
    print(f"Failure log saved to: {FAILURE_CSV}")


if __name__ == "__main__":
    main()
