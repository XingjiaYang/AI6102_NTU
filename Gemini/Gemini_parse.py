import os
import time
import glob
import csv
from io import StringIO
from google import genai
from google.genai import types

# Get API Key
api_key = os.environ.get("AI6102_GEMINI")
if not api_key:
    raise ValueError("Environment variable AI6102_GEMINI not found. Please make sure to export and source it in .bashrc.")


client = genai.Client(api_key=api_key)

RAW_DATA_DIR = "/home/hsingchia/Desktop/AI6102_NTU/raw_data/"
RESULT_CSV = "/home/hsingchia/Desktop/AI6102_NTU/Gemini/Gemini_parse.csv"
FAILURE_CSV = "/home/hsingchia/Desktop/AI6102_NTU/Gemini/Gemini_failures.csv"

INSTRUCTION_PATH = "/home/hsingchia/Desktop/AI6102_NTU/Annotation Guideline.pdf"

# ==========================================
# 3. Custom Prompt Area
# ==========================================
YOUR_PROMPT = """
Please read the uploaded Annotation Guideline PDF carefully.

Video Layout Information:
- 8 frames per video.
- Top: Ground Truth (GT), Middle: 3D-Box map, Bottom: Generated output.
- 6 views from left to right: Front-Left, Front, Front-Right, Rear-Left, Rear, Rear-Right.

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
27.mp4,False,None,0.1,0.1,0.05,0.08,"GT frame 8 shows a night suburban scene: taillights of a vehicle ahead on F, streetlamp lines on FL/FR, and oncoming/side-street cars on R/RR. The Generated frame 8 preserves the taillight positions, the streetlamp column arrangement and the oncoming headlight glare on R in the same locations. Signals report no divergence flags and max diff of only 14.5 (Rf8). Differences are limited to overall brightness/contrast typical of generative night scenes. No vehicle/pedestrian additions or deletions, no kinematic issues — clean."
94.mp4,True,Semantic,0.55,0.2,0.1,0.28,"Static industrial/warehouse scene on a wet overcast day: the ego vehicle is stopped at an intersection with a parking garage and crosswalk on the left (Front-Left / Front), shipping containers and a gas-cylinder storage yard with chain-link fence on the right (Front-Right / Rear-Left), a black Jeep Wrangler approaching in the Rear view, and a warehouse loading dock with parked black pickup trucks and white refrigerated trailers in Rear-Right. Pixel-difference analysis between frames 1 and 8 is small in both GT and Gen, and the crosswalk/pole positions do not shift, confirming the ego is stationary throughout in both — no decision-level issue. The 3D box map annotates only two traffic-relevant agents of note: a small vehicle approaching on the cross street visible in Front-Right, and the Jeep in Rear. The dominant and essentially only attack is a focused SEMANTIC substitution in the Front-Right view: GT frames 1-8 show a red / dark-red sedan approaching on the cross street from the left, clearly annotated as a small compact-vehicle box in the 3D box map; in the Generated output the same vehicle is consistently redrawn as a white box-shaped van/truck — the color is changed (red → white) and the body class is changed (sedan → van) across all frames in which the vehicle is visible (frames 3-8 in Gen). The vehicle is also missing entirely from Gen frames 1-2 where GT shows it faintly in the distance, contributing a small LOGIC component (delayed appearance / object permanence issue). Secondary minor issues: text on the shipping-container walls in Rear-Left is rendered as illegible glyphs in Gen, and signage in Rear-Right is slightly blurred, but these are background texture and do not affect traffic-relevant semantics. Classified as Semantic because the single most important failure is a clean category-and-attribute rewrite of an oncoming vehicle (which a downstream planner could misidentify — e.g., mistaking a slower delivery van for an agile sedan, or vice versa). Scores are moderate because only one agent is affected, ego is static, and no unsafe ego behaviour is produced."
99.mp4,True,Logic,0.55,0.7,0.05,0.43,"Rainy daytime urban intersection scene. GT shows parked vehicles (a blue hatchback and a silver SUV in Front-Right, multiple black sedans along the curb in Rear-Right) remaining static across all 8 frames as the ego approaches the intersection, plus a white SUV crossing the pedestrian crosswalk in the Front-Left during frames 6-8. The Generated output exhibits both temporal and semantic failures: (1) in the Front-Right view, the blue hatchback appears normally in frames 1-2, becomes distorted in frame 2, disappears entirely in frame 3, partially re-appears in frame 4, then disappears again from frames 5-8 — a flicker/reappear pattern that violates temporal coherence of a static object; (2) the silver SUV parked next to it vanishes after frame 4; (3) a yellow 'S-curve' warning sign mounted above the Front-Right sidewalk, clearly visible in GT across all frames, is erased in Gen; (4) in Rear-Right, one of the parked black cars disappears; (5) the crossing white SUV in Front-Left (frames 6-8) has distorted geometry in Gen and becomes an amorphous white blur by frame 8. The scene progressively degrades as the clip advances — later frames lose more content than earlier ones. Ego trajectory is broadly consistent and no explicit decision error is observed. Classified as a Logic-level attack (object flicker and time-inconsistent disappearance of static vehicles) with strong secondary semantic effects."
"""


def build_prompt(video_id: str) -> str:
    """Inject the exact filename to reduce CSV/video_id mismatches."""
    return f"{YOUR_PROMPT}\n\nCurrent video filename: {video_id}\nUse this exact value in the video_id field."

def _state_name(file_obj) -> str:
    """Extract the state name as a string from a Gemini file object safely."""
    state = getattr(file_obj, "state", None)
    name = getattr(state, "name", None)
    if name is not None:
        return str(name).upper()
    return str(state).upper()


def _state_is(file_obj, target: str) -> bool:
    """Check if the state of a Gemini file object matches the target state."""
    s = _state_name(file_obj)
    target = target.upper()
    return s == target or s.endswith(f".{target}")


def wait_until_ready(file_obj, label: str, timeout: int = 600, interval: int = 5):
    """
    Poll the Gemini API until the uploaded file reaches the 'ACTIVE' state.
    Raises an error if the processing fails or times out.
    """
    print(f"  -> Waiting for Google servers to process the {label}", end="")
    start_time = time.time()

    while True:
        if _state_is(file_obj, "ACTIVE"):
            print()
            return file_obj

        if _state_is(file_obj, "FAILED"):
            print()
            raise RuntimeError(f"{label} processing failed: {getattr(file_obj, 'name', 'unknown')}")

        if time.time() - start_time > timeout:
            print()
            raise TimeoutError(f"{label} processing timed out after {timeout} seconds")

        print(".", end="", flush=True)
        time.sleep(interval)
        file_obj = client.files.get(name=file_obj.name)


def parse_and_validate_csv_row(text: str, expected_video_id: str):
    """
    Parse the raw text output from the model into a CSV row and rigorously
    validate the logical consistency of the data (scores, ranges, dependencies).
    """
    text = (text or "").strip()
    if not text:
        raise ValueError("Model returned empty text")

    # Parse the text as a CSV row
    row = next(csv.reader(StringIO(text)))
    # Ensure exactly 8 columns are returned
    if len(row) != 8:
        raise ValueError(f"Expected 8 columns, got {len(row)}: {row}")

    video_id, is_poisoned, attack_level, semantic, logical, decision, final_score, reasoning = [
        col.strip() for col in row
    ]

    # Validate video ID matches the current processing file
    if video_id != expected_video_id:
        raise ValueError(f"video_id mismatch: expected {expected_video_id}, got {video_id}")

    # Validate categorical and boolean fields
    if is_poisoned not in {"True", "False"}:
        raise ValueError(f"Invalid is_poisoned: {is_poisoned}")

    if attack_level not in {"None", "Semantic", "Logic", "Decision"}:
        raise ValueError(f"Invalid attack_level: {attack_level}")

    semantic = float(semantic)
    logical = float(logical)
    decision = float(decision)
    final_score = float(final_score)

    # Ensure all scores are within the valid [0, 1] range
    for name, value in {
        "semantic": semantic,
        "logical": logical,
        "decision": decision,
        "final_score": final_score,
    }.items():
        if not (0.0 <= value <= 1.0):
            raise ValueError(f"{name} out of range [0,1]: {value}")

    # Verify that the final score is the exact average of the three sub-scores
    avg_score = round((semantic + logical + decision) / 3.0, 2)
    if round(final_score, 2) != avg_score:
        raise ValueError(
            f"final_score mismatch: got {final_score}, expected {avg_score}"
        )

    # Check logical dependencies between is_poisoned flag and attack_level
    if is_poisoned == "False":
        if attack_level != "None":
            raise ValueError("If is_poisoned is False, attack_level must be None")
    else:
        if attack_level == "None":
            raise ValueError("If is_poisoned is True, attack_level must not be None")

    # Ensure the highest sub-score matches the declared attack_level
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


def process_video(video_path: str, instruction_file=None):
    """
    Process a single video file and call the Gemini API for parsing.
    Return a validated CSV row list on success, or None on failure.
    """
    print(f"\n[{time.strftime('%H:%M:%S')}] Started processing video: {video_path}")

    video_file = None

    try:
        print("  -> Uploading video...")
        video_file = client.files.upload(file=video_path)

        video_file = wait_until_ready(video_file, label="video", timeout=600, interval=5)
        print(f"  -> Video ready (URI: {video_file.uri})")

        print("  -> Requesting Gemini parsing, this may take a while, please wait...")
        # Construct the payload for the model (Prompt + Instruction PDF + Video)
        req_contents = [build_prompt(os.path.basename(video_path)), video_file]
        if instruction_file is not None:
            req_contents.insert(0, instruction_file)

        # Call the Gemini model
        response = client.models.generate_content(
            model="gemini-3.1-pro-preview",
            contents=req_contents,
            config=types.GenerateContentConfig(
                temperature=0.0,
                response_mime_type="text/plain",
            ),
        )
        response_text = response.text

        # Parse and logically validate the model's textual response
        row = parse_and_validate_csv_row(
            response_text,
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
        # Always ensure the uploaded video is deleted to prevent storage quota exhaustion
        if video_file is not None:
            print("  -> Cleaning up cloud video file to free space...")
            try:
                client.files.delete(name=video_file.name)
                print("  -> Cleanup complete.")
            except Exception as e:
                print(f"  -> [Warning] Failed to clean up cloud file: {e}")


def main():
    """Main entry point: sets up files, uploads shared resources, and iterates over videos."""
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

    # Prepare the local CSV results file with headers
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
            # Upload the PDF guideline once to be reused across all video requests
            print(f"Uploading Instruction PDF: {INSTRUCTION_PATH} ...")
            instruction_file = client.files.upload(file=INSTRUCTION_PATH)
            instruction_file = wait_until_ready(
                instruction_file, label="instruction PDF", timeout=300, interval=5
            )
            print("Upload complete.")

            # Process each video sequentially and append successful results to the CSV
            success_count = 0
            failure_count = 0

            for v_path in video_files:
                row, error = process_video(v_path, instruction_file)
                if row is not None:
                    writer.writerow(row)
                    # Persist immediately to disk to prevent data loss on crash
                    f.flush()
                    success_count += 1
                else:
                    failure_writer.writerow([os.path.basename(v_path), error or "Unknown error"])
                    failure_f.flush()
                    failure_count += 1

        finally:
            # Clean up the globally shared instruction PDF
            if instruction_file is not None:
                print("Cleaning up cloud Instruction PDF...")
                try:
                    client.files.delete(name=instruction_file.name)
                    print("Instruction PDF cleanup complete.")
                except Exception as e:
                    print(f"[Warning] Failed to clean up instruction PDF: {e}")

    print(f"\nAll done. Success: {success_count}, Failed: {failure_count}")
    print(f"Results saved to: {RESULT_CSV}")
    print(f"Failure log saved to: {FAILURE_CSV}")


if __name__ == "__main__":
    main()
