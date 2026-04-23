from pathlib import Path
import os
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Improvement.common_prompt import IMPROVED_PROMPT, build_openai_prompt  # noqa: E402
import Openai.Openai_parse as base  # noqa: E402


HERE = Path(__file__).resolve().parent

base.RESULT_CSV = str(HERE / "openai_improved_parse.csv")
base.FAILURE_CSV = str(HERE / "openai_improved_failures.csv")
base.YOUR_PROMPT = IMPROVED_PROMPT
base.build_prompt = build_openai_prompt
base.MODEL_NAME = os.environ.get("OPENAI_MODEL", "gpt-5.4")


if __name__ == "__main__":
    base.main()
