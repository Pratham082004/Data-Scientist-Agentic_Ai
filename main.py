# main.py
"""
Main CLI runner for Data Scientist Agentic AI.

Usage:
    python main.py --file data/raw/sample.csv --task "clean data and run eda" --target Outcome
"""

import argparse
import json
from dotenv import load_dotenv

from coordinator.coordinator import Coordinator

load_dotenv()


def safe_convert(obj):
    try:
        return str(obj)
    except:
        return "unserializable"


def run_pipeline(file_path: str, task: str, target: str = None):
    coordinator = Coordinator()
    result = coordinator.run(
        request=task,
        dataset_path=file_path,
        target_column=target,
    )

    safe_output = {
        k: {
            "success": v.success,
            "error": v.error,
            "messages": v.messages,
            "outputs": {kk: safe_convert(vv) for kk, vv in v.outputs.items()},
        }
        for k, v in result.items()
    }

    print(json.dumps(safe_output, indent=4))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data Scientist Agentic AI")

    parser.add_argument("--file", required=True, help="Path to CSV file")
    parser.add_argument("--task", required=True, help="Describe what the AI should do")
    parser.add_argument("--target", required=False, help="Target column")

    args = parser.parse_args()

    run_pipeline(
        file_path=args.file,
        task=args.task,
        target=args.target,
    )
