"""
Main Entry Point — Data Scientist Agentic AI
--------------------------------------------

Upgraded:

✔ Full cleaner → EDA → ML pipeline (as before)
✔ JSON-safe output
✔ Optional PDF report check
✔ NEW: QA mode over EDA + ML insight JSONs
"""

import argparse
import json
import os
from dotenv import load_dotenv

from coordinator.coordinator import Coordinator

load_dotenv()


def safe_convert(value):
    """
    Safely convert objects (DataFrames, models, paths) to JSON-friendly strings.
    """
    try:
        if hasattr(value, "to_dict"):
            return value.to_dict()
        return str(value)
    except Exception:
        return "UNSERIALIZABLE"


def run_pipeline(file_path: str, task: str, target: str = None):
    """
    Run the full multi-agent pipeline and print clean JSON-friendly results.
    """

    print("\n🚀 Starting Data Scientist Agentic AI pipeline...\n")

    coordinator = Coordinator()

    # Run the orchestrated LLM-driven pipeline
    result = coordinator.run(
        request=task,
        dataset_path=file_path,
        target_column=target,
    )

    # Clean results for printing
    safe_output = {}

    for agent_name, agent_result in result.items():
        safe_output[agent_name] = {
            "success": agent_result.success,
            "error": agent_result.error,
            "messages": agent_result.messages,
            "outputs": {
                key: safe_convert(val)
                for key, val in agent_result.outputs.items()
            },
        }

    # Print pretty JSON
    print("\n================ Final Output ================\n")
    print(json.dumps(safe_output, indent=4))
    print("\n==============================================\n")

    # If PDF report exists, show where it is
    pdf_path = os.path.join("outputs", "final", "final_report.pdf")
    if os.path.exists(pdf_path):
        print(f"📄 Final PDF report generated at: {pdf_path}")
    else:
        print("ℹ No final report generated yet.")

    return coordinator  # return so we can reuse for QA if desired


def run_qa(question: str):
    """
    Run QAAgent over the existing EDA + ML insight JSONs.
    Assumes:
      - outputs/ml/ml_insights_report.json
      - outputs/insights/insights_report.json
    already exist.
    """
    print("\n💬 Starting Q&A over existing insight reports...\n")

    coordinator = Coordinator()
    qa_result = coordinator.answer_question(question)

    safe_output = {
        "success": qa_result.success,
        "error": qa_result.error,
        "messages": qa_result.messages,
        "outputs": {
            key: safe_convert(val)
            for key, val in qa_result.outputs.items()
        },
    }

    print("\n================ QA Output ================\n")
    print(json.dumps(safe_output, indent=4))
    print("\n===========================================\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data Scientist Agentic AI CLI")

    parser.add_argument("--file", required=False, help="Path to CSV dataset file")
    parser.add_argument("--task", required=False, help="User instruction to the AI")
    parser.add_argument("--target", required=False, help="Target column (optional)")
    parser.add_argument(
        "--qa_question",
        required=False,
        help="Ask a question based on generated EDA + ML insight JSONs",
    )

    args = parser.parse_args()

    # 1) QA-only mode (no pipeline)
    if args.qa_question and not (args.file or args.task):
        run_qa(args.qa_question)

    # 2) Pipeline + (optional) QA in one run
    elif args.file and args.task:
        coord = run_pipeline(
            file_path=args.file,
            task=args.task,
            target=args.target,
        )

        if args.qa_question:
            # Reuse coordinator instance for QA (same config)
            qa_result = coord.answer_question(args.qa_question)
            safe_output = {
                "success": qa_result.success,
                "error": qa_result.error,
                "messages": qa_result.messages,
                "outputs": {
                    key: safe_convert(val)
                    for key, val in qa_result.outputs.items()
                },
            }
            print("\n================ QA Output ================\n")
            print(json.dumps(safe_output, indent=4))
            print("\n===========================================\n")

    # 3) Invalid combo
    else:
        print(
            "\n❌ Invalid arguments.\n"
            "Either:\n"
            "  - Run full pipeline:\n"
            "      python main.py --file data/raw/sample.csv --task \"run full eda and insights\" [--target TargetCol]\n"
            "  - OR run QA over existing reports only:\n"
            "      python main.py --qa_question \"What is the best model?\"\n"
        )
