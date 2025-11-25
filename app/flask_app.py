# app/flask_app.py
"""
Flask API for Data Scientist Agentic AI.
Allows:
- File upload
- Running the multi-agent pipeline
- Returning results as JSON

Run with:
    python app/flask_app.py
"""

from flask import Flask, request, jsonify
import os
from werkzeug.utils import secure_filename

from coordinator.coordinator import Coordinator
from llm.openrouter_client import OpenRouterClient

# Ensure directories exist
os.makedirs("data/raw", exist_ok=True)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "data/raw"


# --------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------

@app.route("/")
def index():
    return jsonify({"message": "Data Scientist Agentic AI API is running"})


@app.route("/run", methods=["POST"])
def run_pipeline():
    """
    Expected form-data:
    - file: CSV file
    - request: natural language request ("clean data", "run eda", "train model")
    - target_column: optional
    """
    if "file" not in request.files:
        return jsonify({"error": "CSV file not provided"}), 400

    csv_file = request.files["file"]
    user_request = request.form.get("request", "Perform full pipeline")
    target_column = request.form.get("target_column", None)

    # Save file
    filename = secure_filename(csv_file.filename)
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    csv_file.save(file_path)

    # Run Coordinator pipeline
    coordinator = Coordinator()
    result = coordinator.run(
        request=user_request,
        dataset_path=file_path,
        target_column=target_column
    )

    # Convert non-JSON objects (e.g., models, dataframes)
    def safe_convert(obj):
        try:
            return str(obj)
        except:
            return "unserializable"

    safe_output = {
        k: {
            "success": v.success,
            "error": v.error,
            "messages": v.messages,
            "outputs": {
                kk: safe_convert(vv) for kk, vv in v.outputs.items()
            }
        }
        for k, v in result.items()
    }

    return jsonify(safe_output)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
