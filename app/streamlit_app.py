# app/streamlit_app.py
"""
Streamlit UI for Data Scientist Agentic AI.

Run with:
    streamlit run app/streamlit_app.py
"""

import streamlit as st
import pandas as pd
import os

from coordinator.coordinator import Coordinator
from llm.openrouter_client import OpenRouterClient

st.set_page_config(
    page_title="Data Scientist Agentic AI",
    layout="wide",
)

st.title("🤖 Data Scientist Agentic AI Dashboard")
st.write("Upload your dataset and let the multi-agent AI pipeline take over!")


# --------------------------------------------------------------------
# File Upload
# --------------------------------------------------------------------

uploaded_file = st.file_uploader("Upload CSV dataset", type=["csv"])
user_request = st.text_area(
    "What should the AI do?",
    "Clean data, run EDA and train a classification model."
)
target_column = st.text_input("Target Column (optional)", "")

run_button = st.button("Run Pipeline 🚀")


# --------------------------------------------------------------------
# Execute Pipeline
# --------------------------------------------------------------------

if run_button:
    if uploaded_file is None:
        st.error("Please upload a CSV file.")
        st.stop()

    # Save file locally
    os.makedirs("data/raw", exist_ok=True)
    csv_path = os.path.join("data/raw", uploaded_file.name)
    with open(csv_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.info("Processing… This may take a moment depending on model size.")
    st.write("---")

    coordinator = Coordinator()
    result = coordinator.run(
        request=user_request,
        dataset_path=csv_path,
        target_column=target_column if target_column else None
    )

    # ----------------------------------------------------------------
    # Display Results
    # ----------------------------------------------------------------

    for agent_name, res in result.items():
        st.subheader(f"🧠 {agent_name.upper()}")
        st.write(f"**Success:** {res.success}")

        if res.error:
            st.error(res.error)

        st.write("**Messages:**")
        for m in res.messages:
            st.write(f"- {m}")

        st.write("**Outputs:**")
        for k, v in res.outputs.items():
            if isinstance(v, pd.DataFrame):
                st.dataframe(v)
            else:
                st.text(f"{k}: {v}")

        st.write("---")

    st.success("Pipeline completed successfully! 🎉")
