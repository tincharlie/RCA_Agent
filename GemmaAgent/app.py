import streamlit as st
from router import app_graph
from state import RCAState

st.set_page_config(page_title="RCA Reliability System", page_icon="⚙️", layout="wide")

st.title("⚙️ Root Cause Analysis System")

equipment_id = st.text_input("Equipment ID", value="PUMP-K204")
question = st.text_area(
    "Query / Observed Issue",
    value="Why is the pump experiencing high vibration and temperature spikes?",
)

if st.button("Run Diagnostic Graph", type="primary"):
    input_state: RCAState = {
        "question": question,
        "equipment_id": equipment_id,
    }

    with st.spinner("Executing agent graph..."):
        final_state = app_graph.invoke(input_state)

    st.success("Analysis Complete")

    col1, col2, col3 = st.columns(3)
    col1.metric("Confidence Score", f"{final_state.get('confidence', 0.0):.2f}")
    col2.metric("Faithfulness Score", f"{final_state.get('faithfulness_score', 0.0):.2f}")
    col3.metric("Hallucination Flag", str(final_state.get("is_hallucinated", False)))

    st.subheader("Final Report")
    st.code(final_state.get("report", ""), language="text")