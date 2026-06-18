import streamlit as st
import requests

BACKEND_URL = "http://localhost:8000"

st.set_page_config(
    page_title="Facet Evaluator"
)

st.title(
    "Conversation Evaluation"
)

conversation = st.text_area(
    "Enter Conversation",
    height=250
)

if st.button("Evaluate"):

    response = requests.post(
        f"{BACKEND_URL}/evaluate",
        json={
            "conversation":
            conversation
        }
    )

    result = response.json()

    st.success(
        "Evaluation Complete"
    )

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Score",
            result["score"]
        )

    with col2:
        st.metric(
            "Confidence",
            result["confidence"]
        )

    st.subheader(
        "Selected Facets"
    )

    st.write(
        result["selected_facets"]
    )

    st.subheader(
        "Reason"
    )

    st.write(
        result["reason"]
    )