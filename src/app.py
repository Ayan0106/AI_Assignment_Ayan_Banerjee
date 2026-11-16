import streamlit as st
from predict import predict_ticket

RESPONSES = {
    # paste your 11-category dictionary here
}

st.title("SAMADHAN SUCHAK- AI-Powered Customer Support Classification")
text = st.text_area("Ticket Description")
if st.button("Predict") and text.strip():
    cat, conf = predict_ticket(text)
    st.success(f"Predicted Category: {cat}")
    st.write(f"Confidence: {conf:.1f}%")
    st.info("Suggested Response: " + RESPONSES.get(cat, "Our support team will review your ticket shortly."))