import streamlit as st
from utils import extract_text_from_pdf, summarize_and_classify

st.set_page_config(page_title="AI News Summarizer", layout="centered")
st.title("📰 AI Newspaper Summarizer for B-School Students")

uploaded_file = st.file_uploader("Upload a newspaper PDF", type="pdf")

HUGGINGFACE_API_KEY = st.secrets.get("HUGGINGFACE_API_KEY") or st.text_input("Enter your HuggingFace API Key:", type="password")

if uploaded_file and HUGGINGFACE_API_KEY:
    with st.spinner("Processing..."):
        text = extract_text_from_pdf(uploaded_file)
        business, others = summarize_and_classify(text, HUGGINGFACE_API_KEY)

    st.subheader("🧠 Important Business News")
    for headline, summary in business:
        st.markdown(f"**🔹 {headline}**")
        st.write(summary)
        st.markdown("---")

    st.subheader("📰 Other News Headlines")
    for headline, summary in others:
        st.markdown(f"**• {headline}**: {summary}")
