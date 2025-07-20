import streamlit as st
from utils import extract_text_from_pdf, summarize_and_categorize

st.set_page_config(page_title="AI News Summarizer", layout="centered")
st.title("📰 AI Newspaper Summarizer for B-School Students")

uploaded_file = st.file_uploader("Upload a newspaper PDF", type="pdf")

if uploaded_file:
    with st.spinner("Processing..."):
        text = extract_text_from_pdf(uploaded_file)
        business_news, other_news = summarize_and_categorize(text)

    st.subheader("🧠 Important Business News")
    for headline, summary in business_news:
        st.markdown(f"**🔹 {headline}**")
        st.write(summary)
        st.markdown("---")

    st.subheader("📰 Other News Headlines")
    for headline, summary in other_news:
        st.markdown(f"**• {headline}**: {summary}")
