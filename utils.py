import fitz  # PyMuPDF
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
import re

summarizer = pipeline("summarization", model="google/pegasus-xsum")
classifier = pipeline("zero-shot-classification")

def extract_text_from_pdf(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def chunk_text(text, max_length=1000):
    sentences = re.split(r'\. |\n', text)
    chunks, current_chunk = [], ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) < max_length:
            current_chunk += sentence + ". "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + ". "
    chunks.append(current_chunk.strip())
    return chunks

def summarize_and_categorize(text):
    chunks = chunk_text(text)
    business_news, other_news = [], []

    for chunk in chunks:
        if len(chunk.split()) < 20:
            continue

        summary = summarizer(chunk, max_length=100, min_length=30, do_sample=False)[0]['summary_text']
        headline = summary.split(".")[0]

        result = classifier(summary,
                            candidate_labels=["business", "sports", "politics", "entertainment", "education"],
                            multi_label=False)
        
        if result['labels'][0] == "business":
            business_news.append((headline, summary))
        else:
            other_news.append((headline, summary[:80] + "..."))

    return business_news, other_news
