import re
import requests
from PyPDF2 import PdfFileReader

SUMMARIZER_URL = "https://api-inference.huggingface.co/models/google/pegasus-xsum"
CLASSIFIER_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-mnli"

def extract_text_from_pdf(file):
    reader = PdfFileReader(file)
    text = ""
    for page in reader.pages:
        content = page.extract_text()
        if content:
            text += content + "\n"
    return text

def summarize_and_classify(text, api_key):
    headers = {"Authorization": f"Bearer {api_key}"}
    chunks, current, business, others = [], "", [], []

    sentences = re.split(r'\. |\n', text)
    for sentence in sentences:
        if len(current) + len(sentence) < 1000:
            current += sentence + ". "
        else:
            chunks.append(current.strip())
            current = sentence + ". "
    if current:
        chunks.append(current.strip())

    for chunk in chunks:
        if len(chunk.split()) < 20:
            continue

        sum_resp = requests.post(SUMMARIZER_URL, headers=headers, json={"inputs": chunk})
        if not sum_resp.ok: continue
        summary = sum_resp.json()[0]["summary_text"]
        headline = summary.split(".")[0]

        clf_resp = requests.post(CLASSIFIER_URL, headers=headers,
            json={"inputs": summary, "parameters": {"candidate_labels": ["business", "sports", "politics", "entertainment", "education"]}})
        if not clf_resp.ok: continue
        label = clf_resp.json()["labels"][0]

        if label == "business":
            business.append((headline, summary))
        else:
            others.append((headline, summary[:80] + "..."))

    return business, others
