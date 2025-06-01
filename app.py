import streamlit as st
import fitz  # PyMuPDF
import re
import nltk
import os
from together import Together
from dotenv import load_dotenv

nltk.download("punkt")
from nltk.tokenize import sent_tokenize

# Load .env variables
load_dotenv()

# --- Setup ---
st.set_page_config(page_title="Law Case Summarizer", layout="centered")
st.title("📄 Law Case Summarizer App")

st.markdown(
    """
    This tool helps **lawyers** review lengthy legal documents quickly by summarizing key points.
    Just upload a PDF file containing a case or judgment, and get a concise summary instantly.
    """
)

# Load API key from environment variables
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
if not TOGETHER_API_KEY:
    st.error("❌ API key for Together AI not found. Please set it in your .env file.")
    st.stop()

# Initialize Together client
client = Together(api_key=TOGETHER_API_KEY)

# Load PDF
pdf_file = st.file_uploader("Upload a legal PDF file", type=["pdf"])


# Clean extracted text
def clean_text(text):
    text = text.replace("\n", " ")
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"[^\x00-\x7F]+", "", text)
    return text


# Extract text from PDF
def extract_text_from_pdf(file):
    text = ""
    with fitz.open(stream=file.read(), filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()
    return clean_text(text)


# Break into chunks for long texts
def chunk_text(text, max_chunk=1200):
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= max_chunk:
            current_chunk += sentence + " "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + " "
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks


# Summarize chunk using Together AI client
def summarize_chunk_with_together(text):
    try:
        prompt = (
            "state the date of decision"
            "Summarize the following text clearly and concisely "
            
            f"Text:\n{text}\n\nSummary:"
        )
        response = client.completions.create(
            model="meta-llama/Llama-2-70b-hf",
            prompt=prompt,
            max_tokens=100,  # increase to get fuller summary
            temperature=0.4,  # lower temp for focused output
            top_p=0.4,
        )
        return response.choices[0].text.strip()
    except Exception as e:
        return f"⚠️ Error summarizing chunk: {str(e)}"


# Main summarization logic
def summarize_text(text):
    chunks = chunk_text(text)
    summaries = []
    for i, chunk in enumerate(chunks):
        #st.info(f"🔍 Summarizing chunk {i + 1}/{len(chunks)}...")
        summary = summarize_chunk_with_together(chunk)
        summaries.append(summary)
    return "\n\n".join(summaries)


# Run app logic
if pdf_file:
    with st.spinner("📄 Extracting text from PDF..."):
        raw_text = extract_text_from_pdf(pdf_file)

    if st.button("📝 Generate Summary"):
        with st.spinner("✍️ Summarizing..."):
            final_summary = summarize_text(raw_text)

        st.subheader("📌 Summary:")
        st.success(final_summary)
