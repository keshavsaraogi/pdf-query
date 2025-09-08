# Chat with your PDF (Streamlit + FAISS)

A lightweight Streamlit app to chat with the contents of a PDF using Retrieval‑Augmented Generation (RAG). It extracts text from your PDF, embeds it with OpenAI, stores vectors in an in‑memory FAISS index, and answers questions via LangChain.

## Quick Start

1) Create a virtual environment (optional but recommended)

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```

2) Install dependencies

```bash
pip install -r requirements.txt
```

3) Configure environment variables

Create a `.env` file in the project root with your OpenAI key:

```bash
OPENAI_KEY=sk-your-key
```

4) Run the app

```bash
streamlit run streamlit.py
```

Open the provided local URL, upload a PDF, and start asking questions.

## How It Works

- PDF parsing: Reads all pages via PyPDF2 and concatenates text.
- Embeddings: Uses `OpenAIEmbeddings` to convert text to vectors.
- Vector store: Builds an in‑memory FAISS index for similarity search.
- Retrieval + LLM: Uses LangChain to query retrieved chunks with OpenAI.

Key implementation: see `streamlit.py`.

## Configuration

- `OPENAI_KEY`: Your OpenAI API key (required).

Optional files:

- `budget_speech.pdf`: Sample PDF you can try.

## Notes and Limitations

- In‑memory index: FAISS lives in memory; data resets when you restart the app.
- Single‑chunk ingestion: The app currently embeds the entire PDF as one document. For better retrieval, add text chunking (see below).
- Costs: Embedding and LLM calls use your OpenAI credits.

## Recommended Improvements (DIY)

1) Chunk the text for higher‑quality retrieval

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
docs = splitter.split_text(text)
vectordb = FAISS.from_texts(docs, embedding=embedding)
```

2) Persist and reload the FAISS index

```python
# Save
vectordb.save_local("faiss_index")

# Load (in a new session)
from langchain_community.vectorstores import FAISS
vectordb = FAISS.load_local("faiss_index", embedding, allow_dangerous_deserialization=True)
```

3) Show sources for transparency

- Keep metadata for each chunk (e.g., page numbers), and render the top retrieved passages alongside the answer.

4) Add conversational memory

- Use LangChain chat memory to keep context across turns.

## Troubleshooting

- No text extracted: Some PDFs are image‑based. Consider OCR (e.g., `pytesseract`, `ocrmypdf`) or a different parser.
- FAISS install issues: Use `faiss-cpu` (already in `requirements.txt`). Ensure Python ≥ 3.8.
- API key errors: Verify `OPENAI_KEY` is set in `.env` and your shell session is using the right environment.

## Repo Structure

- `streamlit.py` – Streamlit app, RAG logic
- `requirements.txt` – Dependencies (OpenAI, LangChain, FAISS, PyPDF2, Streamlit)
- `budget_speech.pdf` – Example PDF
- `.env` – Environment variables (not committed)

