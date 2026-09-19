# Agriculture Knowledge RAG AI

A strict document-grounded RAG chatbot for agriculture knowledge.

## What it does

Upload agriculture-related:
- PDF books
- PDF notes
- DOCX documents
- TXT files
- JPG/PNG/WEBP images

The application extracts text, splits it into chunks, creates embeddings, stores them in Chroma, retrieves relevant chunks for each question, and sends only that retrieved context to Gemini.

## Strict rule

The chatbot is NOT a general agriculture chatbot.

It must answer only from retrieved content from the user's uploaded documents.

If the answer is not supported by the uploaded documents, it responds:

"I couldn't find this information in the uploaded agriculture document(s)."

No internet search or external source is implemented.

## Local setup

Windows PowerShell:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Create `.env`:

```text
GEMINI_API_KEY=YOUR_GEMINI_API_KEY
GEMINI_MODEL=gemini-3.5-flash
```

Run:

```powershell
python app.py
```

Open:

```text
http://127.0.0.1:5000
```

## Image OCR

Image files are processed using Tesseract OCR. Install Tesseract separately and make sure the `tesseract` command is available in PATH.

## RAG architecture

Agriculture document
→ text extraction/OCR
→ chunking
→ embeddings
→ Chroma vector database
→ semantic retrieval
→ Gemini
→ document-grounded answer

## Deployment

`render.yaml` is included for Render.

Set `GEMINI_API_KEY` in Render environment variables.

### Production persistence note

The sample uses local Chroma and local uploaded files. Cloud deployments may use ephemeral storage. If uploaded documents must survive restarts/redeployments, use persistent storage plus a hosted vector database (for example, a managed Chroma-compatible/vector service). The RAG logic remains the same.
