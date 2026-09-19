import os
import uuid
from pathlib import Path

from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from pypdf import PdfReader
from docx import Document
from PIL import Image
import pytesseract

from google import genai
from google.genai import types
import chromadb
from chromadb.utils import embedding_functions

load_dotenv()

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 30 * 1024 * 1024

UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise RuntimeError("GEMINI_API_KEY is missing. Add it to .env")

client = genai.Client(api_key=API_KEY)

# Persistent local Chroma vector store for development.
chroma_client = chromadb.PersistentClient(path="data/chroma")
embedding_fn = embedding_functions.DefaultEmbeddingFunction()
collection = chroma_client.get_or_create_collection(
    name="agriculture_documents",
    embedding_function=embedding_fn,
)

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt", ".png", ".jpg", ".jpeg", ".webp"}

SYSTEM_PROMPT = """You are Agriculture Knowledge RAG AI, a strict document-grounded agriculture assistant.

Your ONLY source of information is the DOCUMENT CONTEXT supplied below.
You MUST NOT use:
- your own general knowledge
- internet information
- external websites
- outside databases
- information not present in the supplied context

If the requested information is not supported by the supplied DOCUMENT CONTEXT, reply:
"I couldn't find this information in the uploaded agriculture document(s)."

Do not guess, fill gaps, or invent facts.
When possible, mention the source filename/page shown in the context.
Answer clearly and use only the retrieved document content.

DOCUMENT CONTEXT:
{context}

USER QUESTION:
{question}
"""

def extract_pages(path: Path):
    ext = path.suffix.lower()
    pages = []

    if ext == ".pdf":
        reader = PdfReader(str(path))
        for i, page in enumerate(reader.pages, 1):
            text = page.extract_text() or ""
            if text.strip():
                pages.append((f"{path.name} - page {i}", text))

    elif ext == ".docx":
        doc = Document(str(path))
        text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
        if text.strip():
            pages.append((path.name, text))

    elif ext == ".txt":
        text = path.read_text(encoding="utf-8", errors="ignore")
        if text.strip():
            pages.append((path.name, text))

    elif ext in {".png", ".jpg", ".jpeg", ".webp"}:
        text = pytesseract.image_to_string(Image.open(path))
        if text.strip():
            pages.append((f"{path.name} - OCR", text))

    return pages

def chunk_text(text, size=1200, overlap=200):
    clean = " ".join(text.split())
    if not clean:
        return []
    chunks = []
    start = 0
    while start < len(clean):
        end = min(start + size, len(clean))
        chunks.append(clean[start:end])
        if end >= len(clean):
            break
        start = end - overlap
    return chunks

def index_file(path):
    pages = extract_pages(path)
    ids, docs, metas = [], [], []

    for source, text in pages:
        for i, chunk in enumerate(chunk_text(text)):
            ids.append(str(uuid.uuid4()))
            docs.append(chunk)
            metas.append({
                "source": source,
                "filename": path.name,
                "chunk": i
            })

    if docs:
        collection.add(ids=ids, documents=docs, metadatas=metas)
    return len(docs)

def retrieve(question, n=6):
    result = collection.query(
        query_texts=[question],
        n_results=n
    )
    docs = result.get("documents", [[]])[0]
    metas = result.get("metadatas", [[]])[0]
    distances = result.get("distances", [[]])[0]
    return docs, metas, distances

@app.get("/")
def home():
    return render_template("index.html")

@app.post("/upload")
def upload():
    files = request.files.getlist("files")
    if not files:
        return jsonify({"error": "Please select at least one agriculture document."}), 400

    results = []
    total = 0

    for file in files:
        if not file.filename:
            continue

        ext = Path(file.filename).suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            return jsonify({
                "error": f"{ext} is not supported. Use PDF, DOCX, TXT, PNG, JPG, JPEG or WEBP."
            }), 400

        safe_name = f"{uuid.uuid4().hex}_{Path(file.filename).name}"
        path = UPLOAD_DIR / safe_name
        file.save(path)

        try:
            chunks = index_file(path)
            results.append({"name": file.filename, "chunks": chunks})
            total += chunks
        except Exception as exc:
            return jsonify({"error": f"Could not process {file.filename}: {exc}"}), 500

    return jsonify({
        "message": "Agriculture documents indexed successfully.",
        "files": results,
        "chunks": total
    })

@app.post("/chat")
def chat():
    data = request.get_json(silent=True) or {}
    question = (data.get("question") or "").strip()

    if not question:
        return jsonify({"error": "Please enter an agriculture question."}), 400

    docs, metas, distances = retrieve(question)

    # Conservative retrieval gate. This prevents weak matches from being
    # presented to the LLM as authoritative document context.
    useful = []
    for doc, meta, distance in zip(docs, metas, distances):
        if distance is None or distance <= 1.35:
            useful.append((doc, meta))

    if not useful:
        return jsonify({
            "answer": "I couldn't find this information in the uploaded agriculture document(s).",
            "sources": []
        })

    context = "\n\n".join(
        f"[Source: {meta.get('source', 'uploaded agriculture document')}]\n{doc}"
        for doc, meta in useful
    )

    prompt = SYSTEM_PROMPT.format(context=context, question=question)

    try:
        response = client.models.generate_content(
            model=os.getenv("GEMINI_MODEL", "gemini-3.5-flash"),
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.0,
                max_output_tokens=900
            )
        )
        answer = (response.text or "").strip()
    except Exception as exc:
        return jsonify({"error": f"Gemini error: {exc}"}), 500

    sources = []
    seen = set()
    for _, meta in useful:
        source = meta.get("source", "uploaded agriculture document")
        if source not in seen:
            seen.add(source)
            sources.append(source)

    return jsonify({"answer": answer, "sources": sources})

@app.post("/clear")
def clear():
    global collection

    try:
        chroma_client.delete_collection("agriculture_documents")
    except Exception:
        pass

    collection = chroma_client.get_or_create_collection(
        name="agriculture_documents",
        embedding_function=embedding_fn
    )

    for path in UPLOAD_DIR.glob("*"):
        try:
            path.unlink()
        except Exception:
            pass

    return jsonify({"message": "Agriculture documents and RAG index cleared."})

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.getenv("PORT", "5000")),
        debug=True
    )
