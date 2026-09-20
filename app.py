import logging
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


# =========================================================
# ENVIRONMENT
# =========================================================

load_dotenv()


# =========================================================
# LOGGING
# =========================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

logger = logging.getLogger(__name__)


# =========================================================
# FLASK APP
# =========================================================

app = Flask(__name__)

# Maximum upload size: 30 MB
app.config["MAX_CONTENT_LENGTH"] = 30 * 1024 * 1024


# =========================================================
# DIRECTORIES
# =========================================================

UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

CHROMA_DIR = Path("data/chroma")
CHROMA_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# GEMINI CONFIGURATION
# =========================================================

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY is missing. "
        "Add GEMINI_API_KEY in Render Environment Variables."
    )

client = genai.Client(api_key=API_KEY)


# Main chat / answer model
GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.1-flash-lite"
)


# Gemini embedding model for RAG
EMBEDDING_MODEL = "gemini-embedding-001"

# Smaller embedding size reduces storage and processing cost.
# Google supports 768 / 1536 / 3072 dimensions.
EMBEDDING_DIMENSION = 768


# =========================================================
# CHROMA CONFIGURATION
# =========================================================

chroma_client = None
collection = None

# New collection name.
# This prevents conflict with the previous local embedding
# collection that may already exist.
COLLECTION_NAME = "agriculture_documents_v2"


def get_collection():
    """
    Initialize Chroma lazily.

    Chroma is created only when upload/chat/clear actually
    needs it. This keeps Gunicorn startup lightweight.
    """

    global chroma_client
    global collection

    if collection is not None:
        return collection

    logger.info("Initializing Chroma vector database...")

    chroma_client = chromadb.PersistentClient(
        path=str(CHROMA_DIR)
    )

    collection = chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={
            "hnsw:space": "cosine"
        }
    )

    logger.info(
        "Chroma initialized. Existing chunks: %s",
        collection.count()
    )

    return collection


# =========================================================
# ALLOWED FILE TYPES
# =========================================================

ALLOWED_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".txt",
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
}


# =========================================================
# STRICT RAG SYSTEM PROMPT
# =========================================================

SYSTEM_PROMPT = """
You are Agriculture Knowledge RAG AI, a strict document-grounded
agriculture assistant.

Your ONLY source of information is the DOCUMENT CONTEXT supplied below.

You MUST NOT use:
- your own general knowledge
- internet information
- external websites
- outside databases
- information not present in the supplied context

If the requested information is not supported by the supplied
DOCUMENT CONTEXT, reply exactly:

"I couldn't find this information in the uploaded agriculture document(s)."

Do not guess.
Do not fill gaps.
Do not invent facts.

When possible, mention the source filename and page shown
in the document context.

Answer clearly using ONLY the retrieved document content.

DOCUMENT CONTEXT:
{context}

USER QUESTION:
{question}
"""


# =========================================================
# TEXT EXTRACTION
# =========================================================

def extract_pages(path: Path):
    """
    Extract text from PDF, DOCX, TXT and image files.

    Returns:
        [
            ("filename - page 1", "text..."),
            ("filename - page 2", "text...")
        ]
    """

    ext = path.suffix.lower()
    pages = []

    logger.info("Extracting text from: %s", path.name)

    try:

        # -------------------------------------------------
        # PDF
        # -------------------------------------------------

        if ext == ".pdf":

            reader = PdfReader(str(path))

            logger.info(
                "PDF %s contains %s pages",
                path.name,
                len(reader.pages)
            )

            for page_number, page in enumerate(
                reader.pages,
                start=1
            ):

                text = page.extract_text() or ""

                if text.strip():

                    pages.append(
                        (
                            f"{path.name} - page {page_number}",
                            text
                        )
                    )


        # -------------------------------------------------
        # DOCX
        # -------------------------------------------------

        elif ext == ".docx":

            doc = Document(str(path))

            text = "\n".join(
                paragraph.text
                for paragraph in doc.paragraphs
                if paragraph.text.strip()
            )

            if text.strip():

                pages.append(
                    (
                        path.name,
                        text
                    )
                )


        # -------------------------------------------------
        # TXT
        # -------------------------------------------------

        elif ext == ".txt":

            text = path.read_text(
                encoding="utf-8",
                errors="ignore"
            )

            if text.strip():

                pages.append(
                    (
                        path.name,
                        text
                    )
                )


        # -------------------------------------------------
        # IMAGE OCR
        # -------------------------------------------------

        elif ext in {
            ".png",
            ".jpg",
            ".jpeg",
            ".webp"
        }:

            image = Image.open(path)

            text = pytesseract.image_to_string(
                image
            )

            if text.strip():

                pages.append(
                    (
                        f"{path.name} - OCR",
                        text
                    )
                )


    except Exception as exc:

        logger.exception(
            "Text extraction failed for %s",
            path.name
        )

        raise RuntimeError(
            f"Could not extract text from {path.name}: {exc}"
        )


    logger.info(
        "Extracted %s text sections from %s",
        len(pages),
        path.name
    )

    return pages


# =========================================================
# TEXT CHUNKING
# =========================================================

def chunk_text(
    text,
    size=1200,
    overlap=200
):
    """
    Split text into overlapping chunks.

    Character-based chunking is used to keep the implementation
    simple and lightweight for Render.
    """

    clean = " ".join(
        text.split()
    )

    if not clean:
        return []

    chunks = []

    start = 0

    while start < len(clean):

        end = min(
            start + size,
            len(clean)
        )

        chunk = clean[start:end]

        if chunk.strip():
            chunks.append(
                chunk.strip()
            )

        if end >= len(clean):
            break

        start = end - overlap

    return chunks


# =========================================================
# GEMINI DOCUMENT EMBEDDINGS
# =========================================================

def generate_document_embeddings(texts):
    """
    Generate Gemini embeddings for document chunks.

    Documents use RETRIEVAL_DOCUMENT because these vectors
    will later be searched using RETRIEVAL_QUERY embeddings.
    """

    if not texts:
        return []

    all_embeddings = []

    # Small batches help reduce memory usage.
    batch_size = 20

    total = len(texts)

    for start in range(
        0,
        total,
        batch_size
    ):

        batch = texts[
            start:start + batch_size
        ]

        logger.info(
            "Generating document embeddings: %s-%s of %s",
            start + 1,
            min(
                start + len(batch),
                total
            ),
            total
        )

        try:

            response = client.models.embed_content(
                model=EMBEDDING_MODEL,
                contents=batch,
                config=types.EmbedContentConfig(
                    task_type="RETRIEVAL_DOCUMENT",
                    output_dimensionality=EMBEDDING_DIMENSION,
                ),
            )

        except Exception as exc:

            logger.exception(
                "Gemini document embedding failed"
            )

            raise RuntimeError(
                f"Gemini document embedding failed: {exc}"
            )


        if not response.embeddings:

            raise RuntimeError(
                "Gemini returned no document embeddings."
            )


        for embedding in response.embeddings:

            all_embeddings.append(
                embedding.values
            )


    if len(all_embeddings) != len(texts):

        raise RuntimeError(
            "Embedding count does not match "
            "document chunk count."
        )


    return all_embeddings


# =========================================================
# GEMINI QUERY EMBEDDING
# =========================================================

def generate_query_embedding(question):
    """
    Generate a Gemini embedding for the user's question.

    Queries use RETRIEVAL_QUERY because the indexed documents
    use RETRIEVAL_DOCUMENT.
    """

    try:

        response = client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=question,
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_QUERY",
                output_dimensionality=EMBEDDING_DIMENSION,
            ),
        )

    except Exception as exc:

        logger.exception(
            "Gemini query embedding failed"
        )

        raise RuntimeError(
            f"Gemini query embedding failed: {exc}"
        )


    if not response.embeddings:

        raise RuntimeError(
            "Gemini returned no query embedding."
        )


    return response.embeddings[0].values


# =========================================================
# INDEX DOCUMENT
# =========================================================

def index_file(path: Path):

    collection_obj = get_collection()

    pages = extract_pages(path)

    ids = []
    docs = []
    metas = []


    # -----------------------------------------------------
    # Create chunks
    # -----------------------------------------------------

    for source, text in pages:

        chunks = chunk_text(text)

        for chunk_number, chunk in enumerate(
            chunks
        ):

            ids.append(
                str(uuid.uuid4())
            )

            docs.append(
                chunk
            )

            metas.append(
                {
                    "source": source,
                    "filename": path.name,
                    "chunk": chunk_number,
                }
            )


    # -----------------------------------------------------
    # No text found
    # -----------------------------------------------------

    if not docs:

        logger.warning(
            "No readable text found in %s",
            path.name
        )

        return 0


    logger.info(
        "Prepared %s chunks from %s",
        len(docs),
        path.name
    )


    # -----------------------------------------------------
    # Generate Gemini embeddings
    # -----------------------------------------------------

    embeddings = generate_document_embeddings(
        docs
    )


    # -----------------------------------------------------
    # Add to Chroma
    # -----------------------------------------------------

    logger.info(
        "Adding %s chunks to Chroma...",
        len(docs)
    )

    try:

        collection_obj.add(
            ids=ids,
            documents=docs,
            embeddings=embeddings,
            metadatas=metas,
        )

    except Exception as exc:

        logger.exception(
            "Chroma indexing failed"
        )

        raise RuntimeError(
            f"Chroma indexing failed: {exc}"
        )


    logger.info(
        "Successfully indexed %s (%s chunks)",
        path.name,
        len(docs)
    )

    return len(docs)


# =========================================================
# RETRIEVAL
# =========================================================

def retrieve(
    question,
    n=6
):

    collection_obj = get_collection()

    count = collection_obj.count()

    if count == 0:

        return [], [], []


    n = min(
        n,
        count
    )


    # -----------------------------------------------------
    # Create query embedding
    # -----------------------------------------------------

    query_embedding = generate_query_embedding(
        question
    )


    # -----------------------------------------------------
    # Search Chroma
    # -----------------------------------------------------

    try:

        result = collection_obj.query(
            query_embeddings=[
                query_embedding
            ],
            n_results=n,
            include=[
                "documents",
                "metadatas",
                "distances",
            ],
        )

    except Exception as exc:

        logger.exception(
            "Chroma retrieval failed"
        )

        raise RuntimeError(
            f"Chroma retrieval failed: {exc}"
        )


    docs = result.get(
        "documents",
        [[]]
    )[0]

    metas = result.get(
        "metadatas",
        [[]]
    )[0]

    distances = result.get(
        "distances",
        [[]]
    )[0]


    return (
        docs,
        metas,
        distances
    )


# =========================================================
# HOME
# =========================================================

@app.get("/")
def home():

    return render_template(
        "index.html"
    )


# =========================================================
# UPLOAD
# =========================================================

@app.post("/upload")
def upload():

    files = request.files.getlist(
        "files"
    )

    if not files:

        return jsonify(
            {
                "error":
                    "Please select at least one "
                    "agriculture document."
            }
        ), 400


    results = []

    total_chunks = 0


    for file in files:

        if not file or not file.filename:
            continue


        original_name = Path(
            file.filename
        ).name

        extension = Path(
            original_name
        ).suffix.lower()


        # -------------------------------------------------
        # Validate extension
        # -------------------------------------------------

        if extension not in ALLOWED_EXTENSIONS:

            return jsonify(
                {
                    "error":
                        f"{extension} is not supported. "
                        "Use PDF, DOCX, TXT, PNG, JPG, "
                        "JPEG or WEBP."
                }
            ), 400


        # -------------------------------------------------
        # Create safe unique filename
        # -------------------------------------------------

        safe_name = (
            f"{uuid.uuid4().hex}_"
            f"{original_name}"
        )

        path = (
            UPLOAD_DIR /
            safe_name
        )


        # -------------------------------------------------
        # Save file
        # -------------------------------------------------

        try:

            file.save(path)

        except Exception as exc:

            logger.exception(
                "Could not save uploaded file"
            )

            return jsonify(
                {
                    "error":
                        f"Could not save "
                        f"{original_name}: {exc}"
                }
            ), 500


        # -------------------------------------------------
        # Process and index
        # -------------------------------------------------

        try:

            chunks = index_file(
                path
            )

            results.append(
                {
                    "name": original_name,
                    "chunks": chunks
                }
            )

            total_chunks += chunks


        except Exception as exc:

            logger.exception(
                "Failed to process %s",
                original_name
            )

            return jsonify(
                {
                    "error":
                        f"Could not process "
                        f"{original_name}: {exc}"
                }
            ), 500


    # -----------------------------------------------------
    # Final response
    # -----------------------------------------------------

    if not results:

        return jsonify(
            {
                "error":
                    "No valid agriculture documents "
                    "were uploaded."
            }
        ), 400


    return jsonify(
        {
            "message":
                "Agriculture documents indexed successfully.",
            "files":
                results,
            "chunks":
                total_chunks
        }
    )


# =========================================================
# CHAT
# =========================================================

@app.post("/chat")
def chat():

    data = request.get_json(
        silent=True
    ) or {}


    question = (
        data.get("question")
        or ""
    ).strip()


    if not question:

        return jsonify(
            {
                "error":
                    "Please enter an agriculture question."
            }
        ), 400


    # -----------------------------------------------------
    # Retrieve relevant chunks
    # -----------------------------------------------------

    try:

        docs, metas, distances = retrieve(
            question,
            n=6
        )

    except Exception as exc:

        logger.exception(
            "Retrieval error"
        )

        return jsonify(
            {
                "error":
                    str(exc)
            }
        ), 500


    # -----------------------------------------------------
    # No documents
    # -----------------------------------------------------

    if not docs:

        return jsonify(
            {
                "answer":
                    "I couldn't find this information "
                    "in the uploaded agriculture document(s).",
                "sources": []
            }
        )


    # -----------------------------------------------------
    # Select useful retrieved chunks
    # -----------------------------------------------------

    useful = []


    for doc, meta, distance in zip(
        docs,
        metas,
        distances
    ):

        # Cosine distance:
        # lower value = more similar.
        #
        # We keep reasonably relevant chunks.
        # If distance is unavailable, keep the chunk.
        if (
            distance is None
            or distance <= 0.75
        ):

            useful.append(
                (
                    doc,
                    meta
                )
            )


    # -----------------------------------------------------
    # Retrieval gate
    # -----------------------------------------------------

    if not useful:

        return jsonify(
            {
                "answer":
                    "I couldn't find this information "
                    "in the uploaded agriculture document(s).",
                "sources": []
            }
        )


    # -----------------------------------------------------
    # Build document context
    # -----------------------------------------------------

    context_parts = []


    for doc, meta in useful:

        source = meta.get(
            "source",
            "uploaded agriculture document"
        )

        context_parts.append(
            f"[Source: {source}]\n{doc}"
        )


    context = "\n\n".join(
        context_parts
    )


    # -----------------------------------------------------
    # Create strict RAG prompt
    # -----------------------------------------------------

    prompt = SYSTEM_PROMPT.format(
        context=context,
        question=question
    )


    # -----------------------------------------------------
    # Gemini answer generation
    # -----------------------------------------------------

    try:

        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.0,
                max_output_tokens=900,
            ),
        )


        answer = (
            response.text
            or ""
        ).strip()


    except Exception as exc:

        logger.exception(
            "Gemini answer generation failed"
        )

        return jsonify(
            {
                "error":
                    f"Gemini error: {exc}"
            }
        ), 500


    if not answer:

        answer = (
            "I couldn't find this information "
            "in the uploaded agriculture document(s)."
        )


    # -----------------------------------------------------
    # Collect unique sources
    # -----------------------------------------------------

    sources = []

    seen = set()


    for _, meta in useful:

        source = meta.get(
            "source",
            "uploaded agriculture document"
        )


        if source not in seen:

            seen.add(
                source
            )

            sources.append(
                source
            )


    # -----------------------------------------------------
    # Return answer
    # -----------------------------------------------------

    return jsonify(
        {
            "answer":
                answer,
            "sources":
                sources
        }
    )


# =========================================================
# CLEAR DOCUMENTS
# =========================================================

@app.post("/clear")
def clear():

    global chroma_client
    global collection


    try:

        # Initialize Chroma if necessary
        collection_obj = get_collection()


        # Delete current collection
        try:

            chroma_client.delete_collection(
                COLLECTION_NAME
            )

        except Exception:

            pass


        # Re-create empty collection
        collection = chroma_client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={
                "hnsw:space": "cosine"
            }
        )


        # -------------------------------------------------
        # Delete uploaded files
        # -------------------------------------------------

        for path in UPLOAD_DIR.glob("*"):

            try:

                if path.is_file():

                    path.unlink()

            except Exception as exc:

                logger.warning(
                    "Could not delete %s: %s",
                    path,
                    exc
                )


        logger.info(
            "Agriculture documents and RAG index cleared."
        )


        return jsonify(
            {
                "message":
                    "Agriculture documents and "
                    "RAG index cleared."
            }
        )


    except Exception as exc:

        logger.exception(
            "Clear operation failed"
        )

        return jsonify(
            {
                "error":
                    f"Could not clear documents: {exc}"
            }
        ), 500


# =========================================================
# FILE TOO LARGE
# =========================================================

@app.errorhandler(413)
def file_too_large(error):

    return jsonify(
        {
            "error":
                "File is too large. Maximum upload size is 30 MB."
        }
    ), 413


# =========================================================
# GENERAL ERROR HANDLER
# =========================================================

@app.errorhandler(Exception)
def handle_unexpected_error(error):

    logger.exception(
        "Unexpected application error"
    )

    return jsonify(
        {
            "error":
                "An unexpected server error occurred."
        }
    ), 500


# =========================================================
# LOCAL DEVELOPMENT
# =========================================================

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=int(
            os.getenv(
                "PORT",
                "5000"
            )
        ),
        debug=False
    )