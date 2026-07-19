"""
03_rag_setup.py  —  Build ChromaDB vector store from rubric documents
IT9204 | Student 202507410 Faiza Malik

Professor's guidance implemented:
  - RAG knowledge base contains ONLY rubric docs, grading criteria,
    source texts, and feedback guidance.
  - The essay dataset itself is NOT indexed here.
  - Scoring comes from the ML model (deterministic, reproducible).
  - RAG/LLM handles explanation and feedback only.

Documents to place in rubrics/ folder:
  asap_scoring_rubric.docx  ← official ASAP rubric (from GitHub repo)
  writing_criteria.txt      ← writing quality criteria (create manually)
  feedback_phrases.txt      ← feedback language by level (create manually)
  IN6_the_face_on_mars.pdf  ← source text for "The Face on Mars" prompt
  [other source text PDFs]  ← one PDF per prompt (7 total)
    Naming convention from dataset prompt_names:
      - "A Cowboy Who Rode the Waves"
      - The Face on Mars            ← IN6_the_face_on_mars.pdf
      - Does the electoral college work?
      - Exploring Venus
      - Facial action coding system
      - Driverless cars
      - Car-free cities

Run ONCE before starting the agent:
    python 03_rag_setup.py

Re-run any time you add new files to rubrics/.
"""

import os
import chromadb
from sentence_transformers import SentenceTransformer


RUBRICS_DIR = "rubrics"
CHROMA_DIR  = "chroma_db"


# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Load all rubric and source documents
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 65)
print("Step 1: Loading rubric and source documents from rubrics/")
print()
print("Expected files:")
print("  asap_scoring_rubric.docx   (official ASAP rubric, score 1-6)")
print("  writing_criteria.txt       (writing quality criteria)")
print("  feedback_phrases.txt       (feedback language by score level)")
print("  *.pdf                      (source texts, one per prompt)")
print("=" * 65)

if not os.path.exists(RUBRICS_DIR):
    print(f"\nERROR: '{RUBRICS_DIR}' folder not found.")
    exit(1)

all_text_by_source = {}

for fname in sorted(os.listdir(RUBRICS_DIR)):
    fpath   = os.path.join(RUBRICS_DIR, fname)
    content = None

    # ── Plain text (.txt) ────────────────────────────────────────────
    if fname.endswith(".txt"):
        try:
            content = open(fpath, encoding="utf-8").read()
        except Exception as e:
            print(f"  ERROR reading {fname}: {e}")

    # ── Word documents (.docx) — covers asap_scoring_rubric.docx ───
    elif fname.endswith(".docx"):
        try:
            from docx import Document
            doc     = Document(fpath)
            content = "\n".join(
                p.text for p in doc.paragraphs if p.text.strip()
            )
        except ImportError:
            print(f"  Skipping {fname} — run: pip install python-docx")
        except Exception as e:
            print(f"  ERROR reading {fname}: {e}")

    # ── PDF files — covers source texts like IN6_the_face_on_mars.pdf ──
    elif fname.endswith(".pdf"):
        try:
            from pypdf import PdfReader
            reader  = PdfReader(fpath)
            content = "\n".join(
                page.extract_text() for page in reader.pages
                if page.extract_text()
            )
        except ImportError:
            print(f"  Skipping {fname} — run: pip install pypdf")
        except Exception as e:
            print(f"  ERROR reading {fname}: {e}")

    if content and content.strip():
        all_text_by_source[fname] = content.strip()
        words = len(content.split())
        print(f"  Loaded: {fname}  ({words} words)")

if not all_text_by_source:
    print("\nERROR: No documents loaded.")
    exit(1)

# ── Check which source texts are present ─────────────────────────────────────
prompt_names = [
    "A Cowboy Who Rode the Waves",
    "The Face on Mars",
    "Does the electoral college work?",
    "Exploring Venus",
    "Facial action coding system",
    "Driverless cars",
    "Car-free cities",
]
source_pdfs = [f for f in all_text_by_source if f.endswith(".pdf")]
if source_pdfs:
    print(f"\nSource text PDFs found: {len(source_pdfs)}")
    for f in source_pdfs:
        print(f"  {f}")
    if len(source_pdfs) < 7:
        print(f"\nNOTE: {7 - len(source_pdfs)} of 7 source text PDFs not yet added.")
        print("Add remaining PDFs to rubrics/ and re-run for full coverage.")
else:
    print("\nWARNING: No source text PDFs found.")
    print("Add the source text PDFs to rubrics/ for better feedback.")

print(f"\nTotal documents loaded: {len(all_text_by_source)}")


# ─────────────────────────────────────────────────────────────────────────────
# Step 2: Split into chunks (500 chars, 50 overlap)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("Step 2: Splitting documents into chunks")
print("=" * 65)

CHUNK_SIZE = 500
OVERLAP    = 50

all_chunks   = []
all_metadata = []

for fname, text in all_text_by_source.items():
    doc_chunks = []
    start = 0
    while start < len(text):
        chunk = text[start : start + CHUNK_SIZE].strip()
        if chunk:
            doc_chunks.append(chunk)
            all_chunks.append(chunk)
            doc_type = (
                "rubric"      if "rubric" in fname.lower()   else
                "source_text" if fname.endswith(".pdf")       else
                "feedback"    if "feedback" in fname.lower()  else
                "criteria"
            )
            all_metadata.append({
                "source"   : fname,
                "doc_type" : doc_type,
                "chunk_idx": len(doc_chunks) - 1,
            })
        start += CHUNK_SIZE - OVERLAP

    print(f"  {fname}: {len(doc_chunks)} chunks  [{doc_type}]")

total_chunks = len(all_chunks)
print(f"\nTotal chunks: {total_chunks}")


# ─────────────────────────────────────────────────────────────────────────────
# Step 3: Generate embeddings
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("Step 3: Generating embeddings")
print("(Downloads ~90 MB model on first run)")
print("=" * 65)

embed_model = SentenceTransformer("all-MiniLM-L6-v2")
print(f"Embedding {total_chunks} chunks...")

embeddings = embed_model.encode(
    all_chunks, show_progress_bar=True, batch_size=32
)
print(f"Shape: {embeddings.shape}")


# ─────────────────────────────────────────────────────────────────────────────
# Step 4: Store in ChromaDB
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("Step 4: Storing in ChromaDB")
print("=" * 65)

os.makedirs(CHROMA_DIR, exist_ok=True)
client = chromadb.PersistentClient(path=CHROMA_DIR)

try:
    client.delete_collection("rubrics")
    print("Deleted previous collection (rebuilding).")
except Exception:
    pass

collection = client.create_collection(
    name     = "rubrics",
    metadata = {
        "description": (
            "ASAP 2.0 rubric (score 1-6), writing criteria, "
            "feedback guidance, and source texts. "
            "Used for LLM explanation only — NOT for scoring. "
            "Scoring is handled by the Azure ML model."
        )
    }
)

collection.add(
    documents  = all_chunks,
    embeddings = embeddings.tolist(),
    metadatas  = all_metadata,
    ids        = [f"chunk_{i}" for i in range(total_chunks)],
)

stored = collection.count()
print(f"Stored {stored} chunks.")

type_counts = {}
for m in all_metadata:
    t = m["doc_type"]
    type_counts[t] = type_counts.get(t, 0) + 1
print("\nChunks by type:")
for doc_type, count in sorted(type_counts.items()):
    print(f"  {doc_type:<15}: {count} chunks")


# ─────────────────────────────────────────────────────────────────────────────
# Step 5: Test retrieval
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("Step 5: Testing retrieval")
print("=" * 65)

test_queries = [
    "score of 5 demonstrates mastery with strong critical thinking",
    "feedback for student with weak evidence and poor organisation",
    "Face on Mars article natural landform alien artifact",
]
for q in test_queries:
    q_vec   = embed_model.encode([q])[0].tolist()
    results = collection.query(query_embeddings=[q_vec], n_results=1)
    src     = results["metadatas"][0][0]["source"]
    typ     = results["metadatas"][0][0]["doc_type"]
    top     = results["documents"][0][0][:100]
    print(f"\nQuery: '{q[:60]}...'")
    print(f"  Source: {src}  [type: {typ}]")
    print(f"  Result: {top}...")

print("\n" + "=" * 65)
print("RAG setup complete!")
print()
print("Knowledge base contains:")
print(f"  rubric       : {type_counts.get('rubric', 0)} chunks")
print(f"  criteria     : {type_counts.get('criteria', 0)} chunks")
print(f"  feedback     : {type_counts.get('feedback', 0)} chunks")
print(f"  source_text  : {type_counts.get('source_text', 0)} chunks")
print()
print("IMPORTANT:")
print("  Score scale in ASAP 2.0 is 1-6 (normalised to 0-10 in ML).")
print("  The essay dataset is NOT in this knowledge base.")
print("  Scoring is handled by the Azure ML endpoint (deterministic).")
print("  RAG is used ONLY for feedback explanation.")
print()
print("Next step: run python agent.py")
print("=" * 65)
