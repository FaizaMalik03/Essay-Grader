"""
rag_tool.py  —  RAG retrieval tool for the essay grading agent
IT9204 | Student 202507410 Faiza Malik

Professor's guidance implemented:
  - This tool retrieves rubric criteria, source texts, and feedback
    guidance to support the EXPLANATION side of grading.
  - It does NOT score essays. Scoring is handled by the Azure ML model.
  - The LLM uses retrieved content to generate grounded feedback only.

Prerequisites:
    1. Run 03_rag_setup.py first to build chroma_db/
    2. pip install chromadb sentence-transformers
"""

import chromadb
from sentence_transformers import SentenceTransformer


# ── Load once at module import time ───────────────────────────────────────────
_embed_model   = SentenceTransformer("all-MiniLM-L6-v2")
_chroma_client = chromadb.PersistentClient(path="chroma_db")

try:
    _collection = _chroma_client.get_collection("rubrics")
    print(f"RAG ready — {_collection.count()} chunks loaded.")
    print("  (rubric criteria + source texts + feedback guidance)")
except Exception as e:
    print(f"WARNING: RAG collection not found: {e}")
    print("Run 03_rag_setup.py first.")
    _collection = None


def retrieve_rubric_criteria(query: str, k: int = 3) -> str:
    """
    Search the RAG knowledge base for rubric criteria and source content
    relevant to the query.

    ROLE: This function supports the EXPLANATION side of grading.
          The score itself always comes from the Azure ML model.
          This function retrieves the rubric criteria and source context
          that the LLM uses to explain WHY the score was given and
          HOW the student can improve.

    Knowledge base contains:
      - Official ASAP 2.0 scoring rubric (asap_scoring_rubric.docx)
      - Writing quality criteria (writing_criteria.txt)
      - Feedback phrase guidance (feedback_phrases.txt)
      - Source texts for all 7 prompts (source_text_*.txt)
        These are the passages students read before writing their essays.
        Including them means the LLM can reference the actual source
        material when explaining whether evidence was used effectively.

    Args:
        query: Natural language query, e.g.:
               "criteria for proficient essay scoring 7 to 8"
               "feedback for student with weak thesis"
               "source text about technology in education"
               "how to improve evidence and argumentation"
        k:     Number of chunks to retrieve (default 3)

    Returns:
        Formatted string of matching rubric/source excerpts with labels.
        Ready to inject into the LLM prompt as grounding context.
    """
    if _collection is None:
        return (
            "ERROR: RAG knowledge base not available. "
            "Run 03_rag_setup.py first."
        )

    query_vec = _embed_model.encode([query])[0].tolist()
    n         = min(k, _collection.count())
    results   = _collection.query(query_embeddings=[query_vec], n_results=n)

    chunks  = results["documents"][0]
    metas   = results["metadatas"][0]

    formatted_parts = []
    for i, (chunk, meta) in enumerate(zip(chunks, metas)):
        source   = meta["source"]
        doc_type = meta.get("doc_type", "document")
        label    = (
            f"Rubric criteria"  if doc_type == "rubric"      else
            f"Source text"      if doc_type == "source_text" else
            f"Feedback guidance"if doc_type == "feedback"    else
            f"Writing criteria" if doc_type == "criteria"    else
            f"Document"
        )
        formatted_parts.append(
            f"[{label} — {source}]\n{chunk}"
        )

    return "\n\n---\n\n".join(formatted_parts)


# ── Quick self-test when run directly ────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("RAG tool self-test")
    print("=" * 60)

    tests = [
        "criteria for exemplary essay score 9 or 10",
        "feedback for student with poor organisation and weak evidence",
        "source text that students used for their writing prompt",
    ]

    for q in tests:
        print(f"\nQuery: '{q}'")
        result = retrieve_rubric_criteria(q, k=2)
        lines  = result.split("\n")
        print("\n".join(lines[:5]))
        print("...")

    print("\nRAG tool working correctly.")
