from langsmith import traceable
from retrieval.vector_store import retrieve_relevant_chunks


@traceable(name="evidence_agent")
def run_evidence_agent(query, doc_type=None, limit=3):
    """Document Evidence Agent.
    Responsibility: search Qdrant for document chunks (RAG) relevant to the
    claim, giving the Verdict Agent document-backed context to reason over.
    """
    return retrieve_relevant_chunks(query, doc_type=doc_type, limit=limit)