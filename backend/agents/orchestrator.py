from langsmith import traceable

from agents.evidence_agent import run_evidence_agent
from agents.research_agent import run_research_agent
from agents.verdict_agent import run_verdict_agent


@traceable(name="verification_orchestrator")
def run_verification_workflow(claim, doc_type=None):
    """Supervisor / Orchestrator Agent.
    Routes the claim through: Document Evidence Agent -> builds a
    identity-aware web query -> Web Research Agent -> Verdict Agent.
    Appears in LangSmith as one chain: orchestrator -> evidence_agent ->
    research_agent -> verdict_agent.
    """
    raw_chunks = run_evidence_agent(claim, doc_type=doc_type)

    document_evidence = [
        {
            "text": r.payload.get("text"),
            "score": r.score,
            "filename": r.payload.get("filename"),
            "source_name": r.payload.get("candidate") or r.payload.get("supplier")
        }
        for r in raw_chunks
    ]

    context_snippet = ""
    identity_context = claim
    if document_evidence:
        context_snippet = (document_evidence[0]["text"] or "")[:150]
        identity_context = f"{claim} | Related document context: {context_snippet}"

    subject_name = ""
    if document_evidence and document_evidence[0]["text"]:
        subject_name = document_evidence[0]["text"].split("\n")[0].strip()

    if subject_name:
        search_query = f'"{subject_name}" {claim}'.strip()
    else:
        search_query = f"{claim} {context_snippet}".strip()

    web_evidence = run_research_agent(search_query)

    verdict = run_verdict_agent(claim, document_evidence, web_evidence, identity_context)

    return {
        "verdict": verdict,
        "document_evidence": document_evidence,
        "web_evidence": web_evidence
    }