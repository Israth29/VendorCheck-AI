from langsmith import traceable
import google.generativeai as genai

from config import GENERATION_MODEL
from prompts.verification_prompts import build_verification_prompt


@traceable(name="verdict_agent")
def run_verdict_agent(claim, document_evidence, web_evidence, identity_context=""):
    """Verdict / Decision Agent.
    Responsibility: weigh Document Evidence Agent output against Research
    Agent output, apply the identity-matching safeguard, and produce the
    final STATUS/EXPLANATION verdict.
    """
    doc_text = "\n\n".join([f"- {c['text']}" for c in document_evidence]) or "None found."
    web_text = "\n\n".join([f"- {w['title']}: {w['content']} (Source: {w['url']})" for w in web_evidence]) or "None found."

    prompt = build_verification_prompt(claim, doc_text, web_text, identity_context)

    model = genai.GenerativeModel(GENERATION_MODEL)
    response = model.generate_content(prompt)
    return response.text


# Backward-compatible alias — old imports keep working
generate_verification = run_verdict_agent