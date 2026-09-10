import re
from langsmith import traceable
import google.generativeai as genai

from config import GENERATION_MODEL
from prompts.verification_prompts import build_verification_prompt, build_email_resolution_prompt
from tools.web_search import web_search
from database.store import find_company_by_name

EMAIL_REGEX = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")


@traceable(name="generate_verification")
def generate_verification(claim, document_evidence, web_evidence, identity_context=""):
    """The core Evidence Verification Agent: compares a claim against
    document evidence (from RAG) and web evidence (from search), applying
    identity-matching caution, and returns a structured STATUS/EXPLANATION
    verdict."""
    doc_text = "\n\n".join([f"- {c['text']}" for c in document_evidence]) or "None found."
    web_text = "\n\n".join([f"- {w['title']}: {w['content']} (Source: {w['url']})" for w in web_evidence]) or "None found."

    prompt = build_verification_prompt(claim, doc_text, web_text, identity_context)

    model = genai.GenerativeModel(GENERATION_MODEL)
    response = model.generate_content(prompt)
    return response.text


@traceable(name="resolve_company_email")
def resolve_company_email_via_ai(company_name):
    """For a company not in our registered database: web-search for its
    official contact info, then ask the model to pick the single best
    official HR/contact email from the search results. Returns None if no
    confident match is found (never invents an address)."""
    results = web_search(f"{company_name} official HR contact email")
    if not results:
        return None

    web_text = "\n\n".join(
        [f"- {r['title']}: {r['content']} (Source: {r['url']})" for r in results]
    )

    prompt = build_email_resolution_prompt(company_name, web_text)

    model = genai.GenerativeModel(GENERATION_MODEL)
    response = model.generate_content(prompt)
    text = (response.text or "").strip()

    match = EMAIL_REGEX.search(text)
    return match.group(0) if match else None


def resolve_recipient(company_name):
    """Resolve a typed company name to an email address (preview only —
    does not send anything):
    1. Check the registered companies database first.
    2. If not registered, web-search + AI-extract an official email.
    The officer reviews/edits this before anything is actually sent."""
    company = find_company_by_name(company_name)
    if company and company.get("contact_email"):
        return {
            "company_name": company["company_name"],
            "company_code": company["company_code"],
            "email": company["contact_email"],
            "source": "registered"
        }

    email = resolve_company_email_via_ai(company_name)
    return {
        "company_name": company_name,
        "company_code": None,
        "email": email,
        "source": "web_search_ai" if email else "not_found"
    }