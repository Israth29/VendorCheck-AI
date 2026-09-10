import re
from langsmith import traceable
import google.generativeai as genai

from config import GENERATION_MODEL
from prompts.verification_prompts import build_email_resolution_prompt
from agents.research_agent import run_research_agent
from database.store import find_company_by_name

EMAIL_REGEX = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")


@traceable(name="recipient_resolution_agent")
def run_recipient_agent(company_name):
    """Recipient Resolution Agent.
    Responsibility: resolve a typed company name to a contact email —
    registered-companies database first, then Research Agent + AI extraction
    as fallback. Never invents an address; officer still confirms before send.
    """
    company = find_company_by_name(company_name)
    if company and company.get("contact_email"):
        return {
            "company_name": company["company_name"],
            "company_code": company["company_code"],
            "email": company["contact_email"],
            "source": "registered"
        }

    email = _resolve_company_email_via_ai(company_name)
    return {
        "company_name": company_name,
        "company_code": None,
        "email": email,
        "source": "web_search_ai" if email else "not_found"
    }


def _resolve_company_email_via_ai(company_name):
    results = run_research_agent(f"{company_name} official HR contact email")
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


# Backward-compatible aliases
resolve_recipient = run_recipient_agent
resolve_company_email_via_ai = _resolve_company_email_via_ai