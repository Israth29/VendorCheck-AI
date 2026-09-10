def build_verification_prompt(claim, doc_text, web_text, identity_context=""):
    return f"""You are an evidence-verification assistant. Compare the CLAIM below with the AVAILABLE EVIDENCE from two sources.
Do not invent information. Only use what is in the evidence.

IDENTITY CAUTION (very important): The person or entity being verified is identified as: {identity_context if identity_context else "not specified"}.
Web search results may return information about a DIFFERENT person or entity who happens to share the same name.
Before using ANY web evidence, check whether it actually refers to the SAME person/entity — matching context such as company name, role, location, or other identifying details from the document evidence.
If a web result does not clearly match the same individual/entity (e.g., different company, different profession, no matching details), you MUST NOT use it as supporting evidence, and should explicitly note in your explanation that it appears to refer to a different person or entity with the same name.

CLAIM:
{claim}

DOCUMENT EVIDENCE (from uploaded CV/supplier document):
{doc_text}

WEB EVIDENCE (from public web search):
{web_text}

Respond in this exact format:
STATUS: <one of: Strongly Supported, Partially Supported, Not Verified, Unable to Verify>
EXPLANATION: <2-3 sentences explaining why, referencing only the evidence above and stating which source(s) support it. If web evidence was discarded due to identity mismatch, say so explicitly.>
"""


def build_email_resolution_prompt(company_name, web_text):
    return f"""You are extracting a company's official contact email address from web search results.

COMPANY: {company_name}

SEARCH RESULTS:
{web_text}

Look for an official HR, careers, or general contact email address for this exact company.
If you find one you are confident belongs to this company, respond with ONLY that email address and nothing else.
If no email address is present in the results, or you are not confident it belongs to this company, respond with exactly: UNKNOWN
"""