from fastapi import FastAPI, UploadFile, File, Header, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import os
import re
import hashlib
import secrets
import platform
import uuid
import json
from datetime import datetime
from typing import Optional, List
import pytesseract
from PIL import Image
from pdf2image import convert_from_path
from dotenv import load_dotenv
import google.generativeai as genai
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct, Filter, FieldCondition, MatchValue
from tavily import TavilyClient
import resend
from langsmith import traceable

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
resend.api_key = os.getenv("RESEND_API_KEY")

if platform.system() == "Windows":
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
# Linux (Docker container) — apt-installed tesseract is already on PATH, no need to set path

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Persistent JSON storage: companies + employees (replaces the old hardcoded
# EMPLOYEES list). Companies push employee changes through the webhook API
# below, so this file is the live source of truth for who can log in.
# ---------------------------------------------------------------------------

EMPLOYEES_FILE = "employees.json"
COMPANIES_FILE = "companies.json"
EMAIL_THREADS_FILE = "email_threads.json"


def load_json_file(path, default):
    if not os.path.exists(path):
        return default
    with open(path, "r") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return default


def save_json_file(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def get_employees():
    return load_json_file(EMPLOYEES_FILE, [])


def save_employees(employees):
    save_json_file(EMPLOYEES_FILE, employees)


def get_companies():
    return load_json_file(COMPANIES_FILE, [])


def save_companies(companies):
    save_json_file(COMPANIES_FILE, companies)


def find_company_by_code(company_code):
    for c in get_companies():
        if c["company_code"].lower() == company_code.lower():
            return c
    return None


def find_company_by_name(name):
    """Fuzzy match a typed company name against registered companies
    (case-insensitive, substring match either direction)."""
    name_lower = name.strip().lower()
    for c in get_companies():
        company_name_lower = c["company_name"].strip().lower()
        if name_lower == company_name_lower or name_lower in company_name_lower or company_name_lower in name_lower:
            return c
    return None


def find_company_by_email(email):
    email_lower = email.strip().lower()
    for c in get_companies():
        if c.get("contact_email", "").strip().lower() == email_lower:
            return c
    return None


def find_company_by_api_key(api_key):
    if not api_key:
        return None
    for c in get_companies():
        if c.get("api_key") == api_key:
            return c
    return None


def get_email_threads():
    return load_json_file(EMAIL_THREADS_FILE, [])


def add_email_thread(thread):
    threads = get_email_threads()
    thread["id"] = str(uuid.uuid4())
    thread["timestamp"] = datetime.utcnow().isoformat()
    threads.append(thread)
    save_json_file(EMAIL_THREADS_FILE, threads)
    return thread


# ---------------------------------------------------------------------------
# Password hashing (PBKDF2-HMAC-SHA256, built into Python's standard library
# — no plaintext passwords are ever stored). Used for company portal login.
# ---------------------------------------------------------------------------

def hash_password(password: str) -> str:
    salt = secrets.token_hex(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), bytes.fromhex(salt), 100_000)
    return f"{salt}${dk.hex()}"


def verify_password(password: str, stored_hash: str) -> bool:
    try:
        salt, hash_hex = stored_hash.split("$")
        dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), bytes.fromhex(salt), 100_000)
        return secrets.compare_digest(dk.hex(), hash_hex)
    except Exception:
        return False


def generate_unique_company_code(company_name, existing_companies):
    base = re.sub(r"[^A-Z0-9]", "", company_name.upper())[:6] or "COMP"
    existing_codes = {c["company_code"].upper() for c in existing_companies}
    while True:
        candidate = f"{base}-{secrets.randbelow(900) + 100}"
        if candidate.upper() not in existing_codes:
            return candidate


UPLOAD_DIR = "uploads/cv"
os.makedirs(UPLOAD_DIR, exist_ok=True)

UPLOAD_DIR_SUPPLIER = "uploads/supplier"
os.makedirs(UPLOAD_DIR_SUPPLIER, exist_ok=True)

AUDIT_LOG_FILE = "audit_log.json"

qdrant = QdrantClient(path="qdrant_data")
COLLECTION_NAME = "vendorcheck_documents"

if not qdrant.collection_exists(COLLECTION_NAME):
    qdrant.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=3072, distance=Distance.COSINE)
    )


def add_audit_record(record):
    record["timestamp"] = datetime.utcnow().isoformat()
    record["id"] = str(uuid.uuid4())

    records = []
    if os.path.exists(AUDIT_LOG_FILE):
        with open(AUDIT_LOG_FILE, "r") as f:
            try:
                records = json.load(f)
            except json.JSONDecodeError:
                records = []

    records.append(record)

    with open(AUDIT_LOG_FILE, "w") as f:
        json.dump(records, f, indent=2)

    return record


def chunk_text(text, chunk_size=300, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


def extract_text_from_file(file_path):
    """
    Extract text via OCR from either an image (PNG/JPG) or a PDF.
    PDFs are first rendered page-by-page to images with pdf2image,
    then each page is OCR'd with the same pytesseract call used for
    plain images. Text from all pages is joined together.
    """
    if file_path.lower().endswith(".pdf"):
        pages = convert_from_path(file_path)
        text_parts = [pytesseract.image_to_string(page) for page in pages]
        return "\n".join(text_parts)
    else:
        return pytesseract.image_to_string(Image.open(file_path))


@traceable(name="get_embedding")
def get_embedding(text):
    result = genai.embed_content(
        model="models/gemini-embedding-001",
        content=text
    )
    return result["embedding"]


def store_chunks_in_qdrant(chunks, metadata):
    points = []
    for chunk in chunks:
        if not chunk.strip():
            continue
        vector = get_embedding(chunk)
        point = PointStruct(
            id=str(uuid.uuid4()),
            vector=vector,
            payload={"text": chunk, **metadata}
        )
        points.append(point)
    if points:
        qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
    return len(points)


@traceable(name="web_search")
def web_search(query, max_results=3):
    try:
        response = tavily.search(query=query, max_results=max_results)
        results = []
        for r in response.get("results", []):
            results.append({
                "title": r.get("title"),
                "content": r.get("content"),
                "url": r.get("url")
            })
        return results
    except Exception:
        return []


@traceable(name="generate_verification")
def generate_verification(claim, document_evidence, web_evidence, identity_context=""):
    doc_text = "\n\n".join([f"- {c['text']}" for c in document_evidence]) or "None found."
    web_text = "\n\n".join([f"- {w['title']}: {w['content']} (Source: {w['url']})" for w in web_evidence]) or "None found."

    prompt = f"""You are an evidence-verification assistant. Compare the CLAIM below with the AVAILABLE EVIDENCE from two sources.
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

    model = genai.GenerativeModel("models/gemini-3.6-flash")
    response = model.generate_content(prompt)
    return response.text


EMAIL_REGEX = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")


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

    prompt = f"""You are extracting a company's official contact email address from web search results.

COMPANY: {company_name}

SEARCH RESULTS:
{web_text}

Look for an official HR, careers, or general contact email address for this exact company.
If you find one you are confident belongs to this company, respond with ONLY that email address and nothing else.
If no email address is present in the results, or you are not confident it belongs to this company, respond with exactly: UNKNOWN
"""

    model = genai.GenerativeModel("models/gemini-3.6-flash")
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


class LoginRequest(BaseModel):
    employee_id: str
    name: str
    company_code: str


class LogoutRequest(BaseModel):
    employee_id: str
    name: str


class ResolveEmailsRequest(BaseModel):
    company_names: List[str]


class ConfirmedRecipient(BaseModel):
    company_name: str
    company_code: Optional[str] = None
    email: str


class SendEmailRequest(BaseModel):
    recipients: List[ConfirmedRecipient]
    subject: str
    message: str
    requested_by: str
    employee_id: str
    candidate_name: str


class EmployeeSyncItem(BaseModel):
    employee_id: str
    name: str
    role: str  # "hr" or "procurement"
    active: bool = True


class EmployeeSyncRequest(BaseModel):
    employees: List[EmployeeSyncItem]


class CompanyRegisterRequest(BaseModel):
    company_name: str
    contact_email: str
    password: str


class CompanyLoginRequest(BaseModel):
    email: str
    password: str


@app.get("/")
def read_root():
    return {"message": "VendorCheck AI backend is running"}


@app.post("/login")
def login(data: LoginRequest):
    for emp in get_employees():
        if (emp["employee_id"].lower() == data.employee_id.lower()
            and emp["name"].lower() == data.name.lower()
            and emp["company_code"].lower() == data.company_code.lower()):
            if not emp["active"]:
                return {"success": False, "message": "Employee is not active"}

            add_audit_record({
                "action": "login",
                "performed_by": emp["name"],
                "employee_id": emp["employee_id"],
                "company_code": emp["company_code"],
                "role": emp["role"]
            })

            return {"success": True, "role": emp["role"], "name": emp["name"]}
    return {"success": False, "message": "Invalid credentials"}


@app.post("/logout")
def logout(data: LogoutRequest):
    add_audit_record({
        "action": "logout",
        "performed_by": data.name,
        "employee_id": data.employee_id
    })
    return {"success": True}


# ---------------------------------------------------------------------------
# Company self-registration + company portal login. A company signs up
# itself (no admin approval step yet — that's a possible future addition),
# gets a unique company_code and api_key, and can log in with its email +
# password to manage its own employees via the portal.
# ---------------------------------------------------------------------------

@app.post("/api/v1/companies/register")
def register_company(data: CompanyRegisterRequest):
    companies = get_companies()

    if find_company_by_email(data.contact_email):
        raise HTTPException(status_code=400, detail="A company is already registered with this email")

    if len(data.password) < 8:
        raise HTTPException(status_code=400, detail="Password must be at least 8 characters")

    company_code = generate_unique_company_code(data.company_name, companies)
    new_company = {
        "company_code": company_code,
        "company_name": data.company_name,
        "contact_email": data.contact_email,
        "password_hash": hash_password(data.password),
        "api_key": secrets.token_urlsafe(32),
        "status": "active"
    }
    companies.append(new_company)
    save_companies(companies)

    add_audit_record({
        "action": "company_registered",
        "performed_by": data.company_name,
        "company_code": company_code
    })

    return {
        "success": True,
        "company_code": company_code,
        "company_name": data.company_name
    }


@app.post("/api/v1/companies/login")
def login_company(data: CompanyLoginRequest):
    company = find_company_by_email(data.email)
    if not company or not verify_password(data.password, company.get("password_hash", "")):
        return {"success": False, "message": "Invalid email or password"}

    if company.get("status") != "active":
        return {"success": False, "message": "This company account is not active"}

    add_audit_record({
        "action": "company_login",
        "performed_by": company["company_name"],
        "company_code": company["company_code"]
    })

    return {
        "success": True,
        "company_code": company["company_code"],
        "company_name": company["company_name"],
        "api_key": company["api_key"]
    }


@app.get("/api/v1/companies/me")
def get_my_company(x_api_key: Optional[str] = Header(None)):
    """Company portal: identify the logged-in company from its API key."""
    company = find_company_by_api_key(x_api_key)
    if not company:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return {
        "company_code": company["company_code"],
        "company_name": company["company_name"],
        "contact_email": company.get("contact_email", "")
    }


@app.get("/api/v1/companies/me/employees")
def get_my_company_employees(x_api_key: Optional[str] = Header(None)):
    """Company portal: list only this company's own employees."""
    company = find_company_by_api_key(x_api_key)
    if not company:
        raise HTTPException(status_code=401, detail="Invalid API key")

    employees = get_employees()
    mine = [
        {k: v for k, v in e.items() if k != "company_code"}
        for e in employees
        if e["company_code"].lower() == company["company_code"].lower()
    ]
    return {"company_name": company["company_name"], "employees": mine}


# ---------------------------------------------------------------------------
# Company webhook API — a registered company's HR system (or the company
# portal UI) pushes employee additions/updates here using its API key.
# Sending an existing employee_id with "active": false invalidates that
# employee going forward.
# ---------------------------------------------------------------------------

@app.post("/api/v1/companies/{company_code}/employees")
def sync_employees(company_code: str, data: EmployeeSyncRequest, x_api_key: Optional[str] = Header(None)):
    company = find_company_by_code(company_code)
    if not company:
        raise HTTPException(status_code=404, detail="Unknown company_code")
    if not x_api_key or x_api_key != company["api_key"]:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")

    employees = get_employees()
    employees_by_id = {e["employee_id"].lower(): e for e in employees}

    added, updated = 0, 0
    for item in data.employees:
        key = item.employee_id.lower()
        if key in employees_by_id:
            employees_by_id[key].update({
                "name": item.name,
                "role": item.role,
                "active": item.active,
                "company_code": company["company_code"],
            })
            updated += 1
        else:
            employees_by_id[key] = {
                "employee_id": item.employee_id,
                "name": item.name,
                "company_code": company["company_code"],
                "role": item.role,
                "active": item.active,
            }
            added += 1

    save_employees(list(employees_by_id.values()))

    add_audit_record({
        "action": "employee_sync",
        "performed_by": f"{company['company_name']} (webhook)",
        "company_code": company["company_code"],
        "added": added,
        "updated": updated
    })

    return {"success": True, "added": added, "updated": updated}


@app.get("/api/v1/companies-overview")
def companies_overview():
    companies = get_companies()
    employees = get_employees()

    overview = []
    for c in companies:
        company_employees = [
            {k: v for k, v in e.items() if k != "company_code"}
            for e in employees
            if e["company_code"].lower() == c["company_code"].lower()
        ]
        overview.append({
            "company_code": c["company_code"],
            "company_name": c["company_name"],
            "employees": company_employees
        })

    return {"companies": overview}


@app.get("/api/v1/login-history")
def login_history(action: Optional[str] = None, employee_id: Optional[str] = None):
    if not os.path.exists(AUDIT_LOG_FILE):
        return {"records": []}
    with open(AUDIT_LOG_FILE, "r") as f:
        try:
            records = json.load(f)
        except json.JSONDecodeError:
            records = []

    filtered = [r for r in records if r.get("action") in ("login", "logout")]

    if action:
        filtered = [r for r in filtered if r.get("action") == action]
    if employee_id:
        filtered = [r for r in filtered if r.get("employee_id", "").lower() == employee_id.lower()]

    return {"records": filtered[::-1]}


@app.get("/api/v1/companies-directory")
def companies_directory():
    """List of registered companies (used to power an autocomplete of
    company names when composing a verification email)."""
    companies = get_companies()
    return {
        "companies": [
            {"company_code": c["company_code"], "company_name": c["company_name"]}
            for c in companies
        ]
    }


@app.post("/api/v1/resolve-company-emails")
def resolve_company_emails(data: ResolveEmailsRequest):
    """Step 1 of sending a verification email: resolve each typed company
    name to an email address (registered lookup, or web search + AI) and
    return it for the officer to review/edit — nothing is sent yet."""
    resolved = [resolve_recipient(name) for name in data.company_names]
    return {"resolved": resolved}


@app.post("/send-verification-email")
def send_verification_email(data: SendEmailRequest):
    """Step 2: actually send to the officer-confirmed recipient list. Each
    recipient here is exactly what the officer reviewed/approved on the
    confirmation screen — this endpoint does no further email resolution."""
    results = []
    any_success = False

    for recipient in data.recipients:
        try:
            params = {
                "from": "VendorCheck AI <onboarding@resend.dev>",
                "to": [recipient.email],
                "subject": data.subject,
                "html": f"""
                    <p>{data.message}</p>
                    <hr>
                    <p style="color:#666;font-size:12px;">This verification email was sent with explicit approval from {data.requested_by} via VendorCheck AI.</p>
                """
            }
            email = resend.Emails.send(params)
            results.append({
                "company_name": recipient.company_name,
                "company_code": recipient.company_code,
                "email": recipient.email,
                "status": "sent",
                "email_id": email.get("id")
            })
            any_success = True
        except Exception as e:
            results.append({
                "company_name": recipient.company_name,
                "company_code": recipient.company_code,
                "email": recipient.email,
                "status": "failed",
                "error": str(e)
            })

    add_audit_record({
        "action": "send_verification_email",
        "performed_by": data.requested_by,
        "employee_id": data.employee_id,
        "candidate_name": data.candidate_name,
        "subject": data.subject,
        "recipients": results,
        "email_requested": True,
        "email_sent": any_success
    })

    add_email_thread({
        "employee_id": data.employee_id,
        "performed_by": data.requested_by,
        "candidate_name": data.candidate_name,
        "subject": data.subject,
        "message": data.message,
        "recipients": results
    })

    return {"success": any_success, "recipients": results}


@app.get("/api/v1/my-sent-emails")
def my_sent_emails(employee_id: str):
    """Returns only the verification-email threads sent by this specific
    officer — never another HR/Procurement officer's sent emails."""
    threads = get_email_threads()
    mine = [t for t in threads if t.get("employee_id", "").lower() == employee_id.lower()]
    return {"threads": mine[::-1]}


@app.post("/upload-cv")
async def upload_cv(candidate_name: str, file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)

    extracted_text = extract_text_from_file(file_path)
    chunks = chunk_text(extracted_text)

    stored_count = store_chunks_in_qdrant(chunks, {
        "type": "cv",
        "candidate": candidate_name,
        "filename": file.filename
    })

    return {
        "success": True,
        "filename": file.filename,
        "candidate": candidate_name,
        "extracted_text": extracted_text,
        "chunks": chunks,
        "stored_in_qdrant": stored_count
    }


@app.post("/upload-supplier-doc")
async def upload_supplier_doc(supplier_name: str, file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR_SUPPLIER, file.filename)
    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)

    extracted_text = extract_text_from_file(file_path)
    chunks = chunk_text(extracted_text)

    stored_count = store_chunks_in_qdrant(chunks, {
        "type": "supplier",
        "supplier": supplier_name,
        "filename": file.filename
    })

    return {
        "success": True,
        "filename": file.filename,
        "supplier": supplier_name,
        "extracted_text": extracted_text,
        "chunks": chunks,
        "stored_in_qdrant": stored_count
    }


@app.get("/list-models")
def list_models():
    embed_models = []
    generate_models = []
    for m in genai.list_models():
        if "embedContent" in m.supported_generation_methods:
            embed_models.append(m.name)
        if "generateContent" in m.supported_generation_methods:
            generate_models.append(m.name)
    return {"embedding_models": embed_models, "generation_models": generate_models}


@app.get("/search")
def search(query: str, doc_type: str = None):
    query_vector = get_embedding(query)

    search_filter = None
    if doc_type:
        search_filter = Filter(
            must=[FieldCondition(key="type", match=MatchValue(value=doc_type))]
        )

    results = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        query_filter=search_filter,
        limit=3
    ).points

    matches = []
    for r in results:
        matches.append({
            "score": r.score,
            "text": r.payload.get("text"),
            "metadata": {k: v for k, v in r.payload.items() if k != "text"}
        })

    return {"query": query, "results": matches}


@app.get("/verify-claim")
def verify_claim(claim: str, doc_type: str = None, requested_by: str = "Unknown"):
    # ---- AI verification logic below is UNCHANGED ----
    query_vector = get_embedding(claim)

    search_filter = None
    if doc_type:
        search_filter = Filter(
            must=[FieldCondition(key="type", match=MatchValue(value=doc_type))]
        )

    results = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        query_filter=search_filter,
        limit=3
    ).points

    # Only addition here: carry along filename/candidate/supplier metadata that
    # was already stored in Qdrant's payload, so the audit trail can show which
    # uploaded document backed this verification. Does not affect the AI logic.
    document_evidence = [
        {
            "text": r.payload.get("text"),
            "score": r.score,
            "filename": r.payload.get("filename"),
            "source_name": r.payload.get("candidate") or r.payload.get("supplier")
        }
        for r in results
    ]

    # Build a more specific web search query using context from the document evidence
    # so results are less likely to match an unrelated person/entity with the same name.
    context_snippet = ""
    identity_context = claim
    if document_evidence:
        context_snippet = document_evidence[0]["text"][:150]
        identity_context = f"{claim} | Related document context: {context_snippet}"

    subject_name = ""
    if document_evidence and document_evidence[0]["text"]:
        subject_name = document_evidence[0]["text"].split("\n")[0].strip()

    if subject_name:
        search_query = f'"{subject_name}" {claim}'.strip()
    else:
        search_query = f"{claim} {context_snippet}".strip()

    web_evidence = web_search(search_query)

    verdict = generate_verification(claim, document_evidence, web_evidence, identity_context)

    status_line = verdict.split("\n")[0] if verdict else "Unknown"
    # ---- end of unchanged AI verification logic ----

    # Only addition: pull filename/source_name out of document_evidence for the audit log.
    source_files = list({d["filename"] for d in document_evidence if d.get("filename")})
    source_names = list({d["source_name"] for d in document_evidence if d.get("source_name")})

    add_audit_record({
        "action": "verify_claim",
        "performed_by": requested_by,
        "verification_type": doc_type or "unspecified",
        "claim": claim,
        "result_status": status_line,
        "source_files": source_files,
        "source_names": source_names,
        "email_requested": False,
        "email_sent": False
    })

    return {
        "claim": claim,
        "document_evidence": document_evidence,
        "web_evidence": web_evidence,
        "verification_result": verdict
    }


@app.get("/audit-log")
def get_audit_log():
    if not os.path.exists(AUDIT_LOG_FILE):
        return {"records": []}
    with open(AUDIT_LOG_FILE, "r") as f:
        try:
            records = json.load(f)
        except json.JSONDecodeError:
            records = []
    return {"records": records[::-1]}