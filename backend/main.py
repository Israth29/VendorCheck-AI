from fastapi import FastAPI, UploadFile, File, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
from typing import Optional
import secrets

import config
from models import (
    LoginRequest, LogoutRequest, ResolveEmailsRequest, SendEmailRequest,
    EmployeeSyncRequest, CompanyRegisterRequest, CompanyLoginRequest
)
from database.store import (
    get_employees, save_employees, get_companies, save_companies,
    find_company_by_code, find_company_by_email, find_company_by_api_key,
    add_email_thread, get_email_threads, add_audit_record, get_audit_records
)
from auth.security import hash_password, verify_password, generate_unique_company_code
from retrieval.vector_store import chunk_text, store_chunks_in_qdrant, retrieve_relevant_chunks
from tools.ocr import extract_text_from_file
from tools.email_sender import send_email
from agents.verification_agent import generate_verification, resolve_recipient

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def read_root():
    return {"message": "VendorCheck AI backend is running"}


# ---------------------------------------------------------------------------
# Employee login / logout
# ---------------------------------------------------------------------------

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
# Company self-registration + company portal login
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
# Company webhook API — a registered company pushes employee add/updates
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
    records = get_audit_records()
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
    """Step 2: actually send to the officer-confirmed recipient list."""
    results = []
    any_success = False

    for recipient in data.recipients:
        try:
            email = send_email(recipient.email, data.subject, data.message, data.requested_by)
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


# ---------------------------------------------------------------------------
# Document upload + OCR + RAG storage
# ---------------------------------------------------------------------------

@app.post("/upload-cv")
async def upload_cv(candidate_name: str, file: UploadFile = File(...)):
    file_path = os.path.join(config.UPLOAD_DIR, file.filename)
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
    file_path = os.path.join(config.UPLOAD_DIR_SUPPLIER, file.filename)
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


@app.get("/search")
def search(query: str, doc_type: str = None):
    results = retrieve_relevant_chunks(query, doc_type)

    matches = []
    for r in results:
        matches.append({
            "score": r.score,
            "text": r.payload.get("text"),
            "metadata": {k: v for k, v in r.payload.items() if k != "text"}
        })

    return {"query": query, "results": matches}


# ---------------------------------------------------------------------------
# Evidence verification (RAG + web search + AI agent)
# ---------------------------------------------------------------------------

@app.get("/verify-claim")
def verify_claim(claim: str, doc_type: str = None, requested_by: str = "Unknown"):
    results = retrieve_relevant_chunks(claim, doc_type)

    document_evidence = [
        {
            "text": r.payload.get("text"),
            "score": r.score,
            "filename": r.payload.get("filename"),
            "source_name": r.payload.get("candidate") or r.payload.get("supplier")
        }
        for r in results
    ]

    # Build a more specific web search query using context from the document
    # evidence so results are less likely to match an unrelated person/entity
    # with the same name.
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

    from tools.web_search import web_search
    web_evidence = web_search(search_query)

    verdict = generate_verification(claim, document_evidence, web_evidence, identity_context)
    status_line = verdict.split("\n")[0] if verdict else "Unknown"

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
    return {"records": get_audit_records()[::-1]}