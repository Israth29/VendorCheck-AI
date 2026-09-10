from fastapi import FastAPI, UploadFile, File, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
import logging
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
from agents.verification_agent import resolve_recipient
from agents.orchestrator import run_verification_workflow

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("vendorcheck")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

ALLOWED_UPLOAD_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg"}
MAX_UPLOAD_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB


@app.get("/")
def read_root():
    return {"message": "VendorCheck AI backend is running"}


# ---------------------------------------------------------------------------
# Employee login / logout
# ---------------------------------------------------------------------------

@app.post("/login")
def login(data: LoginRequest):
    try:
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
    except Exception as e:
        logger.exception("Login failed")
        raise HTTPException(status_code=500, detail="Login failed due to a server error") from e


@app.post("/logout")
def logout(data: LogoutRequest):
    try:
        add_audit_record({
            "action": "logout",
            "performed_by": data.name,
            "employee_id": data.employee_id
        })
        return {"success": True}
    except Exception as e:
        logger.exception("Logout failed")
        raise HTTPException(status_code=500, detail="Logout failed due to a server error") from e


# ---------------------------------------------------------------------------
# Company self-registration + company portal login
# ---------------------------------------------------------------------------

@app.post("/api/v1/companies/register")
def register_company(data: CompanyRegisterRequest):
    if not data.company_name or not data.company_name.strip():
        raise HTTPException(status_code=400, detail="Company name is required")
    if not data.contact_email or "@" not in data.contact_email:
        raise HTTPException(status_code=400, detail="A valid contact email is required")
    if len(data.password) < 8:
        raise HTTPException(status_code=400, detail="Password must be at least 8 characters")

    try:
        companies = get_companies()

        if find_company_by_email(data.contact_email):
            raise HTTPException(status_code=400, detail="A company is already registered with this email")

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
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Company registration failed")
        raise HTTPException(status_code=500, detail="Company registration failed due to a server error") from e


@app.post("/api/v1/companies/login")
def login_company(data: CompanyLoginRequest):
    try:
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
    except Exception as e:
        logger.exception("Company login failed")
        raise HTTPException(status_code=500, detail="Company login failed due to a server error") from e


@app.get("/api/v1/companies/me")
def get_my_company(x_api_key: Optional[str] = Header(None)):
    """Company portal: identify the logged-in company from its API key."""
    if not x_api_key:
        raise HTTPException(status_code=401, detail="Missing API key")

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
    if not x_api_key:
        raise HTTPException(status_code=401, detail="Missing API key")

    company = find_company_by_api_key(x_api_key)
    if not company:
        raise HTTPException(status_code=401, detail="Invalid API key")

    try:
        employees = get_employees()
        mine = [
            {k: v for k, v in e.items() if k != "company_code"}
            for e in employees
            if e["company_code"].lower() == company["company_code"].lower()
        ]
        return {"company_name": company["company_name"], "employees": mine}
    except Exception as e:
        logger.exception("Failed to load company employees")
        raise HTTPException(status_code=500, detail="Could not load employees due to a server error") from e


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

    try:
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
    except Exception as e:
        logger.exception("Employee sync failed")
        raise HTTPException(status_code=500, detail="Employee sync failed due to a server error") from e


@app.get("/api/v1/companies-overview")
def companies_overview():
    try:
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
    except Exception as e:
        logger.exception("Failed to load companies overview")
        raise HTTPException(status_code=500, detail="Could not load companies overview") from e


@app.get("/api/v1/login-history")
def login_history(action: Optional[str] = None, employee_id: Optional[str] = None):
    try:
        records = get_audit_records()
        filtered = [r for r in records if r.get("action") in ("login", "logout")]

        if action:
            filtered = [r for r in filtered if r.get("action") == action]
        if employee_id:
            filtered = [r for r in filtered if r.get("employee_id", "").lower() == employee_id.lower()]

        return {"records": filtered[::-1]}
    except Exception as e:
        logger.exception("Failed to load login history")
        raise HTTPException(status_code=500, detail="Could not load login history") from e


@app.get("/api/v1/companies-directory")
def companies_directory():
    """List of registered companies (used to power an autocomplete of
    company names when composing a verification email)."""
    try:
        companies = get_companies()
        return {
            "companies": [
                {"company_code": c["company_code"], "company_name": c["company_name"]}
                for c in companies
            ]
        }
    except Exception as e:
        logger.exception("Failed to load companies directory")
        raise HTTPException(status_code=500, detail="Could not load companies directory") from e


@app.post("/api/v1/resolve-company-emails")
def resolve_company_emails(data: ResolveEmailsRequest):
    """Step 1 of sending a verification email: resolve each typed company
    name to an email address (registered lookup, or web search + AI) and
    return it for the officer to review/edit — nothing is sent yet."""
    if not data.company_names:
        raise HTTPException(status_code=400, detail="At least one company name is required")

    resolved = []
    for name in data.company_names:
        if not name or not name.strip():
            continue
        try:
            resolved.append(resolve_recipient(name))
        except Exception:
            logger.exception(f"Failed to resolve recipient for company '{name}'")
            resolved.append({
                "company_name": name,
                "company_code": None,
                "email": None,
                "source": "error"
            })

    if not resolved:
        raise HTTPException(status_code=400, detail="No valid company names were provided")

    return {"resolved": resolved}


@app.post("/send-verification-email")
def send_verification_email(data: SendEmailRequest):
    """Step 2: actually send to the officer-confirmed recipient list."""
    if not data.recipients:
        raise HTTPException(status_code=400, detail="At least one recipient is required")

    results = []
    any_success = False

    for recipient in data.recipients:
        if not recipient.email or "@" not in recipient.email:
            results.append({
                "company_name": recipient.company_name,
                "company_code": recipient.company_code,
                "email": recipient.email,
                "status": "failed",
                "error": "Invalid or missing email address"
            })
            continue
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
            logger.exception(f"Failed to send email to {recipient.email}")
            results.append({
                "company_name": recipient.company_name,
                "company_code": recipient.company_code,
                "email": recipient.email,
                "status": "failed",
                "error": str(e)
            })

    try:
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
    except Exception:
        logger.exception("Failed to record audit/email thread after send attempt")

    return {"success": any_success, "recipients": results}


@app.get("/api/v1/my-sent-emails")
def my_sent_emails(employee_id: str):
    """Returns only the verification-email threads sent by this specific
    officer — never another HR/Procurement officer's sent emails."""
    if not employee_id or not employee_id.strip():
        raise HTTPException(status_code=400, detail="employee_id is required")

    try:
        threads = get_email_threads()
        mine = [t for t in threads if t.get("employee_id", "").lower() == employee_id.lower()]
        return {"threads": mine[::-1]}
    except Exception as e:
        logger.exception("Failed to load sent emails")
        raise HTTPException(status_code=500, detail="Could not load sent emails") from e


# ---------------------------------------------------------------------------
# Document upload + OCR + RAG storage
# ---------------------------------------------------------------------------

def _validate_upload(file: UploadFile, content: bytes):
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in ALLOWED_UPLOAD_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed types: {', '.join(sorted(ALLOWED_UPLOAD_EXTENSIONS))}"
        )
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")
    if len(content) > MAX_UPLOAD_SIZE_BYTES:
        raise HTTPException(status_code=400, detail="Uploaded file exceeds the 10 MB size limit")


@app.post("/upload-cv")
async def upload_cv(candidate_name: str, file: UploadFile = File(...)):
    if not candidate_name or not candidate_name.strip():
        raise HTTPException(status_code=400, detail="candidate_name is required")

    try:
        content = await file.read()
    except Exception as e:
        logger.exception("Failed to read uploaded file")
        raise HTTPException(status_code=400, detail="Could not read the uploaded file") from e

    _validate_upload(file, content)

    file_path = os.path.join(config.UPLOAD_DIR, file.filename)
    try:
        with open(file_path, "wb") as f:
            f.write(content)
    except Exception as e:
        logger.exception("Failed to save uploaded CV to disk")
        raise HTTPException(status_code=500, detail="Failed to save the uploaded file") from e

    try:
        extracted_text = extract_text_from_file(file_path)
    except Exception as e:
        logger.exception("OCR extraction failed for CV upload")
        raise HTTPException(status_code=500, detail="Failed to extract text from the document (OCR error)") from e

    if not extracted_text or not extracted_text.strip():
        raise HTTPException(status_code=422, detail="No readable text could be extracted from this document")

    chunks = chunk_text(extracted_text)

    try:
        stored_count = store_chunks_in_qdrant(chunks, {
            "type": "cv",
            "candidate": candidate_name,
            "filename": file.filename
        })
    except Exception as e:
        logger.exception("Failed to store CV chunks in Qdrant")
        raise HTTPException(status_code=500, detail="Failed to index the document for retrieval") from e

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
    if not supplier_name or not supplier_name.strip():
        raise HTTPException(status_code=400, detail="supplier_name is required")

    try:
        content = await file.read()
    except Exception as e:
        logger.exception("Failed to read uploaded file")
        raise HTTPException(status_code=400, detail="Could not read the uploaded file") from e

    _validate_upload(file, content)

    file_path = os.path.join(config.UPLOAD_DIR_SUPPLIER, file.filename)
    try:
        with open(file_path, "wb") as f:
            f.write(content)
    except Exception as e:
        logger.exception("Failed to save uploaded supplier document to disk")
        raise HTTPException(status_code=500, detail="Failed to save the uploaded file") from e

    try:
        extracted_text = extract_text_from_file(file_path)
    except Exception as e:
        logger.exception("OCR extraction failed for supplier document upload")
        raise HTTPException(status_code=500, detail="Failed to extract text from the document (OCR error)") from e

    if not extracted_text or not extracted_text.strip():
        raise HTTPException(status_code=422, detail="No readable text could be extracted from this document")

    chunks = chunk_text(extracted_text)

    try:
        stored_count = store_chunks_in_qdrant(chunks, {
            "type": "supplier",
            "supplier": supplier_name,
            "filename": file.filename
        })
    except Exception as e:
        logger.exception("Failed to store supplier document chunks in Qdrant")
        raise HTTPException(status_code=500, detail="Failed to index the document for retrieval") from e

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
    if not query or not query.strip():
        raise HTTPException(status_code=400, detail="query is required")

    try:
        results = retrieve_relevant_chunks(query, doc_type)
    except Exception as e:
        logger.exception("Retrieval failed for /search")
        raise HTTPException(status_code=500, detail="Search failed due to a server error") from e

    matches = []
    for r in results:
        matches.append({
            "score": r.score,
            "text": r.payload.get("text"),
            "metadata": {k: v for k, v in r.payload.items() if k != "text"}
        })

    return {"query": query, "results": matches}


# ---------------------------------------------------------------------------
# Evidence verification (multi-agent: Evidence Agent + Research Agent +
# Verdict Agent, coordinated by the Verification Orchestrator)
# ---------------------------------------------------------------------------

@app.get("/verify-claim")
def verify_claim(claim: str, doc_type: str = None, requested_by: str = "Unknown"):
    if not claim or not claim.strip():
        raise HTTPException(status_code=400, detail="claim is required")

    try:
        result = run_verification_workflow(claim, doc_type=doc_type)
    except Exception as e:
        logger.exception("Verification workflow failed")
        raise HTTPException(status_code=500, detail="Verification failed due to a server error") from e

    document_evidence = result["document_evidence"]
    web_evidence = result["web_evidence"]
    verdict = result["verdict"]

    status_line = verdict.split("\n")[0] if verdict else "Unknown"

    source_files = list({d["filename"] for d in document_evidence if d.get("filename")})
    source_names = list({d["source_name"] for d in document_evidence if d.get("source_name")})

    try:
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
    except Exception:
        logger.exception("Failed to write audit record for verify-claim (verdict still returned)")

    return {
        "claim": claim,
        "document_evidence": document_evidence,
        "web_evidence": web_evidence,
        "verification_result": verdict
    }


@app.get("/audit-log")
def get_audit_log():
    try:
        return {"records": get_audit_records()[::-1]}
    except Exception as e:
        logger.exception("Failed to load audit log")
        raise HTTPException(status_code=500, detail="Could not load audit log") from e