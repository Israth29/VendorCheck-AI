import os
import json
import uuid
from datetime import datetime

from config import EMPLOYEES_FILE, COMPANIES_FILE, EMAIL_THREADS_FILE, AUDIT_LOG_FILE


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


# ---- Employees ----

def get_employees():
    return load_json_file(EMPLOYEES_FILE, [])


def save_employees(employees):
    save_json_file(EMPLOYEES_FILE, employees)


# ---- Companies ----

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


# ---- Email threads (sent verification emails) ----

def get_email_threads():
    return load_json_file(EMAIL_THREADS_FILE, [])


def add_email_thread(thread):
    threads = get_email_threads()
    thread["id"] = str(uuid.uuid4())
    thread["timestamp"] = datetime.utcnow().isoformat()
    threads.append(thread)
    save_json_file(EMAIL_THREADS_FILE, threads)
    return thread


# ---- Audit log ----

def add_audit_record(record):
    record["timestamp"] = datetime.utcnow().isoformat()
    record["id"] = str(uuid.uuid4())

    records = load_json_file(AUDIT_LOG_FILE, [])
    records.append(record)
    save_json_file(AUDIT_LOG_FILE, records)

    return record


def get_audit_records():
    return load_json_file(AUDIT_LOG_FILE, [])