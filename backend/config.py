import os
import platform
from dotenv import load_dotenv
import google.generativeai as genai
from tavily import TavilyClient
import resend

load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
resend.api_key = os.getenv("RESEND_API_KEY")

if platform.system() == "Windows":
    import pytesseract
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
# Linux (Docker container) — apt-installed tesseract is already on PATH, no need to set path

# ---- File paths (JSON "database") ----
EMPLOYEES_FILE = "employees.json"
COMPANIES_FILE = "companies.json"
EMAIL_THREADS_FILE = "email_threads.json"
AUDIT_LOG_FILE = "audit_log.json"

# ---- Upload directories ----
UPLOAD_DIR = "uploads/cv"
UPLOAD_DIR_SUPPLIER = "uploads/supplier"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR_SUPPLIER, exist_ok=True)

# ---- Vector DB ----
QDRANT_PATH = "qdrant_data"
COLLECTION_NAME = "vendorcheck_documents"
EMBEDDING_MODEL = "models/gemini-embedding-001"
EMBEDDING_SIZE = 3072

# ---- LLM ----
GENERATION_MODEL = "models/gemini-3.6-flash"

# ---- Email ----
EMAIL_FROM = "VendorCheck AI <onboarding@resend.dev>"