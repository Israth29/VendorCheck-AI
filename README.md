# VendorCheck AI

VendorCheck AI is a verification platform for HR and Procurement teams. It checks claims about candidates and suppliers against uploaded documents and live web results, using a small team of AI agents and a RAG pipeline backed by a vector database.

**Live App:** https://vendorcheck-ai-1.onrender.com

**Backend API:** https://vendorcheck-ai.onrender.com

**Youtube Link:** https://youtu.be/BdllEp0zKkA

**Registration Companey:** https://vendorcheck-ai-1.onrender.com/company-register.html

**Admin:** https://vendorcheck-ai-1.onrender.com/company-overview.html

---

## 1. Problem Statement

### Who the target user is
HR Officers and Procurement Officers. They're the people who have to sign off on a candidate's background or a supplier's credentials before a hiring or contracting decision goes through.

### What problem the user currently faces
A lot of the claims that land on an HR or Procurement desk are hard to check by hand. A candidate says they worked somewhere for five years. A supplier claims a certain registration or certification. Confirming any of this means going through the uploaded document, then searching the web, then figuring out if what you found on the web is even about the right person or company (a name match isn't the same as an identity match). Doing this manually for every claim is slow and it's easy to get wrong.

### How the system solves or reduces that problem
The officer uploads a document, types in the claim, and the system does the cross-checking for them. It pulls the relevant part of the document, searches the web, and reasons over both to produce a verdict with an explanation. A human still reviews the result and approves anything that goes out (like a verification email) — the AI narrows the work down, it doesn't replace the judgment call.

### Why AI is useful here
This isn't really a lookup problem, it's a judgment problem. Deciding whether a web result is actually about the same person or company named in the document takes reasoning, not a database query. That's the part an LLM agent is good at, as long as it's grounded in evidence that was actually retrieved rather than guessing from what it already "knows." That's also why there's an identity-matching check built into the verdict prompt — without it, a same-name mismatch could easily get treated as confirmation.

---

### The end-to-end pipeline
1. OCR extracts text from the uploaded document, which gets chunked and embedded.
2. When a claim comes in, the system retrieves the most relevant chunks from that document (RAG).
3. It also searches the live web for anything relevant.
4. An LLM agent weighs both sources and gives one of four verdicts — **Strongly Supported**, **Partially Supported**, **Not Verified**, or **Unable to Verify** — depending on how much of the claim the retrieved evidence actually backs up, while explicitly discarding web evidence that looks like it's about a different person or company with the same name.
5. Everything gets logged to an audit trail, and the officer can route the case to a verification email if needed.

---

## 2. Project Requirements Coverage

### 3.1 Real-World Problem
Covered in Section 1 above — target user (HR/Procurement officers), the problem they face (manual, error-prone claim verification), how the system solves it (RAG + web search + agent reasoning, with a human still approving anything sent out), and why AI is needed (it's a judgment call, not a lookup).

### 3.2 Front-End Application
Plain HTML, CSS, and JavaScript (Tailwind for styling, no framework). It's a full working interface — login, upload, claim verification, email sending, company portal, and admin views — not just a chat box wrapped around an LLM. See [Front-End](#35-front-end) for details.

### 3.3 Back-End
FastAPI, with the codebase split into clearly separate folders: `agents/` for AI logic, `tools/` for OCR/search/email, `retrieval/` for the vector DB, `database/` for storage, `prompts/` for prompt templates, `config.py` for settings, and `main.py` for the API routes that tie it all together. See [Back-End](#36-back-end) and the [Folder Structure](#34-folder-structure).

### 4. AI and Agent Requirements
Four agents, each with one job: an Evidence Agent (pulls document evidence via RAG), a Research Agent (searches the web), a Verdict Agent (reasons over both and gives the final verdict), and a Recipient Agent (resolves a company name to a contact email). An Orchestrator coordinates the first three for claim verification. No agent here is decorative — each one exists because that specific step needed its own clear responsibility, not to inflate the agent count.

---

## 3. Architecture

### 3.1 Overall Application Flow

![Application Flow](docs/architecture.png)

Login checks the employee against their company code and role, then sends them to either the HR or Procurement dashboard. Both roles get the same three-step workflow (upload, verify, email) plus a shared "My Sent Emails" history.

### 3.2 Feature-Level Structure

![Feature Structure](docs/diagram2.png)

- **Upload** — OCR (external) pulls text out of the file, which gets chunked and embedded (Gemini API, external) and stored in Qdrant (internal).
- **Verify a Claim** — the Orchestrator hands the claim to three agents: Evidence Agent (internal, reads from Qdrant), Research Agent (external, Tavily search), and Verdict Agent (external, Gemini reasoning). The result gets written to the audit log.
- **Verification Email** — the Recipient Agent checks the internal company database first, and falls back to a web search + AI extraction if the company isn't registered. The officer confirms before anything is sent through Resend.

### 3.3 Company & Admin Side

![Company and Admin Structure](docs/diagram3.png)

Company passwords are hashed with PBKDF2 (`auth/security.py`) and only the hash gets stored in `companies.json`. Login re-hashes whatever was typed and compares it to the stored hash — the plaintext password is never saved or logged anywhere. A password-reset flow is sketched out in the diagram above but isn't built yet (see [Known Limitations](#12-known-limitations--future-work)).

### 3.4 Folder Structure

```
VendorCheck-AI/
├── backend/
│   ├── agents/
│   │   ├── evidence_agent.py       # RAG retrieval agent (Qdrant)
│   │   ├── research_agent.py       # Web search agent (Tavily)
│   │   ├── verdict_agent.py        # Reasoning agent (Gemini) - final STATUS/EXPLANATION
│   │   ├── recipient_agent.py      # Resolves company name -> contact email
│   │   ├── orchestrator.py         # Chains evidence -> research -> verdict agents
│   │   └── verification_agent.py   # Backward-compatible re-export shim
│   ├── auth/
│   │   └── security.py             # Password hashing (PBKDF2), company code generation
│   ├── database/
│   │   └── store.py                # JSON read/write: employees, companies, audit log, email threads
│   ├── retrieval/
│   │   └── vector_store.py         # Qdrant client, chunking, embeddings, retrieval
│   ├── tools/
│   │   ├── ocr.py                  # Tesseract OCR text extraction
│   │   ├── web_search.py           # Tavily API wrapper
│   │   └── email_sender.py         # Resend API wrapper
│   ├── prompts/
│   │   └── verification_prompts.py # Prompt templates for verdict + email resolution
│   ├── uploads/
│   │   ├── cv/                     # Uploaded candidate CVs
│   │   └── supplier/               # Uploaded supplier documents
│   ├── qdrant_data/                # Local persistent vector DB storage
│   ├── config.py                   # Env var loading, model names, paths
│   ├── models.py                   # Pydantic request/response schemas
│   ├── main.py                     # FastAPI app - all API routes
│   ├── requirements.txt
│   ├── .env / .env.example
│   ├── companies.json / employees.json / audit_log.json
│   └── Dockerfile
├── frontend/
│   ├── index.html                  # Employee login (landing page)
│   ├── hr-dashboard.html           # HR Officer dashboard
│   ├── procurement-dashboard.html  # Procurement Officer dashboard
│   ├── my-sent-emails.html         # Per-officer sent email history
│   ├── company-register.html       # Company self-registration
│   ├── company-login.html          # Company portal login
│   ├── company-portal.html         # Company self-service (view employees)
│   ├── company-overview.html       # Internal admin: all companies
│   ├── activity-log.html           # Internal admin: audit log
│   └── script.js
├── docker-compose.yml
└── .gitignore
```

### 3.5 Front-End

plain HTML, CSS, and JavaScript, styled with Tailwind loaded from a CDN. Each page is its own `.html` file (`index.html`, `hr-dashboard.html`, `procurement-dashboard.html`, etc.), with `script.js` handling the logic: reading form inputs, calling the backend API with `fetch`, and updating the DOM with whatever comes back — showing the verdict, the resolved email, the upload confirmation, and so on. Login state (employee name, role, company code) is kept in `localStorage` so it survives a page refresh, since there's no framework-level state management here. There's no build step — the browser runs these files as-is, which is also why the frontend has to be served through a local HTTP server during development rather than opened directly as a file.

### 3.6 Back-End

FastAPI, running as a single app defined in `main.py`. Every feature is exposed as its own route — for example `POST /upload-cv`, `GET /verify-claim`, `POST /api/v1/resolve-company-emails`, `POST /send-verification-email`. A route's job is kept narrow: validate the input, call the right function from `agents/`, `retrieval/`, `tools/`, or `database/` to do the actual work, then return a JSON response. None of the AI logic, database logic, or file handling lives inside `main.py` itself — it only wires things together, which is what keeps the file readable even with this many routes. CORS is enabled so the static frontend (served from a different origin on Render) can call the API directly.

### 3.7 Data & AI Stack

| Component | Technology |
|---|---|
| LLM (reasoning + generation) | Google Gemini |
| Embeddings | Google Gemini |
| Vector Database | Qdrant (local persistent storage) |
| OCR | Tesseract |
| Web Search / Grounding | Tavily API |
| Email | Resend API |
| Tracing | LangSmith (every agent, embedding, and retrieval call is traced) |
| Containerization | Docker |
| Hosting | Render (Web Service for backend, Static Site for frontend) |

---

## 4. Why This Counts as a Complete Workflow, Not a Single Feature

The brief describes a few example architectures — Router → Agent → Tool/RAG, or Upload → OCR → Retrieval → Analysis, or Planner → Multiple Agents → Aggregation. This project ends up combining pieces of all three, because the underlying problem naturally has three connected parts:

- **Document upload** follows the OCR → Retrieval pattern: extract text, chunk it, embed it, store it for later retrieval.
- **Claim verification** follows the Planner → Multiple Agents pattern: the Orchestrator hands the claim to the Evidence and Research agents in parallel, then the Verdict Agent aggregates both into one answer.
- **Verification email** follows the Router → Tool pattern: the Recipient Agent checks the internal database first, and only reaches for the external search tool if that lookup comes up empty.

These aren't three separate, disconnected demos. The document indexed in step one is what the Evidence Agent reads from in step two, and the verdict from step two is what the officer reviews before triggering step three. It's one pipeline — verify, decide, notify — built out of the pieces each part actually needed.

---

## 5. Screenshots

_Add screenshots below — see the "Screenshots to add" checklist at the end of this file for exactly which ones._

### Login & Dashboard
`docs/screenshots/login.png`
`docs/screenshots/hr-dashboard.png`

### Document Upload
`docs/screenshots/upload-cv.png`

### Claim Verification
`docs/screenshots/verify-claim-supported.png`
`docs/screenshots/verify-claim-not-verified.png`

### Verification Email
`docs/screenshots/resolve-email.png`
`docs/screenshots/send-email.png`

### Company Portal / Admin
`docs/screenshots/company-register.png`
`docs/screenshots/company-overview.png`

---

## 6. Key Technical Decisions

- **FastAPI for the backend.** It's fast to write routes in, validates request bodies automatically using the Pydantic models in `models.py`, and gives free interactive API docs (Swagger UI) at `/docs` — useful for testing routes like `/verify-claim` directly without needing the frontend running.
- **Docker for the backend.** The backend depends on Tesseract (OCR) and a specific Python setup that isn't guaranteed to exist on any machine it gets deployed to. The `Dockerfile` packages the FastAPI app together with Tesseract and all Python dependencies into one image, so it runs the same way locally and on Render — no "works on my machine" gap. `docker-compose.yml` just makes it a one-command start (`docker compose up -d`) instead of typing out a long `docker run` command with all the port and volume flags. On Render, the backend is deployed by building this same Docker image directly, which is why the Dockerfile is what actually defines the production environment.
- **Split into multiple agents instead of one big function.** Verification logic used to live in a single `verification_agent.py` function. It's now an Orchestrator coordinating four agents with one job each, mostly so each piece shows up as its own span in LangSmith and can be reasoned about on its own.
- **Tavily instead of a raw search API.** Simpler to integrate, and the JSON it returns plugs straight into the verdict prompt without much cleanup.
- **Qdrant in local persistent mode, not a hosted cluster.** It's a real vector database with real similarity search, and it's enough for what this project needs without adding another external service to manage.
- **JSON files instead of a SQL database.** Given the scope here (companies, employees, an audit log), a JSON store keeps things simple without giving up the separation the project needed. This would be the first thing to swap out in a production version.
- **The identity-matching check lives in the prompt, not a separate model.** Keeping it inside the Verdict Agent's reasoning made it easier to get right and keeps the whole decision, safeguard included, visible in one trace.

---

## 7. Error Handling

Every backend route validates its input before doing any real work, and wraps external calls (OCR, Qdrant, Resend) in try/except so a failure in one step returns a clean error instead of crashing the app.

There are two layers of validation working together, and the screenshot below shows the first one:

**Browser-level (frontend):** required fields use plain HTML `required`, so trying to submit an empty claim, for example, gets caught immediately with a native browser message — before any request even reaches the backend.

`docs/screenshots/error-handling-frontend.png`

**API-level (backend):** the same check exists again on the server, independent of the frontend, because the frontend validation can always be bypassed (e.g. hitting the URL directly). For example, calling `/verify-claim` with an empty `claim` parameter directly returns:
```json
{"detail": "claim is required"}
```
instead of a 500 error or a silent failure. The same pattern — validate first, wrap external calls in try/except, return a structured `400`/`422`/`500` — is used across upload, claim verification, and email routes.

---

## 8. Course Concept Mapping

| Requirement | Implementation |
|---|---|
| AI Agents (multi-agent) | `agents/orchestrator.py` coordinates `evidence_agent.py` (RAG), `research_agent.py` (web search), `verdict_agent.py` (reasoning), and `recipient_agent.py` (email resolution) — each with one clear job |
| Tool Integration | OCR, web search, and email each live in their own module under `tools/`, called by whichever agent needs them |
| Search Grounding | Tavily web search stands in for Google Search grounding — the Verdict Agent is required to reason from retrieved evidence, not memory |
| RAG | `retrieval/vector_store.py` chunks and embeds documents into Qdrant; `retrieve_relevant_chunks` pulls the closest matches before generation |
| Vector Database | Qdrant, storing embedded document chunks per verification session |
| OCR | Tesseract (`tools/ocr.py`) extracts text from uploaded CVs and supplier documents before chunking |
| Observability | LangSmith traces the orchestrator, every sub-agent, and the embedding/retrieval calls (project: `VendorCheck-AI`) — see [LangSmith Traces](#9-langsmith-traces) |
| Evaluation / Safety | The identity-matching check in the Verdict Agent's prompt stops same-name, different-entity web results from counting as support |
| Error Handling | Every route validates its input (empty claims, bad emails, unsupported file types, oversized uploads) and wraps external calls (OCR, Qdrant, Resend) in try/except, so failures come back as clean `400`/`422`/`500` responses instead of crashes |

---

## 9. LangSmith Traces

These are from a single end-to-end test pass, in order: uploading a candidate CV, verifying it as Strongly Supported / Partially Supported / Not Verified, sending a verification email, then uploading a supplier PDF and verifying it as Not Verified / Unable to Verify, then sending a second verification email. (Note: the actual email-send call isn't wrapped in a LangSmith trace — the traced step for "email sent" is the Recipient Resolution Agent that runs just before it.)

1. `get_embedding` (CV upload): https://smith.langchain.com/public/23aa6230-2ffe-4b1f-9ab0-e9c819b0ed7c/r/01a090cb-81e6-7ad1-9d2f-c4261b77e043
2. `verification_orchestrator` (Strongly Supported): https://smith.langchain.com/public/0afd130a-525c-4475-95e3-3986608aebb2/r/01a090ce-29f0-7153-8dad-df47702328f2
3. `verification_orchestrator` (Partially Supported): https://smith.langchain.com/public/374b584b-92e7-4cd6-b84a-75e9f0e838d7/r/01a090d1-1e8d-7633-a057-152db19a8e43
4. `verification_orchestrator` (Not Verified): https://smith.langchain.com/public/84febe0a-9f9e-4123-9284-b57307c26911/r/01a090d3-73cb-7e51-93eb-d24416f4fb01
5. `recipient_resolution_agent` (email sent): https://smith.langchain.com/public/822ee8ab-c5c0-4ae8-ab4a-5cc0b0bdbff9/r/01a090d4-b6f5-7133-ae6c-027b39a85d4d
6. `get_embedding` (supplier PDF upload, chunk 1): https://smith.langchain.com/public/a5e7f9ca-c5ad-4f25-bf84-f88612628405/r/01a090d8-d27d-7901-9270-524f17d916f2
7. `get_embedding` (supplier PDF upload, chunk 2): https://smith.langchain.com/public/3e034a09-9c6b-41cb-85d9-6dc184ca3a28/r/01a090d8-d34a-7cc1-a586-e2a1811d3344
8. `get_embedding` (supplier PDF upload, chunk 3): https://smith.langchain.com/public/32d1eb79-014f-419f-8325-7b32d43c1314/r/01a090d8-d3f5-76e0-84a5-09f0ad7aadc6
9. `get_embedding` (supplier PDF upload, chunk 4): https://smith.langchain.com/public/c6b29336-8883-449d-a21e-a42e0e90ca64/r/01a090d8-d4be-7d23-9dba-4dae67872352
10. `get_embedding` (supplier PDF upload, chunk 5): https://smith.langchain.com/public/2d071b79-466d-43ae-9bd7-e124e153129d/r/01a090d8-d58b-7f20-803f-0ae90bb52e11
11. `verification_orchestrator` (supplier, Not Verified): https://smith.langchain.com/public/25276c31-2ec0-4f13-bd50-aff3560235c1/r/01a090d9-f82b-7562-92f1-e542c6006e25
12. `verification_orchestrator` (supplier, Unable to Verify): https://smith.langchain.com/public/8d78cc0c-9695-4c4f-93cc-d9d0e5258839/r/01a090db-a70d-7133-8fff-c46e5382b7e3
13. `recipient_resolution_agent` (second email sent): https://smith.langchain.com/public/426fa5e2-dc75-464d-90ec-3e0cb5e2a5fc/r/01a090dc-d4f7-7ce2-991e-20c3f266668f

---

## 10. Setup Instructions

### Backend

```powershell
cd backend
python -m venv ../venv
..\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in your keys:
```
GEMINI_API_KEY=your_key
TAVILY_API_KEY=your_key
RESEND_API_KEY=your_key
LANGCHAIN_API_KEY=your_key
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=VendorCheck-AI
```

Run locally:
```powershell
uvicorn main:app --reload
```

### Frontend

Serve through a local HTTP server — opening `index.html` directly breaks `localStorage` and `fetch`:
```powershell
cd frontend
python -m http.server 5500
```

### Docker (backend)

```powershell
docker compose up -d
```

---

## 11. Deployment

- **Backend:** Render Web Service (Docker), auto-deploys from the `main` branch of [Israth29/VendorCheck-AI](https://github.com/Israth29/VendorCheck-AI).
- **Frontend:** Render Static Site, pointed at the live backend URL.

---

## 12. Known Limitations / Future Work

- **Employee login** is currently a plaintext match on ID, name, and company code — no password or session yet. That's planned for a later pass.
- **Company password reset** is designed (see the diagram in 2.3) but not built. It would email a time-limited reset link.
- **Custom email domain**: Resend's free tier only lets you send to your own verified email until you verify a real domain, so the "Verification Email" and "My Sent Emails → reply" features are demoed using the developer's own email as a stand-in. A real domain is the fix for production use.

---

## 13. Inspiration

The idea for this project grew out of some NLP research I've been doing — one part on detecting bug priority, and a separate part on detecting severity from reviews. Working on that got me thinking about the same underlying problem in a different setting: taking a claim someone makes and figuring out, with evidence, how much weight it actually deserves. Around the same time I'd been thinking about how companies handle hiring and vendor onboarding — how much of that verification is still manual, and how easy it is to accept a claim just because a web search returned something with a matching name. Combining those two threads is where VendorCheck AI came from.

VendorCheck AI Developed By : Israth Jahan Worthy
