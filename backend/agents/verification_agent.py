"""
Backward-compatible re-exports.

Verification logic is now split into focused agents:
- evidence_agent.py    -> Document Evidence Agent (RAG)
- research_agent.py    -> Web Research Agent (search grounding)
- verdict_agent.py     -> Verdict / Decision Agent
- recipient_agent.py   -> Recipient Resolution Agent
- orchestrator.py      -> Supervisor Agent chaining the above

This file keeps the original function names importable so main.py
does not need to change.
"""
from agents.verdict_agent import run_verdict_agent, generate_verification
from agents.recipient_agent import run_recipient_agent, resolve_recipient, resolve_company_email_via_ai
from agents.orchestrator import run_verification_workflow