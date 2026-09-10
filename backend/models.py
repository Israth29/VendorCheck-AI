from pydantic import BaseModel
from typing import Optional, List


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