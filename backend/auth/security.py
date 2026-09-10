import re
import hashlib
import secrets


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