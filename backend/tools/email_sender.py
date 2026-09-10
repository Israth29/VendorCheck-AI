import resend
from config import EMAIL_FROM


def send_email(to_email: str, subject: str, message: str, requested_by: str):
    """Sends one verification email via Resend. Raises on failure so the
    caller can record success/failure per recipient."""
    params = {
        "from": EMAIL_FROM,
        "to": [to_email],
        "subject": subject,
        "html": f"""
            <p>{message}</p>
            <hr>
            <p style="color:#666;font-size:12px;">This verification email was sent with explicit approval from {requested_by} via VendorCheck AI.</p>
        """
    }
    return resend.Emails.send(params)