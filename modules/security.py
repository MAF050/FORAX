import smtplib
import os
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv

load_dotenv()

# Gmail SMTP via App Passwords (set via env/.env; do NOT hardcode secrets)
GMAIL_ADDRESS = (os.getenv('FORAX_GMAIL_ADDRESS') or os.getenv('GMAIL_ADDRESS') or '').strip()
GMAIL_APP_PASS = (os.getenv('FORAX_GMAIL_APP_PASS') or os.getenv('GMAIL_APP_PASS') or '').strip()

_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
_EMAIL_TEMPLATE_PATH = os.path.normpath(
    os.path.join(_MODULE_DIR, '..', 'templates', 'emails', 'email_template.html')
)

MAX_ATTEMPTS   = 5
BLOCK_MINUTES  = 15

def get_failed_attempts(ip):
    from modules.database import get_rate_limit
    return get_rate_limit(ip)

def record_failed_attempt(ip):
    from modules.database import get_rate_limit, set_rate_limit
    row = get_rate_limit(ip)
    now = int(datetime.now().timestamp())
    if not row:
        # First failure — record it, no block yet
        set_rate_limit(ip, 1, None)
        return False, 1
    
    blocked_until = int(row['blocked_until']) if row['blocked_until'] else 0
    
    # If already blocked, reject immediately
    if row['attempts'] >= MAX_ATTEMPTS and now < blocked_until:
        return True, row['attempts']
    
    # Increment attempts
    attempts = row['attempts'] + 1
    
    # Block if threshold reached
    if attempts >= MAX_ATTEMPTS:
        set_rate_limit(ip, attempts, now + BLOCK_MINUTES * 60)
        return True, attempts
    else:
        set_rate_limit(ip, attempts, None)
        return False, attempts

def is_ip_blocked(ip):
    from modules.database import get_rate_limit
    row = get_rate_limit(ip)
    if not row: return False, 0
    now = int(datetime.now().timestamp())
    
    blocked_until = int(row['blocked_until']) if row['blocked_until'] else 0
    blocked = row['attempts'] >= MAX_ATTEMPTS and now < blocked_until
    mins = max(1, (blocked_until - now) // 60) if blocked else 0
    return blocked, mins

def reset_failed_attempts(ip):
    from modules.database import clear_rate_limit as db_clear
    db_clear(ip)

def clear_rate_limit(ip):
    """Wrapper for app.py import compatibility."""
    reset_failed_attempts(ip)

def get_attempts_left(ip):
    from modules.database import get_rate_limit
    row = get_rate_limit(ip)
    if not row: return MAX_ATTEMPTS
    return max(0, MAX_ATTEMPTS - row['attempts'])

def generate_otp():
    import random
    return str(random.randint(100000, 999999))

def generate_math_captcha():
    import random
    a = random.randint(1, 10)
    b = random.randint(1, 10)
    return f"{a} + {b}", str(a + b)

# ── E-MAIL LOGIC ───────────────────
def _get_base_html():
    try:
        with open(_EMAIL_TEMPLATE_PATH, 'r', encoding='utf-8') as f:
            return f.read()
    except OSError:
        return "<p>{body_content}</p>" # Minimal fallback

def send_email(to_email, subject, title, body_content):
    if not GMAIL_ADDRESS or not GMAIL_APP_PASS:
        return False, "Email not configured. Set FORAX_GMAIL_ADDRESS and FORAX_GMAIL_APP_PASS (or GMAIL_ADDRESS/GMAIL_APP_PASS)."
    try:
        msg = MIMEMultipart('alternative')
        msg['Subject'] = subject
        msg['From']    = f"FORAX Security <{GMAIL_ADDRESS}>"
        msg['To']      = to_email
        
        html = _get_base_html().format(title=title, body_content=body_content)
        msg.attach(MIMEText(html, 'html'))
        
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(GMAIL_ADDRESS, GMAIL_APP_PASS)
            server.sendmail(GMAIL_ADDRESS, to_email, msg.as_string())
        return True, "Sent"
    except Exception as e:
        return False, str(e)

def email_otp(to_email, otp, name='Investigator'):
    body = f"<p>Hello <b>{name}</b>,</p><p>Your password reset code is: <h2 style='color:#0f5f46;'>{otp}</h2></p><p>Valid for 10 minutes.</p>"
    return send_email(to_email, "FORAX - Password Reset Code", "Password Reset", body)

def email_welcome(to_email, name, badge_id, temp_password=None):
    pw_str = f"<p>Temp Password: <b>{temp_password}</b></p>" if temp_password else ""
    body = f"<p>Hello <b>{name}</b>, your account (Badge {badge_id}) has been created by the Chief.</p>{pw_str}"
    return send_email(to_email, f"FORAX - Account Created [{badge_id}]", "Account Created", body)

def email_login_alert(to_email, name, ip, time_str):
    body = f"<p>Hello <b>{name}</b>, a successful login was detected from IP: <code>{ip}</code> at {time_str}.</p>"
    return send_email(to_email, "FORAX - New Login Alert", "New Login", body)

def email_lockout_alert(to_email, name, ip):
    body = f"<p>Hello <b>{name}</b>, your account has been temporarily locked following {MAX_ATTEMPTS} failed attempts from IP <code>{ip}</code>.</p>"
    return send_email(to_email, "FORAX - Account Locked", "Account Locked", body)
