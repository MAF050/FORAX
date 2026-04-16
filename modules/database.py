"""
╔══════════════════════════════════════════════════════════════════╗
║  FORAX — Database Module (SQLite)                                ║
║  Manages all data persistence for the forensic platform          ║
╚══════════════════════════════════════════════════════════════════╝

PURPOSE:
  This module handles ALL database operations for FORAX using SQLite.
  It provides functions for managing:
  - Users (investigators, chief)
  - Cases (forensic investigations)
  - Evidence (files extracted from devices)
  - Activity logs (audit trail)
  - OTP store (password reset codes)
  - Rate limiting (brute-force protection)

DATABASE SCHEMA:
  ┌─────────────┐    ┌──────────────┐    ┌──────────────┐
  │   users      │───▶│    cases      │───▶│   evidence   │
  │ id (PK)      │    │ id (PK)      │    │ id (PK)      │
  │ name         │    │ case_id      │    │ case_id (FK)  │
  │ email        │    │ title        │    │ device_slot   │
  │ password     │    │ investigator │    │ filename      │
  │ role         │    │ device1_*    │    │ sha256        │
  │ badge_id     │    │ device2_*    │    │ risk_level    │
  │ is_active    │    │ threat_level │    │ ai_result     │
  └─────────────┘    └──────────────┘    │ gps_lat/lng   │
                                          └──────────────┘
  ┌─────────────┐    ┌──────────────┐    ┌──────────────┐
  │ activity_log │    │  otp_store   │    │ rate_limits  │
  │ user_id      │    │ email (PK)   │    │ ip (PK)      │
  │ action       │    │ otp          │    │ attempts     │
  │ detail       │    │ expires_at   │    │ blocked_until│
  │ ip_address   │    │ attempts     │    │ last_attempt │
  └─────────────┘    └──────────────┘    └──────────────┘

DEFAULT CREDENTIALS (seeded on first run):
  Chief:     chief@forax.gov / Chief@FORAX2025

ROLES:
  chief        — Full access: manages investigators, views all cases
  investigator — Case access: creates cases, extracts & analyzes evidence

SECURITY NOTES:
  - All passwords are hashed with bcrypt (salt auto-generated)
  - OTPs expire after 10 minutes with 3 attempt limit
  - Rate limiting blocks IPs after 5 failed login attempts
"""

import sqlite3   # Lightweight embedded database (no server needed)
import bcrypt     # Password hashing library (industry standard)
import os         # For file path operations
from datetime import datetime  # For timestamps
import zlib       # For evidence data compression
import base64     # For encoding compressed binaries

# ── DATABASE FILE LOCATION ─────────────────────────────────────
# The database file (forax.db) is stored in the project root directory.
# os.path.dirname(__file__) gives us the 'modules/' folder,
# then '..' goes up one level to the project root.
DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'forax.db')


def _ensure_users_columns(conn):
    """Add missing columns to legacy users tables without dropping data."""
    c = conn.cursor()
    cols = {r[1] for r in c.execute("PRAGMA table_info(users)").fetchall()}
    required = [
        ('department', "TEXT"),
        ('assigned_by', "INTEGER"),
        ('is_active', "INTEGER DEFAULT 1"),
        ('created_at', "TEXT DEFAULT CURRENT_TIMESTAMP"),
        ('last_login', "TEXT"),
    ]
    for col, col_type in required:
        if col not in cols:
            c.execute(f"ALTER TABLE users ADD COLUMN {col} {col_type}")


def _ensure_cases_columns(conn):
    """Add missing columns to legacy cases tables without dropping data."""
    c = conn.cursor()
    cols = {r[1] for r in c.execute("PRAGMA table_info(cases)").fetchall()}
    required = [
        ('status', "TEXT DEFAULT 'active'"),
        ('device1_serial', "TEXT"),
        ('device1_model', "TEXT"),
        ('device1_android', "TEXT"),
        ('device2_serial', "TEXT"),
        ('device2_model', "TEXT"),
        ('device2_android', "TEXT"),
        ('total_files', "INTEGER DEFAULT 0"),
        ('flagged_files', "INTEGER DEFAULT 0"),
        ('total_size_mb', "REAL DEFAULT 0"),
        ('threat_level', "TEXT DEFAULT 'LOW'"),
        ('created_at', "TEXT DEFAULT CURRENT_TIMESTAMP"),
        ('updated_at', "TEXT DEFAULT CURRENT_TIMESTAMP"),
    ]
    for col, col_type in required:
        if col not in cols:
            c.execute(f"ALTER TABLE cases ADD COLUMN {col} {col_type}")


def _ensure_evidence_columns(conn):
    """Add missing columns to legacy evidence tables without dropping data."""
    c = conn.cursor()
    cols = {r[1] for r in c.execute("PRAGMA table_info(evidence)").fetchall()}
    required = [
        ('device_slot', "TEXT DEFAULT 'device1'"),
        ('file_type', "TEXT"),
        ('file_size', "TEXT"),
        ('sha256', "TEXT"),
        ('risk_level', "TEXT DEFAULT 'LOW'"),
        ('ai_result', "TEXT"),
        ('gps_lat', "REAL"),
        ('gps_lng', "REAL"),
        ('exif_date', "TEXT"),
        ('file_path', "TEXT"),
        ('extracted_at', "TEXT DEFAULT CURRENT_TIMESTAMP"),
        ('override_risk', "TEXT"),
        ('override_by', "INTEGER"),
        ('override_note', "TEXT"),
        ('override_at', "TEXT"),
        ('confidence', "INTEGER DEFAULT 0"),
    ]
    for col, col_type in required:
        if col not in cols:
            c.execute(f"ALTER TABLE evidence ADD COLUMN {col} {col_type}")


# ══════════════════════════════════════════════════════════════════
# CONNECTION HELPER
# ══════════════════════════════════════════════════════════════════

def get_db():
    """
    Open a connection to the SQLite database.
    
    Uses sqlite3.Row as row_factory so results can be accessed 
    by column name (e.g., user['email']) instead of index (user[2]).
    
    IMPORTANT: Every function that calls get_db() must also close 
    the connection after use to prevent database locks.
    
    Returns:
        sqlite3.Connection: Active database connection
    """
    conn = sqlite3.connect(DB_PATH, isolation_level=None, check_same_thread=False, timeout=20)
    conn.row_factory = sqlite3.Row  # Enable dict-like row access
    return conn

# ══════════════════════════════════════════════════════════════════
# COMPRESSION HELPER
# ══════════════════════════════════════════════════════════════════

def compress_text(text):
    """Compresses large text logs into a zlib base64 string to save DB space."""
    if not text: return text
    try:
        compressed = zlib.compress(text.encode('utf-8'))
        return "ZLIB:" + base64.b64encode(compressed).decode('utf-8')
    except:
        return text

def decompress_text(text):
    """Decompresses zlib base64 strings back to text."""
    if not text or not str(text).startswith("ZLIB:"): return text
    try:
        compressed = base64.b64decode(text[5:])
        return zlib.decompress(compressed).decode('utf-8')
    except:
        return text


# ══════════════════════════════════════════════════════════════════
# DATABASE INITIALIZATION — Called once on app startup
# ══════════════════════════════════════════════════════════════════

def init_db():
    """
    Initialize the database — create all tables and seed default users.
    
    This function is SAFE to call multiple times (uses CREATE IF NOT EXISTS).
    Called by:
      - desktop.py on app startup
      - app.py when run directly
    
    Tables created:
      1. users         — Investigators, Chief, Authority accounts
      2. cases         — Forensic investigation cases
      3. evidence      — Extracted files with AI analysis results
      4. activity_logs — Audit trail of all user actions
      5. otp_store     — Temporary OTP codes for password reset
      6. rate_limits   — IP-based brute-force protection
    
    Default users seeded:
      - Chief Investigator (chief@forax.gov) — manages investigators
      - Senior Authority (authority@forax.gov) — oversight/CMS access
    """
    conn = get_db(); c = conn.cursor()
    
    # ── USERS TABLE ────────────────────────────────────────────
    # Stores all user accounts (investigators, chief, authority)
    # Passwords are bcrypt-hashed, never stored in plaintext
    c.execute('''CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,                         -- Full name
        email TEXT UNIQUE NOT NULL,                 -- Login email (unique)
        password TEXT NOT NULL,                     -- Bcrypt hash
        role TEXT NOT NULL DEFAULT 'investigator',  -- 'investigator' or 'chief'
        badge_id TEXT UNIQUE,                       -- e.g., INV-001, CHIEF-001
        department TEXT,                            -- Department name
        assigned_by INTEGER,                        -- User ID of chief who created this account
        is_active INTEGER DEFAULT 1,                -- 1=active, 0=disabled
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,  -- Account creation time
        last_login TEXT)''')  # Last login timestamp

    # ── MIGRATIONS ─────────────────────────────────────────────
    # Ensure all expected columns exist in existing databases.
    _ensure_users_columns(conn)

    
    # ── CASES TABLE ────────────────────────────────────────────
    # Each case represents a forensic investigation
    # Supports up to 2 devices per case (device1 and device2)
    c.execute('''CREATE TABLE IF NOT EXISTS cases (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        case_id TEXT UNIQUE NOT NULL,       -- e.g., FORAX-20250407143000
        title TEXT NOT NULL,                -- Case title
        description TEXT,                   -- Case description
        investigator_id INTEGER NOT NULL,   -- FK → users.id
        status TEXT DEFAULT 'active',       -- 'active' or 'closed'
        device1_serial TEXT,                -- Device 1 serial number
        device1_model TEXT,                 -- Device 1 model name
        device1_android TEXT,               -- Device 1 Android version
        device2_serial TEXT,                -- Device 2 serial (optional)
        device2_model TEXT,                 -- Device 2 model
        device2_android TEXT,               -- Device 2 Android version
        total_files INTEGER DEFAULT 0,      -- Total evidence files count
        flagged_files INTEGER DEFAULT 0,    -- HIGH risk files count
        total_size_mb REAL DEFAULT 0,       -- Total evidence size in MB
        threat_level TEXT DEFAULT 'LOW',    -- Overall: 'HIGH', 'MED', or 'LOW'
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        updated_at TEXT DEFAULT CURRENT_TIMESTAMP)''')
    _ensure_cases_columns(conn)
    
    # ── EVIDENCE TABLE ─────────────────────────────────────────
    # Each row = one extracted file from a device
    # Linked to a case via case_id, and a device via device_slot
    c.execute('''CREATE TABLE IF NOT EXISTS evidence (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        case_id TEXT NOT NULL,              -- FK → cases.case_id
        device_slot TEXT DEFAULT 'device1', -- 'device1' or 'device2' (separates evidence per device)
        filename TEXT NOT NULL,             -- Original filename
        file_type TEXT,                     -- 'image', 'chat', 'sms', 'call_log', etc.
        file_size TEXT,                     -- Human-readable size (e.g., '2.4MB')
        sha256 TEXT,                        -- SHA-256 hash for chain of custody
        risk_level TEXT DEFAULT 'LOW',      -- AI-determined: 'HIGH', 'MED', 'LOW'
        ai_result TEXT,                     -- AI analysis description
        gps_lat REAL,                       -- GPS latitude from EXIF (if image)
        gps_lng REAL,                       -- GPS longitude from EXIF (if image)
        exif_date TEXT,                     -- Camera date from EXIF metadata
        file_path TEXT,                     -- Local filesystem path to the file
        extracted_at TEXT DEFAULT CURRENT_TIMESTAMP)''')
    _ensure_evidence_columns(conn)
    
    # ── ACTIVITY LOGS TABLE ────────────────────────────────────
    # Audit trail — records every important action for accountability
    c.execute('''CREATE TABLE IF NOT EXISTS activity_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,            -- FK → users.id (NULL for system events)
        action TEXT NOT NULL,       -- e.g., 'LOGIN', 'CASE_CREATED', 'EVIDENCE_SAVED'
        detail TEXT,                -- Additional context
        ip_address TEXT,            -- Client IP address
        logged_at TEXT DEFAULT CURRENT_TIMESTAMP)''')
    
    # ── OTP STORE TABLE ────────────────────────────────────────
    # Temporary storage for password reset OTP codes
    # OTPs expire after 10 minutes and allow max 3 verification attempts
    c.execute('''CREATE TABLE IF NOT EXISTS otp_store (
        email TEXT PRIMARY KEY,     -- User's email
        otp TEXT,                   -- 6-digit OTP code
        expires_at TEXT,            -- Expiration timestamp
        attempts INTEGER DEFAULT 0)''')  # Verification attempts count
    
    # ── RATE LIMITS TABLE ──────────────────────────────────────
    # Tracks failed login attempts per IP to prevent brute-force attacks
    # After 5 failures, the IP is blocked for 15 minutes
    c.execute('''CREATE TABLE IF NOT EXISTS rate_limits (
        ip TEXT PRIMARY KEY,        -- Client IP address
        attempts INTEGER DEFAULT 0, -- Failed attempts count
        blocked_until TEXT,         -- Timestamp when block expires
        last_attempt TEXT)''')  # Last attempt timestamp
    
    # ── SEED DEFAULT USERS ─────────────────────────────────────
    # Only creates if they don't already exist (prevents duplicates)
    
    # Chief Investigator — manages and assigns investigators
    if not conn.execute("SELECT id FROM users WHERE role='chief'").fetchone():
        h = bcrypt.hashpw(b'Chief@FORAX2025', bcrypt.gensalt()).decode()
        c.execute("INSERT INTO users (name,email,password,role,badge_id,department) VALUES (?,?,?,?,?,?)",
                  ('Chief Investigator','chief@forax.gov',h,'chief','CHIEF-001','Cyber Crimes HQ'))
    
    conn.commit(); conn.close()
    print("[FORAX] DB ready  |  chief@forax.gov / Chief@FORAX2025")


# ══════════════════════════════════════════════════════════════════
# USER MANAGEMENT FUNCTIONS
# ══════════════════════════════════════════════════════════════════

def verify_user(email, password):
    """
    Verify user credentials for login.
    
    Checks:
      1. Email exists in database
      2. Account is active (is_active=1)
      3. Password matches bcrypt hash
    
    Args:
        email (str): User's email address
        password (str): Plaintext password to verify
    
    Returns:
        tuple: (user_dict, message) on success
               (None, error_message) on failure
    """
    conn = get_db()
    u = conn.execute('SELECT * FROM users WHERE email=? AND is_active=1',(email,)).fetchone()
    conn.close()
    if not u: return None, "Invalid email or password"
    # bcrypt.checkpw compares plaintext password against stored hash
    if bcrypt.checkpw(password.encode(), u['password'].encode()): return dict(u), "OK"
    return None, "Invalid email or password"

def update_last_login(uid):
    """Update the last_login timestamp for a user after successful login."""
    conn=get_db()
    conn.execute('UPDATE users SET last_login=? WHERE id=?',
                 (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), uid))
    conn.commit(); conn.close()

def get_user_by_id(uid):
    """Fetch a single user by their database ID. Returns dict or None."""
    conn=get_db()
    u=conn.execute('SELECT * FROM users WHERE id=?',(uid,)).fetchone()
    conn.close()
    return dict(u) if u else None

def get_all_users():
    """Fetch all users (for CMS/admin views). Returns list of dicts."""
    conn=get_db()
    rows=conn.execute('SELECT * FROM users ORDER BY created_at DESC').fetchall()
    conn.close()
    return [dict(r) for r in rows]

def get_all_investigators():
    """Fetch only investigator-role users (for chief panel). Returns list of dicts."""
    conn=get_db()
    rows=conn.execute("SELECT * FROM users WHERE role='investigator' ORDER BY created_at DESC").fetchall()
    conn.close()
    return [dict(r) for r in rows]

def next_badge_id():
    """
    Generate the next badge ID in sequence.
    
    Badge IDs follow the pattern INV-001, INV-002, etc.
    Looks at the most recent badge_id and increments by 1.
    
    Returns:
        str: Next badge ID (e.g., 'INV-003')
    """
    conn=get_db()
    row=conn.execute("SELECT badge_id FROM users WHERE badge_id LIKE 'INV-%' ORDER BY id DESC LIMIT 1").fetchone()
    conn.close()
    if not row: return 'INV-001'  # First investigator
    try: return f"INV-{int(row['badge_id'].split('-')[1])+1:03d}"
    except: return 'INV-001'

def create_investigator(name, email, password, badge_id, department, assigned_by):
    """
    Create a new investigator account.
    
    Called by the Chief Investigator from the chief panel.
    Password is hashed with bcrypt before storage.
    
    Args:
        name (str): Full name
        email (str): Login email
        password (str): Plaintext password (will be hashed)
        badge_id (str): Badge ID (e.g., INV-003)
        department (str): Department name
        assigned_by (int): User ID of the chief who created this account
    
    Returns:
        tuple: (True, "Created") on success
               (False, error_message) on failure
    """
    conn=get_db()
    try:
        _ensure_users_columns(conn)
        h=bcrypt.hashpw(password.encode(),bcrypt.gensalt()).decode()
        conn.execute("INSERT INTO users (name,email,password,role,badge_id,department,assigned_by) VALUES (?,?,?,?,?,?,?)",
                     (name,email,h,'investigator',badge_id,department,assigned_by))
        conn.commit(); return True,"Created"
    except sqlite3.IntegrityError as e:
        # Handle unique constraint violations
        if 'email' in str(e): return False,"Email already registered"
        if 'badge_id' in str(e): return False,"Badge ID conflict"
        return False,str(e)
    finally: conn.close()

def toggle_investigator(uid, active):
    """Enable or disable an investigator account. active=1 to enable, 0 to disable."""
    conn=get_db()
    conn.execute('UPDATE users SET is_active=? WHERE id=?',(active,uid))
    conn.commit(); conn.close()


# ══════════════════════════════════════════════════════════════════
# CASE MANAGEMENT FUNCTIONS
# ══════════════════════════════════════════════════════════════════

def create_case(case_id, title, description, investigator_id):
    """
    Create a new forensic investigation case.
    
    Case IDs are auto-generated as FORAX-YYYYMMDDHHmmSS by app.py.
    Each case is linked to the investigator who created it.
    
    Returns:
        bool: True if created successfully, False on error
    """
    conn=get_db()
    try:
        _ensure_cases_columns(conn)
        conn.execute("INSERT INTO cases (case_id,title,description,investigator_id) VALUES (?,?,?,?)",
                     (case_id,title,description,investigator_id))
        conn.commit(); return True
    except: return False
    finally: conn.close()

def get_cases(investigator_id=None):
    """
    Fetch cases with investigator name joined.
    
    If investigator_id is provided, only returns THEIR cases.
    If None, returns ALL cases (used by chief/authority views).
    
    Returns:
        list[dict]: List of case records with 'investigator_name' field
    """
    conn=get_db()
    _ensure_cases_columns(conn)
    if investigator_id:
        rows=conn.execute("SELECT c.*,u.name as investigator_name FROM cases c JOIN users u ON c.investigator_id=u.id WHERE c.investigator_id=? ORDER BY c.created_at DESC",(investigator_id,)).fetchall()
    else:
        rows=conn.execute("SELECT c.*,u.name as investigator_name FROM cases c JOIN users u ON c.investigator_id=u.id ORDER BY c.created_at DESC").fetchall()
    conn.close(); return [dict(r) for r in rows]

def get_case_data(case_id):
    """Fetch all details for a single case by its string case_id."""
    conn=get_db()
    _ensure_cases_columns(conn)
    row = conn.execute("SELECT c.*, u.name as investigator_name FROM cases c JOIN users u ON c.investigator_id=u.id WHERE c.case_id=?", (case_id,)).fetchone()
    conn.close()
    return dict(row) if row else None

def update_case_stats(case_id, total, flagged, size_mb, threat):
    """
    Update a case's aggregate statistics after evidence is saved or analyzed.
    
    Called after extraction save or AI analysis completion.
    """
    conn=get_db()
    _ensure_cases_columns(conn)
    conn.execute("UPDATE cases SET total_files=?,flagged_files=?,total_size_mb=?,threat_level=?,updated_at=? WHERE case_id=?",
                 (total,flagged,size_mb,threat,datetime.now().strftime('%Y-%m-%d %H:%M:%S'),case_id))
    conn.commit(); conn.close()

def update_case_device(case_id, slot, serial, model, android):
    """
    Store device info (serial, model, android version) in the case record.
    
    Each case supports 2 devices: device1 and device2.
    The slot parameter must be '1' or '2' to prevent SQL injection
    (we use it in a column name, so we validate it strictly).
    """
    if str(slot) not in ('1','2'): return  # Strict validation — prevent SQL injection
    col = str(slot)
    conn=get_db()
    try:
        _ensure_cases_columns(conn)
        conn.execute(f"UPDATE cases SET device{col}_serial=?,device{col}_model=?,device{col}_android=? WHERE case_id=?",
                     (serial,model,android,case_id))
        conn.commit()
    except sqlite3.OperationalError as e:
        # One more migration pass for stale runtime DB handles, then retry once.
        if 'no such column' in str(e).lower():
            _ensure_cases_columns(conn)
            conn.execute(f"UPDATE cases SET device{col}_serial=?,device{col}_model=?,device{col}_android=? WHERE case_id=?",
                         (serial,model,android,case_id))
            conn.commit()
        else:
            raise
    finally:
        conn.close()


def delete_case(case_id):
    """
    Permanently delete a case and all its associated evidence from the database.
    Returns a list of file paths for physical cleanup.
    """
    conn = get_db()
    try:
        # 1. Get all file paths for physical deletion
        rows = conn.execute("SELECT file_path FROM evidence WHERE case_id=?", (case_id,)).fetchall()
        paths = [r['file_path'] for r in rows if r['file_path']]
        
        # 2. Delete evidence
        conn.execute("DELETE FROM evidence WHERE case_id=?", (case_id,))
        
        # 3. Delete case
        conn.execute("DELETE FROM cases WHERE case_id=?", (case_id,))
        
        conn.commit()
        return paths
    finally:
        conn.close()


# ══════════════════════════════════════════════════════════════════
# EVIDENCE MANAGEMENT FUNCTIONS
# ══════════════════════════════════════════════════════════════════

def add_evidence(case_id, device_slot, filename, file_type, file_size, sha256, risk_level, ai_result, gps_lat=None, gps_lng=None, exif_date=None, file_path=''):
    """
    Insert a single evidence record into the database.
    
    Called after extraction is complete and user clicks "Save to Case".
    Each file gets its own row with SHA-256 hash for chain of custody.
    
    Args:
        case_id (str): Case ID this evidence belongs to
        device_slot (str): 'device1' or 'device2' — which phone it came from
        filename (str): Original filename from the device
        file_type (str): Category — 'image', 'chat', 'sms', 'call_log', etc.
        sha256 (str): SHA-256 hash computed at extraction time
        risk_level (str): AI-determined risk — 'HIGH', 'MED', or 'LOW'
        ai_result (str): AI analysis description text
        gps_lat/gps_lng (float): GPS coordinates from EXIF metadata (images only)
        exif_date (str): Camera date from EXIF metadata
        file_path (str): Local path where the file is stored
    """
    conn=get_db()
    # Check for existing duplicate (Filename + SHA256 + CaseID)
    try:
        existing = conn.execute("SELECT id FROM evidence WHERE case_id=? AND filename=? AND sha256=?", 
                               (case_id, filename, sha256)).fetchone()
        if existing:
            conn.close()
            return # Skip duplicate
    except: pass

    # Compress potentially large AI analysis strings
    compressed_ai_result = compress_text(ai_result)
    try:
        _ensure_evidence_columns(conn)
        conn.execute("INSERT INTO evidence (case_id,device_slot,filename,file_type,file_size,sha256,risk_level,ai_result,gps_lat,gps_lng,exif_date,file_path) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (case_id,device_slot,filename,file_type,file_size,sha256,risk_level,compressed_ai_result,gps_lat,gps_lng,exif_date,file_path))
        conn.commit()
    except sqlite3.OperationalError as e:
        # Retry once after migration for legacy DBs loaded with older schema.
        if 'no such column' in str(e).lower():
            _ensure_evidence_columns(conn)
            conn.execute("INSERT INTO evidence (case_id,device_slot,filename,file_type,file_size,sha256,risk_level,ai_result,gps_lat,gps_lng,exif_date,file_path) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (case_id,device_slot,filename,file_type,file_size,sha256,risk_level,compressed_ai_result,gps_lat,gps_lng,exif_date,file_path))
            conn.commit()
        else:
            raise
    finally:
        conn.close()

def get_evidence(case_id, device_slot=None):
    """
    Fetch evidence records for a case, optionally filtered by device slot.
    
    When device_slot is provided, only returns evidence from that device,
    enabling per-device evidence separation in the dashboard.
    
    Returns:
        list[dict]: List of evidence records
    """
    conn=get_db()
    _ensure_evidence_columns(conn)
    if device_slot:
        rows=conn.execute("SELECT * FROM evidence WHERE case_id=? AND device_slot=? ORDER BY extracted_at DESC",
                          (case_id,device_slot)).fetchall()
    else:
        rows=conn.execute("SELECT * FROM evidence WHERE case_id=? ORDER BY device_slot,extracted_at DESC",
                          (case_id,)).fetchall()
    conn.close()
    
    # Decompress AI results for usage
    results = []
    for r in rows:
        d = dict(r)
        d['ai_result'] = decompress_text(d.get('ai_result'))
        results.append(d)
    return results

def delete_device_evidence(case_id, device_slot):
    """
    Deletes evidence entries for a specific device in a case.
    Returns a list of local file paths so the caller can delete the files from disk.
    """
    conn = get_db()
    try:
        _ensure_evidence_columns(conn)
        rows = conn.execute("SELECT file_path FROM evidence WHERE case_id=? AND device_slot=?", (case_id, device_slot)).fetchall()
        paths = [row['file_path'] for row in rows if row['file_path']]
        conn.execute("DELETE FROM evidence WHERE case_id=? AND device_slot=?", (case_id, device_slot))
        conn.commit()
        return paths
    finally:
        conn.close()

# ══════════════════════════════════════════════════════════════════
# ACTIVITY LOGGING — Audit Trail
# ══════════════════════════════════════════════════════════════════

def log_activity(user_id, action, detail='', ip=''):
    """
    Record an activity in the audit log.
    
    Examples of logged actions:
      LOGIN, LOGOUT, CASE_CREATED, EVIDENCE_SAVED,
      AI_ANALYSIS, REPORT_GENERATED, INVESTIGATOR_CREATED, etc.
    
    Used for accountability and forensic audit trail.
    """
    conn=get_db()
    conn.execute("INSERT INTO activity_logs (user_id,action,detail,ip_address) VALUES (?,?,?,?)",
                 (user_id,action,detail,ip))
    conn.commit(); conn.close()

def get_activity_logs(limit=100):
    """Fetch recent activity logs with user names joined. For chief/CMS views."""
    conn=get_db()
    rows=conn.execute("SELECT a.*,u.name as user_name,u.role FROM activity_logs a LEFT JOIN users u ON a.user_id=u.id ORDER BY a.logged_at DESC LIMIT ?",(limit,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]

def get_system_stats():
    """
    Fetch aggregate system statistics for the CMS dashboard.
    
    Returns dict with:
      total_investigators, total_cases, active_cases,
      total_evidence, flagged_evidence
    """
    conn=get_db()
    _ensure_cases_columns(conn)
    _ensure_evidence_columns(conn)
    s={
        'total_investigators':conn.execute("SELECT COUNT(*) FROM users WHERE role='investigator'").fetchone()[0],
        'total_cases':conn.execute("SELECT COUNT(*) FROM cases").fetchone()[0],
        'active_cases':conn.execute("SELECT COUNT(*) FROM cases WHERE status='active'").fetchone()[0],
        'total_evidence':conn.execute("SELECT COUNT(*) FROM evidence").fetchone()[0],
        'flagged_evidence':conn.execute("SELECT COUNT(*) FROM evidence WHERE risk_level='HIGH'").fetchone()[0],
    }
    conn.close(); return s


# ══════════════════════════════════════════════════════════════════
# OTP (One-Time Password) Functions — For Password Reset
# ══════════════════════════════════════════════════════════════════

def store_otp(email, otp, expires_at):
    """Store a new OTP for password reset. Replaces any existing OTP for this email."""
    otp = ''.join(ch for ch in str(otp) if ch.isdigit())[:6].zfill(6)
    conn=get_db()
    conn.execute("INSERT OR REPLACE INTO otp_store (email,otp,expires_at,attempts) VALUES (?,?,?,0)",
                 (email,otp,expires_at))
    conn.commit(); conn.close()

def get_otp(email):
    """Retrieve stored OTP data for an email. Returns dict or None."""
    conn=get_db()
    row=conn.execute("SELECT * FROM otp_store WHERE email=?",(email,)).fetchone()
    conn.close()
    return dict(row) if row else None

def increment_otp_attempts(email):
    """Increment the verification attempt counter for an OTP (max 3 allowed)."""
    conn=get_db()
    conn.execute("UPDATE otp_store SET attempts=attempts+1 WHERE email=?",(email,))
    conn.commit(); conn.close()

def delete_otp(email):
    """Delete an OTP record after successful verification or expiry."""
    conn=get_db()
    conn.execute("DELETE FROM otp_store WHERE email=?",(email,))
    conn.commit(); conn.close()


# ══════════════════════════════════════════════════════════════════
# RATE LIMITING Functions — Brute-Force Protection
# ══════════════════════════════════════════════════════════════════

def get_rate_limit(ip):
    """Get the current rate limit record for an IP address. Returns dict or None."""
    conn=get_db()
    row=conn.execute("SELECT * FROM rate_limits WHERE ip=?",(ip,)).fetchone()
    conn.close()
    return dict(row) if row else None

def set_rate_limit(ip, attempts, blocked_until=None):
    """
    Update or create a rate limit record for an IP.
    
    After 5 failed attempts, blocked_until is set to 15 minutes from now.
    Uses INSERT OR REPLACE to upsert (update if exists, insert if new).
    """
    conn=get_db()
    conn.execute("INSERT OR REPLACE INTO rate_limits (ip,attempts,blocked_until,last_attempt) VALUES (?,?,?,?)",
        (ip,attempts,blocked_until,datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    conn.commit(); conn.close()

def clear_rate_limit(ip):
    """Clear rate limiting after a successful login. Removes the IP from the table."""
    conn=get_db()
    conn.execute("DELETE FROM rate_limits WHERE ip=?",(ip,))
    conn.commit(); conn.close()


# ── Standalone execution — initialize the database ─────────────
if __name__ == '__main__': init_db()
