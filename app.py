"""
FORAX — Main Application (Flask Backend)
Routes: login, dashboard, chief panel, extraction API, AI analysis, reports.
Run via desktop.py (native window) or directly for web mode.
"""
from flask import (Flask, render_template, request, jsonify,
                   session, redirect, url_for, Response, send_file)
from functools import wraps
from datetime import datetime, timedelta
import os, json, threading, bcrypt
from werkzeug.utils import secure_filename

from modules.database import (
    init_db, verify_user, update_last_login, get_user_by_id,
    get_all_users, get_all_investigators, create_investigator,
    toggle_investigator, next_badge_id, log_activity, get_activity_logs,
    create_case, get_cases, get_case_data, update_case_stats, update_case_device,
    add_evidence, get_evidence, get_system_stats, delete_case,
    store_otp, get_otp, increment_otp_attempts, delete_otp,
    get_db
)
from modules.ai_image import classify_image, get_image_summary
from modules.ai_nlp   import analyze_file, get_threat_summary
from modules.adb_extractor import (
    get_devices, get_device_info, full_extraction_realtime,
    get_state, reset_state, get_event_queue,
    stop_extraction, pause_extraction, resume_extraction, is_paused,
    get_uploads_size, cleanup_uploads, get_file_type, get_sha256
)
from modules.security import (
    is_ip_blocked, record_failed_attempt, clear_rate_limit,
    get_attempts_left, generate_otp, generate_math_captcha,
    email_otp, email_welcome, email_login_alert, email_lockout_alert
)
from modules.pdf_report import generate_pdf_report, generate_html_report

app = Flask(__name__)
app.secret_key = 'FORAX-SECRET-LGU-AHSAN-2025-BSDFCS'  # Session encryption key
UPLOADS_DIR = os.path.join(os.path.dirname(__file__), 'uploads')

# ── SESSION TIMEOUT (30 minutes) ──────────────────────────────
app.permanent_session_lifetime = timedelta(minutes=30)

@app.before_request
def refresh_session():
    session.permanent = True
    session.modified  = True

# ── DECORATORS ─────────────────────────────────────────────────
def login_required(f):  # Redirects unauthenticated users to login
    @wraps(f)
    def d(*a, **kw):
        if 'user_id' not in session: return redirect(url_for('login'))
        return f(*a, **kw)
    return d

def role_required(*roles):  # Restricts routes to specific roles (chief, investigator)
    def dec(f):
        @wraps(f)
        def d(*a, **kw):
            if 'user_id' not in session: return redirect(url_for('login'))
            if session.get('role') not in roles: return redirect(url_for('dashboard'))
            return f(*a, **kw)
        return d
    return dec

# ── ROUTES ─────────────────────────────────────────────────────
@app.route('/')  # Root — redirects to appropriate page based on role
def index():
    if 'user_id' not in session: return redirect(url_for('login'))
    r = session.get('role')
    if r == 'chief':     return redirect(url_for('chief_panel'))
    return redirect(url_for('dashboard'))

@app.route('/login', methods=['GET','POST'])
def login():
    if request.method == 'GET': return render_template('login.html')
    ip   = request.remote_addr
    data = request.get_json() or {}
    email    = data.get('email','').strip().lower()
    password = data.get('password','')

    blocked, mins = is_ip_blocked(ip)
    if blocked:
        return jsonify({'success':False,'blocked':True,'minutes':mins,
                        'error':f'Too many attempts. Try again in {mins} min.'})

    user, msg = verify_user(email, password)
    if not user:
        was_blocked, _ = record_failed_attempt(ip)
        left = get_attempts_left(ip)
        log_activity(None,'LOGIN_FAILED',f'{email} from {ip}',ip)
        if was_blocked:
            # Send lockout email
            try:
                conn = get_db()
                u = conn.execute('SELECT * FROM users WHERE email=?',(email,)).fetchone()
                conn.close()
                if u: threading.Thread(target=email_lockout_alert, args=(email,u['name'],ip), daemon=True).start()
            except: pass
            return jsonify({'success':False,'blocked':True,'minutes':15,
                            'error':'Account locked for 15 minutes.'})
        return jsonify({'success':False,'error':msg,'attempts_left':left})

    clear_rate_limit(ip)
    session['user_id']   = user['id']
    session['user_name'] = user['name']
    session['role']      = user['role']
    session['email']     = user['email']
    update_last_login(user['id'])
    log_activity(user['id'],'LOGIN',f'from {ip}',ip)

    # Login alert email (non-blocking)
    try:
        ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        threading.Thread(target=email_login_alert,
                         args=(email, user['name'], ip, ts), daemon=True).start()
    except: pass

    routes = {'chief': url_for('chief_panel')}
    return jsonify({'success':True,'role':user['role'],
                    'redirect': routes.get(user['role'], url_for('dashboard'))})

@app.route('/logout')
def logout():
    log_activity(session.get('user_id'),'LOGOUT','')
    session.clear()
    return redirect(url_for('login'))

# ── CAPTCHA ────────────────────────────────────────────────────
@app.route('/api/captcha/math')
def captcha_math():
    q, a = generate_math_captcha()
    session['math_answer'] = a
    return jsonify({'question': q, 'answer': a})

# ── FORGOT PASSWORD / OTP ──────────────────────────────────────
@app.route('/api/forgot-password', methods=['POST'])
def forgot_password():
    data  = request.get_json() or {}
    email = data.get('email','').strip().lower()
    conn  = get_db()
    u     = conn.execute('SELECT * FROM users WHERE email=?',(email,)).fetchone()
    conn.close()
    if not u: return jsonify({'success':False,'error':'Email not found'})
    otp     = generate_otp()
    expires = (datetime.now().timestamp() + 600)  # 10 min
    expires_str = datetime.fromtimestamp(expires).strftime('%Y-%m-%d %H:%M:%S')
    store_otp(email, otp, expires_str)
    ok, msg = email_otp(email, otp, u['name'])
    if not ok:
        return jsonify({'success':False,'error':f'Email failed: {msg}'})
    log_activity(u['id'],'OTP_SENT',f'Reset OTP sent to {email}',request.remote_addr)
    return jsonify({'success':True,'message':'OTP sent to your email'})

@app.route('/api/verify-otp', methods=['POST'])
def verify_otp():
    data  = request.get_json() or {}
    email = data.get('email','').strip().lower()
    otp   = ''.join(ch for ch in str(data.get('otp','')) if ch.isdigit())[:6]
    row   = get_otp(email)
    if not row: return jsonify({'success':False,'error':'OTP not found. Request again.'})
    if len(otp) != 6:
        return jsonify({'success':False,'error':'Enter a valid 6-digit OTP'})
    if row['attempts'] >= 3:
        delete_otp(email)
        return jsonify({'success':False,'error':'Too many attempts. Request a new OTP.'})
    expires_dt = datetime.strptime(row['expires_at'],'%Y-%m-%d %H:%M:%S')
    if datetime.now() > expires_dt:
        delete_otp(email)
        return jsonify({'success':False,'error':'OTP expired. Request a new one.'})
    stored_otp = ''.join(ch for ch in str(row['otp']) if ch.isdigit())[:6]
    otp_ok = (stored_otp == otp)

    # Legacy compatibility: older sqlite schemas/config may coerce OTP to numeric
    # and drop leading zeros (e.g., 012345 -> 12345). Accept numeric-equivalent match.
    if not otp_ok and stored_otp.isdigit() and otp.isdigit():
        try:
            otp_ok = int(stored_otp) == int(otp)
        except ValueError:
            otp_ok = False

    if not otp_ok:
        increment_otp_attempts(email)
        left = 3 - row['attempts'] - 1
        return jsonify({'success':False,'error':f'Wrong OTP. {left} attempts left.'})
    delete_otp(email)
    session['reset_email'] = email  # mark as verified
    return jsonify({'success':True})

@app.route('/api/reset-password', methods=['POST'])
def reset_password():
    data     = request.get_json() or {}
    email    = data.get('email','').strip().lower()
    new_pw   = data.get('new_password','')
    if len(new_pw) < 8: return jsonify({'success':False,'error':'Password must be 8+ characters'})
    if session.get('reset_email') != email:
        return jsonify({'success':False,'error':'OTP not verified'})
    hashed = bcrypt.hashpw(new_pw.encode(), bcrypt.gensalt()).decode()
    conn   = get_db()
    conn.execute('UPDATE users SET password=? WHERE email=?',(hashed,email))
    conn.commit()
    u = conn.execute('SELECT id FROM users WHERE email=?',(email,)).fetchone()
    conn.close()
    session.pop('reset_email', None)
    if u: log_activity(u['id'],'PASSWORD_RESET','via OTP',request.remote_addr)
    return jsonify({'success':True})

# ── CHIEF ──────────────────────────────────────────────────────
@app.route('/chief')
@role_required('chief')
def chief_panel():
    user   = get_user_by_id(session['user_id'])
    invs   = get_all_investigators()
    logs   = get_activity_logs(60)
    stats  = get_system_stats()
    next_b = next_badge_id()
    
    conn = get_db()
    flagged = conn.execute("SELECT * FROM evidence WHERE risk_level='HIGH' ORDER BY extracted_at DESC").fetchall()
    global_cases = conn.execute('''
        SELECT cases.*, users.name as investigator_name 
        FROM cases 
        JOIN users ON cases.investigator_id = users.id 
        ORDER BY cases.created_at DESC
    ''').fetchall()
    conn.close()
    
    return render_template('chief.html', user=user, investigators=invs,
                           logs=logs, stats=stats, next_badge=next_b,
                           flagged=flagged, cases=global_cases,
                           now=datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

@app.route('/api/chief/create-investigator', methods=['POST'])
@role_required('chief')
def api_create_investigator():
    d = request.get_json() or {}
    name   = d.get('name','').strip()
    email  = d.get('email','').strip().lower()
    pw     = d.get('password','').strip()
    badge  = d.get('badge_id','').strip()
    dept   = d.get('department','').strip()
    if not all([name,email,pw,badge]):
        return jsonify({'success':False,'error':'All fields required'})
    if len(pw) < 8:
        return jsonify({'success':False,'error':'Password must be 8+ characters'})
    ok, msg = create_investigator(name,email,pw,badge,dept,session['user_id'])
    if ok:
        log_activity(session['user_id'],'INVESTIGATOR_CREATED',f'{email} | {badge}',request.remote_addr)
        threading.Thread(target=email_welcome, args=(email,name,badge), daemon=True).start()
    return jsonify({'success':ok,'error':msg if not ok else None})

@app.route('/api/chief/toggle-investigator', methods=['POST'])
@role_required('chief')
def api_toggle_investigator():
    d = request.get_json() or {}
    uid, active = d.get('user_id'), d.get('active',1)
    toggle_investigator(uid, active)
    log_activity(session['user_id'],f"INV_{'ENABLED' if active else 'DISABLED'}",f'uid={uid}',request.remote_addr)
    return jsonify({'success':True})

@app.route('/api/chief/next-badge')
@role_required('chief')
def api_next_badge():
    return jsonify({'badge_id': next_badge_id()})

# ── DASHBOARD ──────────────────────────────────────────────────
@app.route('/dashboard')
@role_required('investigator','chief')
def dashboard():
    user  = get_user_by_id(session['user_id'])
    cases = get_cases(session['user_id'])
    return render_template('dashboard.html', user=user, cases=cases,
                           now=datetime.now().strftime('%H:%M:%S'))

@app.route('/api/cases/create', methods=['POST'])
@login_required
def api_create_case():
    d = request.get_json() or {}
    case_id = 'FORAX-' + datetime.now().strftime('%Y%m%d%H%M%S')
    ok = create_case(case_id, d.get('title','Untitled'), d.get('description',''), session['user_id'])
    if ok:
        log_activity(session['user_id'],'CASE_CREATED',case_id,request.remote_addr)
        return jsonify({'success':True,'case_id':case_id})
    return jsonify({'success':False,'error':'Failed to create case'})

@app.route('/api/cases')
@login_required
def api_get_cases():
    return jsonify(get_cases(session['user_id']))

@app.route('/api/cases/delete', methods=['POST'])
@login_required
def api_delete_case():
    import shutil
    d       = request.get_json() or {}
    case_id = d.get('case_id')
    user_id = session.get('user_id')
    role    = session.get('role')

    if not case_id:
        return jsonify({'success':False, 'error':'Missing case_id'}), 400

    # Verification: Only Chief can delete anything, Investigator only their own
    case_data = get_case_data(case_id)
    if not case_data:
        return jsonify({'success':False, 'error':'Case not found'}), 404

    if role != 'chief' and case_data['investigator_id'] != user_id:
        return jsonify({'success':False, 'error':'Permission denied. You can only delete your own cases.'}), 403

    # Delete from Database
    paths = delete_case(case_id)
    
    # Delete from Filesystem (uploads folder)
    case_dir = os.path.join(UPLOADS_DIR, secure_filename(case_id))
    if os.path.exists(case_dir):
        try:
            shutil.rmtree(case_dir)
        except Exception as e:
            print(f"Error deleting case directory: {e}")

    # Also clean up individual paths if they were stored elsewhere
    for p in paths:
        if p and os.path.exists(p):
            try:
                # If it's a file, remove it. If it's a dir, rmtree.
                if os.path.isfile(p): os.remove(p)
            except: pass

    log_activity(user_id, 'CASE_DELETED', case_id, request.remote_addr)
    return jsonify({'success':True, 'message':f'Case {case_id} and all related data deleted.'})

# ── DEVICE ─────────────────────────────────────────────────────
@app.route('/api/devices/check')
@login_required
def api_check_devices():
    devices, status = get_devices()
    if status == 'adb_missing':
        return jsonify({'connected':False,'error':'ADB not found. Install Android SDK Platform Tools and add to PATH.'})
    if not devices:
        return jsonify({'connected':False,'error':'No device found. Enable USB Debugging on your phone.'})
    result = []
    for serial in devices[:2]:
        info = get_device_info(serial)
        result.append({'serial':serial,'info':info})
    return jsonify({'connected':True,'devices':result,'count':len(result)})

# ── EXTRACTION ─────────────────────────────────────────────────
@app.route('/extraction')  # Standalone extraction page (kept for backward compat)
@login_required
def extraction_page():
    return render_template('extraction.html')

@app.route('/api/extract/start-realtime', methods=['POST'])  # Starts extraction in background thread
@login_required
def api_start_realtime():
    d       = request.get_json() or {}
    serial  = d.get('serial')
    slot    = d.get('slot','device1')
    case_id = d.get('case_id','unknown')
    if not serial: return jsonify({'error':'Serial required'}),400
    # update case device info
    info = get_device_info(serial)
    slot_num = slot[-1]
    update_case_device(case_id, slot_num, serial,
                       f"{info.get('brand','')} {info.get('model','')}",
                       info.get('android',''))
    log_activity(session['user_id'],'EXTRACTION_STARTED',
                 f'{slot} serial={serial} case={case_id}',request.remote_addr)
    reset_state(slot)
    full_extraction_realtime(serial, slot, case_id)
    return jsonify({'started':True,'slot':slot,'stream_url':f'/api/extract/stream/{slot}'})

@app.route('/api/extract/stream/<slot>')  # SSE endpoint — streams live extraction events to UI
@login_required
def api_extract_stream(slot):
    if slot not in ('device1','device2'):
        return Response("data: {\"type\":\"error\"}\n\n", mimetype='text/event-stream')
    def generate():
        q = get_event_queue(slot)
        while True:
            try:
                ev = q.get(timeout=25)
                yield f"data: {json.dumps(ev)}\n\n"
                if ev.get('type') == 'status':
                    if ev.get('data',{}).get('status') in ('complete','error','stopped'):
                        break
            except:
                yield "data: {\"type\":\"ping\"}\n\n"
    return Response(generate(), mimetype='text/event-stream',
                    headers={'Cache-Control':'no-cache','X-Accel-Buffering':'no','Connection':'keep-alive'})

@app.route('/api/extract/status/<slot>')
@login_required
def api_extract_status(slot):
    return jsonify(get_state(slot))

@app.route('/api/extract/reset/<slot>', methods=['POST'])
@login_required
def api_extract_reset(slot):
    reset_state(slot)
    return jsonify({'reset':True})

@app.route('/api/extract/stop/<slot>', methods=['POST'])
@login_required
def api_extract_stop(slot):
    """Signal the running extraction thread to stop after the current step."""
    if slot not in ('device1','device2'):
        return jsonify({'error':'Invalid slot'}), 400
    stop_extraction(slot)
    log_activity(session['user_id'],'EXTRACTION_STOPPED',
                 f'{slot}',request.remote_addr)
    return jsonify({'stopped':True,'slot':slot})

@app.route('/api/extract/pause/<slot>', methods=['POST'])
@login_required
def api_extract_pause(slot):
    """Pause or resume the extraction thread between steps."""
    if slot not in ('device1','device2'):
        return jsonify({'error':'Invalid slot'}), 400
    action = request.get_json(force=True).get('action','pause')  # 'pause' | 'resume'
    if action == 'resume':
        resume_extraction(slot)
        log_activity(session['user_id'],'EXTRACTION_RESUMED',f'{slot}',request.remote_addr)
        return jsonify({'paused':False,'slot':slot})
    else:
        pause_extraction(slot)
        log_activity(session['user_id'],'EXTRACTION_PAUSED',f'{slot}',request.remote_addr)
        return jsonify({'paused':True,'slot':slot})

@app.route('/api/extract/pause-status/<slot>')
@login_required
def api_extract_pause_status(slot):
    """Query whether a slot is currently paused."""
    if slot not in ('device1','device2'):
        return jsonify({'error':'Invalid slot'}), 400
    return jsonify({'paused': is_paused(slot), 'slot': slot})

@app.route('/api/extract/save', methods=['POST'])
@login_required
def api_save_evidence():
    d       = request.get_json() or {}
    case_id = d.get('case_id')
    slot    = d.get('slot','device1')
    files   = d.get('files',[])
    saved, flagged = 0, 0
    for f in files:
        risk = f.get('risk_level','LOW')
        if risk == 'HIGH': flagged += 1
        add_evidence(case_id, slot, f.get('filename',''),
                     f.get('type',''), str(round(f.get('size',0)/1024,1))+'KB',
                     f.get('sha256',''), risk, f.get('ai_result',''),
                     f.get('gps_lat'), f.get('gps_lng'),
                     f.get('exif_date'), f.get('path',''))
        saved += 1
    total_size = round(sum(f.get('size',0) for f in files)/(1024*1024), 2)
    threat     = 'HIGH' if flagged > 0 else 'LOW'
    update_case_stats(case_id, saved, flagged, total_size, threat)
    log_activity(session['user_id'],'EVIDENCE_SAVED',
                 f'{saved} files | {slot} | {case_id}',request.remote_addr)
    return jsonify({'success':True,'saved':saved})



# ── LOCAL EVIDENCE UPLOAD ───────────────────────────────────────
@app.route('/api/evidence/upload_local', methods=['POST'])
@login_required
def api_upload_local():
    case_id = request.form.get('case_id')
    slot = request.form.get('slot', 'LocalSystem')
    
    if not case_id or 'file' not in request.files:
        return jsonify({'error': 'Missing case_id or file'}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
        
    upload_dir = os.path.join(UPLOADS_DIR, secure_filename(case_id), 'LocalSystem')
    os.makedirs(upload_dir, exist_ok=True)
    
    filename = secure_filename(file.filename)
    filepath = os.path.join(upload_dir, filename)
    file.save(filepath)
    
    from modules.ai_nlp import score_text
    
    ftype = get_file_type(filepath)
    fsize = os.path.getsize(filepath)
    sha256 = get_sha256(filepath)
    
    # ── SUSPECT FILENAME DETECTION ────────────────────────────
    # Instantly flag files with suspicious names (keywords from forensic_intel.json)
    fn_lower = filename.lower()
    fn_score = score_text(fn_lower)
    
    risk_level = fn_score.get('risk_level', 'LOW')
    reasoning = "The file was scanned upon upload. No immediate risk indicators found in the filename."
    
    if risk_level != 'LOW':
        matched_kws = [m['keyword'] for m in fn_score.get('matched_keywords', [])]
        ai_result = f"LOCAL: Suspect filename — {', '.join(matched_kws[:3])}"
        reasoning = (f"Initial Screening Alert: This file was flagged upon upload because the filename '{filename}' "
                    f"contains markers matching {len(matched_kws)} indicator(s) in our threat database: {', '.join(matched_kws[:2])}. "
                    "Full forensic analysis of the file contents is required to confirm the risk level.")
        print(f"[FORAX] Suspect file detected during upload: {filename} (Risk: {risk_level})")
    else:
        ai_result = "Pending Analysis"
        
    # Save to database
    add_evidence(
        case_id=case_id, device_slot=slot, filename=filename,
        file_type=ftype, file_size=f"{round(fsize/1024, 1)}KB",
        sha256=sha256, risk_level=risk_level, ai_result=ai_result,
        gps_lat=None, gps_lng=None, exif_date=None, file_path=filepath
    )
    
    # Store forensic reasoning narrative in the DB
    conn = get_db()
    conn.execute("UPDATE evidence SET copilot_reasoning=? WHERE sha256=? AND case_id=?", (reasoning, sha256, case_id))
    conn.commit()
    conn.close()
    
    # Update Stats
    conn = get_db()
    remaining = conn.execute("SELECT * FROM evidence WHERE case_id=?",(case_id,)).fetchall()
    total = len(remaining)
    flagged = sum(1 for r in remaining if r['risk_level'] == 'HIGH')
    size_mb = sum(
        float(str(dict(r).get('file_size','') or '').replace('KB','').replace('MB','').replace('B','') or 0)
        for r in remaining
    ) / 1024
    threat = 'HIGH' if flagged > 0 else 'LOW'
    conn.close()
    update_case_stats(case_id, total, flagged, round(size_mb, 2), threat)
    
    log_activity(session['user_id'], 'LOCAL_FILE_UPLOAD', f'Uploaded {filename} to {case_id}', request.remote_addr)
    return jsonify({'success': True, 'filename': filename, 'risk': risk_level})


@app.route('/api/evidence/<case_id>')
@login_required
def api_get_evidence(case_id):
    slot = request.args.get('slot')
    return jsonify(get_evidence(case_id, slot))

# ── AI ANALYSIS ─────────────────────────────────────────────────
@app.route('/api/analyze', methods=['POST'])  # Runs Gemini AI analysis on saved evidence
@login_required
def api_analyze():
    d            = request.get_json() or {}
    case_id      = d.get('case_id')
    slot         = d.get('slot', 'device1')
    evidence_ids = d.get('evidence_ids') # Optional: list of specific IDs to analyze
    
    # If slot is 'all', we pass None to get_evidence to fetch everything for the case
    search_slot = None if slot == 'all' else slot
    files       = get_evidence(case_id, search_slot)
    
    # If specific IDs were requested, filter the files list
    if evidence_ids and isinstance(evidence_ids, list):
        files = [f for f in files if f['id'] in evidence_ids]
        
    risks   = {'HIGH': 0, 'MED': 0, 'LOW': 0}
    conn    = get_db()
    nlp_results = []

    for f in files:
        fname   = f.get('filename', '').lower()
        ftype   = f.get('file_type', '')
        fpath   = f.get('file_path', '')
        risk    = 'LOW'
        res     = 'Analyzed'
        res_reasoning = 'Standard forensic analysis complete.'
        gps_lat = f.get('gps_lat')
        gps_lng = f.get('gps_lng')
        exif_dt = f.get('exif_date')

        # ── CNN IMAGE & VIDEO ANALYSIS ──────────────────────────
        if ftype in ('image', 'video'):
            if fpath and os.path.exists(fpath):
                try:
                    img_result = classify_image(fpath)
                    risk    = img_result.get('risk_level', 'LOW')
                    res     = img_result.get('ai_result', 'CNN: Analyzed')
                    res_reasoning = img_result.get('copilot_reasoning', '')
                    gps_lat = img_result.get('gps_lat') or gps_lat
                    gps_lng = img_result.get('gps_lng') or gps_lng
                    exif_dt = img_result.get('exif_date') or exif_dt
                except Exception as e:
                    res  = f'CNN: Error — {str(e)[:60]}'
                    risk = 'LOW'
                    res_reasoning = ''
            else:
                # File not on disk
                res  = 'File missing'
                risk = 'LOW'

        # ── NLP TEXT ANALYSIS ───────────────────────────────────
        elif ftype in ('chat', 'sms', 'call_log', 'contacts', 'browser', 'document', 'text'):
            if fpath and os.path.exists(fpath):
                try:
                    nlp_result = analyze_file(fpath, ftype)
                    risk = nlp_result.get('risk_level', 'LOW')
                    res  = nlp_result.get('ai_result', 'NLP: Analyzed')
                    res_reasoning = nlp_result.get('copilot_reasoning', '')
                    nlp_results.append(nlp_result)
                except Exception as e:
                    res  = f'NLP: Error — {str(e)[:60]}'
                    risk = 'LOW'
                    res_reasoning = ''
            else:
                # File not on disk
                res  = 'NLP: File not on disk'
                risk = 'LOW'

        # ── DOWNLOADS / EXECUTABLE / APK ───────────────────────
        elif ftype == 'download':
            ext = fname.rsplit('.', 1)[-1] if '.' in fname else ''
            if ext in ('apk', 'exe', 'bat', 'sh', 'ps1', 'vbs', 'cmd'):
                risk = 'HIGH'
                res  = f'HIGH: Executable file (.{ext}) — manual inspection required'
                res_reasoning = f'Critical Alert: An executable file with extension .{ext} was identified. Such files can contain malicious payloads or unauthorized tools. Manual disassembly or sandboxing is recommended.'
            else:
                res  = 'Download analyzed'
                res_reasoning = 'The file was identified in a download directory. Standard verification recommended.'

        # ── APP DATA ────────────────────────────────────────────
        elif ftype == 'app_data':
            res = 'App inventory catalogued'

        # ── RECOVERED FILES ─────────────────────────────────────
        elif ftype == 'recovered':
            risk = 'MED'
            res  = 'Recovered deleted file — review recommended'
            res_reasoning = 'Artifact Recovery: This file was retrieved from unallocated space or a recycle bin. Its presence may indicate an attempt to destroy evidence.'

        # ── HIDDEN CONFIG FILES ─────────────────────────────────
        elif ftype == 'config':
            risk = 'MED'
            res  = 'Hidden config file analyzed'
            res_reasoning = 'Sensitive Configuration: File contains settings or metadata that may reveal system credentials, network topology, or user accounts.'
            if any(k in fname for k in ['wpa_supplicant','wifi','vpn','credentials','password','accounts','token','key','secret','auth']):
                risk = 'HIGH'
                res  = f'HIGH RISK: Sensitive config — {fname}'
                res_reasoning = f'Critical Discovery: This configuration file ({fname}) specifically matches patterns for credentials or authentication tokens. Highly sensitive material.'

        else:
            res = 'File analyzed'

        risks[risk] += 1
        conn.execute(
            'UPDATE evidence SET risk_level=?, ai_result=?, copilot_reasoning=?, gps_lat=?, gps_lng=?, exif_date=? WHERE id=?',
            (risk, res, res_reasoning, gps_lat, gps_lng, exif_dt, f['id'])
        )

    conn.commit()
    conn.close()

    # Aggregate NLP threat summary
    if nlp_results:
        summary = get_threat_summary(nlp_results)
        top_cats = ', '.join(c.replace('_',' ').title() for c,_ in summary['threat_cats'][:3])
        log_detail = f"{len(files)} files | {case_id} | {slot} | Top threats: {top_cats or 'None'}"
    else:
        log_detail = f"{len(files)} files | {case_id} | {slot}"

    total  = len(files)
    size_mb = sum(
        float(str(f.get('file_size','')).replace('KB','').replace('MB','').replace('B','') or 0)
        for f in files
    ) / 1024
    threat = 'HIGH' if risks['HIGH'] else 'MED' if risks['MED'] else 'LOW'
    update_case_stats(case_id, total, risks['HIGH'], round(size_mb, 2), threat)
    log_activity(session['user_id'], 'AI_ANALYSIS', log_detail, request.remote_addr)

    return jsonify({
        'success': True, 'total': len(files),
        'high': risks['HIGH'], 'med': risks['MED'], 'low': risks['LOW'],
        'threat': threat
    })

@app.route('/api/analyze/reset', methods=['POST'])
@login_required
def api_analyze_reset():
    """Reset AI analysis results to 'Pending Analysis' and 'LOW' risk."""
    d            = request.get_json() or {}
    case_id      = d.get('case_id')
    evidence_ids = d.get('evidence_ids') # Optional list of IDs
    
    if not case_id: return jsonify({'error':'case_id required'}), 400
    
    conn = get_db()
    if evidence_ids and isinstance(evidence_ids, list):
        # Reset specific files
        placeholders = ','.join(['?'] * len(evidence_ids))
        conn.execute(
            f"UPDATE evidence SET risk_level='LOW', ai_result='Pending Analysis' WHERE case_id=? AND id IN ({placeholders})",
            (case_id, *evidence_ids)
        )
    else:
        # Reset entire case
        conn.execute(
            "UPDATE evidence SET risk_level='LOW', ai_result='Pending Analysis' WHERE case_id=?",
            (case_id,)
        )
    
    conn.commit()
    # Recalculate stats
    remaining = conn.execute("SELECT risk_level, file_size FROM evidence WHERE case_id=?",(case_id,)).fetchall()
    total = len(remaining)
    flagged = sum(1 for r in remaining if r['risk_level'] == 'HIGH')
    size_mb = sum(
        float(str(dict(r).get('file_size','') or '').replace('KB','').replace('MB','').replace('B','') or 0)
        for r in remaining
    ) / 1024
    threat = 'HIGH' if flagged > 0 else 'LOW'
    update_case_stats(case_id, total, flagged, round(size_mb, 2), threat)
    conn.close()
    
    log_activity(session['user_id'], 'AI_RESET', f'Reset evidence for {case_id}', request.remote_addr)
    return jsonify({'success':True})

@app.route('/api/evidence/delete', methods=['POST'])
@login_required
def api_evidence_delete():
    """Securely delete an evidence record from the active database and log to the Chief."""
    d       = request.get_json() or {}
    ev_id   = d.get('evidence_id')
    case_id = d.get('case_id')
    
    if not ev_id or not case_id:
        return jsonify({'error':'Missing identifiers'}), 400
        
    conn = get_db()
    
    # Verify evidence exists and get filename
    ev = conn.execute("SELECT filename, file_size, risk_level FROM evidence WHERE id=? AND case_id=?", (ev_id, case_id)).fetchone()
    if not ev:
        conn.close()
        return jsonify({'error':'Evidence not found'}), 404
        
    filename = ev['filename']
    
    # Delete from active view
    conn.execute("DELETE FROM evidence WHERE id=? AND case_id=?", (ev_id, case_id))
    
    # Recalculate stats
    remaining = conn.execute("SELECT risk_level, file_size FROM evidence WHERE case_id=?",(case_id,)).fetchall()
    total = len(remaining)
    flagged = sum(1 for r in remaining if r['risk_level'] == 'HIGH')
    size_mb = sum(
        float(str(dict(r).get('file_size','') or '').replace('KB','').replace('MB','').replace('B','') or 0)
        for r in remaining
    ) / 1024
    threat = 'HIGH' if flagged > 0 else 'LOW'
    update_case_stats(case_id, total, flagged, round(size_mb, 2), threat)
    conn.commit()
    conn.close()
    
    # Log securely for Chief audit
    log_activity(session['user_id'], 'EVIDENCE_DELETED', f"{case_id} | {filename}", request.remote_addr)
    
    return jsonify({'success': True, 'filename': filename})

# ── REPORTS ─────────────────────────────────────────────────────
@app.route('/api/report/generate', methods=['POST'])
@login_required
def api_generate_report():
    d       = request.get_json() or {}
    case_id = d.get('case_id','UNKNOWN')
    evidence = get_evidence(case_id)
    report = {
        'case_id':      case_id,
        'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'investigator': session.get('user_name'),
        'institution':  'LGU — BSDFCS 2025',
        'total_evidence': len(evidence),
        'flagged': sum(1 for e in evidence if e.get('risk_level')=='HIGH'),
        'integrity': 'SHA-256 verified',
        'evidence':  evidence,
    }
    os.makedirs('reports', exist_ok=True)
    path = f"reports/{case_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}.json"
    with open(path,'w') as f: json.dump(report, f, indent=2)
    log_activity(session['user_id'],'REPORT_GENERATED',path,request.remote_addr)
    return jsonify({'success':True,'case_id':case_id,'path':path})

@app.route('/api/report/generate-advanced', methods=['POST'])
@login_required
def api_generate_advanced_report():
    """Generate an advanced report based on modal filtering and options."""
    d = request.get_json() or {}
    case_id = d.get('case_id', 'UNKNOWN')
    evidence = get_evidence(case_id)
    case_info = get_case_data(case_id)
    
    device_str = "Standard Forensic Extraction"
    if case_info:
        if case_info.get('device1_model'):
            device_str = f"{case_info['device1_model']} (SN: {case_info.get('device1_serial','N/A')})"
        elif case_info.get('device2_model'):
            device_str = f"{case_info['device2_model']} (SN: {case_info.get('device2_serial','N/A')})"
    
    # Extract options for the report generator
    options = {
        'case_id': case_id,
        'examiner': d.get('examiner'),
        'examiner_badge': d.get('examiner_badge'),
        'qualifications': d.get('qualifications'),
        'department': d.get('department'),
        'location': d.get('location'),
        'crime_type': d.get('crime_type'),
        'reviewer': d.get('reviewer'),
        'jurisdiction': d.get('jurisdiction'),
        'warrant_ref': d.get('warrant_ref'),
        'agency': d.get('agency'),
        'fir_ref': d.get('fir_ref'),
        'classification': d.get('classification', 'LAW ENFORCEMENT SENSITIVE'),
        'notes': d.get('notes'),
        'types': d.get('types', []),
        'redact_attachments': d.get('redact_attachments', False),
        'device_info': device_str
    }
    
    fmt = d.get('format', 'pdf')
    
    if fmt == 'pdf':
        try:
            filepath, filename = generate_pdf_report(
                case_id=case_id,
                investigator=options.get('examiner') or session.get('user_name', 'Investigator'),
                evidence=evidence,
                output_dir='reports',
                options=options
            )
            log_activity(session['user_id'],'PDF_ADVANCED_GENERATED',filepath,request.remote_addr)
            return jsonify({'success':True, 'case_id':case_id, 'path':filepath, 'filename':filename})
        except Exception as e:
            return jsonify({'success':False, 'error':str(e)})
    elif fmt == 'html':
        try:
            filepath, filename = generate_html_report(
                case_id=case_id,
                investigator=options.get('examiner') or session.get('user_name', 'Investigator'),
                evidence=evidence,
                output_dir='reports',
                options=options
            )
            log_activity(session['user_id'],'HTML_ADVANCED_GENERATED',filepath,request.remote_addr)
            return jsonify({'success':True, 'case_id':case_id, 'path':filepath, 'filename':filename})
        except Exception as e:
            import traceback; traceback.print_exc()
            return jsonify({'success':False, 'error':str(e)})
    else:
        return jsonify({'success':False, 'error':f"Format {fmt} coming soon"})


@app.route('/api/report/download-pdf/<case_id>')
@login_required
def api_download_pdf(case_id):
    """Download the most recent PDF report for a case or specify path."""
    import glob
    path = request.args.get('path')
    if path and os.path.exists(path):
        filename = os.path.basename(path)
        log_activity(session['user_id'],'PDF_DOWNLOAD',path,request.remote_addr)
        return send_file(path, as_attachment=True, download_name=filename)
        
    # Fallback to generating a fresh one (default standard)
    evidence = get_evidence(case_id)
    inv_name = session.get('user_name', 'Investigator')
    try:
        filepath, filename = generate_pdf_report(
            case_id=case_id,
            investigator=inv_name,
            evidence=evidence,
            output_dir='reports',
            options=None
        )
        log_activity(session['user_id'],'PDF_DOWNLOAD',filepath,request.remote_addr)
        return send_file(filepath, as_attachment=True, download_name=filename)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/report/download-html/<case_id>')
@login_required
def api_download_html(case_id):
    """Download the most recent HTML report for a case or specify path."""
    path = request.args.get('path')
    if path and os.path.exists(path):
        filename = os.path.basename(path)
        log_activity(session['user_id'],'HTML_DOWNLOAD',path,request.remote_addr)
        return send_file(path, as_attachment=True, download_name=filename)
    return jsonify({'error': 'Report not found'}), 404

# ── EVIDENCE DOWNLOAD ────────────────────────────────────────────

@app.route('/api/evidence/download/file/<int:evidence_id>')
@login_required
def download_single_file(evidence_id):
    """Download a single evidence file by its DB id."""
    from flask import send_file, abort
    conn   = get_db()
    row    = conn.execute('SELECT * FROM evidence WHERE id=?',(evidence_id,)).fetchone()
    conn.close()
    if not row: abort(404)
    fp = row['file_path']
    if not fp or not os.path.exists(fp): 
        return "File not found on disk (may be a demo/virtual item).", 404
    log_activity(session['user_id'],'FILE_DOWNLOAD',
                 f'{row["filename"]} | {row["case_id"]}', request.remote_addr)
    return send_file(fp, as_attachment=True,
                     download_name=row['filename'])

@app.route('/api/evidence/preview/<int:evidence_id>')
@login_required
def preview_single_file(evidence_id):
    """Serve a single evidence file inline for previewing."""
    from flask import send_file, abort
    conn   = get_db()
    row    = conn.execute('SELECT * FROM evidence WHERE id=?',(evidence_id,)).fetchone()
    conn.close()
    if not row: abort(404)
    fp = row['file_path']
    if not fp or not os.path.exists(fp): 
        return "Preview Unavailable: File not on disk.", 404
    log_activity(session['user_id'],'FILE_PREVIEW',
                 f'{row["filename"]} | {row["case_id"]}', request.remote_addr)
    return send_file(fp, as_attachment=False)

@app.route('/api/evidence/delete_device/<case_id>/<slot>', methods=['DELETE'])
@login_required
def api_delete_device_evidence(case_id, slot):
    """Delete all evidence for a given device slot inside a case."""
    from flask import jsonify
    from modules.database import delete_device_evidence
    
    # 1. Delete records and retrieve local file paths
    paths = delete_device_evidence(case_id, slot)
    
    # 2. Delete physical files
    deleted_count = 0
    for path in paths:
        if path and os.path.exists(path):
            try:
                os.remove(path)
                deleted_count += 1
            except:
                pass
                
    # 3. Update case stats
    conn = get_db()
    remaining = conn.execute("SELECT * FROM evidence WHERE case_id=?",(case_id,)).fetchall()
    total = len(remaining)
    flagged = sum(1 for r in remaining if r['risk_level'] == 'HIGH')
    size_mb = sum(
        float(str(dict(r).get('file_size','') or '').replace('KB','').replace('MB','').replace('B','') or 0)
        for r in remaining
    ) / 1024
    threat = 'HIGH' if flagged > 0 else 'LOW'  # approximation
    conn.close()
    
    update_case_stats(case_id, total, flagged, round(size_mb, 2), threat)
    
    log_activity(session['user_id'],'DEVICE_EVIDENCE_DELETED',
                 f'Deleted {len(paths)} files from {slot} in {case_id}', request.remote_addr)
    return jsonify({'success': True, 'deleted': deleted_count, 'records_removed': len(paths)})



@app.route('/api/evidence/download/zip/<case_id>')
@login_required
def download_zip(case_id):
    """
    Download all evidence for a case as a ZIP.
    Optional query params:
      ?slot=device1|device2|all  (default: all)
      ?type=image|chat|...       (default: all)
      ?risk=HIGH|MED|LOW         (default: all)
    """
    import zipfile, io
    from flask import send_file

    slot       = request.args.get('slot', 'all')
    ftype      = request.args.get('type', 'all')
    risk       = request.args.get('risk', 'all')
    types_raw  = request.args.get('types', '').strip()
    selected_types = [t.strip() for t in types_raw.split(',') if t.strip()]
    if not selected_types and ftype != 'all':
        selected_types = [ftype]

    conn  = get_db()
    query = 'SELECT * FROM evidence WHERE case_id=?'
    params = [case_id]
    if slot != 'all':
        query += ' AND device_slot=?'
        params.append(slot)
    if selected_types:
        placeholders = ','.join(['?'] * len(selected_types))
        query += f' AND file_type IN ({placeholders})'
        params.extend(selected_types)
    if risk != 'all':
        query += ' AND risk_level=?'
        params.append(risk)
    rows = conn.execute(query, params).fetchall()
    conn.close()

    if not rows:
        return jsonify({'error': 'No evidence files found for this filter'}), 404

    # Build ZIP in memory
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as zf:
        # Add manifest CSV
        manifest_lines = ['filename,device_slot,file_type,file_size,risk_level,ai_result,sha256,extracted_at']
        added = 0
        for row in rows:
            fp = row['file_path']
            manifest_lines.append(
                f'"{row["filename"]}",{row["device_slot"]},{row["file_type"]},'
                f'{row["file_size"]},{row["risk_level"]},"{row["ai_result"] or ""}",{row["sha256"] or ""},'
                f'{row["extracted_at"]}'
            )
            if fp and os.path.exists(fp):
                # Place in subfolder: device_slot/file_type/filename
                arcname = f"{row['device_slot']}/{row['file_type']}/{row['filename']}"
                zf.write(fp, arcname)
                added += 1

        # Always include manifest
        manifest_csv = chr(10).join(manifest_lines)
        zf.writestr('MANIFEST.csv', manifest_csv)
        # Chain of custody report
        coc_lines = [
            'FORAX Chain of Custody Report',
            'Case ID:       ' + case_id,
            'Downloaded:    ' + datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'Investigator:  ' + session.get('user_name', 'Unknown'),
            'Total Records: ' + str(len(rows)),
            'Files in ZIP:  ' + str(added),
            'Filter:        slot=' + slot + ' type=' + ftype + ' risk=' + risk,
            '',
            'SHA-256 hashes are listed in MANIFEST.csv for verification.',
        ]
        coc = chr(10).join(coc_lines)
        zf.writestr('CHAIN_OF_CUSTODY.txt', coc)

    buf.seek(0)
    fname_parts = [case_id]
    if slot  != 'all': fname_parts.append(slot)
    if selected_types:
        fname_parts.append('types-' + str(len(selected_types)))
    if risk  != 'all': fname_parts.append(risk)
    zip_name = '_'.join(fname_parts) + '.zip'

    type_label = ','.join(selected_types) if selected_types else 'all'
    log_activity(session['user_id'], 'EVIDENCE_ZIP_DOWNLOAD',
                 f'{added} files | {case_id} | slot={slot} types={type_label} risk={risk}',
                 request.remote_addr)

    return send_file(buf, mimetype='application/zip',
                     as_attachment=True, download_name=zip_name)


@app.route('/api/evidence/download/csv/<case_id>')
@login_required
def download_csv(case_id):
    """Download evidence table as CSV for external analysis tools."""
    from flask import Response
    slot      = request.args.get('slot', 'all')
    ftype     = request.args.get('type', 'all')
    risk      = request.args.get('risk', 'all')
    types_raw = request.args.get('types', '').strip()
    selected_types = [t.strip() for t in types_raw.split(',') if t.strip()]
    if not selected_types and ftype != 'all':
        selected_types = [ftype]

    conn  = get_db()
    query = 'SELECT * FROM evidence WHERE case_id=?'
    params = [case_id]
    if slot != 'all':
        query += ' AND device_slot=?'
        params.append(slot)
    if selected_types:
        placeholders = ','.join(['?'] * len(selected_types))
        query += f' AND file_type IN ({placeholders})'
        params.extend(selected_types)
    if risk != 'all':
        query += ' AND risk_level=?'
        params.append(risk)
    rows = conn.execute(query, params).fetchall()
    conn.close()

    lines = ['#,Case ID,Device,Filename,Type,Size,AI Result,Risk Level,GPS Lat,GPS Lng,EXIF Date,SHA-256,Extracted At']
    for i, row in enumerate(rows, 1):
        lines.append(
            f'{i},{row["case_id"]},{row["device_slot"]},"{row["filename"]}",'
            f'{row["file_type"]},{row["file_size"]},'
            f'"{(row["ai_result"] or "").replace(chr(34), chr(39))}",'
            f'{row["risk_level"]},{row["gps_lat"] or ""},'
            f'{row["gps_lng"] or ""},{row["exif_date"] or ""},'
            f'{row["sha256"] or ""},{row["extracted_at"]}'
        )

    csv_content = chr(10).join(lines)
    log_activity(session['user_id'], 'CSV_DOWNLOAD',
                 f'{len(rows)} rows | {case_id}', request.remote_addr)

    return Response(
        csv_content,
        mimetype='text/csv',
        headers={'Content-Disposition': f'attachment; filename={case_id}_evidence.csv'}
    )


# ── PASSWORD CHANGE ────────────────────────────────────────────
@app.route('/api/change-password', methods=['POST'])
@login_required
def api_change_password():
    d = request.get_json() or {}
    current_pw = d.get('current_password','')
    new_pw     = d.get('new_password','')
    if len(new_pw) < 8:
        return jsonify({'success':False,'error':'New password must be 8+ characters'})
    conn = get_db()
    u    = conn.execute('SELECT * FROM users WHERE id=?',(session['user_id'],)).fetchone()
    if not u:
        conn.close()
        return jsonify({'success':False,'error':'User not found'})
    if not bcrypt.checkpw(current_pw.encode(), u['password'].encode()):
        conn.close()
        return jsonify({'success':False,'error':'Current password is incorrect'})
    hashed = bcrypt.hashpw(new_pw.encode(), bcrypt.gensalt()).decode()
    conn.execute('UPDATE users SET password=? WHERE id=?',(hashed, session['user_id']))
    conn.commit(); conn.close()
    log_activity(session['user_id'],'PASSWORD_CHANGED','via dashboard',request.remote_addr)
    return jsonify({'success':True})

# ── SEARCH ─────────────────────────────────────────────────────
@app.route('/api/search')
@login_required
def api_search():
    q = request.args.get('q','').strip()
    if not q or len(q) < 2:
        return jsonify({'cases':[],'evidence':[]})
    conn  = get_db()
    like  = f'%{q}%'
    # Search cases
    if session.get('role') in ('chief','authority'):
        cases = conn.execute(
            "SELECT c.*,u.name as investigator_name FROM cases c JOIN users u ON c.investigator_id=u.id "
            "WHERE c.case_id LIKE ? OR c.title LIKE ? OR c.description LIKE ? ORDER BY c.created_at DESC LIMIT 20",
            (like,like,like)).fetchall()
    else:
        cases = conn.execute(
            "SELECT c.*,u.name as investigator_name FROM cases c JOIN users u ON c.investigator_id=u.id "
            "WHERE c.investigator_id=? AND (c.case_id LIKE ? OR c.title LIKE ? OR c.description LIKE ?) "
            "ORDER BY c.created_at DESC LIMIT 20",
            (session['user_id'],like,like,like)).fetchall()
    # Search evidence
    evidence = conn.execute(
        "SELECT * FROM evidence WHERE filename LIKE ? OR ai_result LIKE ? OR sha256 LIKE ? "
        "ORDER BY extracted_at DESC LIMIT 30",
        (like,like,like)).fetchall()
    conn.close()
    return jsonify({
        'cases':    [dict(r) for r in cases],
        'evidence': [dict(r) for r in evidence]
    })

# ── GPS MAP DATA ───────────────────────────────────────────────
@app.route('/api/evidence/gps/<case_id>')
@login_required
def api_gps_evidence(case_id):
    """Get all evidence with GPS coordinates for map display."""
    rows = get_evidence(case_id)
    gps_rows = [
        r for r in rows
        if r.get('gps_lat') not in (None, '') and r.get('gps_lng') not in (None, '')
    ]
    return jsonify(gps_rows)

# ── STORAGE MANAGEMENT ─────────────────────────────────────────
@app.route('/api/storage/status')
@login_required
def api_storage_status():
    """Return total and per-case disk usage of extracted evidence files."""
    total_mb, cases = get_uploads_size()
    return jsonify({'total_mb': total_mb, 'cases': cases})

@app.route('/api/storage/cleanup', methods=['POST'])
@login_required
def api_storage_cleanup():
    """Delete extracted files from disk to reclaim storage.
    Evidence metadata is preserved in the database."""
    d = request.get_json() or {}
    case_id = d.get('case_id')  # None = clean ALL
    ok = cleanup_uploads(case_id)
    detail = f'case={case_id}' if case_id else 'ALL cases'
    log_activity(session['user_id'], 'STORAGE_CLEANUP', detail, request.remote_addr)
    total_mb, _ = get_uploads_size()
    return jsonify({'success': ok, 'remaining_mb': total_mb})

# ── SESSION STATUS ─────────────────────────────────────────────
@app.route('/api/session/status')
def api_session_status():
    """Check if session is still valid — for auto-logout JS."""
    if 'user_id' not in session:
        return jsonify({'active':False})
    return jsonify({'active':True,'user':session.get('user_name'),'role':session.get('role')})

# ── CMS (redirects to chief panel) ──────────────────────────────
@app.route('/cms')
@role_required('chief')
def cms():
    return redirect(url_for('chief_panel'))

# ── AI OVERRIDE — Manual risk level correction by investigator ──
@app.route('/api/evidence/override', methods=['POST'])
@login_required
def api_evidence_override():
    """Allow investigators to manually override AI risk classifications.
    Logs the override in the audit trail for accountability."""
    d = request.get_json() or {}
    ev_id = d.get('evidence_id')
    new_risk = d.get('risk_level', '').upper()
    note = d.get('note', '').strip()
    if not note or len(note) < 3:
        return jsonify({'success': False, 'error': 'Justification note is required (min 3 chars)'})
    if not ev_id or new_risk not in ('HIGH','MED','LOW'):
         return jsonify({'success': False, 'error': 'Invalid evidence ID or risk level'})

    conn = get_db()
    ev = conn.execute('SELECT * FROM evidence WHERE id=?', (ev_id,)).fetchone()
    if not ev:
        conn.close()
        return jsonify({'success': False, 'error': 'Evidence not found'})
    
    old_risk = ev['risk_level']
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    conn.execute('''UPDATE evidence SET risk_level=?, override_risk=?, override_by=?,
                    override_note=?, override_at=? WHERE id=?''',
                 (new_risk, new_risk, session['user_id'], note, now, ev_id))
    conn.commit()
    conn.close()
    log_activity(session['user_id'], 'AI_OVERRIDE',
                 f"Evidence #{ev_id} '{ev['filename']}': {old_risk}→{new_risk} | Note: {note}",
                 request.remote_addr)
    return jsonify({'success': True, 'old_risk': old_risk, 'new_risk': new_risk})


# ── AI FORENSIC REASONING ──────────────────────────────────────
@app.route('/api/forensic/reasoning/<int:ev_id>')
@app.route('/api/copilot/reasoning/<int:ev_id>')  # backward compat
@login_required
def api_forensic_reasoning(ev_id):
    """Generate detailed forensic reasoning for a specific detection."""
    conn = get_db()
    row = conn.execute('SELECT * FROM evidence WHERE id=?', (ev_id,)).fetchone()
    conn.close()
    
    if not row:
        return jsonify({'success': False, 'error': 'Evidence not found'})
    
    fpath = row['file_path']
    ftype = row['file_type']
    
    if not fpath or not os.path.exists(fpath):
         return jsonify({'success': True, 'reasoning': 'Detailed reasoning unavailable: Physical file has been moved or deleted from the forensic volume.'})

    # Run analysis to get fresh forensic reasoning narrative
    try:
        if ftype in ('image', 'video'):
            res = classify_image(fpath)
        else:
            res = analyze_file(fpath, ftype)
        
        reasoning = res.get('copilot_reasoning', 'No detailed reasoning available for this artifact.')
    except Exception as e:
        reasoning = f"Analysis Error: Failed to generate reasoning narrative. {str(e)}"

    return jsonify({
        'success': True,
        'filename': row['filename'],
        'risk_level': row['risk_level'],
        'reasoning': reasoning
    })

# ── VERIFY INTEGRITY — Re-hash evidence file and compare with stored hash ──
@app.route('/api/evidence/verify/<int:ev_id>')
@login_required
def api_evidence_verify(ev_id):
    """Re-compute SHA-256 of an evidence file and compare with stored hash.
    Used to verify chain of custody integrity."""
    conn = get_db()
    ev = conn.execute('SELECT * FROM evidence WHERE id=?', (ev_id,)).fetchone()
    conn.close()
    if not ev:
        return jsonify({'success': False, 'error': 'Evidence not found'})
    fpath = ev['file_path']
    if not fpath or not os.path.exists(fpath):
        return jsonify({'success': False, 'error': 'File not found on disk', 'status': 'MISSING'})
    current_hash = get_sha256(fpath)
    stored_hash = ev['sha256']
    match = (current_hash == stored_hash)
    log_activity(session['user_id'], 'INTEGRITY_CHECK',
                 f"Evidence #{ev_id} '{ev['filename']}': {'PASS' if match else 'FAIL'}",
                 request.remote_addr)
    return jsonify({
        'success': True,
        'match': match,
        'status': 'INTACT' if match else 'TAMPERED',
        'stored_hash': stored_hash,
        'current_hash': current_hash,
        'filename': ev['filename']
    })

# ── EXTRACTION STATE RECOVERY (for desktop app) ────────────────
@app.route('/api/extract/reconnect/<slot>')
@login_required
def api_extract_reconnect(slot):
    """Return current extraction state so UI can recover after internal navigation."""
    if slot not in ('device1','device2'):
        return jsonify({'error':'Invalid slot'}), 400
    state = get_state(slot)
    return jsonify({
        'status': state.get('status','idle'),
        'progress': state.get('progress',0),
        'step': state.get('step',''),
        'files': state.get('files',[]),
        'device': state.get('device',{}),
        'summary': state.get('summary',{}),
    })

# ── START ───────────────────────────────────────────────────────
if __name__ == '__main__':
    init_db()
    os.makedirs('uploads', exist_ok=True)
    os.makedirs('reports', exist_ok=True)
    print("\n" + "="*56)
    print("  FORAX — Forensic Analysis & eXtraction Platform")
    print()
    print("  NOTE: For desktop mode, run:  python desktop.py")
    print("  Web mode: http://127.0.0.1:5000")
    print()
    print("  Chief:  chief@forax.gov / Chief@FORAX2025")
    print("="*56 + "\n")
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)
