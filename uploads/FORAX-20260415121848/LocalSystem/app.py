from flask import (Flask, render_template, request, jsonify,
                   session, redirect, url_for, Response)
from functools import wraps
from datetime import datetime
import os, json, threading, bcrypt

from modules.database import (
    init_db, verify_user, update_last_login, get_user_by_id,
    get_all_users, get_all_investigators, create_investigator,
    toggle_investigator, next_badge_id, log_activity, get_activity_logs,
    create_case, get_cases, update_case_stats, update_case_device,
    add_evidence, get_evidence, get_system_stats,
    store_otp, get_otp, increment_otp_attempts, delete_otp,
    get_db
)
from modules.ai_image import classify_image, classify_image_batch
from modules.ai_nlp   import analyze_file, get_threat_summary
from modules.adb_extractor import (
    get_devices, get_device_info, full_extraction_realtime,
    get_state, reset_state, get_event_queue
)
from modules.security import (
    is_ip_blocked, record_failed_attempt, clear_rate_limit,
    get_attempts_left, generate_otp, generate_math_captcha,
    email_otp, email_welcome, email_login_alert, email_lockout_alert
)

app = Flask(__name__)
app.secret_key = 'FORAX-SECRET-LGU-AHSAN-2025-BSDFCS'

# ── DECORATORS ─────────────────────────────────────────────────
def login_required(f):
    @wraps(f)
    def d(*a, **kw):
        if 'user_id' not in session: return redirect(url_for('login'))
        return f(*a, **kw)
    return d

def role_required(*roles):
    def dec(f):
        @wraps(f)
        def d(*a, **kw):
            if 'user_id' not in session: return redirect(url_for('login'))
            if session.get('role') not in roles: return redirect(url_for('dashboard'))
            return f(*a, **kw)
        return d
    return dec

# ── ROUTES ─────────────────────────────────────────────────────
@app.route('/')
def index():
    if 'user_id' not in session: return redirect(url_for('login'))
    r = session.get('role')
    if r == 'chief':     return redirect(url_for('chief_panel'))
    if r == 'authority': return redirect(url_for('cms'))
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

    routes = {'chief': url_for('chief_panel'), 'authority': url_for('cms')}
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
    otp   = data.get('otp','').strip()
    row   = get_otp(email)
    if not row: return jsonify({'success':False,'error':'OTP not found. Request again.'})
    if row['attempts'] >= 3:
        delete_otp(email)
        return jsonify({'success':False,'error':'Too many attempts. Request a new OTP.'})
    expires_dt = datetime.strptime(row['expires_at'],'%Y-%m-%d %H:%M:%S')
    if datetime.now() > expires_dt:
        delete_otp(email)
        return jsonify({'success':False,'error':'OTP expired. Request a new one.'})
    if row['otp'] != otp:
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
    return render_template('chief.html', user=user, investigators=invs,
                           logs=logs, stats=stats, next_badge=next_b,
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
@app.route('/extraction')
@login_required
def extraction_page():
    return render_template('extraction.html')

@app.route('/api/extract/start-realtime', methods=['POST'])
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

@app.route('/api/extract/stream/<slot>')
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
                    if ev.get('data',{}).get('status') in ('complete','error'):
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

@app.route('/api/evidence/<case_id>')
@login_required
def api_get_evidence(case_id):
    slot = request.args.get('slot')
    return jsonify(get_evidence(case_id, slot))

# ── AI ANALYSIS ─────────────────────────────────────────────────
@app.route('/api/analyze', methods=['POST'])
@login_required
def api_analyze():
    d       = request.get_json() or {}
    case_id = d.get('case_id')
    slot    = d.get('slot', 'device1')
    files   = get_evidence(case_id, slot)
    risks   = {'HIGH': 0, 'MED': 0, 'LOW': 0}
    conn    = get_db()
    nlp_results = []

    for f in files:
        fname   = f.get('filename', '').lower()
        ftype   = f.get('file_type', '')
        fpath   = f.get('file_path', '')
        risk    = 'LOW'
        res     = 'Analyzed'
        gps_lat = f.get('gps_lat')
        gps_lng = f.get('gps_lng')
        exif_dt = f.get('exif_date')

        # ── CNN IMAGE ANALYSIS ──────────────────────────────────
        if ftype == 'image' and fpath and os.path.exists(fpath):
            try:
                img_result = classify_image(fpath, f.get('filename',''))
                risk    = img_result.get('risk_level', 'LOW')
                res     = img_result.get('ai_result', 'CNN: Analyzed')
                gps_lat = img_result.get('gps_lat') or gps_lat
                gps_lng = img_result.get('gps_lng') or gps_lng
                exif_dt = img_result.get('exif_date') or exif_dt
            except Exception as e:
                res  = f'CNN: Error — {str(e)[:60]}'
                risk = 'LOW'

        # ── NLP TEXT ANALYSIS ───────────────────────────────────
        elif ftype in ('chat', 'sms', 'call_log', 'contacts', 'browser', 'download'):
            if fpath and os.path.exists(fpath):
                try:
                    nlp_result = analyze_file(fpath, ftype)
                    risk = nlp_result.get('risk_level', 'LOW')
                    res  = nlp_result.get('ai_result', 'NLP: Analyzed')
                    nlp_results.append(nlp_result)
                except Exception as e:
                    res  = f'NLP: Error — {str(e)[:60]}'
                    risk = 'LOW'
            else:
                # File not on disk — use filename-based analysis
                from modules.ai_nlp import scan_keywords
                kw = scan_keywords(fname)
                if kw:
                    risk = 'HIGH' if any(
                        v['risk']=='HIGH' for k,v in __import__('modules.ai_nlp', fromlist=['THREAT_KEYWORDS']).THREAT_KEYWORDS.items()
                        if k in kw) else 'MED'
                    res  = f"NLP: Filename analysis — {sum(len(v) for v in kw.values())} keywords"
                else:
                    res  = 'NLP: Filename analyzed — no threats'

        # ── EXECUTABLE / APK ────────────────────────────────────
        elif ftype == 'download':
            ext = fname.rsplit('.', 1)[-1] if '.' in fname else ''
            if ext in ('apk', 'exe', 'bat', 'sh', 'ps1', 'vbs', 'cmd'):
                risk = 'HIGH'
                res  = f'HIGH: Executable file (.{ext}) — manual inspection required'
            else:
                res  = 'Download analyzed'

        # ── APP DATA ────────────────────────────────────────────
        elif ftype == 'app_data':
            res = 'App inventory catalogued'

        # ── RECOVERED FILES ─────────────────────────────────────
        elif ftype == 'recovered':
            risk = 'MED'
            res  = 'Recovered deleted file — review recommended'

        else:
            res = 'File analyzed'

        risks[risk] += 1
        conn.execute(
            'UPDATE evidence SET risk_level=?, ai_result=?, gps_lat=?, gps_lng=?, exif_date=? WHERE id=?',
            (risk, res, gps_lat, gps_lng, exif_dt, f['id'])
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
        'success': True, 'total': total,
        'high': risks['HIGH'], 'med': risks['MED'], 'low': risks['LOW'],
        'threat': threat
    })

# ── REPORT ─────────────────────────────────────────────────────
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
    if not fp or not os.path.exists(fp): abort(404)
    log_activity(session['user_id'],'FILE_DOWNLOAD',
                 f'{row["filename"]} | {row["case_id"]}', request.remote_addr)
    return send_file(fp, as_attachment=True,
                     download_name=row['filename'])


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

    slot      = request.args.get('slot', 'all')
    ftype     = request.args.get('type', 'all')
    risk      = request.args.get('risk', 'all')

    conn  = get_db()
    query = 'SELECT * FROM evidence WHERE case_id=?'
    params = [case_id]
    if slot != 'all':  query += ' AND device_slot=?'; params.append(slot)
    if ftype != 'all': query += ' AND file_type=?';   params.append(ftype)
    if risk  != 'all': query += ' AND risk_level=?';  params.append(risk)
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
    if ftype != 'all': fname_parts.append(ftype)
    if risk  != 'all': fname_parts.append(risk)
    zip_name = '_'.join(fname_parts) + '.zip'

    log_activity(session['user_id'], 'EVIDENCE_ZIP_DOWNLOAD',
                 f'{added} files | {case_id} | slot={slot} type={ftype} risk={risk}',
                 request.remote_addr)

    return send_file(buf, mimetype='application/zip',
                     as_attachment=True, download_name=zip_name)


@app.route('/api/evidence/download/csv/<case_id>')
@login_required
def download_csv(case_id):
    """Download evidence table as CSV for external analysis tools."""
    from flask import Response
    slot  = request.args.get('slot', 'all')
    ftype = request.args.get('type', 'all')
    risk  = request.args.get('risk', 'all')

    conn  = get_db()
    query = 'SELECT * FROM evidence WHERE case_id=?'
    params = [case_id]
    if slot  != 'all': query += ' AND device_slot=?'; params.append(slot)
    if ftype != 'all': query += ' AND file_type=?';   params.append(ftype)
    if risk  != 'all': query += ' AND risk_level=?';  params.append(risk)
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


# ── CMS ─────────────────────────────────────────────────────────
@app.route('/cms')
@role_required('authority','chief')
def cms():
    user  = get_user_by_id(session['user_id'])
    stats = get_system_stats()
    users = get_all_users()
    cases = get_cases()
    logs  = get_activity_logs(100)
    return render_template('cms.html', user=user, stats=stats,
                           users=users, cases=cases, logs=logs)

# ── START ───────────────────────────────────────────────────────
if __name__ == '__main__':
    init_db()
    os.makedirs('uploads', exist_ok=True)
    os.makedirs('reports', exist_ok=True)
    print("\n" + "="*56)
    print("  FORAX — Forensic Analysis & eXtraction Platform")
    print("  http://127.0.0.1:5000")
    print()
    print("  Chief:     chief@forax.gov     / Chief@FORAX2025")
    print("  Authority: authority@forax.gov / Authority@123")
    print("="*56 + "\n")
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)
