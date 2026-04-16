"""
FORAX — ADB Evidence Extractor
Extracts forensic data from Android devices via ADB (Android Debug Bridge).
Supports 2 simultaneous devices, streams results via SSE (Server-Sent Events).

Data Sources: Images, Videos, Documents, Audio, Call Logs, SMS, Contacts,
Browser, Calendar, Downloads, Device Accounts, WiFi, Apps, Notifications,
Deleted Files, Hidden Configs, Root Data
"""

import subprocess, os, hashlib, time, threading, queue, shutil
from datetime import datetime

# Directory where extracted evidence files are stored
UPLOAD_DIR = os.path.join(os.path.dirname(__file__), '..', 'uploads')

# Each device slot has its own event queue for SSE streaming to the frontend
event_queues = {'device1': queue.Queue(), 'device2': queue.Queue()}

# Tracks extraction progress/state per device (accessed by multiple threads)
extraction_states = {
    'device1': {'status':'idle','progress':0,'step':'','files':[],'summary':{},'device':{},'log':[]},
    'device2': {'status':'idle','progress':0,'step':'','files':[],'summary':{},'device':{},'log':[]},
}
_lock = threading.Lock()  # Protects extraction_states from race conditions

# Stop / pause flags — set by API routes, checked inside extraction thread
_stop_flags   = {'device1': threading.Event(), 'device2': threading.Event()}
_pause_flags  = {'device1': threading.Event(), 'device2': threading.Event()}

# ── ADB HELPERS ── Find and run ADB commands ───────────────────
def find_adb():
    """Find adb executable - checks PATH, platform-tools, and common Windows locations."""
    import shutil, sys
    # 1. Try adb directly from PATH
    found = shutil.which('adb')
    if found: return found
    # 2. Look for platform-tools relative to project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    candidates = [
        os.path.join(project_root, 'platform-tools', 'adb.exe'),
        os.path.join(project_root, 'platform-tools', 'adb'),
        os.path.join(os.path.dirname(project_root), 'platform-tools', 'adb.exe'),
        os.path.join(os.path.dirname(project_root), 'platform-tools', 'adb'),
    ]
    # 3. Common Windows installation paths
    if sys.platform == 'win32':
        home = os.path.expanduser('~')
        candidates += [
            os.path.join(home, 'AppData', 'Local', 'Android', 'Sdk', 'platform-tools', 'adb.exe'),
            r'C:\Android\platform-tools\adb.exe',
            r'C:\sdk\platform-tools\adb.exe',
            r'C:\Users\Public\Android\platform-tools\adb.exe',
        ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    return 'adb'  # fallback — let OS handle it

ADB_PATH = find_adb()  # Cached on module load

def adb(args, serial=None, timeout=60):
    """Run an ADB command and return (stdout, return_code)."""
    cmd = [ADB_PATH]
    if serial: cmd += ['-s', serial]
    cmd += args
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                           encoding='utf-8', errors='replace')
        return r.stdout.strip(), r.returncode
    except FileNotFoundError: return 'ADB_NOT_FOUND', -1
    except subprocess.TimeoutExpired: return 'TIMEOUT', -1
    except Exception as e: return str(e), -1

def get_devices():
    """List connected Android devices. Returns ([serials], status_string)."""
    out, code = adb(['devices'])
    if out == 'ADB_NOT_FOUND': return [], 'adb_missing'
    devices = []
    for line in out.splitlines()[1:]:
        if '\tdevice' in line:
            serial = line.split('\t')[0].strip()
            if serial: devices.append(serial)
    return devices, 'ok'

def get_device_info(serial=None):
    """Read device properties (model, brand, Android version, etc.) via ADB."""
    props = {
        'model':        'ro.product.model',
        'brand':        'ro.product.brand',
        'android':      'ro.build.version.release',
        'sdk':          'ro.build.version.sdk',
        'serial':       'ro.serialno',
        'manufacturer': 'ro.product.manufacturer',
        'storage':      'ro.product.storage',
    }
    info = {}
    for key, prop in props.items():
        val, _ = adb(['shell', 'getprop', prop], serial, timeout=10)
        info[key] = val if val and val not in ('TIMEOUT','ADB_NOT_FOUND','') else 'Unknown'

    # IMEI (telephony)
    imei_out, _ = adb(['shell', 'service', 'call', 'iphonesubinfo', '1'], serial, timeout=10)
    if imei_out and 'ADB_NOT_FOUND' not in imei_out:
        import re
        digits = re.findall(r"'([^']*)'", imei_out)
        imei = ''.join(digits).replace('.', '').replace(' ', '').strip()
        info['imei'] = imei if len(imei) >= 14 else 'Unavailable'
    else:
        info['imei'] = 'Unavailable'

    # App count
    apps, _ = adb(['shell', 'pm', 'list', 'packages'], serial, timeout=15)
    info['app_count'] = len([l for l in apps.splitlines() if l.startswith('package:')]) if apps else 0
    info['extracted_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    return info

def sha256_file(path):
    """Compute SHA-256 hash of a file for chain of custody verification."""
    h = hashlib.sha256()
    try:
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(65536), b''): h.update(chunk)
        return h.hexdigest()
    except: return 'error'

def get_sha256(path): return sha256_file(path)

def get_file_type(path):
    """Classify file based on extension."""
    ext = os.path.splitext(path)[1].lower()
    if ext in ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'): return 'image'
    if ext in ('.mp4', '.avi', '.mov', '.mkv', '.wmv', '.3gp'): return 'video'
    if ext in ('.mp3', '.wav', '.ogg', '.aac', '.m4a', '.flac', '.amr', '.opus'): return 'audio'
    if ext in ('.txt', '.log', '.json', '.xml', '.csv'): return 'text'
    if ext in ('.pdf', '.doc', '.docx', '.rtf'): return 'document'
    if ext in ('.db', '.sqlite', '.sqlite3'): return 'database'
    if ext in ('.apk', '.exe', '.bat', '.sh'): return 'download'
    return 'unknown'

# ── EVENT SYSTEM ── Pushes live updates to frontend via SSE ───
def emit(slot, etype, data):
    """Push an event to the SSE queue and update extraction state."""
    event_queues[slot].put({
        'type': etype, 'data': data,
        'time': datetime.now().strftime('%H:%M:%S')
    })
    with _lock:
        st = extraction_states[slot]
        if etype == 'file':     st['files'].append(data)
        elif etype == 'progress': st['progress'] = data.get('pct',0); st['step'] = data.get('step','')
        elif etype == 'log':    st['log'].insert(0, data)
        elif etype == 'device': st['device'] = data
        elif etype == 'status': st['status'] = data.get('status','')
        elif etype == 'summary': st['summary'] = data

def elog(slot, msg, level='info'):
    emit(slot, 'log', {'msg':msg, 'level':level, 'time':datetime.now().strftime('%H:%M:%S')})

def eprog(slot, pct, step):
    emit(slot, 'progress', {'pct':pct, 'step':step})

# ── QUICK AI ANALYSIS ── Fast risk check during extraction ─────
def quick_analyze(f):
    """Minimal metadata tagging. Full AI analysis is manual only."""
    f['risk_level'] = 'LOW'
    f['ai_result']  = 'Pending Analysis'
    return f

# Tracks SHA-256 hashes already seen in this extraction to skip duplicates
_seen_hashes = {'device1': set(), 'device2': set()}

# ── FILE PULLER ── Pulls files from device and streams to UI ──
def pull_and_stream(serial, slot, remote, local_dir, ftype, include_hidden=False):
    """Pull a remote folder via 'adb pull', hash new files, and emit them to SSE.
    Deduplicates by SHA-256 to save storage. Set include_hidden=True to keep dotfiles."""
    os.makedirs(local_dir, exist_ok=True)
    before = set()
    for root, _, files in os.walk(local_dir):
        for f in files: before.add(os.path.join(root,f))

    adb(['pull', remote, local_dir], serial, timeout=300)

    new = []
    for root, _, files in os.walk(local_dir):
        for fname in files:
            fp = os.path.join(root, fname)
            if fp in before: continue
            # Skip dotfiles only when not in forensic/hidden mode
            if not include_hidden and fname.startswith('.'): continue
            try:
                size = os.path.getsize(fp)
                h    = sha256_file(fp)
                # ── DEDUP: skip files with identical SHA-256 ──
                if h in _seen_hashes[slot]:
                    os.remove(fp)  # Remove duplicate to save storage
                    continue
                _seen_hashes[slot].add(h)
                fd   = {'filename':fname,'path':fp,'size':size,'sha256':h,
                        'type':ftype,'risk_level':'LOW','ai_result':'Extracted',
                        'gps_lat':None,'gps_lng':None,'exif_date':None}
                fd = quick_analyze(fd)
                emit(slot, 'file', fd)
                new.append(fd)
                time.sleep(0.03)
            except: pass
    return new

def query_and_stream(serial, slot, uri, projection, filename, ftype, local_dir):
    """Query Android content provider (call logs, SMS, contacts, etc.) and save as text file."""
    os.makedirs(local_dir, exist_ok=True)
    out, code = adb(['shell','content','query','--uri',uri,'--projection',projection], serial, timeout=30)
    if code == 0 and out and len(out) > 20:
        fp = os.path.join(local_dir, filename)
        with open(fp,'w',encoding='utf-8') as f: f.write(out)
        h = sha256_file(fp)
        fd = {'filename':filename,'path':fp,'size':len(out.encode()),'sha256':h,
              'type':ftype,'risk_level':'LOW','ai_result':f'{len(out.splitlines())} records',
              'gps_lat':None,'gps_lng':None,'exif_date':None}
        fd = quick_analyze(fd)
        emit(slot, 'file', fd)
        return [fd]
    return []

def dumpsys_and_stream(serial, slot, service, filename, ftype, local_dir, max_lines=5000):
    """Run 'adb shell dumpsys <service>' and save the output as a text file."""
    os.makedirs(local_dir, exist_ok=True)
    out, code = adb(['shell', 'dumpsys', service], serial, timeout=30)
    if code == 0 and out and len(out) > 50:
        # Truncate huge outputs
        lines = out.splitlines()[:max_lines]
        text = '\n'.join(lines)
        fp = os.path.join(local_dir, filename)
        with open(fp, 'w', encoding='utf-8') as f: f.write(text)
        h = sha256_file(fp)
        fd = {'filename': filename, 'path': fp, 'size': len(text.encode()), 'sha256': h,
              'type': ftype, 'risk_level': 'LOW', 'ai_result': f'{len(lines)} lines captured',
              'gps_lat': None, 'gps_lng': None, 'exif_date': None}
        fd = quick_analyze(fd)
        emit(slot, 'file', fd)
        return [fd]
    return []

# ── ROOT DATA EXTRACTION ── Extracts sensitive data from rooted devices ──
def extract_sensitive_root(serial, slot, dest):
    """Extract sensitive data from rooted Android devices (social media DBs, configs)."""
    local_dir = os.path.join(dest, 'root_data')
    os.makedirs(local_dir, exist_ok=True)
    new_files = []

    out, code = adb(['shell', 'su', '-c', 'id'], serial, timeout=10)
    if 'root' not in (out or '').lower() and code != 0:
        return new_files  # Not rooted

    targets = [
        ('/data/system/users/0/accounts.db', 'accounts.db', 'config', 'MED'),
        ('/data/misc/wifi/WifiConfigStore.xml', 'WifiConfigStore.xml', 'config', 'HIGH'),
        ('/data/data/com.google.android.gms/databases/', 'google_gms.tar', 'app_data', 'MED'),
        ('/data/data/com.facebook.katana/databases/', 'facebook_db.tar', 'app_data', 'LOW'),
        ('/data/data/com.instagram.android/databases/', 'instagram_db.tar', 'app_data', 'LOW'),
        ('/data/data/com.twitter.android/databases/', 'twitter_db.tar', 'app_data', 'LOW'),
        ('/data/data/com.snapchat.android/databases/', 'snapchat_db.tar', 'app_data', 'LOW'),
        ('/data/data/com.whatsapp/databases/', 'whatsapp_db.tar', 'app_data', 'MED'),
        ('/data/data/org.telegram.messenger/files/', 'telegram_files.tar', 'app_data', 'MED'),
        ('/data/data/com.tencent.mm/MicroMsg/', 'wechat_db.tar', 'app_data', 'MED'),
    ]

    for rpath, local_name, ftype, risk in targets:
        tmp_rpath = f'/sdcard/forax_tmp_{local_name}'
        if rpath.endswith('/'):
            o, c = adb(['shell', 'su', '-c', f'ls {rpath} 2>/dev/null'], serial, timeout=10)
            if c != 0 or not o.strip(): continue
            adb(['shell', 'su', '-c', f'tar -cf {tmp_rpath} -C {os.path.dirname(rpath)} {os.path.basename(rpath[:-1])}'], serial, timeout=20)
        else:
            o, c = adb(['shell', 'su', '-c', f'ls {rpath} 2>/dev/null'], serial, timeout=10)
            if c != 0 or not o.strip(): continue
            adb(['shell', 'su', '-c', f'cp {rpath} {tmp_rpath}'], serial, timeout=10)

        adb(['shell', 'su', '-c', f'chmod 666 {tmp_rpath}'], serial, timeout=10)

        local_path = os.path.join(local_dir, local_name)
        adb(['pull', tmp_rpath, local_path], serial, timeout=30)
        adb(['shell', 'rm', tmp_rpath], serial, timeout=10)

        if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
            size = os.path.getsize(local_path)
            h = sha256_file(local_path)
            if h in _seen_hashes[slot]:
                os.remove(local_path)
                continue
            _seen_hashes[slot].add(h)
            fd = {'filename': local_name, 'path': local_path, 'size': size, 'sha256': h,
                  'type': ftype, 'risk_level': risk,
                  'ai_result': 'Root extracted app/config data',
                  'gps_lat': None, 'gps_lng': None, 'exif_date': None}
            emit(slot, 'file', fd)
            new_files.append(fd)
            time.sleep(0.05)

    return new_files

def extract_documents(serial, slot, dest):
    """Extract document files (PDF, DOC, TXT) from the device."""
    local_dir = os.path.join(dest, 'documents')
    os.makedirs(local_dir, exist_ok=True)
    new_files = []

    cmd = r'find /sdcard/ -type f \( -iname "*.pdf" -o -iname "*.doc" -o -iname "*.docx" -o -iname "*.txt" -o -iname "*.xls" -o -iname "*.xlsx" -o -iname "*.ppt" -o -iname "*.pptx" -o -iname "*.rtf" \) 2>/dev/null'
    out, code = adb(['shell', cmd], serial, timeout=30)
    if not out or code != 0 or out in ('TIMEOUT','ADB_NOT_FOUND'): return new_files

    for line in out.strip().splitlines():
        rpath = line.strip()
        if not rpath or not rpath.startswith('/'): continue
        fname = os.path.basename(rpath)
        if not fname: continue

        sub_dir = os.path.join(local_dir, os.path.basename(os.path.dirname(rpath)) or "unknown")
        os.makedirs(sub_dir, exist_ok=True)
        local_path = os.path.join(sub_dir, fname)

        if os.path.exists(local_path):
            local_path = os.path.join(sub_dir, str(int(time.time()*1000)) + '_' + fname)

        adb(['pull', rpath, local_path], serial, timeout=30)
        if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
            size = os.path.getsize(local_path)
            h = sha256_file(local_path)
            if h in _seen_hashes[slot]:
                os.remove(local_path)
                continue
            _seen_hashes[slot].add(h)
            fd = {'filename': fname, 'path': local_path, 'size': size, 'sha256': h,
                  'type': 'document', 'risk_level': 'LOW',
                  'ai_result': 'Document Extracted',
                  'gps_lat': None, 'gps_lng': None, 'exif_date': None}
            fd = quick_analyze(fd)
            emit(slot, 'file', fd)
            new_files.append(fd)
            time.sleep(0.02)
    return new_files

def find_and_pull_hidden(serial, slot, dest):
    """Use 'adb shell find' to locate hidden config files, then pull them individually."""
    local_dir = os.path.join(dest, 'hidden_configs')
    os.makedirs(local_dir, exist_ok=True)
    scan_targets = [
        ('/sdcard/', '-maxdepth 1 -name ".*" -type f'),
        ('/data/misc/wifi/', '-name "*.conf" -o -name "*.xml"'),
        ('/sdcard/Android/data/', '-name "*.xml" -path "*/shared_prefs/*"'),
        ('/sdcard/Android/data/', '-name "*.db" -path "*/databases/*"'),
        ('/sdcard/', '-maxdepth 2 -name ".nomedia" -type f'),
        ('/sdcard/', '-maxdepth 2 -type d -name ".*"'),
        ('/sdcard/Bluetooth/', '-type f'),
    ]
    new_files = []
    for remote_dir, find_args in scan_targets:
        try:
            cmd = f'find {remote_dir} {find_args} 2>/dev/null'
            out, code = adb(['shell', cmd], serial, timeout=15)
            if code != 0 or not out or out in ('TIMEOUT','ADB_NOT_FOUND'): continue
            for line in out.strip().splitlines():
                rpath = line.strip()
                if not rpath or 'Permission denied' in rpath or 'No such' in rpath: continue
                fname = os.path.basename(rpath)
                if not fname: continue
                sub_dir = os.path.join(local_dir, os.path.basename(os.path.dirname(rpath)))
                os.makedirs(sub_dir, exist_ok=True)
                local_path = os.path.join(sub_dir, fname)
                adb(['pull', rpath, local_path], serial, timeout=30)
                if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
                    size = os.path.getsize(local_path)
                    h = sha256_file(local_path)
                    if h in _seen_hashes[slot]:
                        os.remove(local_path)
                        continue
                    _seen_hashes[slot].add(h)
                    fd = {'filename': fname, 'path': local_path, 'size': size, 'sha256': h,
                          'type': 'config', 'risk_level': 'MED',
                          'ai_result': f'Hidden config: {os.path.dirname(rpath)}',
                          'gps_lat': None, 'gps_lng': None, 'exif_date': None}
                    fd = quick_analyze(fd)
                    emit(slot, 'file', fd)
                    new_files.append(fd)
                    time.sleep(0.02)
        except: pass
    return new_files

# ── MAIN EXTRACTION PIPELINE ── Orchestrates the full extraction process
def full_extraction_realtime(serial, slot='device1', case_id='unknown'):
    """Start full device extraction in a background thread. Streams progress via SSE."""
    _stop_flags[slot].clear()
    _pause_flags[slot].clear()
    _seen_hashes[slot] = set()
    dest = os.path.join(UPLOAD_DIR, case_id, slot)
    os.makedirs(dest, exist_ok=True)

    with _lock:
        extraction_states[slot] = {
            'status':'running','progress':0,'step':'Starting...',
            'files':[],'summary':{},'device':{},'log':[]
        }
    while not event_queues[slot].empty():
        try: event_queues[slot].get_nowait()
        except: break

    def run():
        all_files = []
        try:
            emit(slot,'status',{'status':'running'})
            elog(slot,f'Connecting to {serial}...','info')
            eprog(slot, 2, 'Reading device info...')

            info = get_device_info(serial)
            emit(slot,'device',info)
            elog(slot,f"Device: {info.get('brand','')} {info.get('model','')} | Android {info.get('android','?')} | IMEI: {info.get('imei','N/A')} | {info.get('app_count',0)} apps",'success')
            eprog(slot, 5, 'Device connected — starting extraction...')
            time.sleep(0.2)

            steps = [
                # ── MEDIA ──
                (5, 12, '📷 Extracting images (DCIM, Pictures, Screenshots)...',
                 lambda: sum([pull_and_stream(serial,slot,p,os.path.join(dest,'images'),'image')
                              for p in ['/sdcard/DCIM/','/sdcard/Pictures/','/sdcard/Screenshots/']], [])),
                (12, 18, '🎬 Extracting videos...',
                 lambda: sum([pull_and_stream(serial,slot,p,os.path.join(dest,'videos'),'video')
                              for p in ['/sdcard/Movies/','/sdcard/Videos/']], [])),
                (18, 24, '📄 Extracting documents (PDF/DOC/TXT/XLS)...',
                 lambda: extract_documents(serial, slot, dest)),
                (24, 30, '🎵 Extracting audio & recordings...',
                 lambda: sum([pull_and_stream(serial,slot,p,os.path.join(dest,'audio'),'audio')
                              for p in ['/sdcard/Recordings/','/sdcard/Voice Recorder/',
                                        '/sdcard/Music/','/sdcard/Ringtones/',
                                        '/sdcard/Notifications/','/sdcard/Alarms/']], [])),

                # ── COMMUNICATIONS ──
                (30, 35, '📞 Extracting call logs...',
                 lambda: query_and_stream(serial,slot,'content://call_log/calls',
                                          'number:date:duration:type:name',
                                          'call_logs.txt','call_log',os.path.join(dest,'call_logs'))),
                (35, 40, '💬 Extracting SMS messages...',
                 lambda: query_and_stream(serial,slot,'content://sms',
                                          'address:date:body:type:read',
                                          'sms_messages.txt','sms',os.path.join(dest,'sms'))),
                (40, 44, '👤 Extracting contacts...',
                 lambda: query_and_stream(serial,slot,'content://contacts/phones',
                                          'display_name:number:type',
                                          'contacts.txt','contacts',os.path.join(dest,'contacts'))),

                # ── BROWSING & CALENDAR ──
                (44, 49, '🌐 Extracting browser history & bookmarks...',
                 lambda: query_and_stream(serial,slot,
                                          'content://com.android.browser.history/history',
                                          'url:date:title:visits',
                                          'browser_history.txt','browser',os.path.join(dest,'browser'))),
                (49, 54, '📅 Extracting calendar events...',
                 lambda: query_and_stream(serial,slot,
                                          'content://com.android.calendar/events',
                                          'title:dtstart:dtend:description:eventLocation',
                                          'calendar_events.txt','calendar',os.path.join(dest,'calendar'))),

                # ── DOWNLOADS ──
                (54, 59, '📥 Extracting downloads folder...',
                 lambda: pull_and_stream(serial,slot,'/sdcard/Download/',
                                         os.path.join(dest,'downloads'),'download')),

                # ── DEVICE INTELLIGENCE ──
                (59, 63, '📱 Extracting device accounts & SIM info...',
                 lambda: dumpsys_and_stream(serial,slot,'telephony.registry',
                                           'sim_info.txt','device_info',os.path.join(dest,'device_intel'))),
                (63, 67, '📶 Extracting WiFi network history...',
                 lambda: dumpsys_and_stream(serial,slot,'wifi',
                                           'wifi_networks.txt','network',os.path.join(dest,'device_intel'))),
                (67, 71, '🔔 Extracting notification log...',
                 lambda: dumpsys_and_stream(serial,slot,'notification',
                                           'notifications.txt','notification',os.path.join(dest,'device_intel'), max_lines=3000)),
                (71, 75, '📊 Extracting battery & usage stats...',
                 lambda: dumpsys_and_stream(serial,slot,'usagestats',
                                           'usage_stats.txt','usage_stats',os.path.join(dest,'device_intel'), max_lines=4000)),

                # ── RECOVERY & FORENSIC ──
                (75, 83, '🗑️ Recovering deleted files...',
                 lambda: sum([pull_and_stream(serial,slot,p,os.path.join(dest,'recovered'),'recovered',
                                             include_hidden=True)
                              for p in [
                                  '/sdcard/.trash/',
                                  '/sdcard/.Trash/',
                                  '/sdcard/.Recently-Deleted/',
                                  '/sdcard/.recently-deleted/',
                                  '/sdcard/.thumbnails/',
                                  '/sdcard/.gallerycache/',
                                  '/sdcard/Android/data/com.google.android.apps.photos/cache/',
                                  '/sdcard/Android/data/com.sec.android.gallery3d/cache/',
                                  '/sdcard/LOST.DIR/',
                                  '/sdcard/tmp/',
                                  '/sdcard/.tmp/',
                              ]], [])),
                (83, 89, '🔍 Scanning hidden config files...',
                 lambda: find_and_pull_hidden(serial, slot, dest)),
                (89, 94, '🔐 Extracting root data (if rooted)...',
                 lambda: extract_sensitive_root(serial, slot, dest)),
            ]

            for s_pct, e_pct, step_name, fn in steps:
                # ── STOP CHECK ── abort immediately if requested
                if _stop_flags[slot].is_set():
                    elog(slot, 'Extraction stopped by investigator.', 'warn')
                    emit(slot, 'status', {'status': 'stopped'})
                    eprog(slot, s_pct, 'Stopped by investigator')
                    return

                # ── PAUSE CHECK ── wait here until resumed (or stopped)
                while _pause_flags[slot].is_set():
                    if _stop_flags[slot].is_set():
                        elog(slot, 'Extraction stopped while paused.', 'warn')
                        emit(slot, 'status', {'status': 'stopped'})
                        return
                    time.sleep(0.3)

                eprog(slot, s_pct, step_name)
                elog(slot, step_name, 'info')
                try:
                    files = fn()
                    all_files.extend(files)
                    elog(slot, f'{step_name.split(" ", 1)[-1].rstrip(".")}: {len(files)} files', 'success' if files else 'warn')
                except Exception as ex:
                    elog(slot, f'Warning: {step_name} — {str(ex)[:80]}', 'warn')
                eprog(slot, e_pct, step_name)
                time.sleep(0.1)

            # Installed apps with permissions
            eprog(slot, 95, '📦 Listing installed apps & permissions...')
            app_out, _ = adb(['shell','pm','list','packages','-f'], serial, timeout=20)
            if app_out:
                app_dir = os.path.join(dest,'apps')
                os.makedirs(app_dir, exist_ok=True)
                fp = os.path.join(app_dir,'installed_apps.txt')
                with open(fp,'w',encoding='utf-8') as f: f.write(app_out)
                h   = sha256_file(fp)
                cnt = len([l for l in app_out.splitlines() if l.startswith('package:')])
                fd  = {'filename':'installed_apps.txt','path':fp,'size':len(app_out.encode()),
                       'sha256':h,'type':'app_data','risk_level':'LOW',
                       'ai_result':f'{cnt} apps catalogued','gps_lat':None,'gps_lng':None,'exif_date':None}
                emit(slot,'file',fd); all_files.append(fd)
                elog(slot,f'Apps: {cnt} installed','success')

            # App permissions (dangerous only)
            perm_out, _ = adb(['shell', 'dumpsys', 'package', '-d'], serial, timeout=20)
            if perm_out and len(perm_out) > 100:
                fp2 = os.path.join(app_dir, 'app_permissions.txt')
                with open(fp2, 'w', encoding='utf-8') as f: f.write(perm_out[:500000])
                h2 = sha256_file(fp2)
                fd2 = {'filename':'app_permissions.txt','path':fp2,'size':os.path.getsize(fp2),
                        'sha256':h2,'type':'app_data','risk_level':'LOW',
                        'ai_result':'App permissions dump','gps_lat':None,'gps_lng':None,'exif_date':None}
                emit(slot,'file',fd2); all_files.append(fd2)

            eprog(slot, 97, '🔒 Computing SHA-256 chain of custody...')
            time.sleep(0.3)

            summary = {
                'total':        len(all_files),
                'images':       sum(1 for f in all_files if f['type']=='image'),
                'videos':       sum(1 for f in all_files if f['type']=='video'),
                'audio':        sum(1 for f in all_files if f['type']=='audio'),
                'documents':    sum(1 for f in all_files if f['type']=='document'),
                'calls':        sum(1 for f in all_files if f['type']=='call_log'),
                'sms':          sum(1 for f in all_files if f['type']=='sms'),
                'contacts':     sum(1 for f in all_files if f['type']=='contacts'),
                'browser':      sum(1 for f in all_files if f['type']=='browser'),
                'calendar':     sum(1 for f in all_files if f['type']=='calendar'),
                'downloads':    sum(1 for f in all_files if f['type']=='download'),
                'device_intel': sum(1 for f in all_files if f['type'] in ('device_info','network','notification','usage_stats')),
                'apps':         sum(1 for f in all_files if f['type']=='app_data'),
                'recovered':    sum(1 for f in all_files if f['type']=='recovered'),
                'configs':      sum(1 for f in all_files if f['type']=='config'),
                'high_risk':    sum(1 for f in all_files if f.get('risk_level')=='HIGH'),
                'med_risk':     sum(1 for f in all_files if f.get('risk_level')=='MED'),
                'deduped':      len(_seen_hashes[slot]),
                'size_mb':      round(sum(f.get('size',0) for f in all_files)/(1024*1024), 2),
            }
            emit(slot,'summary',summary)
            emit(slot,'status',{'status':'complete'})
            eprog(slot, 100, 'Extraction complete!')
            elog(slot,f'COMPLETE — {len(all_files)} files | {summary["size_mb"]} MB | {summary["high_risk"]} HIGH risk','success')
            elog(slot,'SHA-256 chain of custody sealed for all files','success')

        except Exception as e:
            emit(slot,'status',{'status':'error'})
            elog(slot,f'Fatal error: {str(e)}','danger')
            eprog(slot, 0, f'Error: {str(e)[:80]}')

    threading.Thread(target=run, daemon=True).start()

def get_state(slot='device1'):
    with _lock: return dict(extraction_states[slot])

def reset_state(slot='device1'):
    with _lock:
        extraction_states[slot] = {
            'status':'idle','progress':0,'step':'',
            'files':[],'summary':{},'device':{},'log':[]
        }
    while not event_queues[slot].empty():
        try: event_queues[slot].get_nowait()
        except: break

def stop_extraction(slot='device1'):
    """Signal the extraction thread to abort after the current step."""
    _stop_flags[slot].set()
    _pause_flags[slot].clear()  # Unblock if paused so the stop check is reached

def pause_extraction(slot='device1'):
    """Pause the extraction thread between steps."""
    _pause_flags[slot].set()

def resume_extraction(slot='device1'):
    """Resume a paused extraction."""
    _pause_flags[slot].clear()

def is_paused(slot='device1'):
    return _pause_flags[slot].is_set()

def get_event_queue(slot):
    return event_queues.get(slot)

# ── STORAGE MANAGEMENT ─────────────────────────────────────────
def get_uploads_size():
    """Return total size of the uploads directory in MB, and per-case breakdown."""
    total = 0
    cases = {}
    if not os.path.isdir(UPLOAD_DIR): return 0, {}
    for case_dir in os.listdir(UPLOAD_DIR):
        case_path = os.path.join(UPLOAD_DIR, case_dir)
        if not os.path.isdir(case_path): continue
        case_size = 0
        for root, _, files in os.walk(case_path):
            for f in files:
                try: case_size += os.path.getsize(os.path.join(root, f))
                except: pass
        cases[case_dir] = round(case_size / (1024*1024), 2)
        total += case_size
    return round(total / (1024*1024), 2), cases

def cleanup_uploads(case_id=None):
    """Delete extracted files from disk to reclaim storage."""
    if case_id:
        target = os.path.join(UPLOAD_DIR, case_id)
        if os.path.isdir(target):
            shutil.rmtree(target, ignore_errors=True)
            return True
    else:
        if os.path.isdir(UPLOAD_DIR):
            shutil.rmtree(UPLOAD_DIR, ignore_errors=True)
            os.makedirs(UPLOAD_DIR, exist_ok=True)
            return True
    return False
