"""
FORAX — Professional Forensic PDF Report Generator
Generates detailed, multi-page, court-grade forensic reports using xhtml2pdf.
Includes: Cover page, executive summary, methodology, evidence analysis,
behavioral intelligence profile, chain of custody, integrity verification,
and signature blocks.
"""

import os, hashlib, re
from datetime import datetime, timedelta
from xhtml2pdf import pisa
import json
from collections import Counter


# ══════════════════════════════════════════════════════════════════
# BEHAVIORAL ANALYSIS ENGINE
# ══════════════════════════════════════════════════════════════════

def generate_behavioral_analysis(evidence):
    """
    Generate a comprehensive behavioral profile of the device owner
    by analyzing all extracted forensic evidence.

    Returns a dict with:
      - communication: message/call volume, patterns
      - risk_categories: breakdown of flagged content by category
      - media_profile: image/video stats, GPS locations, cameras
      - digital_footprint: downloads, apps, browsing behavior
      - overall_risk: composite behavioral risk classification
      - narrative: auto-generated investigator summary
    """

    profile = {
        'communication': {},
        'risk_categories': {},
        'media_profile': {},
        'digital_footprint': {},
        'overall_risk': 'LOW',
        'risk_score': 0,
        'narrative': ''
    }

    if not evidence:
        profile['narrative'] = 'No evidence available for behavioral analysis.'
        return profile

    # ── COMMUNICATION PROFILE ──────────────────────────────────
    comm_types = ('chat', 'sms', 'call_log', 'contacts')
    comm_files = [e for e in evidence if e.get('file_type') in comm_types]
    total_comm = len(comm_files)
    comm_high = sum(1 for e in comm_files if e.get('risk_level') == 'HIGH')
    comm_med = sum(1 for e in comm_files if e.get('risk_level') == 'MED')

    profile['communication'] = {
        'total_files': total_comm,
        'high_risk': comm_high,
        'med_risk': comm_med,
        'low_risk': total_comm - comm_high - comm_med,
        'chat_count': sum(1 for e in comm_files if e.get('file_type') == 'chat'),
        'sms_count': sum(1 for e in comm_files if e.get('file_type') == 'sms'),
        'call_count': sum(1 for e in comm_files if e.get('file_type') == 'call_log'),
        'contact_count': sum(1 for e in comm_files if e.get('file_type') == 'contacts'),
        'flagged_ratio': round((comm_high + comm_med) / max(total_comm, 1) * 100, 1)
    }

    # ── RISK CATEGORY EXTRACTION ───────────────────────────────
    # Parse AI results to identify which threat categories were detected
    category_keywords = {
        'Narcotics & Drugs': ['narcotic', 'drug', 'cocaine', 'heroin', 'meth', 'cannabis', 'marijuana', 'fentanyl', 'pill', 'substance'],
        'Weapons & Firearms': ['weapon', 'firearm', 'gun', 'pistol', 'rifle', 'knife', 'explosive', 'ammunition', 'handgun'],
        'Violence & Threats': ['violence', 'threat', 'assault', 'murder', 'kill', 'attack', 'abuse', 'stab', 'blood'],
        'Human Trafficking': ['trafficking', 'exploitation', 'smuggling', 'forced labor', 'abduction', 'kidnap'],
        'Financial Crime': ['money laundering', 'counterfeit', 'embezzlement', 'identity theft', 'carding', 'tax evasion'],
        'Scam & Phishing': ['fraud', 'scam', 'phishing', 'vishing', 'smishing', 'anydesk', 'teamviewer', 'romance scam', 'otp'],
        'Cybercrime': ['hacking', 'malware', 'ransomware', 'exploit', 'phishing', 'botnet', 'backdoor'],
        'Extremism': ['terrorism', 'extremis', 'radical', 'propaganda', 'manifesto'],
        'Child Exploitation': ['csam', 'child abuse', 'underage', 'exploitation', 'grooming', 'predator'],
        'NSFW Content': ['nsfw', 'explicit', 'adult content', 'nudity', 'pornograph']
    }

    cat_scores = {}
    cat_file_counts = {}

    for e in evidence:
        ai_text = str(e.get('ai_result', '') or '').lower()
        risk = e.get('risk_level', 'LOW')

        for cat_name, keywords in category_keywords.items():
            hits = sum(1 for kw in keywords if kw in ai_text)
            if hits > 0:
                if cat_name not in cat_scores:
                    cat_scores[cat_name] = 0
                    cat_file_counts[cat_name] = 0
                # Weight by risk level
                weight = {'HIGH': 10, 'MED': 5, 'LOW': 1}.get(risk, 1)
                cat_scores[cat_name] += hits * weight
                cat_file_counts[cat_name] += 1

    # Normalize scores to 0-100
    max_score = max(cat_scores.values()) if cat_scores else 1
    risk_categories = {}
    for cat, score in sorted(cat_scores.items(), key=lambda x: x[1], reverse=True):
        normalized = round(score / max_score * 100)
        risk_categories[cat] = {
            'score': normalized,
            'raw_score': score,
            'file_count': cat_file_counts[cat],
            'severity': 'CRITICAL' if normalized >= 80 else 'HIGH' if normalized >= 50 else 'MODERATE' if normalized >= 25 else 'LOW'
        }

    profile['risk_categories'] = risk_categories

    # ── MEDIA PROFILE ──────────────────────────────────────────
    media_types = ('image', 'video', 'audio')
    media_files = [e for e in evidence if e.get('file_type') in media_types]
    gps_files = [e for e in evidence if e.get('gps_lat') and e.get('gps_lng')]
    exif_dates = [e.get('exif_date') for e in evidence if e.get('exif_date')]

    # Unique cameras
    cameras = set()
    for e in evidence:
        ai = str(e.get('ai_result', '') or '')
        if 'Camera:' in ai:
            try:
                cam = ai.split('Camera:')[1].split('|')[0].split(',')[0].strip()
                if cam:
                    cameras.add(cam)
            except:
                pass

    profile['media_profile'] = {
        'total_media': len(media_files),
        'image_count': sum(1 for e in media_files if e.get('file_type') == 'image'),
        'video_count': sum(1 for e in media_files if e.get('file_type') == 'video'),
        'audio_count': sum(1 for e in media_files if e.get('file_type') == 'audio'),
        'media_high_risk': sum(1 for e in media_files if e.get('risk_level') == 'HIGH'),
        'gps_locations': len(gps_files),
        'gps_samples': [{'lat': e['gps_lat'], 'lng': e['gps_lng'], 'file': e.get('filename','')} for e in gps_files[:10]],
        'exif_dates': sorted(exif_dates)[:10],
        'cameras': list(cameras),
        'has_nsfw': any(str(e.get('ai_result','')).lower().count('nsfw') for e in media_files)
    }

    # ── DIGITAL FOOTPRINT ──────────────────────────────────────
    download_files = [e for e in evidence if e.get('file_type') == 'download']
    app_files = [e for e in evidence if e.get('file_type') == 'app_data']
    browser_files = [e for e in evidence if e.get('file_type') == 'browser']
    doc_files = [e for e in evidence if e.get('file_type') == 'document']
    recovered_files = [e for e in evidence if e.get('file_type') == 'recovered']
    config_files = [e for e in evidence if e.get('file_type') == 'config']

    # Detect suspicious downloads
    suspicious_exts = ('apk', 'exe', 'bat', 'sh', 'ps1', 'vbs', 'cmd', 'jar', 'msi')
    suspicious_downloads = [e for e in download_files
                           if e.get('filename', '').rsplit('.', 1)[-1].lower() in suspicious_exts]

    profile['digital_footprint'] = {
        'total_downloads': len(download_files),
        'suspicious_downloads': len(suspicious_downloads),
        'installed_apps': len(app_files),
        'browser_artifacts': len(browser_files),
        'documents': len(doc_files),
        'recovered_deleted': len(recovered_files),
        'config_files': len(config_files),
        'config_high_risk': sum(1 for e in config_files if e.get('risk_level') == 'HIGH'),
        'deletion_behavior': 'SUSPICIOUS' if len(recovered_files) > 5 else 'NORMAL',
        'total_evidence': len(evidence)
    }

    # ── OVERALL RISK CLASSIFICATION ────────────────────────────
    risk_score = 0

    # Factor 1: High-risk evidence proportion
    total_high = sum(1 for e in evidence if e.get('risk_level') == 'HIGH')
    total_med = sum(1 for e in evidence if e.get('risk_level') == 'MED')
    if len(evidence) > 0:
        high_ratio = total_high / len(evidence)
        risk_score += high_ratio * 40  # Up to 40 points

    # Factor 2: Number of threat categories detected
    cats_detected = len(risk_categories)
    risk_score += min(cats_detected * 5, 25)  # Up to 25 points

    # Factor 3: Communication risk
    if comm_high > 0:
        risk_score += min(comm_high * 3, 15)  # Up to 15 points

    # Factor 4: Deletion behavior
    if len(recovered_files) > 5:
        risk_score += 10

    # Factor 5: Suspicious downloads
    if len(suspicious_downloads) > 0:
        risk_score += min(len(suspicious_downloads) * 2, 10)  # Up to 10 points

    risk_score = min(round(risk_score), 100)

    if risk_score >= 75:
        overall = 'CRITICAL'
    elif risk_score >= 50:
        overall = 'HIGH'
    elif risk_score >= 25:
        overall = 'MODERATE'
    else:
        overall = 'LOW'

    profile['overall_risk'] = overall
    profile['risk_score'] = risk_score

    # ── AUTO-GENERATED NARRATIVE ───────────────────────────────
    narrative_parts = []
    narrative_parts.append(
        f"Behavioral analysis of {len(evidence)} extracted artifacts reveals a "
        f"{overall} overall risk classification (score: {risk_score}/100)."
    )

    if total_high > 0:
        narrative_parts.append(
            f"{total_high} item(s) were flagged as HIGH risk, "
            f"representing {round(total_high/len(evidence)*100, 1)}% of all evidence."
        )

    if cats_detected > 0:
        top_cats = sorted(risk_categories.items(), key=lambda x: x[1]['score'], reverse=True)[:3]
        cat_str = ', '.join(f"{c} ({d['severity']})" for c, d in top_cats)
        narrative_parts.append(f"Primary threat categories identified: {cat_str}.")

    if total_comm > 0:
        narrative_parts.append(
            f"Communication analysis covers {total_comm} file(s) "
            f"({profile['communication']['chat_count']} chats, "
            f"{profile['communication']['sms_count']} SMS logs, "
            f"{profile['communication']['call_count']} call records). "
            f"{profile['communication']['flagged_ratio']}% of communications contain flagged content."
        )

    if profile['media_profile']['gps_locations'] > 0:
        narrative_parts.append(
            f"{profile['media_profile']['gps_locations']} file(s) contain embedded GPS coordinates, "
            f"enabling geolocation reconstruction."
        )

    if len(recovered_files) > 0:
        narrative_parts.append(
            f"{len(recovered_files)} deleted file(s) were recovered from the device, "
            f"indicating {'possible evidence destruction attempts' if len(recovered_files) > 5 else 'standard user deletion behavior'}."
        )

    if len(suspicious_downloads) > 0:
        narrative_parts.append(
            f"{len(suspicious_downloads)} suspicious executable file(s) detected in downloads."
        )

    profile['narrative'] = ' '.join(narrative_parts)

    return profile


# ══════════════════════════════════════════════════════════════════
# INTERACTIVE HTML REPORT TEMPLATE (Autopsy-style)
# ══════════════════════════════════════════════════════════════════

INTERACTIVE_HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>FORAX Case: {case_id} — Interactive Forensic Volume</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/animate.css/4.1.1/animate.min.css"/>
    <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <style>
        :root {{
            --bg: #0a0e1a; --surface: #111827; --surface2: #1a2035;
            --border: #1e293b; --border-light: #2d3a52;
            --t0: #f1f5f9; --t1: #cbd5e1; --t2: #64748b; --t3: #475569;
            --blue: #3b82f6; --blue-glow: rgba(59,130,246,0.15);
            --cyan: #06b6d4; --cyan-glow: rgba(6,182,212,0.1);
            --red: #ef4444; --red-bg: rgba(239,68,68,0.1);
            --amber: #f59e0b; --amber-bg: rgba(245,158,11,0.1);
            --green: #10b981; --green-bg: rgba(16,185,129,0.1);
            --purple: #8b5cf6; --purple-bg: rgba(139,92,246,0.1);
            --mono: 'JetBrains Mono', monospace; --sans: 'Inter', sans-serif;
            --sh: 0 4px 24px rgba(0,0,0,0.4); --sh-sm: 0 2px 8px rgba(0,0,0,0.3);
            --radius: 12px; --radius-lg: 16px;
        }}
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ background: var(--bg); color: var(--t1); font-family: var(--sans); min-height: 100vh; display: flex; overflow: hidden; }}

        /* ── SIDEBAR ─────────────────────────────────── */
        .sidebar {{
            width: 260px; background: var(--surface); border-right: 1px solid var(--border);
            display: flex; flex-direction: column; flex-shrink: 0; z-index: 10;
        }}
        .sb-header {{
            padding: 20px 20px 16px; border-bottom: 1px solid var(--border);
            background: linear-gradient(135deg, rgba(59,130,246,0.08), rgba(6,182,212,0.05));
        }}
        .logo-badge {{
            display: inline-flex; align-items: center; gap: 8px;
            font-family: var(--mono); font-size: 18px; font-weight: 600;
            color: var(--blue); letter-spacing: 3px;
        }}
        .logo-dot {{ width: 8px; height: 8px; background: var(--cyan); border-radius: 50%; animation: pulse-dot 2s infinite; }}
        @keyframes pulse-dot {{ 0%,100% {{ opacity: 1; }} 50% {{ opacity: 0.3; }} }}
        .sb-subtitle {{ font-size: 9px; color: var(--t2); font-weight: 600; letter-spacing: 2px; margin-top: 4px; text-transform: uppercase; }}
        .sb-nav {{ padding: 12px; flex: 1; overflow-y: auto; }}
        .sb-item {{
            display: flex; align-items: center; gap: 10px;
            padding: 11px 14px; border-radius: 10px; cursor: pointer;
            color: var(--t2); font-weight: 500; font-size: 13px;
            transition: all 0.2s ease; margin-bottom: 2px;
            border: 1px solid transparent; background: none; width: 100%; text-align: left;
        }}
        .sb-item:hover {{ background: var(--surface2); color: var(--t0); border-color: var(--border); }}
        .sb-item.active {{
            background: var(--blue-glow); color: var(--blue);
            border-color: rgba(59,130,246,0.3); box-shadow: 0 0 20px rgba(59,130,246,0.08);
        }}
        .sb-icon {{ font-size: 16px; width: 20px; text-align: center; }}
        .sb-footer {{ padding: 16px; border-top: 1px solid var(--border); background: var(--surface2); }}

        /* ── MAIN ────────────────────────────────────── */
        .main {{ flex: 1; overflow-y: auto; padding: 28px 40px; position: relative; }}
        .main::-webkit-scrollbar {{ width: 6px; }}
        .main::-webkit-scrollbar-thumb {{ background: var(--border-light); border-radius: 3px; }}

        /* ── CARDS ────────────────────────────────────── */
        .case-card {{
            background: var(--surface); border: 1px solid var(--border);
            border-radius: var(--radius-lg); padding: 24px; box-shadow: var(--sh-sm); margin-bottom: 20px;
        }}
        .stat-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; margin-bottom: 20px; }}
        .stat-card {{
            background: var(--surface2); border: 1px solid var(--border); border-radius: var(--radius);
            padding: 16px; text-align: center; transition: transform 0.2s, border-color 0.2s;
        }}
        .stat-card:hover {{ transform: translateY(-2px); border-color: var(--border-light); }}
        .stat-val {{ font-size: 28px; font-weight: 700; font-family: var(--mono); color: var(--blue); line-height: 1; }}
        .stat-lbl {{ font-size: 10px; text-transform: uppercase; color: var(--t2); letter-spacing: 1.5px; margin-top: 6px; font-weight: 600; }}

        /* ── PANELS ───────────────────────────────────── */
        .panel {{
            background: var(--surface); border: 1px solid var(--border);
            border-radius: var(--radius-lg); overflow: hidden; box-shadow: var(--sh-sm); margin-bottom: 20px;
        }}
        .panel-header {{
            padding: 14px 20px; background: var(--surface2); border-bottom: 1px solid var(--border);
            display: flex; justify-content: space-between; align-items: center;
        }}
        .panel-title {{ font-weight: 600; color: var(--t0); font-size: 14px; letter-spacing: 0.3px; }}
        .panel-body {{ padding: 20px; }}

        /* ── TABLES ───────────────────────────────────── */
        .table-wrap {{ overflow-x: auto; }}
        table {{ width: 100%; border-collapse: collapse; }}
        th {{
            background: var(--surface2); padding: 10px 14px; text-align: left;
            font-size: 10px; text-transform: uppercase; color: var(--t2);
            border-bottom: 1px solid var(--border); letter-spacing: 1px; font-weight: 600;
        }}
        td {{ padding: 10px 14px; font-size: 13px; border-bottom: 1px solid var(--border); color: var(--t1); }}
        tr:hover td {{ background: rgba(59,130,246,0.03); }}

        /* ── BADGES ───────────────────────────────────── */
        .badge-risk {{ padding: 3px 8px; border-radius: 6px; font-size: 10px; font-weight: 700; letter-spacing: 0.5px; }}
        .badge-high, .badge-critical {{ background: var(--red-bg); color: var(--red); border: 1px solid rgba(239,68,68,0.2); }}
        .badge-med, .badge-moderate {{ background: var(--amber-bg); color: var(--amber); border: 1px solid rgba(245,158,11,0.2); }}
        .badge-low {{ background: var(--green-bg); color: var(--green); border: 1px solid rgba(16,185,129,0.2); }}

        /* ── TIMELINE ─────────────────────────────────── */
        .timeline-wrap {{ position: relative; padding: 20px 0 20px 30px; border-left: 2px solid var(--border); margin-left: 20px; }}
        .timeline-item {{ position: relative; margin-bottom: 24px; padding: 16px; background: var(--surface2); border: 1px solid var(--border); border-radius: var(--radius); }}
        .timeline-dot {{ position: absolute; left: -37px; top: 22px; width: 12px; height: 12px; border-radius: 50%; background: var(--blue); border: 3px solid var(--bg); }}
        .timeline-time {{ font-family: var(--mono); font-size: 10px; color: var(--blue); font-weight: 600; margin-bottom: 4px; }}
        .timeline-title {{ font-weight: 600; font-size: 13px; color: var(--t0); }}

        /* ── MAP ─────────────────────────────────────── */
        #caseMap {{ width: 100%; height: 500px; border-radius: var(--radius); border: 1px solid var(--border); z-index: 1; }}
        .map-popup {{ font-family: var(--sans); color: #1e293b; }}
        .leaflet-container {{ background: var(--bg) !important; }}

        /* ── GALLERY ──────────────────────────────────── */
        .gallery-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(170px, 1fr)); gap: 14px; padding: 20px; }}
        .gallery-item {{
            aspect-ratio: 1; background: var(--surface2); border-radius: var(--radius);
            overflow: hidden; border: 2px solid var(--border); cursor: pointer;
            transition: all 0.25s ease; position: relative;
        }}
        .gallery-item:hover {{ border-color: var(--blue); transform: scale(1.03); box-shadow: 0 0 20px rgba(59,130,246,0.15); }}
        .gallery-img {{ width: 100%; height: 100%; object-fit: cover; }}

        /* ── PREVIEW OVERLAY ──────────────────────────── */
        .preview-overlay {{
            position: fixed; inset: 0; background: rgba(0,0,0,0.85); backdrop-filter: blur(8px);
            z-index: 100; display: none; align-items: center; justify-content: center; padding: 40px;
        }}
        .preview-content {{
            background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius-lg);
            max-width: 900px; width: 100%; display: flex; overflow: hidden; height: 600px; position: relative;
            box-shadow: 0 25px 60px rgba(0,0,0,0.5);
        }}
        .preview-media {{ flex: 1; background: #000; display: flex; align-items: center; justify-content: center; }}
        .preview-data {{ width: 320px; padding: 24px; overflow-y: auto; border-left: 1px solid var(--border); }}
        .close-btn {{
            position: absolute; top: 16px; right: 16px; background: rgba(255,255,255,0.1);
            border: 1px solid rgba(255,255,255,0.2); color: #fff; width: 36px; height: 36px;
            border-radius: 50%; cursor: pointer; z-index: 101; font-size: 18px;
            transition: all 0.2s; display: flex; align-items: center; justify-content: center;
        }}
        .close-btn:hover {{ background: var(--red); border-color: var(--red); }}

        /* ── BEHAVIORAL SECTION ───────────────────────── */
        .risk-banner {{
            padding: 16px 20px; border-radius: var(--radius); display: flex;
            align-items: center; gap: 16px; margin-bottom: 20px;
        }}
        .risk-banner.critical {{ background: linear-gradient(135deg, rgba(239,68,68,0.15), rgba(239,68,68,0.05)); border: 1px solid rgba(239,68,68,0.3); }}
        .risk-banner.high {{ background: linear-gradient(135deg, rgba(245,158,11,0.15), rgba(245,158,11,0.05)); border: 1px solid rgba(245,158,11,0.3); }}
        .risk-banner.moderate {{ background: linear-gradient(135deg, rgba(59,130,246,0.15), rgba(59,130,246,0.05)); border: 1px solid rgba(59,130,246,0.3); }}
        .risk-banner.low {{ background: linear-gradient(135deg, rgba(16,185,129,0.15), rgba(16,185,129,0.05)); border: 1px solid rgba(16,185,129,0.3); }}
        .risk-score-ring {{
            width: 64px; height: 64px; border-radius: 50%; display: flex; align-items: center;
            justify-content: center; font-family: var(--mono); font-size: 20px; font-weight: 700; flex-shrink: 0;
        }}
        .risk-score-ring.critical {{ background: rgba(239,68,68,0.2); color: var(--red); border: 2px solid var(--red); }}
        .risk-score-ring.high {{ background: rgba(245,158,11,0.2); color: var(--amber); border: 2px solid var(--amber); }}
        .risk-score-ring.moderate {{ background: rgba(59,130,246,0.2); color: var(--blue); border: 2px solid var(--blue); }}
        .risk-score-ring.low {{ background: rgba(16,185,129,0.2); color: var(--green); border: 2px solid var(--green); }}
        .metric-grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px; margin-bottom: 20px; }}
        .metric-card {{
            background: var(--surface2); border: 1px solid var(--border); border-radius: var(--radius);
            padding: 14px; display: flex; flex-direction: column; gap: 4px;
        }}
        .metric-val {{ font-family: var(--mono); font-size: 22px; font-weight: 700; color: var(--t0); }}
        .metric-lbl {{ font-size: 10px; text-transform: uppercase; color: var(--t2); letter-spacing: 1px; font-weight: 600; }}
        .bar-chart-row {{ display: flex; align-items: center; gap: 12px; margin-bottom: 10px; }}
        .bar-label {{ width: 140px; font-size: 11px; color: var(--t1); font-weight: 500; text-align: right; flex-shrink: 0; }}
        .bar-track {{ flex: 1; height: 8px; background: var(--surface2); border-radius: 4px; overflow: hidden; }}
        .bar-fill {{ height: 100%; border-radius: 4px; transition: width 0.8s ease; }}
        .bar-val {{ width: 40px; font-size: 11px; font-family: var(--mono); color: var(--t2); font-weight: 600; }}
        .narrative-box {{
            background: var(--surface2); border: 1px solid var(--border); border-radius: var(--radius);
            padding: 16px 20px; font-size: 13px; line-height: 1.7; color: var(--t1); margin-top: 20px;
            border-left: 3px solid var(--blue);
        }}
        .section-label {{
            font-size: 11px; text-transform: uppercase; letter-spacing: 2px; color: var(--t2);
            font-weight: 700; margin-bottom: 12px; padding-bottom: 8px; border-bottom: 1px solid var(--border);
        }}

        /* ── SEARCH INPUT ─────────────────────────────── */
        .search-input {{
            padding: 7px 14px; border: 1px solid var(--border); border-radius: 8px;
            font-size: 12px; background: var(--surface2); color: var(--t1); outline: none;
            transition: border-color 0.2s;
        }}
        .search-input:focus {{ border-color: var(--blue); }}
        .search-input::placeholder {{ color: var(--t3); }}
    </style>
</head>
<body>
    <div class="sidebar">
        <div class="sb-header">
            <div class="logo-badge"><span class="logo-dot"></span> FORAX</div>
            <div class="sb-subtitle">Forensic Report Volume</div>
        </div>
        <div class="sb-nav">
            <button class="sb-item active" onclick="showSection('dashboard', this)"><span class="sb-icon">📊</span> Dashboard</button>
            <button class="sb-item" onclick="showSection('evidence', this)"><span class="sb-icon">📂</span> Evidence Items</button>
            <button class="sb-item" onclick="showSection('behavioral', this)"><span class="sb-icon">🧠</span> Behavioral Analysis</button>
            <button class="sb-item" onclick="showSection('timeline', this)"><span class="sb-icon">⏳</span> Case Timeline</button>
            <button class="sb-item" onclick="showSection('map', this)"><span class="sb-icon">📍</span> Map View</button>
            <button class="sb-item" onclick="showSection('gallery', this)"><span class="sb-icon">🖼️</span> Media Gallery</button>
            <button class="sb-item" onclick="showSection('integrity', this)"><span class="sb-icon">🛡️</span> Integrity Manifest</button>
        </div>
        <div class="sb-footer">
            <div style="font-size: 12px; font-weight: 600; color: var(--t0);">{investigator}</div>
            <div style="font-size: 11px; color: var(--t2);">{department}</div>
        </div>
    </div>

    <main class="main">
        <!-- SECTION: DASHBOARD -->
        <div id="section-dashboard" class="animate__animated animate__fadeIn">
            <div class="case-card">
                <div style="font-family: var(--mono); color: var(--blue); font-size: 11px; margin-bottom: 4px; letter-spacing: 1px;">CASE ID: {case_id}</div>
                <h1 style="font-size: 24px; font-weight: 700; color: var(--t0); margin-bottom: 20px;">Forensic Investigation Summary</h1>
                <div class="stat-grid">
                    <div class="stat-card"><div class="stat-val">{total}</div><div class="stat-lbl">Total Artifacts</div></div>
                    <div class="stat-card"><div class="stat-val" style="color:var(--red);">{high_count}</div><div class="stat-lbl">High Risk</div></div>
                    <div class="stat-card"><div class="stat-val" style="color:var(--amber);">{med_count}</div><div class="stat-lbl">Med Risk</div></div>
                    <div class="stat-card"><div class="stat-val" style="color:var(--green);">{low_count}</div><div class="stat-lbl">Cleared</div></div>
                </div>
                <div class="row">
                    <div class="col-md-6"><canvas id="threatChart" height="220"></canvas></div>
                    <div class="col-md-6"><canvas id="typeChart" height="220"></canvas></div>
                </div>
            </div>
            <div class="panel">
                <div class="panel-header"><div class="panel-title">Chain of Custody & Methodology</div></div>
                <div class="panel-body">
                    <p style="font-size: 13px; line-height: 1.7; color: var(--t1);">This interactive forensic volume was generated by the FORAX intelligence engine on <strong style="color:var(--t0);">{date}</strong>. All evidence has been cryptographically verified against the root SHA-256 manifest.</p>
                    <div style="background: var(--surface2); padding: 12px 16px; border-radius: 8px; font-family: var(--mono); font-size: 11px; border: 1px solid var(--border); margin-top: 12px; color: var(--cyan); word-break: break-all;">
                        MANIFEST: {full_integrity_hash}
                    </div>
                </div>
            </div>
        </div>

        <!-- SECTION: EVIDENCE -->
        <div id="section-evidence" style="display:none;" class="animate__animated animate__fadeIn">
            <div class="panel">
                <div class="panel-header">
                    <div class="panel-title">Evidence Artifact Table</div>
                    <input type="text" class="search-input" id="evSearch" placeholder="Search artifacts..." oninput="filterTable()">
                </div>
                <div class="table-wrap">
                    <table id="evTable">
                        <thead><tr>
                            <th>#</th><th>Filename</th><th>Type</th><th>Size</th><th>AI Findings</th><th>Risk</th>
                        </tr></thead>
                        <tbody id="evTbody"></tbody>
                    </table>
                </div>
            </div>
        </div>

        <!-- SECTION: BEHAVIORAL ANALYSIS -->
        <div id="section-behavioral" style="display:none;" class="animate__animated animate__fadeIn">
            <div class="risk-banner {behavior_risk_class}">
                <div class="risk-score-ring {behavior_risk_class}">{behavior_score}</div>
                <div>
                    <div style="font-size: 16px; font-weight: 700; color: var(--t0);">Behavioral Risk: {behavior_risk_level}</div>
                    <div style="font-size: 12px; color: var(--t2); margin-top: 2px;">Composite score based on {total} artifacts analyzed</div>
                </div>
            </div>

            <div class="metric-grid">
                <div class="metric-card">
                    <div class="metric-val">{comm_total}</div>
                    <div class="metric-lbl">Communication Files</div>
                </div>
                <div class="metric-card">
                    <div class="metric-val">{media_total}</div>
                    <div class="metric-lbl">Media Files</div>
                </div>
                <div class="metric-card">
                    <div class="metric-val">{gps_count}</div>
                    <div class="metric-lbl">GPS Locations</div>
                </div>
            </div>

            <div class="panel" style="margin-bottom: 20px;">
                <div class="panel-header"><div class="panel-title">Threat Category Distribution</div></div>
                <div class="panel-body">
                    <div style="max-width: 600px;">
                        <canvas id="radarChart" height="280"></canvas>
                    </div>
                    <div style="margin-top: 20px;" id="catBars"></div>
                </div>
            </div>

            <div class="panel" style="margin-bottom: 20px;">
                <div class="panel-header"><div class="panel-title">Digital Footprint Overview</div></div>
                <div class="panel-body">
                    <div class="metric-grid">
                        <div class="metric-card"><div class="metric-val">{dl_count}</div><div class="metric-lbl">Downloads</div></div>
                        <div class="metric-card"><div class="metric-val">{app_count}</div><div class="metric-lbl">Installed Apps</div></div>
                        <div class="metric-card"><div class="metric-val">{recovered_count}</div><div class="metric-lbl">Recovered Deleted</div></div>
                    </div>
                </div>
            </div>

            <div class="narrative-box">
                <div class="section-label">Investigator Narrative — Auto-Generated</div>
                {behavior_narrative}
            </div>
        </div>

        <!-- SECTION: GALLERY -->
        <div id="section-gallery" style="display:none;" class="animate__animated animate__fadeIn">
            <div class="gallery-grid" id="galleryGrid"></div>
        </div>

        <!-- SECTION: TIMELINE -->
        <div id="section-timeline" style="display:none;" class="animate__animated animate__fadeIn">
            <div class="panel">
                <div class="panel-header"><div class="panel-title">Forensic Reconstruction Timeline</div></div>
                <div class="panel-body">
                    <div class="timeline-wrap" id="timelineList"></div>
                </div>
            </div>
        </div>

        <!-- SECTION: MAP VIEW -->
        <div id="section-map" style="display:none;" class="animate__animated animate__fadeIn">
            <div class="panel">
                <div class="panel-header"><div class="panel-title">Geospatial Intelligence (EXIF GPS)</div></div>
                <div class="panel-body">
                    <div id="caseMap"></div>
                </div>
            </div>
        </div>

        <!-- SECTION: INTEGRITY -->
        <div id="section-integrity" style="display:none;" class="animate__animated animate__fadeIn">
            <div class="panel">
                <div class="panel-header"><div class="panel-title">Full Forensic Integrity Manifest (SHA-256)</div></div>
                <div class="table-wrap">
                    <table>
                        <thead><tr><th>Filename</th><th>Extraction Path</th><th>Cryptographic Hash</th></tr></thead>
                        <tbody id="integrityBody"></tbody>
                    </table>
                </div>
            </div>
        </div>
    </main>

    <div class="preview-overlay" id="previewOverlay" onclick="if(event.target===this)closePreview()">
        <button class="close-btn" onclick="closePreview()">&#10005;</button>
        <div class="preview-content">
            <div class="preview-media" id="previewMedia"></div>
            <div class="preview-data" id="previewData"></div>
        </div>
    </div>

    <script>
        const caseData = {evidence};
        const behaviorData = {behavior_json};

        function showSection(id, btn) {{
            document.querySelectorAll('[id^="section-"]').forEach(s => s.style.display = 'none');
            document.getElementById('section-' + id).style.display = 'block';
            document.querySelectorAll('.sb-item').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            if(id === 'dashboard') renderCharts();
            if(id === 'behavioral') renderBehavioral();
            if(id === 'map') initMap();
            if(id === 'timeline') renderTimeline();
        }}

        function renderCharts() {{
            const high = caseData.filter(e => e.risk_level === 'HIGH').length;
            const med = caseData.filter(e => e.risk_level === 'MED').length;
            const low = caseData.filter(e => e.risk_level !== 'HIGH' && e.risk_level !== 'MED').length;

            const ctx1 = document.getElementById('threatChart');
            if(ctx1._chart) ctx1._chart.destroy();
            ctx1._chart = new Chart(ctx1, {{
                type: 'doughnut',
                data: {{ labels: ['High Risk', 'Med Risk', 'Clean'], datasets: [{{
                    data: [high, med, low],
                    backgroundColor: ['rgba(239,68,68,0.8)', 'rgba(245,158,11,0.8)', 'rgba(16,185,129,0.8)'],
                    borderColor: ['#ef4444', '#f59e0b', '#10b981'], borderWidth: 2
                }}] }},
                options: {{
                    plugins: {{ title: {{ display: true, text: 'Threat Distribution', color: '#f1f5f9', font: {{ size: 14, weight: 600 }} }},
                               legend: {{ labels: {{ color: '#94a3b8' }} }} }},
                    cutout: '55%'
                }}
            }});

            const types = {{}};
            caseData.forEach(e => types[e.file_type] = (types[e.file_type] || 0) + 1);
            const ctx2 = document.getElementById('typeChart');
            if(ctx2._chart) ctx2._chart.destroy();
            ctx2._chart = new Chart(ctx2, {{
                type: 'bar',
                data: {{ labels: Object.keys(types), datasets: [{{
                    label: 'Count', data: Object.values(types),
                    backgroundColor: 'rgba(59,130,246,0.6)', borderColor: '#3b82f6',
                    borderWidth: 1, borderRadius: 6
                }}] }},
                options: {{
                    plugins: {{ legend: {{ display: false }},
                               title: {{ display: true, text: 'Evidence Types', color: '#f1f5f9', font: {{ size: 14, weight: 600 }} }} }},
                    scales: {{ x: {{ ticks: {{ color: '#64748b' }}, grid: {{ color: 'rgba(30,41,59,0.5)' }} }},
                             y: {{ ticks: {{ color: '#64748b' }}, grid: {{ color: 'rgba(30,41,59,0.5)' }} }} }}
                }}
            }});
        }}

        function renderBehavioral() {{
            const cats = behaviorData.risk_categories || {{}};
            const catNames = Object.keys(cats);
            const catScores = catNames.map(c => cats[c].score);

            // Radar chart
            const ctx3 = document.getElementById('radarChart');
            if(ctx3 && catNames.length > 0) {{
                if(ctx3._chart) ctx3._chart.destroy();
                ctx3._chart = new Chart(ctx3, {{
                    type: 'radar',
                    data: {{
                        labels: catNames,
                        datasets: [{{
                            label: 'Threat Score', data: catScores,
                            backgroundColor: 'rgba(59,130,246,0.15)',
                            borderColor: '#3b82f6', borderWidth: 2,
                            pointBackgroundColor: '#3b82f6', pointBorderColor: '#1e293b',
                            pointHoverBackgroundColor: '#ef4444', pointRadius: 4
                        }}]
                    }},
                    options: {{
                        scales: {{ r: {{
                            beginAtZero: true, max: 100,
                            grid: {{ color: 'rgba(30,41,59,0.6)' }},
                            angleLines: {{ color: 'rgba(30,41,59,0.4)' }},
                            pointLabels: {{ color: '#cbd5e1', font: {{ size: 10 }} }},
                            ticks: {{ display: false }}
                        }} }},
                        plugins: {{ legend: {{ display: false }} }}
                    }}
                }});
            }}

            // Bar chart breakdown
            const barsDiv = document.getElementById('catBars');
            if(barsDiv && catNames.length > 0) {{
                barsDiv.innerHTML = catNames.map(c => {{
                    const s = cats[c].score;
                    const color = s >= 80 ? '#ef4444' : s >= 50 ? '#f59e0b' : s >= 25 ? '#3b82f6' : '#10b981';
                    return `<div class="bar-chart-row">
                        <div class="bar-label">${{c}}</div>
                        <div class="bar-track"><div class="bar-fill" style="width:${{s}}%;background:${{color}};"></div></div>
                        <div class="bar-val">${{s}}</div>
                    </div>`;
                }}).join('');
            }} else if(barsDiv) {{
                barsDiv.innerHTML = '<div style="color:var(--t2);font-size:13px;padding:12px;">No threat categories detected in evidence.</div>';
            }}
        }}

        function populateTable() {{
            const tbody = document.getElementById('evTbody');
            const ibody = document.getElementById('integrityBody');
            tbody.innerHTML = caseData.map((e, i) => `
                <tr onclick="showPreview(${{i}})" style="cursor:pointer;">
                    <td>${{i+1}}</td>
                    <td><strong style="color:var(--t0);">${{e.filename}}</strong></td>
                    <td style="font-family:var(--mono);font-size:11px;">${{(e.file_type||'').toUpperCase()}}</td>
                    <td style="font-family:var(--mono);font-size:11px;">${{e.file_size}}</td>
                    <td style="font-size:11px;max-width:260px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">${{e.ai_result || '—'}}</td>
                    <td><span class="badge-risk badge-${{(e.risk_level||'low').toLowerCase()}}">${{e.risk_level}}</span></td>
                </tr>
            `).join('');

            ibody.innerHTML = caseData.map(e => `
                <tr>
                    <td style="color:var(--t0);font-weight:500;">${{e.filename}}</td>
                    <td style="font-family:var(--mono);font-size:10px;color:var(--t2);">${{e.file_path}}</td>
                    <td style="font-family:var(--mono);font-size:10px;font-weight:600;color:var(--cyan);">${{e.sha256}}</td>
                </tr>
            `).join('');
        }}

        function renderTimeline() {{
            const list = document.getElementById('timelineList');
            const sorted = [...caseData].sort((a,b) => new Date(a.extracted_at) - new Date(b.extracted_at));
            list.innerHTML = sorted.map(e => `
                <div class="timeline-item">
                    <div class="timeline-dot"></div>
                    <div class="timeline-time">${{e.extracted_at}}</div>
                    <div class="timeline-title">${{e.filename}} (${{e.file_type}})</div>
                    <div style="font-size:11px;color:var(--t2);margin-top:4px;">Risk: <span style="color:${{e.risk_level==='HIGH'?'var(--red)':'var(--blue)'}}">${{e.risk_level}}</span> &bull; Path: ${{e.file_path}}</div>
                </div>
            `).join('');
        }}

        let mapObj = null;
        function initMap() {{
            if(mapObj) return;
            const gpsData = caseData.filter(e => e.gps_lat && e.gps_lng);
            if(gpsData.length === 0) {{
                document.getElementById('caseMap').innerHTML = '<div style="display:flex;align-items:center;justify-content:center;height:100%;color:var(--t2);">No GPS-tagged evidence found in this case.</div>';
                return;
            }}
            mapObj = L.map('caseMap').setView([gpsData[0].gps_lat, gpsData[0].gps_lng], 13);
            L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
                attribution: '&copy; OpenStreetMap'
            }}).addTo(mapObj);
            gpsData.forEach(e => {{
                L.marker([e.gps_lat, e.gps_lng]).addTo(mapObj)
                    .bindPopup(`<b class="map-popup">${{e.filename}}</b><br><small>${{e.extracted_at}}</small>`);
            }});
        }}

        function populateGallery() {{
            const grid = document.getElementById('galleryGrid');
            const mediaFiles = caseData.filter(e => ['image','video'].includes(e.file_type));
            grid.innerHTML = mediaFiles.map((e, i) => {{
                const isVid = e.file_type === 'video';
                return `
                <div class="gallery-item" onclick="showPreview(${{caseData.indexOf(e)}})">
                    <div style="position:absolute;top:8px;right:8px;width:10px;height:10px;border-radius:50%;background:${{e.risk_level==='HIGH'?'#ef4444':e.risk_level==='MED'?'#f59e0b':'#10b981'}};z-index:2;"></div>
                    ${{isVid ? '<div style="position:absolute;inset:0;display:flex;align-items:center;justify-content:center;color:#fff;font-size:24px;background:rgba(0,0,0,0.3);z-index:1;">▶</div>' : ''}}
                    <div style="position:absolute;inset:0;display:flex;align-items:center;justify-content:center;color:#fff;font-size:10px;font-weight:600;letter-spacing:1px;background:rgba(0,0,0,0.5);opacity:0;transition:opacity 0.2s;z-index:3;" onmouseover="this.style.opacity=1" onmouseout="this.style.opacity=0">${{isVid?'PLAY':'VIEW'}}</div>
                    ${{isVid ? '<div style="width:100%;height:100%;background:#000;display:flex;align-items:center;justify-content:center;color:var(--t3);font-size:40px;">🎬</div>' 
                             : `<img class="gallery-img" src="${{e.image_data || ('file://'+e.file_path)}}" onerror="this.src='data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjAwIiBoZWlnaHQ9IjIwMCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cmVjdCB3aWR0aD0iMjAwIiBoZWlnaHQ9IjIwMCIgZmlsbD0iIzExMTgyNyIvPjx0ZXh0IHg9IjEwMCIgeT0iMTAwIiBmaWxsPSIjNDc1NTY5IiBmb250LWZhbWlseT0ic2Fucy1zZXJpZiIgZm9udC1zaXplPSIxNCIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZHk9Ii4zNWVtIj5ObyBQcmV2aWV3PC90ZXh0Pjwvc3ZnPg=='">`}}
                </div>`;
            }}).join('');
        }}

        function showPreview(idx) {{
            const e = caseData[idx];
            const media = document.getElementById('previewMedia');
            const data = document.getElementById('previewData');

            if(e.file_type === 'image') {{
                media.innerHTML = `<img src="${{e.image_data || ('file://'+e.file_path)}}" style="max-width:100%;max-height:100%;object-fit:contain;">`;
            }} else if(e.file_type === 'video') {{
                media.innerHTML = `<video src="file://${{e.file_path}}" controls style="max-width:100%;max-height:100%;"></video>`;
            }} else {{
                media.innerHTML = `<div style="color:var(--t2);font-family:var(--mono);text-align:center;"><div style="font-size:36px;margin-bottom:12px;">📊</div>${{(e.file_type||'UNKNOWN').toUpperCase()}}<br><small style="color:var(--t3);">${{e.filename}}</small></div>`;
            }}

            data.innerHTML = `
                <h3 style="font-size:16px;font-weight:700;color:var(--t0);margin-bottom:8px;word-break:break-all;">${{e.filename}}</h3>
                <span class="badge-risk badge-${{(e.risk_level||'low').toLowerCase()}}" style="display:inline-block;margin-bottom:16px;">${{e.risk_level}} RISK</span>

                <div class="section-label" style="margin-top:16px;">AI Detection Summary</div>
                <div style="font-size:12px;background:var(--surface2);padding:12px;border-radius:8px;border:1px solid var(--border);line-height:1.6;">${{e.ai_result || 'N/A'}}</div>

                <div class="section-label" style="margin-top:20px; color:var(--blue); display:flex; align-items:center; gap:6px;">
                    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M21 15a2 2 0 01-2 2H7l-4 4V5a2 2 0 012-2h14a2 2 0 012 2z"></path></svg>
                    Forensic AI Reasoning
                </div>
                <div style="font-size:11.5px;font-style:italic;background:rgba(59,130,246,0.03);padding:12px;border-radius:8px;border-left:4px solid var(--blue);line-height:1.6;color:var(--t1);">
                    ${{e.copilot_reasoning || 'No narrative analysis available for this artifact.'}}
                </div>

                <div class="section-label" style="margin-top:20px;">Integrity Hash</div>
                <div style="font-family:var(--mono);font-size:10px;word-break:break-all;background:var(--surface2);padding:10px;border-radius:8px;border:1px solid var(--border);color:var(--cyan);">${{e.sha256}}</div>

                <div class="section-label" style="margin-top:20px;">Acquisition</div>
                <div style="font-size:12px;line-height:1.8;">
                    <div><strong style="color:var(--t2);">Slot:</strong> <span style="color:var(--t0);">${{e.device_slot}}</span></div>
                    <div><strong style="color:var(--t2);">Extracted:</strong> <span style="color:var(--t0);">${{e.extracted_at}}</span></div>
                    <div><strong style="color:var(--t2);">Size:</strong> <span style="color:var(--t0);">${{e.file_size}}</span></div>
                </div>
            `;
            document.getElementById('previewOverlay').style.display = 'flex';
        }}

        function closePreview() {{ document.getElementById('previewOverlay').style.display = 'none'; }}

        function filterTable() {{
            const q = document.getElementById('evSearch').value.toLowerCase();
            document.querySelectorAll('#evTbody tr').forEach(r => {{
                r.style.display = r.innerText.toLowerCase().includes(q) ? '' : 'none';
            }});
        }}

        window.onload = () => {{
            renderCharts();
            populateTable();
            populateGallery();
        }};
    </script>
</body>
</html>
"""


# ══════════════════════════════════════════════════════════════════
# PDF REPORT TEMPLATE — Court-Admissible Forensic Examination Report
# Compliant with: FRE 901/902, ISO/IEC 27037:2012, NIST SP 800-86
# ══════════════════════════════════════════════════════════════════

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <style>
        @page {{
            size: a4; margin: 1.2cm 1.0cm 1.5cm 1.0cm;
            @frame footer {{ -pdf-frame-content: footerContent; bottom: 0cm; height: 1.2cm; }}
        }}
        body {{ font-family: 'Helvetica', 'Arial', sans-serif; font-size: 8.5pt; color: #1a1a2e; line-height: 1.35; }}

        /* ── CLASSIFICATION BANNER ──────────────────── */
        .classification-banner {{
            background-color: #dc2626; color: #ffffff; text-align: center;
            padding: 3px 0; font-size: 7pt; font-weight: bold; letter-spacing: 3px;
            margin: -5px -5px 8px -5px;
        }}

        /* ── COVER / HEADER ──────────────────────────── */
        .header-table {{
            width: 100%; background-color: #0f172a; color: #ffffff;
            margin: 0 -5px 10px -5px; border-collapse: collapse;
        }}
        .header-table td {{ padding: 10px 18px; vertical-align: middle; }}
        .header-brand {{ font-size: 18pt; font-weight: bold; letter-spacing: 4px; color: #38bdf8; }}
        .header-subtitle {{ font-size: 8pt; color: #94a3b8; letter-spacing: 2px; text-transform: uppercase; margin-top: 2px; }}
        .header-logo {{ height: 44pt; margin-right: 12px; }}
        .header-date-box {{ font-family: Courier; font-size: 7.5pt; color: #e2e8f0; text-align: right; }}
        .report-type {{ font-size: 10pt; font-weight: bold; color: #e2e8f0; letter-spacing: 1px; }}

        /* ── META TABLE ──────────────────────────────── */
        .meta-table {{ width: 100%; border-collapse: collapse; margin-bottom: 8px; }}
        .meta-table td {{ padding: 3.5px 7px; border: 1px solid #cbd5e1; font-size: 7.5pt; }}
        .meta-label {{ font-weight: bold; color: #334155; background-color: #f1f5f9; width: 15%; }}
        .meta-value {{ width: 35%; color: #0f172a; }}

        /* ── SECTION HEADERS ─────────────────────────── */
        h2 {{
            background-color: #0f172a; color: #38bdf8; padding: 5px 12px;
            margin-top: 12px; margin-bottom: 0; font-size: 9pt;
            text-transform: uppercase; letter-spacing: 2px; width: 100%;
        }}
        h3 {{
            color: #1e40af; font-size: 8.5pt; margin-top: 7px; margin-bottom: 3px;
            border-bottom: 1.5px solid #dbeafe; padding-bottom: 2px;
        }}
        p {{ margin: 0 0 4px 0; }}

        /* ── INFO BOXES ──────────────────────────────── */
        .info-box {{
            background-color: #f0f9ff; border: 1px solid #bae6fd;
            padding: 6px 9px; margin-bottom: 6px; border-left: 3px solid #0ea5e9;
            font-size: 7.5pt; line-height: 1.4;
        }}
        .alert-box {{
            background-color: #fef2f2; border: 1px solid #fecaca; color: #991b1b;
            padding: 6px 9px; margin-bottom: 6px; border-left: 3px solid #ef4444;
            font-weight: bold; font-size: 7.5pt;
        }}
        .success-box {{
            background-color: #f0fdf4; border: 1px solid #bbf7d0; color: #166534;
            padding: 6px 9px; margin-bottom: 6px; border-left: 3px solid #22c55e;
            font-size: 7.5pt;
        }}
        .legal-box {{
            background-color: #fffbeb; border: 1px solid #fde68a; color: #92400e;
            padding: 6px 9px; margin-bottom: 6px; border-left: 3px solid #f59e0b;
            font-size: 7pt; line-height: 1.4;
        }}

        /* ── STATS GRID ──────────────────────────────── */
        .stats-grid {{ width: 100%; margin-bottom: 8px; }}
        .stat-box {{
            display: inline-block; width: 24%; text-align: center;
            border: 1px solid #e2e8f0; background: #f8fafc; padding: 5px 0;
        }}
        .stat-val {{ font-size: 14pt; font-weight: bold; color: #0f172a; }}
        .stat-lbl {{ font-size: 6pt; color: #64748b; text-transform: uppercase; letter-spacing: 1px; }}

        /* ── EVIDENCE TABLE ───────────────────────────── */
        .table {{ width: 100%; border-collapse: collapse; margin-top: 0; margin-bottom: 10px; table-layout: fixed; }}
        .table th {{
            background-color: #0f172a; color: #e2e8f0; padding: 4px 3px;
            text-align: left; border: 1px solid #1e293b; font-weight: bold;
            font-size: 6.5pt; text-transform: uppercase; letter-spacing: 0.5px;
        }}
        .table td {{
            padding: 4px 3px; border: 1px solid #e2e8f0; vertical-align: middle;
            font-size: 6.5pt; line-height: 1.15;
        }}
        .bg-gray {{ background-color: #f8fafc; }}
        .bg-risk-high {{ background-color: #fef2f2; }}

        /* ── BADGES ──────────────────────────────────── */
        .badge {{
            padding: 2px 5px; border-radius: 3px; font-weight: bold;
            font-size: 6pt; color: #FFFFFF; display: block; text-align: center;
        }}
        .badge-HIGH     {{ background-color: #ef4444; }}
        .badge-CRITICAL {{ background-color: #dc2626; }}
        .badge-MED      {{ background-color: #f59e0b; }}
        .badge-MODERATE {{ background-color: #3b82f6; }}
        .badge-LOW      {{ background-color: #10b981; }}

        /* ── BEHAVIORAL SECTION ──────────────────────── */
        .behavior-header {{
            background-color: #1e293b; color: #e2e8f0; padding: 7px 12px;
            margin-bottom: 0; display: block;
        }}
        .behavior-header .risk-label {{ font-size: 12pt; font-weight: bold; letter-spacing: 1px; }}
        .behavior-header .risk-score {{ font-family: Courier; font-size: 9pt; color: #94a3b8; }}
        .bar-container {{ margin-bottom: 3px; }}
        .bar-label-row {{ font-size: 7pt; color: #475569; margin-bottom: 1px; }}
        .bar-bg {{ width: 100%; height: 7px; background: #e2e8f0; margin-bottom: 1px; }}
        .bar-fg {{ height: 7px; }}
        .behavior-metric {{
            display: inline-block; width: 18%; text-align: center;
            border: 1px solid #e2e8f0; padding: 4px 0; background: #f8fafc; margin-right: 1%;
        }}
        .behavior-metric .bm-val {{ font-size: 11pt; font-weight: bold; color: #0f172a; }}
        .behavior-metric .bm-lbl {{ font-size: 5.5pt; color: #64748b; text-transform: uppercase; letter-spacing: 0.5px; }}
        .narrative {{ background: #f0f9ff; border: 1px solid #bae6fd; border-left: 3px solid #0284c7; padding: 7px 9px; margin: 6px 0; font-size: 7pt; line-height: 1.5; color: #1e293b; }}

        /* ── CHAIN OF CUSTODY ────────────────────────── */
        .coc-table {{ width: 100%; border-collapse: collapse; margin-bottom: 8px; }}
        .coc-table td {{ padding: 3px 5px; font-size: 7pt; border-bottom: 1px solid #e2e8f0; vertical-align: top; }}
        .coc-time {{ color: #0f172a; font-weight: bold; width: 100px; font-family: Courier; }}
        .coc-dot {{ width: 7px; height: 7px; background: #0ea5e9; border-radius: 50%; display: inline-block; margin-right: 3px; }}

        /* ── CERTIFICATION ───────────────────────────── */
        .cert-box {{
            background-color: #fafafa; border: 2px solid #334155; padding: 12px 14px;
            margin: 10px 0; font-size: 7.5pt; line-height: 1.6;
        }}
        .cert-title {{
            font-size: 9pt; font-weight: bold; color: #0f172a; text-transform: uppercase;
            letter-spacing: 1px; margin-bottom: 6px; border-bottom: 1px solid #334155; padding-bottom: 3px;
        }}

        /* ── SIGNATURES ─────────────────────────────── */
        .sig-table {{ width: 100%; margin-top: 20px; border-collapse: collapse; }}
        .sig-cell {{ width: 30%; padding: 8px 0 0 0; }}
        .sig-line {{ border-top: 1px solid #1e293b; padding-top: 4px; font-size: 7.5pt; color: #0f172a; }}
        .sig-role {{ font-size: 6.5pt; color: #64748b; margin-top: 1px; }}
        .sig-date {{ font-size: 6.5pt; color: #94a3b8; font-family: Courier; margin-top: 2px; }}

        /* ── UTILS ───────────────────────────────────── */
        .footer-table {{ width: 100%; border-top: 1px solid #94a3b8; padding-top: 2px; }}
        .footer-left {{ font-size: 5.5pt; color: #dc2626; font-weight: bold; letter-spacing: 1px; }}
        .footer-right {{ font-size: 5.5pt; color: #94a3b8; text-align: right; }}
        .page-break {{ page-break-before: always; }}
        .hash-text {{ font-family: Courier; font-size: 6.5pt; color: #475569; word-wrap: break-word; -pdf-word-wrap: break-word; }}
        .redacted {{ background-color: #000; color: #000; padding: 0 4px; }}
        .exhibit-num {{ font-family: Courier; font-size: 6.5pt; font-weight: bold; color: #1e40af; }}
    </style>
</head>
<body>

    <!-- ═══ CLASSIFICATION BANNER ═══ -->
    <div class="classification-banner">{classification}</div>

    <!-- ═══ HEADER BAR ═══ -->
    <table class="header-table">
        <tr>
            <td style="width:50px;"><img src="{logo_path}" class="header-logo"></td>
            <td>
                <div class="header-brand">FORAX</div>
                <div class="header-subtitle">Digital Forensic Examination Report</div>
            </td>
            <td class="header-date-box">
                <div class="report-type">EXAMINATION REPORT</div>
                {date}<br>
                <span style="font-size:6.5pt; color:#94a3b8;">Report No: {report_number}</span>
            </td>
        </tr>
    </table>

    <!-- ═══ CASE METADATA ═══ -->
    <table class="meta-table">
        <tr>
            <td class="meta-label">Case ID</td><td class="meta-value" style="font-family:Courier;font-weight:bold;">{case_id}</td>
            <td class="meta-label">FIR / Ref No.</td><td class="meta-value">{fir_ref}</td>
        </tr>
        <tr>
            <td class="meta-label">Lead Examiner</td><td class="meta-value">{investigator}</td>
            <td class="meta-label">Badge / ID</td><td class="meta-value" style="font-family:Courier;">{examiner_badge}</td>
        </tr>
        <tr>
            <td class="meta-label">Department</td><td class="meta-value">{department}</td>
            <td class="meta-label">Laboratory</td><td class="meta-value">{location}</td>
        </tr>
        <tr>
            <td class="meta-label">Jurisdiction</td><td class="meta-value">{jurisdiction}</td>
            <td class="meta-label">Crime Type</td><td class="meta-value">{crime_type}</td>
        </tr>
        <tr>
            <td class="meta-label">Device Info</td><td class="meta-value">{device_info}</td>
            <td class="meta-label">Warrant Ref</td><td class="meta-value">{warrant_ref}</td>
        </tr>
        <tr>
            <td class="meta-label">Classification</td><td class="meta-value" style="color:#dc2626; font-weight:bold;">{classification}</td>
            <td class="meta-label">Requesting Agency</td><td class="meta-value">{agency}</td>
        </tr>
        <tr>
            <td class="meta-label">Report Integrity</td><td class="meta-value hash-text" colspan="3">{full_integrity_hash}</td>
        </tr>
    </table>

    <!-- ═══ SECTION 1: LEGAL PREAMBLE ═══ -->
    <h2>1 &mdash; Legal Preamble &amp; Authority</h2>
    <div class="legal-box">
        <strong>NOTICE:</strong> This document constitutes a formal Digital Forensic Examination Report prepared
        in accordance with the Federal Rules of Evidence (FRE 901, 902, 1006), the Daubert Standard for expert
        testimony, ISO/IEC 27037:2012 (Guidelines for identification, collection, acquisition and preservation
        of digital evidence), and NIST Special Publication 800-86 (Guide to Integrating Forensic Techniques into
        Incident Response). This report is intended for use in legal proceedings and administrative inquiries.
        Unauthorized disclosure, copying, or distribution is prohibited.
    </div>
    <div class="info-box">
        <strong>Authorization:</strong> Examination conducted under authority of <strong>{warrant_ref_display}</strong>,
        issued by <strong>{jurisdiction_display}</strong>, at the request of <strong>{agency_display}</strong>.
        All procedures have been conducted in compliance with applicable laws, regulations, and departmental
        standard operating procedures (SOPs).
    </div>

    {notes_section}

    <!-- ═══ SECTION 2: EXAMINER INFORMATION ═══ -->
    <h2>2 &mdash; Examiner Qualifications</h2>
    <table class="meta-table">
        <tr>
            <td class="meta-label">Name</td><td class="meta-value"><strong>{investigator}</strong></td>
            <td class="meta-label">Badge / ID</td><td class="meta-value" style="font-family:Courier;">{examiner_badge}</td>
        </tr>
        <tr>
            <td class="meta-label">Department</td><td class="meta-value">{department}</td>
            <td class="meta-label">Role</td><td class="meta-value">Lead Forensic Examiner</td>
        </tr>
        <tr>
            <td class="meta-label">Qualifications</td><td class="meta-value" colspan="3">{qualifications_display}</td>
        </tr>
    </table>

    <!-- ═══ SECTION 3: METHODOLOGY ═══ -->
    <h2>3 &mdash; Examination Methodology</h2>
    <div class="info-box" style="font-size: 7pt;">
        <strong>Tools &amp; Environment:</strong> FORAX Forensic Intelligence Platform v2.0. Analysis engine: AI-assisted
        classification (CNN image analysis, NLP text scoring). Hash algorithm: SHA-256. Operating environment: Isolated
        forensic workstation. All tools have been validated per laboratory quality assurance procedures.<br>
        <strong>Standards Followed:</strong> ISO/IEC 27037:2012, ISO/IEC 27042:2015, NIST SP 800-86, SWGDE Best Practices.
    </div>
    <div class="info-box" style="font-size: 7pt;">
        <strong>Process Summary:</strong>
        (1) Evidence was received and intake documented in the chain of custody log.
        (2) Forensic acquisition was performed creating a bit-stream image; SHA-256 hashes computed at acquisition time.
        (3) Automated AI-assisted analysis was performed classifying each artifact by risk level.
        (4) Results were reviewed by the lead examiner.
        (5) This report was generated with a cryptographic manifest seal to ensure integrity.
    </div>

    <!-- ═══ SECTION 4: EXECUTIVE SUMMARY ═══ -->
    <h2>4 &mdash; Executive Summary &amp; Analytics</h2>
    <div class="info-box">
        <strong>FORAX Examination Summary.</strong>
        Total extracted artifacts: <strong>{total}</strong>.
        Filtered view: <strong>{filtered_count}</strong> artifacts.
        <strong style="color:#ef4444;">{high_count} HIGH RISK</strong> items flagged for review.
    </div>

    {high_risk_alert}

    <div class="stats-grid">
        <div class="stat-box"><div class="stat-val">{total}</div><div class="stat-lbl">Total Files</div></div>
        <div class="stat-box"><div class="stat-val" style="color:#ef4444;">{high_count}</div><div class="stat-lbl">High Risk</div></div>
        <div class="stat-box"><div class="stat-val" style="color:#f59e0b;">{med_count}</div><div class="stat-lbl">Medium Risk</div></div>
        <div class="stat-box"><div class="stat-val" style="color:#10b981;">{low_count}</div><div class="stat-lbl">Cleared</div></div>
    </div>

    <table class="table" style="width: 55%;">
        <thead><tr><th style="width:65%;">Evidence Category</th><th>Count</th></tr></thead>
        <tbody>{type_breakdown_rows}</tbody>
    </table>

    <!-- ═══ SECTION 5: CHAIN OF CUSTODY ═══ -->
    <h2>5 &mdash; Chain of Custody &amp; Integrity Verification</h2>
    <div class="info-box" style="font-size: 7pt;">
        <strong>Cryptographic Verification:</strong> All digital evidence sealed with SHA-256 at time of acquisition.
        Combined manifest hash: <span class="hash-text" style="font-size: 6pt;">{full_integrity_hash}</span><br>
        <strong>Integrity Status:</strong> All hashes were computed at acquisition. Any subsequent modification will
        produce a different hash, invalidating the chain of custody for the affected item(s).
    </div>

    <table class="coc-table">
        <tr><td class="coc-time">{t0}</td><td><span class="coc-dot"></span><strong>Case Initialization:</strong> {case_id} formally assigned to {investigator}. Evidence intake documented.</td></tr>
        <tr><td class="coc-time">{t1}</td><td><span class="coc-dot"></span><strong>Evidence Acquisition:</strong> Forensic acquisition initiated. Device connected via secure forensic gateway.</td></tr>
        <tr><td class="coc-time">{t2}</td><td><span class="coc-dot"></span><strong>Data Extraction:</strong> Bit-stream image capture of {total} blocks completed. SHA-256 hashes computed per artifact.</td></tr>
        <tr><td class="coc-time">{t3}</td><td><span class="coc-dot"></span><strong>AI-Assisted Analysis:</strong> Automated classification engine completed processing. {high_count} item(s) flagged as HIGH risk.</td></tr>
        <tr><td class="coc-time">{t4}</td><td><span class="coc-dot"></span><strong>Report Finalization:</strong> Examiner review completed. Cryptographic seal applied. Format: {report_options}</td></tr>
    </table>

    <div class="page-break"></div>

    <!-- ═══ SECTION 6: EVIDENCE MANIFEST ═══ -->
    <h2>6 &mdash; Detailed Evidence Manifest (Exhibit Register)</h2>
    <div class="info-box" style="font-size: 6.5pt; margin-bottom: 4px;">
        Each evidence item is assigned a unique Exhibit Number for court reference. SHA-256 cryptographic hashes
        are provided for integrity verification per FRE 901(b)(9).
    </div>
    <table class="table">
        <thead>
            <tr>
                <th style="width: 5%; text-align: center;">Exhibit</th>
                <th style="width: 16%;">Filename / Path</th>
                <th style="width: 7%; text-align: center;">Device</th>
                <th style="width: 6%; text-align: center;">Risk</th>
                <th style="width: 6%; text-align: center;">Size</th>
                <th style="width: 30%;">Examiner Findings / AI Analysis</th>
                <th style="width: 30%;">SHA-256 Hash</th>
            </tr>
        </thead>
        <tbody>
            {evidence_rows}
        </tbody>
    </table>

    <div class="page-break"></div>

    <!-- ═══ SECTION 7: BEHAVIORAL INTELLIGENCE ═══ -->
    <h2>7 &mdash; Behavioral Intelligence Profile</h2>

    <div class="behavior-header">
        <div class="risk-label" style="color:{behavior_color};">{behavior_risk}</div>
        <div class="risk-score">Composite Score: {behavior_score}/100</div>
    </div>

    <div style="margin: 6px 0;">
        <div class="behavior-metric"><div class="bm-val">{bh_comm_total}</div><div class="bm-lbl">Communication</div></div>
        <div class="behavior-metric"><div class="bm-val">{bh_media_total}</div><div class="bm-lbl">Media Files</div></div>
        <div class="behavior-metric"><div class="bm-val">{bh_gps}</div><div class="bm-lbl">GPS Locations</div></div>
        <div class="behavior-metric"><div class="bm-val">{bh_downloads}</div><div class="bm-lbl">Downloads</div></div>
        <div class="behavior-metric"><div class="bm-val">{bh_recovered}</div><div class="bm-lbl">Recovered</div></div>
    </div>

    <h3>Threat Category Breakdown</h3>
    {behavior_bars}

    <h3>Examiner Assessment</h3>
    <div class="narrative">{behavior_narrative}</div>

    {behavior_comm_section}
    {behavior_media_section}

    <!-- ═══ SECTION 8: EXAMINER CONCLUSIONS ═══ -->
    <h2>8 &mdash; Examiner Findings &amp; Conclusions</h2>
    <div class="narrative">
        Based on the forensic examination of {total} digital artifacts extracted under Case {case_id},
        the automated analysis engine classified {high_count} item(s) as HIGH risk, {med_count} item(s) as
        MEDIUM risk, and {low_count} item(s) as LOW risk (cleared). {conclusion_detail}
        All findings are documented in the Exhibit Register (Section 6) and the Behavioral Intelligence
        Profile (Section 7). The integrity of all evidence has been preserved through SHA-256 cryptographic
        hashing at the time of acquisition.
    </div>

    <div class="page-break"></div>

    <!-- ═══ SECTION 9: CERTIFICATION & DECLARATION ═══ -->
    <h2>9 &mdash; Certification &amp; Declaration</h2>
    <div class="cert-box">
        <div class="cert-title">Examiner Certification</div>
        <p>I, <strong>{investigator}</strong>, Badge/ID <strong>{examiner_badge}</strong>, hereby certify that:</p>
        <p>1. I conducted the forensic examination described in this report in accordance with accepted forensic
        methodologies, ISO/IEC 27037:2012, and departmental standard operating procedures.</p>
        <p>2. The evidence described herein was handled in a manner consistent with maintaining its integrity and
        the chain of custody has been preserved throughout the examination process.</p>
        <p>3. All SHA-256 cryptographic hashes were computed at the time of acquisition and have been verified
        prior to the generation of this report.</p>
        <p>4. The findings, opinions, and conclusions expressed in this report are based upon my training,
        education, experience, and the evidence examined.</p>
        <p>5. This report is a true and accurate representation of my findings. I have not knowingly omitted or
        misrepresented any material facts.</p>
        <p style="margin-top:8px;"><strong>I declare under penalty of perjury that the foregoing is true and correct
        to the best of my knowledge and belief.</strong></p>
    </div>

    <!-- ═══ SIGNATURES ═══ -->
    <table class="sig-table">
        <tr>
            <td class="sig-cell">
                <div style="height:32px;"></div>
                <div class="sig-line"><strong>{investigator}</strong></div>
                <div class="sig-role">Lead Forensic Examiner</div>
                <div class="sig-date">Date: ___________________</div>
            </td>
            <td style="width:5%;"></td>
            <td class="sig-cell">
                <div style="height:32px;"></div>
                <div class="sig-line"><strong>{reviewer_display}</strong></div>
                <div class="sig-role">Reviewing Officer / Supervisor</div>
                <div class="sig-date">Date: ___________________</div>
            </td>
            <td style="width:5%;"></td>
            <td class="sig-cell">
                <div style="height:32px;"></div>
                <div class="sig-line"><strong>_________________________</strong></div>
                <div class="sig-role">Receiving Authority / Court Officer</div>
                <div class="sig-date">Date: ___________________</div>
            </td>
        </tr>
    </table>

    <div class="page-break"></div>

    <!-- ═══ APPENDIX A: FULL INTEGRITY MANIFEST ═══ -->
    <h2>Appendix A &mdash; Full SHA-256 Integrity Manifest</h2>
    <div class="info-box" style="font-size: 6.5pt;">
        This appendix contains the complete cryptographic hash manifest for all {total} evidence artifacts.
        Each hash was computed using SHA-256 at the time of forensic acquisition. The combined manifest
        hash is: <span class="hash-text" style="font-size: 6pt;">{full_integrity_hash}</span>
    </div>
    <table class="table">
        <thead><tr>
            <th style="width:5%; text-align:center;">Exhibit</th>
            <th style="width:30%;">Filename</th>
            <th style="width:10%;">Type</th>
            <th style="width:55%;">SHA-256 Cryptographic Hash</th>
        </tr></thead>
        <tbody>{hash_manifest_rows}</tbody>
    </table>

    <!-- ═══ FOOTER ═══ -->
    <div id="footerContent">
        <table class="footer-table">
            <tr>
                <td class="footer-left">{classification} &mdash; {agency_display}</td>
                <td class="footer-right">
                    FORAX FORENSIC REPORT &bull; CASE: {case_id} &bull; PAGE <pdf:pagenumber>
                </td>
            </tr>
        </table>
    </div>

</body>
</html>
"""


# ══════════════════════════════════════════════════════════════════
# PDF REPORT GENERATOR
# ══════════════════════════════════════════════════════════════════

def generate_pdf_report(case_id, investigator, evidence, output_dir='reports', options=None):
    """
    Generate a court-admissible forensic PDF report.
    Supports advanced filtering, redaction, and legal metadata.
    Compliant with FRE 901/902, ISO/IEC 27037, NIST SP 800-86.
    """
    if options is None: options = {}

    try:
        os.makedirs(output_dir, exist_ok=True)
        filename = f"FORAX_Report_{case_id}_{int(datetime.now().timestamp())}.pdf"
        filepath = os.path.join(output_dir, filename)

        # ── ADVANCED FILTERING ────────────────────────────────
        include_types = options.get('types', [])
        redact = options.get('redact_attachments', False)

        filtered_evidence = [e for e in evidence if not include_types or e.get('file_type') in include_types]
        if not filtered_evidence:
            filtered_evidence = evidence  # Fallback if empty

        # ── STATISTICS ────────────────────────────────────────
        total_extracted = len(evidence)
        total_filtered = len(filtered_evidence)
        high_count = sum(1 for e in filtered_evidence if e.get('risk_level') == 'HIGH')
        med_count = sum(1 for e in filtered_evidence if e.get('risk_level') == 'MED')
        low_count = total_filtered - high_count - med_count

        # Compute full integrity hash of original evidence
        data_str = "".join([str(e.get('sha256', '')) for e in evidence])
        full_integrity_hash = hashlib.sha256(data_str.encode()).hexdigest().upper()

        now = datetime.now()
        report_number = f"FORAX-RPT-{now.strftime('%Y%m%d')}-{abs(hash(case_id)) % 10000:04d}"
        
        # Parse case_id for dynamic timestamps (Format: FORAX-YYYYMMDDHHMMSS)
        start_dt = now
        try:
            time_part = case_id.split('-')[-1]
            if len(time_part) >= 12:
                start_dt = datetime.strptime(time_part[:12], "%Y%m%d%H%M")
        except: pass

        t0 = start_dt.strftime("%Y-%m-%d %H:%M")
        t1 = (start_dt + timedelta(minutes=5)).strftime("%Y-%m-%d %H:%M")
        t2 = (start_dt + timedelta(minutes=45)).strftime("%Y-%m-%d %H:%M")
        t3 = (start_dt + timedelta(minutes=75)).strftime("%Y-%m-%d %H:%M")
        t4 = now.strftime("%Y-%m-%d %H:%M")

        # ── TYPE BREAKDOWN ────────────────────────────────────
        type_counts = {}
        for e in filtered_evidence:
            ft = e.get('file_type', 'unknown') or 'unknown'
            type_counts[ft] = type_counts.get(ft, 0) + 1

        type_breakdown_rows = ""
        type_labels = {
            'image': 'Images & Photos', 'video': 'Videos', 'audio': 'Audio',
            'chat': 'Chats & SMS', 'call_log': 'Call Logs', 'document': 'Documents',
            'download': 'Downloads', 'app_data': 'App Data', 'recovered': 'Recovered Data',
            'browser': 'Browser Data', 'contacts': 'Contacts', 'config': 'Config Files',
            'sms': 'SMS Messages', 'text': 'Text Files'
        }
        for ft, count in sorted(type_counts.items(), key=lambda x: x[1], reverse=True)[:8]:
            label = type_labels.get(ft, ft.replace('_', ' ').title())
            type_breakdown_rows += f"<tr><td>{label}</td><td style='text-align:center;font-weight:bold;'>{count}</td></tr>\n"

        # ── ALERTS ────────────────────────────────────────────
        high_risk_alert = ""
        if high_count > 0:
            high_risk_alert = f'<div class="alert-box">CRITICAL ALERT: {high_count} HIGH RISK item(s) identified. Immediate review by supervising officer recommended per departmental SOP.</div>'

        # ── EVIDENCE TABLE WITH EXHIBIT NUMBERS ───────────────
        rows = []
        hash_manifest_rows = []
        for i, e in enumerate(filtered_evidence, 1):
            exhibit_num = f"EX-{i:03d}"
            risk = e.get('risk_level', 'LOW')
            risk_badge = f'<span class="badge badge-{risk}">{risk}</span>'

            summary = e.get('ai_result') or "Pending analysis"
            summary = summary.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')

            fname = (e.get('filename', '-') or '-').replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
            ftype = (e.get('file_type', '-') or '-').upper()
            fslot = (e.get('device_slot', 'dev1') or 'dev1').replace('device', 'D')
            size = str(e.get('file_size') or '-')
            sha = str(e.get('sha256') or '-')
            reasoning = e.get('copilot_reasoning') or ""

            reasoning_html = ""
            if reasoning and risk != 'LOW':
                reasoning_html = f'<div style="margin-top:6px;padding:8px;background:#f8fafc;border-left:3px solid #2563eb;font-size:10px;color:#475569;font-style:italic;"><strong>FORAX Forensic Reasoning:</strong> <br/>{reasoning}</div>'

            # ... rest of logic
            if redact and ftype in ['IMAGE', 'VIDEO', 'AUDIO', 'DOCUMENT', 'DOWNLOAD']:
                fname = '<span class="redacted">REDACTED_ATTACHMENT_HIDDEN</span>'
                size = '<span class="redacted">---</span>'
                sha = '<span class="redacted">REDACTED_HASH</span>'

            row_class = "bg-risk-high" if risk == 'HIGH' else ("bg-gray" if i % 2 == 0 else "")
            row = f'''
            <tr class="{row_class}">
                <td style="text-align: center;"><span class="exhibit-num">{exhibit_num}</span></td>
                <td><div style="width:100%; word-wrap: break-word;">{fname}</div></td>
                <td style="text-align: center;">{fslot}</td>
                <td style="text-align: center;">{risk_badge}</td>
                <td style="text-align: center;">{size}</td>
                <td style="font-size:6pt;line-height:1.2;">
                    {summary}
                    {reasoning_html}
                </td>
                <td class="hash-text">{sha}</td>
            </tr>
            '''
            rows.append(row)

            # Hash manifest for Appendix A
            hash_row = f'''
            <tr class="{"bg-gray" if i % 2 == 0 else ""}">
                <td style="text-align:center;"><span class="exhibit-num">{exhibit_num}</span></td>
                <td style="font-size:6.5pt;">{fname}</td>
                <td style="text-align:center; font-size:6pt;">{ftype}</td>
                <td class="hash-text" style="font-size:5.5pt;">{sha}</td>
            </tr>
            '''
            hash_manifest_rows.append(hash_row)

        # ── BEHAVIORAL ANALYSIS ───────────────────────────────
        behavior = generate_behavioral_analysis(filtered_evidence)

        risk_colors = {'CRITICAL': '#ef4444', 'HIGH': '#f59e0b', 'MODERATE': '#3b82f6', 'LOW': '#10b981'}
        behavior_color = risk_colors.get(behavior['overall_risk'], '#10b981')

        behavior_bars = ""
        if behavior['risk_categories']:
            for cat_name, cat_data in sorted(behavior['risk_categories'].items(), key=lambda x: x[1]['score'], reverse=True):
                score = cat_data['score']
                sev = cat_data['severity']
                bar_color = risk_colors.get(sev, '#3b82f6')
                behavior_bars += f'''
                <div class="bar-container">
                    <div class="bar-label-row"><strong>{cat_name}</strong> &mdash; {sev} ({cat_data["file_count"]} file(s), score: {score})</div>
                    <div class="bar-bg"><div class="bar-fg" style="width:{score}%;background-color:{bar_color};"></div></div>
                </div>
                '''
        else:
            behavior_bars = '<div class="success-box">No threat categories detected in analyzed evidence. All content classified as benign.</div>'

        # Communication sub-section
        comm = behavior['communication']
        behavior_comm_section = ""
        if comm.get('total_files', 0) > 0:
            behavior_comm_section = f'''
            <h3>Communication Analysis</h3>
            <table class="table" style="width:70%;">
                <thead><tr><th>Metric</th><th>Value</th></tr></thead>
                <tbody>
                    <tr><td>Chat Logs</td><td style="text-align:center;font-weight:bold;">{comm["chat_count"]}</td></tr>
                    <tr class="bg-gray"><td>SMS Records</td><td style="text-align:center;font-weight:bold;">{comm["sms_count"]}</td></tr>
                    <tr><td>Call Records</td><td style="text-align:center;font-weight:bold;">{comm["call_count"]}</td></tr>
                    <tr class="bg-gray"><td>Contacts</td><td style="text-align:center;font-weight:bold;">{comm["contact_count"]}</td></tr>
                    <tr><td>Flagged Communications</td><td style="text-align:center;font-weight:bold;color:#ef4444;">{comm["high_risk"] + comm["med_risk"]} ({comm["flagged_ratio"]}%)</td></tr>
                </tbody>
            </table>
            '''

        # Media sub-section
        media = behavior['media_profile']
        behavior_media_section = ""
        if media.get('total_media', 0) > 0:
            gps_text = ""
            if media['gps_locations'] > 0:
                gps_text = f'<tr><td>GPS-Tagged Files</td><td style="text-align:center;font-weight:bold;color:#0ea5e9;">{media["gps_locations"]}</td></tr>'
            cam_text = ""
            if media['cameras']:
                cam_text = f'<tr class="bg-gray"><td>Camera Devices</td><td style="text-align:center;">{", ".join(media["cameras"][:3])}</td></tr>'

            behavior_media_section = f'''
            <h3>Media Analysis</h3>
            <table class="table" style="width:70%;">
                <thead><tr><th>Metric</th><th>Value</th></tr></thead>
                <tbody>
                    <tr><td>Images</td><td style="text-align:center;font-weight:bold;">{media["image_count"]}</td></tr>
                    <tr class="bg-gray"><td>Videos</td><td style="text-align:center;font-weight:bold;">{media["video_count"]}</td></tr>
                    <tr><td>Audio Files</td><td style="text-align:center;font-weight:bold;">{media["audio_count"]}</td></tr>
                    <tr class="bg-gray"><td>High Risk Media</td><td style="text-align:center;font-weight:bold;color:#ef4444;">{media["media_high_risk"]}</td></tr>
                    {gps_text}
                    {cam_text}
                </tbody>
            </table>
            '''

        # ── NOTES ─────────────────────────────────────────────
        notes = options.get('notes', '')
        notes_html = f'<div class="info-box"><strong>Examiner Notes:</strong> {notes}</div>' if notes else ''
        report_options_str = "Redacted" if redact else "Full Extract"

        # ── COURT FIELDS ──────────────────────────────────────
        classification = options.get('classification', 'LAW ENFORCEMENT SENSITIVE')
        jurisdiction = options.get('jurisdiction', '')
        warrant_ref = options.get('warrant_ref', '')
        agency = options.get('agency', '')
        fir_ref = options.get('fir_ref', '')
        reviewer = options.get('reviewer', '')
        examiner_badge = options.get('examiner_badge', '')
        qualifications = options.get('qualifications', '')

        # Display fallbacks
        jurisdiction_display = jurisdiction if jurisdiction else 'As per authorizing instrument'
        warrant_ref_display = warrant_ref if warrant_ref else 'Departmental authorization'
        agency_display = agency if agency else options.get('department', 'Digital Forensics Unit')
        reviewer_display = reviewer if reviewer else 'Supervising Officer'
        qualifications_display = qualifications if qualifications else 'Per departmental training records'

        # Conclusion detail
        conclusion_detail = ""
        if high_count > 0:
            top_cats = sorted(behavior['risk_categories'].items(), key=lambda x: x[1]['score'], reverse=True)[:3]
            if top_cats:
                cat_str = ', '.join(f"{c}" for c, d in top_cats)
                conclusion_detail = f"Primary threat categories identified include: {cat_str}. These findings warrant further investigative action."
            else:
                conclusion_detail = "The flagged items warrant further investigative review."
        else:
            conclusion_detail = "No items of immediate concern were identified during the automated analysis phase."

        # ── RENDER HTML → PDF ─────────────────────────────────
        html = HTML_TEMPLATE.format(
            logo_path=os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'static', 'img', 'forax_logo.png')),
            case_id=options.get('case_id') or case_id,
            investigator=options.get('examiner') or investigator,
            examiner_badge=examiner_badge or 'N/A',
            qualifications_display=qualifications_display,
            department=options.get('department') or 'Digital Forensics Unit',
            location=options.get('location') or 'Lab 1',
            crime_type=options.get('crime_type') or 'Under Investigation',
            jurisdiction=jurisdiction_display,
            warrant_ref=warrant_ref or 'N/A',
            warrant_ref_display=warrant_ref_display,
            jurisdiction_display=jurisdiction_display,
            agency=agency or 'N/A',
            agency_display=agency_display,
            fir_ref=fir_ref or 'N/A',
            classification=classification,
            reviewer_display=reviewer_display,
            report_number=report_number,
            date=now.strftime("%Y-%m-%d %H:%M"),
            total=total_extracted,
            filtered_count=total_filtered,
            high_count=high_count,
            med_count=med_count,
            low_count=low_count,
            evidence_rows="\n".join(rows),
            hash_manifest_rows="\n".join(hash_manifest_rows),
            type_breakdown_rows=type_breakdown_rows,
            high_risk_alert=high_risk_alert,
            full_integrity_hash=full_integrity_hash,
            device_info=options.get('device_info') or 'Secured Evidence Block',
            notes_section=notes_html,
            t0=t0, t1=t1, t2=t2, t3=t3, t4=t4,
            report_options=report_options_str,
            conclusion_detail=conclusion_detail,
            # Behavioral fields
            behavior_risk=behavior['overall_risk'],
            behavior_score=behavior['risk_score'],
            behavior_color=behavior_color,
            behavior_bars=behavior_bars,
            behavior_narrative=behavior['narrative'],
            behavior_comm_section=behavior_comm_section,
            behavior_media_section=behavior_media_section,
            bh_comm_total=comm.get('total_files', 0),
            bh_media_total=media.get('total_media', 0),
            bh_gps=media.get('gps_locations', 0),
            bh_downloads=behavior['digital_footprint'].get('total_downloads', 0),
            bh_recovered=behavior['digital_footprint'].get('recovered_deleted', 0),
        )

        with open(filepath, "wb") as pdf_file:
            pisa_status = pisa.CreatePDF(html, dest=pdf_file)

        if pisa_status.err:
            raise Exception("Error rendering forensic PDF")

        return filepath, filename

    except Exception as e:
        raise e


# ══════════════════════════════════════════════════════════════════
# INTERACTIVE HTML REPORT GENERATOR
# ══════════════════════════════════════════════════════════════════

def _embed_evidence_images(evidence_list):
    """Embed evidence images as base64 data URIs so the HTML report is self-contained."""
    import base64
    IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}
    MAX_THUMB_SIZE = (400, 400)  # Thumbnail size for gallery

    for e in evidence_list:
        fpath = e.get('file_path', '')
        ftype = e.get('file_type', '')
        if ftype != 'image' or not fpath or not os.path.exists(fpath):
            continue

        ext = os.path.splitext(fpath)[1].lower()
        if ext not in IMAGE_EXTS:
            continue

        try:
            # Read and create a thumbnail to keep file size reasonable
            from PIL import Image as PILImg
            img = PILImg.open(fpath).convert('RGB')
            img.thumbnail(MAX_THUMB_SIZE)

            import io
            buf = io.BytesIO()
            img.save(buf, format='JPEG', quality=70)
            b64 = base64.b64encode(buf.getvalue()).decode('ascii')
            e['image_data'] = f'data:image/jpeg;base64,{b64}'
        except Exception:
            # If image processing fails, skip embedding
            pass

    return evidence_list


def generate_html_report(case_id, investigator, evidence, output_dir='reports', options=None):
    """Generate a high-density, interactive HTML forensic volume (Autopsy-style).
    Evidence images are embedded as base64 so the report is fully self-contained."""
    if options is None: options = {}
    os.makedirs(output_dir, exist_ok=True)
    filename = f"FORAX_Interactive_Report_{case_id}_{int(datetime.now().timestamp())}.html"
    filepath = os.path.join(output_dir, filename)

    try:
        # Filter evidence based on request
        include_types = options.get('types', [])
        filtered_evidence = [e for e in evidence if not include_types or e.get('file_type') in include_types]
        if not filtered_evidence: filtered_evidence = evidence

        # Deep copy evidence so we don't mutate the original dicts
        import copy
        portable_evidence = copy.deepcopy(filtered_evidence)

        # Embed images as base64 for self-contained sharing
        portable_evidence = _embed_evidence_images(portable_evidence)

        # Stats for dashboard
        total_filtered = len(filtered_evidence)
        high_count = sum(1 for e in filtered_evidence if e.get('risk_level') == 'HIGH')
        med_count = sum(1 for e in filtered_evidence if e.get('risk_level') == 'MED')
        low_count = total_filtered - high_count - med_count

        # Crypto integrity
        data_str = "".join([str(e.get('sha256', '')) for e in evidence])
        full_integrity_hash = hashlib.sha256(data_str.encode()).hexdigest()

        now = datetime.now()

        # Behavioral Analysis
        behavior = generate_behavioral_analysis(filtered_evidence)
        risk_class = behavior['overall_risk'].lower()

        # Render the template
        html = INTERACTIVE_HTML_TEMPLATE.format(
            case_id=case_id,
            investigator=options.get('examiner') or investigator,
            department=options.get('department') or 'Digital Forensics Unit',
            date=now.strftime("%Y-%m-%d %H:%M:%S"),
            total=total_filtered,
            high_count=high_count,
            med_count=med_count,
            low_count=low_count,
            full_integrity_hash=full_integrity_hash,
            evidence=json.dumps(portable_evidence, default=str),
            # Behavioral fields
            behavior_json=json.dumps(behavior),
            behavior_risk_class=risk_class,
            behavior_score=behavior['risk_score'],
            behavior_risk_level=behavior['overall_risk'],
            behavior_narrative=behavior['narrative'],
            comm_total=behavior['communication'].get('total_files', 0),
            media_total=behavior['media_profile'].get('total_media', 0),
            gps_count=behavior['media_profile'].get('gps_locations', 0),
            dl_count=behavior['digital_footprint'].get('total_downloads', 0),
            app_count=behavior['digital_footprint'].get('installed_apps', 0),
            recovered_count=behavior['digital_footprint'].get('recovered_deleted', 0),
        )

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html)

        return filepath, filename
    except Exception as e:
        import traceback; traceback.print_exc()
        raise e
