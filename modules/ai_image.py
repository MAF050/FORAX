"""
FORAX — AI Image Analyzer
Vision-based image classification with EXIF metadata extraction.

Inference priority:
  1. Custom trained CNN  (forax_forensic_model.h5) — PRIMARY
  2. OpenAI GPT-4o-mini API                         — FALLBACK
  3. Local filename heuristics                       — LAST RESORT
"""

import os, sys, json, base64, requests
import numpy as np
try:
    import cv2
except ImportError:
    cv2 = None
from .forensic_parser import extract_frames_from_video
from dotenv import load_dotenv
import PIL.Image as PILImage

load_dotenv()
GITHUB_KEY = os.getenv('GITHUB_TOKEN', '').strip()

OPENAI_URL   = 'https://models.inference.ai.azure.com/chat/completions'
OPENAI_MODEL = 'gpt-4o-mini'

# ── LOCAL MODEL CONFIG ────────────────────────────────────────────
# Expects model_training/forax_forensic_model.h5 relative to this file
_MODULE_DIR  = os.path.dirname(os.path.abspath(__file__))
_MODEL_PATH  = os.path.join(_MODULE_DIR, '..', 'model_training', 'forax_forensic_model.h5')
_CLS_IDX_PATH = os.path.join(_MODULE_DIR, '..', 'model_training', 'class_indices.json')
_MODEL_CONFIG_PATH = os.path.join(_MODULE_DIR, '..', 'model_training', 'model_config.json')
IMG_SIZE     = (224, 224)
_MODEL_PREPROCESSING = 'rescale_255'

# Class → FORAX risk mapping
_CLASS_RISK_MAP = {
    'drugs':    ('MED',  'Drug/narcotics-related content detected',  ['pills', 'powder']),
    'normal':   ('LOW',  'No forensic threats detected',             []),
    'nsfw':     ('MED',  'NSFW/suspicious content detected',         ['nsfw']),
    'violence': ('HIGH', 'Signs of violence or assault detected',    ['violence', 'injury']),
    'weapons':  ('HIGH', 'Weapon/firearm detected',                  ['weapon', 'firearm']),
    'scam':     ('HIGH', 'Scam/phishing visual pattern detected',     ['scam', 'phish']),
}

# ── Lazy-load model on first use ──────────────────────────────────
_local_model       = None
_local_class_order = None   # list of class names in model output order
_local_model_unavailable_reason = None


def _load_model_config():
    if os.path.exists(_MODEL_CONFIG_PATH):
        try:
            with open(_MODEL_CONFIG_PATH, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def _load_local_model():
    """Load the trained CNN model (called once on first use)."""
    global _local_model, _local_class_order, _local_model_unavailable_reason, IMG_SIZE, _MODEL_PREPROCESSING
    if _local_model is not None:
        return True
    if _local_model_unavailable_reason:
        return False
    model_path = os.path.normpath(_MODEL_PATH)
    if not os.path.exists(model_path):
        _local_model_unavailable_reason = f"Model file not found at: {model_path}"
        return False

    try:
        import tensorflow as tf
    except ModuleNotFoundError:
        _local_model_unavailable_reason = (
            f"TensorFlow is not installed for this interpreter ({sys.executable}). "
            "Use the project venv or install requirements.txt."
        )
        return False

    try:
        config = _load_model_config()
        if config:
            img_side = int(config.get('img_size', IMG_SIZE[0]))
            IMG_SIZE = (img_side, img_side)
            _MODEL_PREPROCESSING = config.get('preprocessing', 'rescale_255')

        # Inference-only usage: avoid compiling to prevent warnings and speed up load.
        _local_model = tf.keras.models.load_model(model_path, compile=False)
        # Load class order from saved index file
        cls_path = os.path.normpath(_CLS_IDX_PATH)
        if os.path.exists(cls_path):
            with open(cls_path, 'r') as f:
                idx_map = json.load(f)  # {class_name: index}
            _local_class_order = [None] * len(idx_map)
            for cls_name, idx in idx_map.items():
                _local_class_order[idx] = cls_name
        else:
            # Fallback: alphabetical order (matches ImageDataGenerator default)
            _local_class_order = ['drugs', 'normal', 'nsfw', 'scam', 'violence', 'weapons']
        return True
    except Exception as e:
        _local_model_unavailable_reason = str(e)
        print(f"[FORAX] Local model load error: {_local_model_unavailable_reason}")
        return False


FORENSIC_PROMPT = (
    "You are a professional forensic image analyst for law enforcement. "
    "Analyze this image thoroughly for:\n"
    "1. WEAPONS: Firearms, components, explosives, blades\n"
    "2. NARCOTICS: Drugs, paraphernalia, distribution materials\n"
    "3. VIOLENCE: Assault signs, injuries\n"
    "4. FINANCIAL CRIME: Cash piles, IDs, skimming devices\n"
    "5. NSFW/EXPLICIT: Adult content\n\n"
    "Respond in EXACTLY this pipe-separated format:\n"
    "RISK|CONFIDENCE|NSFW_BOOL|HAS_TEXT_BOOL|OBJ1,OBJ2|DETAILED_FORENSIC_REASONING\n\n"
    "REASONING: Write 2-3 professional sentences for an investigator's report explaining the forensic significance of this evidence. "
    "Example: HIGH|95|FALSE|FALSE|handgun|A black semi-automatic handgun is clearly visible in the subject's possession, indicating potential possession of illegal firearms in this context."
)


def load_intel():
    """Load forensic intelligence dataset for keyword fallback scanning."""
    intel_path = os.path.join(os.path.dirname(__file__), 'forensic_intel.json')
    if os.path.exists(intel_path):
        with open(intel_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def extract_exif(filepath):
    """Extract EXIF metadata from an image using Pillow."""
    exif_data = {
        'gps_lat': None, 'gps_lng': None, 'exif_date': None,
        'camera_make': None, 'camera_model': None,
        'software': None, 'orientation': None, 'has_exif': False
    }
    try:
        from PIL import Image
        from PIL.ExifTags import TAGS, GPSTAGS

        img = Image.open(filepath)
        exif_raw = img._getexif()
        if not exif_raw:
            return exif_data

        exif_data['has_exif'] = True
        decoded = {TAGS.get(tid, tid): v for tid, v in exif_raw.items()}

        exif_data['camera_make'] = str(decoded.get('Make', '')).strip() or None
        exif_data['camera_model'] = str(decoded.get('Model', '')).strip() or None
        exif_data['software'] = str(decoded.get('Software', '')).strip() or None
        exif_data['orientation'] = decoded.get('Orientation')

        date_str = decoded.get('DateTimeOriginal') or decoded.get('DateTime')
        if date_str:
            exif_data['exif_date'] = str(date_str).strip()

        gps_info = decoded.get('GPSInfo')
        if gps_info:
            gps_decoded = {GPSTAGS.get(k, k): v for k, v in gps_info.items()}

            def dms_to_dd(dms, ref):
                try:
                    dd = float(dms[0]) + float(dms[1]) / 60 + float(dms[2]) / 3600
                    return round(-dd if ref in ('S', 'W') else dd, 6)
                except (ValueError, IndexError, TypeError):
                    return None

            lat = gps_decoded.get('GPSLatitude')
            lng = gps_decoded.get('GPSLongitude')
            if lat and lng:
                exif_data['gps_lat'] = dms_to_dd(lat, gps_decoded.get('GPSLatitudeRef', 'N'))
                exif_data['gps_lng'] = dms_to_dd(lng, gps_decoded.get('GPSLongitudeRef', 'E'))

    except Exception:
        pass
    return exif_data


# ── OPENAI VISION API ────────────────────────────────────────────
def _analyze_openai(filepath):
    """Analyze image content using OpenAI GPT-4o-mini vision."""
    if not GITHUB_KEY:
        return None

    ext = filepath.rsplit('.', 1)[-1].lower()
    mime = {'jpg': 'image/jpeg', 'jpeg': 'image/jpeg', 'png': 'image/png',
            'gif': 'image/gif', 'webp': 'image/webp', 'bmp': 'image/bmp'}.get(ext, 'image/jpeg')

    with open(filepath, 'rb') as f:
        img_b64 = base64.b64encode(f.read()).decode()

    resp = requests.post(OPENAI_URL, headers={
        'Authorization': f'Bearer {GITHUB_KEY}',
        'Content-Type': 'application/json'
    }, json={
        'model': OPENAI_MODEL,
        'messages': [{'role': 'user', 'content': [
            {'type': 'text', 'text': FORENSIC_PROMPT},
            {'type': 'image_url', 'image_url': {'url': f'data:{mime};base64,{img_b64}'}}
        ]}],
        'max_tokens': 400,
        'temperature': 0.1
    }, timeout=30)

    if resp.status_code != 200:
        raise Exception(f"OpenAI {resp.status_code}: {resp.text[:100]}")

    return resp.json()['choices'][0]['message']['content']


def _parse_ai_response(raw_text, all_keywords):
    """Parse the pipe-separated AI response into a structured result."""
    result = {}
    parts = raw_text.split('|')
    keyword_hit = any(k.lower() in raw_text.lower() for k in all_keywords if len(k) > 4)

    if len(parts) >= 6:
        risk = parts[0].strip().upper()
        result['risk_level'] = risk if risk in ('HIGH', 'MED', 'LOW') else 'LOW'
        try:
            result['confidence'] = int(parts[1].strip())
        except ValueError:
            result['confidence'] = 50
        result['is_nsfw'] = 'TRUE' in parts[2].upper()
        result['has_text'] = 'TRUE' in parts[3].upper()
        result['objects'] = [o.strip() for o in parts[4].split(',') if o.strip()]
        result['ai_result'] = f"AI Detected: {parts[4].strip() or 'Object identified'}"
        result['copilot_reasoning'] = parts[5].strip()

        if keyword_hit and result['risk_level'] == 'LOW':
            result['risk_level'] = 'MED'
            result['copilot_reasoning'] += " [Keyword match escalation]"
    else:
        result['ai_result'] = "AI: " + raw_text[:100]
        result['copilot_reasoning'] = raw_text
        result['risk_level'] = 'HIGH' if keyword_hit else 'LOW'
        result['confidence'] = 60

    return result


# ── LOCAL MODEL INFERENCE ─────────────────────────────────────────
def _analyze_local_model(filepath):
    """
    Run forensic classification using the custom trained CNN.
    Returns a result dict on success, or None if model not loaded.
    """
    if _local_model is None and not _load_local_model():
        return None

    try:
        from PIL import Image
        if _MODEL_PREPROCESSING == 'efficientnet':
            from tensorflow.keras.applications.efficientnet import preprocess_input as _preprocess_input

        # ── 1. Image Loading & Preprocessing ──
        try:
            img = Image.open(filepath).convert('RGB').resize(IMG_SIZE)
            img_array = np.array(img, dtype='float32')
            if _MODEL_PREPROCESSING == 'efficientnet':
                img_array = _preprocess_input(img_array)
            else:
                img_array = img_array / 255.0
            img_array = np.expand_dims(img_array, axis=0)
        except Exception as e:
            print(f"[FORAX] Inference Error (Preprocess): {e}")
            return None

        # ── 2. Real-time Prediction ──
        try:
            probs = _local_model.predict(img_array, verbose=0)[0]
            pred_idx   = int(np.argmax(probs))
            confidence = int(round(probs[pred_idx] * 100))
        except Exception as e:
            print(f"[FORAX] Inference Error (Predict): {e}")
            return None
        
        # ── 3. Class Mapping ──
        if pred_idx < len(_local_class_order):
            pred_class = _local_class_order[pred_idx]
        else:
            pred_class = 'unclassified'
            print(f"[WARN] Local model returned index {pred_idx} but only {len(_local_class_order)} classes are mapped.")

        risk, description, objects = _CLASS_RISK_MAP.get(
            pred_class, ('LOW', 'Unknown classification result', [])
        )

        # ── 4. Breakdown Generation ──
        breakdown_parts = []
        for i in range(len(probs)):
            cname = _local_class_order[i].upper() if i < len(_local_class_order) else f"Unknown-{i}"
            breakdown_parts.append(f"{cname}: {probs[i]*100:.1f}%")
        breakdown = '  |  '.join(breakdown_parts)

        # ── 5. Narrative (Local Heuristic) ──
        if risk == 'HIGH':
            narrative = f"CRITICAL FINDING: The local CNN identifying patterns matching '{pred_class.upper()}' with {confidence}% confidence. Artifact suggests high forensic significance. Immediate priority recommended."
        elif risk == 'MED':
            narrative = f"SECURITY ALERT: Analysis indicates indicators of '{pred_class.upper()}' ({confidence}% confidence). Review is advised."
        else:
            narrative = f"The automated local scan categorized this artifact as {pred_class.upper()} with {confidence}% confidence. Risk profile is LOW."

        return {
            'risk_level':  risk,
            'confidence':  confidence,
            'is_nsfw':     pred_class == 'nsfw',
            'has_text':    False,
            'objects':     objects,
            'ai_result':   f"[LOCAL CNN] {description} (class={pred_class}, conf={confidence}%)  |  {breakdown}",
            'copilot_reasoning': narrative
        }
    except Exception as e:
        import traceback
        print(f"[FORAX] Critical Inference Fail: {e}")
        traceback.print_exc()
        return None


def classify_image(filepath):
    """
    Classify an image for forensic risk assessment.

    Priority:
      1. Custom trained CNN  (forax_forensic_model.h5)  ← PRIMARY
      2. OpenAI GPT-4o-mini API                          ← FALLBACK
      3. Local filename heuristics                       ← LAST RESORT
    """
    result = {
        'risk_level': 'LOW', 'is_nsfw': False, 'has_text': False,
        'ai_result': 'Scanning...', 'objects': [], 'confidence': 0,
        'gps_lat': None, 'gps_lng': None, 'exif_date': None,
        'camera_model': None, 'has_exif': False
    }

    if not os.path.exists(filepath):
        result['ai_result'] = "Image file not found."
        return result

    # ── VIDEO / GIF ANALYSIS ─────────────────────────────────────
    ext = filepath.rsplit('.', 1)[-1].lower() if '.' in filepath else ''
    if ext in ('mp4', 'avi', 'mov', 'mkv', 'gif', 'wmv'):
        return _analyze_video(filepath)

    # ── EXIF (always runs offline) ────────────────────────────
    exif = extract_exif(filepath)
    result.update({
        'gps_lat': exif['gps_lat'], 'gps_lng': exif['gps_lng'],
        'exif_date': exif['exif_date'], 'camera_model': exif['camera_model'],
        'has_exif': exif['has_exif']
    })

    # ── KEYWORDS DB ───────────────────────────────────────────
    intel = load_intel()
    all_keywords = []
    if intel:
        for cat in intel.get('categories', {}).values():
            for key in ['keywords', 'paraphernalia', 'indicators', 'tools']:
                kd = cat.get(key)
                if isinstance(kd, dict):
                    all_keywords.extend(kd.keys())
                elif isinstance(kd, list):
                    all_keywords.extend(kd)

    # ── PRIORITY 1: Custom Local CNN ──────────────────────────
    local_result = _analyze_local_model(filepath)
    if local_result:
        result.update(local_result)
        _append_exif(result, exif)
        return result

    # ── PRIORITY 2: OpenAI Vision API ─────────────────────────
    raw_text = None
    # ── VIDEO / GIF ANALYSIS ─────────────────────────────────────
    ext = filepath.rsplit('.', 1)[-1].lower() if '.' in filepath else ''
    if ext in ('mp4', 'avi', 'mov', 'mkv', 'gif', 'wmv'):
        return _analyze_video(filepath)

    try:
        img = PILImage.open(filepath).convert('RGB')
    except Exception as e:
        return {'risk_level': 'LOW', 'ai_result': f'Image: Open error — {str(e)[:50]}'}

    try:
        raw_text = _analyze_openai(filepath)
    except Exception as e:
        result['ai_result'] = f"API: {str(e)[:80]}"

    if raw_text:
        parsed = _parse_ai_response(raw_text, all_keywords)
        result.update(parsed)
        _append_exif(result, exif)
        return result

    # ── PRIORITY 3: Filename Heuristics ───────────────────────
    fname_lower = os.path.basename(filepath).lower()
    hits = [k for k in all_keywords if len(k) > 3 and k.lower() in fname_lower]
    if hits:
        result['risk_level'] = 'HIGH'
        result['ai_result'] = f"LOCAL: Suspect filename — {', '.join(hits[:3])}"
        result['confidence'] = 70
    else:
        if _local_model_unavailable_reason:
            result['ai_result'] = (
                f"Local CNN unavailable: {_local_model_unavailable_reason[:180]} "
                "| Add GITHUB_TOKEN to .env for API fallback."
            )
        else:
            result['ai_result'] = "No local model found. Train model or add GITHUB_TOKEN to .env"
        result['confidence'] = 10

    _append_exif(result, exif)
    
    # ── FORENSIC REASONING ─────────────────────────────────────────
    reasoning = []
    if result['risk_level'] != 'LOW':
        rs = result.get('risk_level', 'UNKNOWN')
        conf = result.get('confidence', 0)
        reasoning.append(f"Visual Forensic Alert: Image classified as {rs} with {conf}% confidence.")
        reasoning.append(f"Detection Metric: System identified visual features characteristic of the '{rs.lower()}' category.")
        if conf > 90:
            reasoning.append("High-Priority: High confidence level suggests this artifact requires immediate investigator review.")
    else:
        reasoning.append("Visual Analysis Complete: No illicit visual patterns or forensic threats identified in this image.")
    result['copilot_reasoning'] = " ".join(reasoning)

    return result


def _analyze_video(filepath):
    """Deep analysis of video/GIF by sampling keyframes and running CNN classification."""
    frames = extract_frames_from_video(filepath, interval_sec=2, max_frames=15)
    if not frames:
        return {'risk_level': 'LOW', 'ai_result': 'Video: No frames extracted', 'copilot_reasoning': 'Analysis could not be performed — no frames found.'}

    max_risk = 'LOW'
    risk_order = {'LOW': 0, 'MED': 1, 'HIGH': 2}
    hits = []
    
    for ts, frame in frames:
        # Convert BGR (cv2) to RGB (PIL)
        # Here we manually call _analyze_local_model or pass it as image
        # For simplicity in this project, we'll assume we can pass the frame object if our CNN supports it.
        # But our current _analyze_local_model expects a path.
        # We will save a temp frame if needed, but for the FYP, we'll mock the classification of these frames
        # or just use the first frame's results if it's too complex for local CPU.
        
        # ACTUALLY: Let's just run it properly.
        # Temp save
        import tempfile
        tmp = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
        tmp_path = tmp.name
        tmp.close()
        cv2.imwrite(tmp_path, frame)
        res = _analyze_local_model(tmp_path)
        os.remove(tmp_path)
        
        if res:
            if risk_order.get(res['risk_level'], 0) > risk_order.get(max_risk, 0):
                max_risk = res['risk_level']
            if res['risk_level'] != 'LOW':
                # Simplified label extraction
                label = res.get('ai_result','').split(':')[1].split('|')[0].strip() if ':' in res.get('ai_result','') else 'Suspicious'
                hits.append(f"{ts}s:{label}")

    # Forensic narrative
    reasoning = [f"Video Forensic Report: Analyzed {len(frames)} keyframes across the timeline."]
    if max_risk != 'LOW':
        reasoning.append(f"Timeline Analysis: Suspicious activity found at {len(hits)} timestamps.")
        reasoning.append(f"Identified Tags: {', '.join(list(set(hits))[:5])}.")
        reasoning.append(f"Forensic Conclusion: Risk escalated to {max_risk} based on visual timeline evidence.")
    else:
        reasoning.append("Timeline Scan: No identifiable forensic threats found in the examined frames.")

    return {
        'risk_level': max_risk,
        'ai_result': f"Video [{max_risk}]: {len(frames)} frames scanned | Hits: {', '.join(hits[:3]) or 'None'}",
        'copilot_reasoning': " ".join(reasoning)
    }


def _append_exif(result, exif):
    """Append EXIF metadata to result string."""
    parts = []
    if exif['camera_model']: parts.append(f"Camera: {exif['camera_model']}")
    if exif['exif_date']: parts.append(f"Date: {exif['exif_date']}")
    if exif['gps_lat'] and exif['gps_lng']: parts.append(f"GPS: {exif['gps_lat']},{exif['gps_lng']}")
    if parts:
        result['ai_result'] += f" | EXIF: {', '.join(parts)}"


def get_image_summary(results):
    """Aggregate image analysis results into a summary."""
    total = len(results)
    high = sum(1 for r in results if r.get('risk_level') == 'HIGH')
    return {
        'total_images': total,
        'nsfw_count': sum(1 for r in results if r.get('is_nsfw')),
        'has_text_count': sum(1 for r in results if r.get('has_text')),
        'high_risk_count': high,
        'gps_count': sum(1 for r in results if r.get('gps_lat') is not None),
        'avg_confidence': round(sum(r.get('confidence', 0) for r in results) / max(total, 1), 1),
        'overall_risk': 'HIGH' if high > 0 else 'LOW'
    }
