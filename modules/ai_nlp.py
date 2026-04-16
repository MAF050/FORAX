"""
FORAX — NLP Forensic Text Analyzer
Weighted scoring engine with pattern detection + optional Gemini AI enhancement.
Works fully offline using forensic_intel.json dataset.
"""

import os, re, json, requests
import numpy as np

from .forensic_parser import get_parser_for_ext

# ── AI CONFIG ──────────────────────────────────────────────────
GITHUB_KEY = os.getenv('GITHUB_TOKEN', '').strip()
OPENAI_URL = 'https://models.inference.ai.azure.com/chat/completions'
OPENAI_MODEL = 'gpt-4o-mini'

try:
    import joblib
except ImportError:
    joblib = None

# ── DATASET LOADER ──────────────────────────────────────────────
_intel_cache = None

# ── OPTIONAL ML TEXT MODEL ──────────────────────────────────────
_text_model = None
_text_model_classes = None

_MODULE_DIR = os.path.dirname(__file__)
_TEXT_MODEL_PATH = os.path.normpath(os.path.join(_MODULE_DIR, '..', 'model_training', 'nlp_text_model.joblib'))
_TEXT_LABEL_PATH = os.path.normpath(os.path.join(_MODULE_DIR, '..', 'model_training', 'text_label_map.json'))

_TEXT_CLASS_RISK = {
    'violence': ('HIGH', 'Violence content detected'),
    'threat': ('HIGH', 'Threat or coercion detected'),
    'weapon': ('HIGH', 'Weapon-related content detected'),
    'trafficking': ('HIGH', 'Human trafficking indicators detected'),
    'drugs': ('MED', 'Drug-related content detected'),
    'harassment': ('MED', 'Harassment indicators detected'),
    'fraud': ('HIGH', 'Financial fraud indicators detected'),
    'scam': ('HIGH', 'Scam or phishing indicators detected'),
    'normal': ('LOW', 'No threat indicators detected')
}


def _load_text_model():
    """Load the optional ML text model once at startup."""
    global _text_model, _text_model_classes
    if joblib is None:
        return False
    if _text_model is not None:
        return True
    if not os.path.exists(_TEXT_MODEL_PATH):
        return False
    try:
        _text_model = joblib.load(_TEXT_MODEL_PATH)
        # Try to read classes from model if available
        clf = getattr(_text_model, 'named_steps', {}).get('clf') if hasattr(_text_model, 'named_steps') else None
        if clf is not None and hasattr(clf, 'classes_'):
            _text_model_classes = list(clf.classes_)
        else:
            _text_model_classes = None
        return True
    except Exception:
        return False


def _predict_text_model(text):
    """Predict class and risk level using the trained text model (if available)."""
    if _text_model is None and not _load_text_model():
        return None

    try:
        proba = None
        if hasattr(_text_model, 'predict_proba'):
            proba = _text_model.predict_proba([text])[0]
            classes = getattr(_text_model, 'classes_', None)
            if classes is None:
                classes = _text_model_classes
            else:
                classes = list(classes)
        else:
            pred = _text_model.predict([text])[0]
            classes = _text_model_classes
            return {
                'class': pred,
                'confidence': 50,
                'risk_level': _TEXT_CLASS_RISK.get(pred, ('LOW', 'Text analysis'))[0],
                'summary': _TEXT_CLASS_RISK.get(pred, ('LOW', 'Text analysis'))[1]
            }

        if classes is None:
            return None

        idx = int(np.argmax(proba))
        pred = classes[idx]
        confidence = int(round(proba[idx] * 100))
        risk, summary = _TEXT_CLASS_RISK.get(pred, ('LOW', 'Text analysis'))

        return {
            'class': pred,
            'confidence': confidence,
            'risk_level': risk,
            'summary': summary
        }
    except Exception:
        return None

def load_intel():
    """Load and cache the forensic intelligence dataset."""
    global _intel_cache
    if _intel_cache is not None:
        return _intel_cache
    intel_path = os.path.join(os.path.dirname(__file__), 'forensic_intel.json')
    if os.path.exists(intel_path):
        with open(intel_path, 'r', encoding='utf-8') as f:
            _intel_cache = json.load(f)
            return _intel_cache
    return {}


# ── WEIGHTED SCORING ENGINE ─────────────────────────────────────
def score_text(text):
    """
    Analyze text using the weighted forensic intelligence dataset.
    Returns a detailed scoring breakdown with risk level, matched keywords,
    entity detections, and per-category threat scores.
    """
    intel = load_intel()
    if not intel:
        return {'risk_level': 'LOW', 'score': 0, 'categories': {}, 'entities': {}, 'matched_keywords': []}

    text_lower = text.lower()
    categories = intel.get('categories', {})
    scoring = intel.get('scoring', {})
    LOW_T = scoring.get('LOW_threshold', 10)
    HIGH_T = scoring.get('HIGH_threshold', 25)
    MULTI_MULT = scoring.get('multi_category_multiplier', 1.5)

    total_score = 0
    cat_scores = {}
    all_matches = []
    cats_hit = 0

    for cat_name, cat_data in categories.items():
        cat_score = 0
        cat_matches = []
        severity = cat_data.get('severity', 5)

        # Scan all keyword dictionaries in this category
        keyword_dicts = []
        for key in ['keywords', 'paraphernalia', 'indicators', 'tools']:
            if key in cat_data and isinstance(cat_data[key], dict):
                keyword_dicts.append(cat_data[key])

        for kw_dict in keyword_dicts:
            for keyword, weight in kw_dict.items():
                if keyword and len(keyword) > 2:
                    # Use word boundaries to prevent false positives like 'load' in 'download'
                    pattern = rf"\b{re.escape(keyword.lower())}\b"
                    matches = re.findall(pattern, text_lower)
                    if matches:
                        count = min(len(matches), 5)
                        hit_score = weight * count
                        cat_score += hit_score
                        cat_matches.append({
                            'keyword': keyword,
                            'weight': weight,
                            'count': count,
                            'score': hit_score
                        })

        # Check regex patterns
        for pattern in cat_data.get('patterns', []):
            try:
                matches = re.findall(pattern, text_lower)
                if matches:
                    pat_score = len(matches) * 3
                    cat_score += pat_score
                    cat_matches.append({
                        'keyword': f'[PATTERN] {pattern[:30]}',
                        'weight': 3,
                        'count': len(matches),
                        'score': pat_score
                    })
            except re.error:
                pass

        if cat_score > 0:
            cats_hit += 1
            # Apply severity multiplier
            adjusted = cat_score * (severity / 10)
            cat_scores[cat_name] = {
                'raw_score': cat_score,
                'adjusted_score': round(adjusted, 1),
                'severity': severity,
                'matches': cat_matches
            }
            total_score += adjusted
            all_matches.extend(cat_matches)

    # Multi-category multiplier — hitting multiple categories is worse
    if cats_hit >= 3:
        total_score *= MULTI_MULT
    elif cats_hit >= 2:
        total_score *= 1.2

    # ── ENTITY DETECTION ──────────────────────────────────────
    entity_patterns = intel.get('entity_patterns', {})
    entities = {}
    for ent_name, ent_data in entity_patterns.items():
        try:
            found = re.findall(ent_data['regex'], text)
            if found:
                # Deduplicate
                unique = list(set(found if isinstance(found[0], str) else [f[0] for f in found]))[:20]
                entities[ent_name] = {
                    'count': len(found),
                    'samples': unique[:5],
                    'weight': ent_data.get('weight', 1)
                }
                total_score += ent_data.get('weight', 1) * min(len(found), 3)
        except re.error:
            pass

    # ── RISK DETERMINATION ────────────────────────────────────
    total_score = round(total_score, 1)
    if total_score >= HIGH_T:
        risk = 'HIGH'
    elif total_score >= LOW_T:
        risk = 'MED'
    else:
        risk = 'LOW'

    # Top categories by score
    sorted_cats = sorted(cat_scores.items(), key=lambda x: x[1]['adjusted_score'], reverse=True)

    return {
        'risk_level': risk,
        'score': total_score,
        'categories': cat_scores,
        'entities': entities,
        'matched_keywords': all_matches,
        'top_categories': [(name, data['adjusted_score']) for name, data in sorted_cats[:3]],
        'categories_hit': cats_hit
    }

def _generate_ai_reasoning(text, detections, risk):
    """Generate professional forensic reasoning using GitHub GPT-4o-mini API."""
    if not GITHUB_KEY:
        return None
    
    # Prune text for prompt (first 1000 chars)
    snippet = text[:1000].replace('\n', ' ')
    cat_hits = [c[0].replace('_', ' ').title() for c in detections.get('top_categories', [])]
    
    prompt = (
        "You are a professional forensic analyst. Analyze these findings and provide a professional, "
        "concise forensic narrative for an investigator's report.\n\n"
        f"FINDINGS:\n- Risk Level: {risk}\n- Categories Match: {', '.join(cat_hits)}\n"
        f"- Score: {detections.get('score', 0)}\n\n"
        f"TEXT SNIPPET:\n{snippet}\n\n"
        "GOAL: Write 2-3 formal sentences explaining the forensic significance of these hits. "
        "Do not use markdown formatting. Be professional."
    )
    
    try:
        resp = requests.post(OPENAI_URL, headers={
            'Authorization': f'Bearer {GITHUB_KEY}',
            'Content-Type': 'application/json'
        }, json={
            'model': OPENAI_MODEL,
            'messages': [{'role': 'user', 'content': prompt}],
            'max_tokens': 150,
            'temperature': 0.3
        }, timeout=8)
        
        if resp.status_code == 200:
            return resp.json()['choices'][0]['message']['content'].strip()
    except Exception:
        pass
    return None


# ── FILE ANALYZER ───────────────────────────────────────────────
def analyze_file(filepath, filetype='text'):
    """
    Analyze a text file using the weighted scoring engine + optional Gemini AI.
    Returns a comprehensive forensic analysis result.
    """
    result = {
        'risk_level': 'LOW',
        'ai_result': 'NLP: Analyzed',
        'score': 0,
        'keyword_hits': {},
        'patterns': {},
        'entities': {},
        'line_count': 0,
        'word_count': 0,
        'categories_hit': 0,
        'confidence': 'LOCAL'
    }

    if not os.path.exists(filepath):
        result['ai_result'] = 'NLP: File not found'
        return result

    # ── CONTENT EXTRACTION ────────────────────────────────────
    ext = filepath.rsplit('.', 1)[-1].lower() if '.' in filepath else ''
    parser_func = get_parser_for_ext(ext)
    
    try:
        if parser_func:
            text = parser_func(filepath)
        else:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read(200_000)  # Increased for documentation support
    except Exception as e:
        result['ai_result'] = f'NLP: Read error — {str(e)[:50]}'
        return result

    if not text or not text.strip() or text.startswith("[Parser Error]"):
        result['ai_result'] = text if text.startswith("[Parser Error]") else 'NLP: Empty or unreadable file'
        return result

    result['line_count'] = len(text.splitlines())
    result['word_count'] = len(text.split())

    # ── LOCAL WEIGHTED SCORING ────────────────────────────────
    score_result = score_text(text)
    result['risk_level'] = score_result['risk_level']
    result['score'] = score_result['score']
    result['categories_hit'] = score_result['categories_hit']
    result['entities'] = score_result['entities']

    # ── OPTIONAL ML TEXT MODEL ────────────────────────────────
    ml_result = _predict_text_model(text)
    if ml_result:
        result['ml_class'] = ml_result.get('class')
        result['ml_confidence'] = ml_result.get('confidence')
        result['ml_risk_level'] = ml_result.get('risk_level')

    # Build human-readable AI result
    top_cats = score_result.get('top_categories', [])
    matched = score_result.get('matched_keywords', [])

    if score_result['score'] > 0:
        top_kw = [m['keyword'] for m in sorted(matched, key=lambda x: x['score'], reverse=True)[:5]
                  if not m['keyword'].startswith('[PATTERN]')]
        cat_names = [c.replace('_', ' ').title() for c, _ in top_cats]

        parts = []
        if cat_names:
            parts.append(f"Categories: {', '.join(cat_names)}")
        if top_kw:
            parts.append(f"Keywords: {', '.join(top_kw[:5])}")
        parts.append(f"Score: {score_result['score']}")

        result['ai_result'] = f"NLP [{result['risk_level']}]: {' | '.join(parts)}"
        result['keyword_hits'] = {cat: [m['keyword'] for m in data['matches']]
                                  for cat, data in score_result['categories'].items()}
    else:
        result['ai_result'] = f'NLP: No threats detected ({result["word_count"]} words, {result["line_count"]} lines)'

    # ── ML RESULT MERGE ───────────────────────────────────────
    if ml_result:
        ml_class = ml_result.get('class', 'unknown')
        ml_conf = ml_result.get('confidence', 0)
        result['ai_result'] += f" | ML: {str(ml_class).upper()} ({ml_conf}%)"

        risk_order = {'LOW': 0, 'MED': 1, 'HIGH': 2}
        ml_risk = ml_result.get('risk_level', 'LOW')
        if risk_order.get(ml_risk, 0) > risk_order.get(result['risk_level'], 0):
            result['risk_level'] = ml_risk
            result['ai_result'] += " | ML escalated risk"

    # ── ENTITY SUMMARY ────────────────────────────────────────
    if score_result['entities']:
        ent_parts = []
        for ent_name, ent_data in score_result['entities'].items():
            ent_parts.append(f"{ent_name.replace('_',' ').title()}: {ent_data['count']}")
        if ent_parts:
            result['ai_result'] += f" | Entities: {', '.join(ent_parts[:4])}"

    # ── FORENSIC REASONING ─────────────────────────────────────────
    # Priority: 1. AI Narrative  2. Heuristic Narrative
    ai_reasoning = _generate_ai_reasoning(text, score_result, result['risk_level'])
    if ai_reasoning:
        result['copilot_reasoning'] = ai_reasoning
    else:
        # Heuristic Traceability Narrative
        reasoning = []
        if result['risk_level'] != 'LOW':
            reasoning.append(f"Forensic Indicator: Risk level is {result['risk_level']} based on weighted scoring ({result['score']} points).")
            if top_cats:
                top_cat_list = [c[0].replace('_', ' ').title() for c in top_cats]
                reasoning.append(f"Pattern Match: Primary threat patterns identified: {', '.join(top_cat_list)}.")
            if result['entities']:
                reasoning.append(f"Entity Analysis: Detected {len(result['entities'])} types of sensitive entities.")
        else:
            reasoning.append("The file was scanned for forensic signatures and linguistic patterns. No actionable threats were identified.")
        result['copilot_reasoning'] = " ".join(reasoning)

    return result


def get_threat_summary(results):
    """Aggregate threat summary across multiple analyzed files."""
    total = len(results)
    high_count = sum(1 for r in results if r.get('risk_level') == 'HIGH')
    med_count = sum(1 for r in results if r.get('risk_level') == 'MED')

    # Collect all category hits across files
    all_cats = {}
    for r in results:
        for cat, keywords in r.get('keyword_hits', {}).items():
            if cat not in all_cats:
                all_cats[cat] = set()
            all_cats[cat].update(keywords)

    sorted_cats = sorted(all_cats.items(), key=lambda x: len(x[1]), reverse=True)
    threat_cats = [(cat, len(kws)) for cat, kws in sorted_cats]

    return {
        'total_files': total,
        'high_risk': high_count,
        'med_risk': med_count,
        'low_risk': total - high_count - med_count,
        'threat_cats': threat_cats,
        'overall_risk': 'HIGH' if high_count > 0 else 'MED' if med_count > 0 else 'LOW',
        'total_score': sum(r.get('score', 0) for r in results),
    }
