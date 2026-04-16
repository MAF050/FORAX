"""
FORAX — Unified Forensic Parser
Handles high-fidelity text and metadata extraction from diverse forensic formats:
PDF, DOCX, XLSX, PPTX, Video/GIF.
"""

import os
import re
import json
import logging

try:
    import pdfplumber
except ImportError:
    pdfplumber = None

try:
    from docx import Document as DocxDocument
except ImportError:
    DocxDocument = None

try:
    import openpyxl
except ImportError:
    openpyxl = None

try:
    import cv2
except ImportError:
    cv2 = None

logger = logging.getLogger(__name__)

def extract_text_from_pdf(filepath):
    """Extract full text from PDF using pdfplumber."""
    if not pdfplumber:
        return "[Parser Error] pdfplumber not installed"
    
    text_content = []
    try:
        with pdfplumber.open(filepath) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text_content.append(page_text)
        return "\n".join(text_content)
    except Exception as e:
        logger.error(f"PDF Extraction Error: {e}")
        return f"[Parser Error] Failed to read PDF: {str(e)}"

def extract_text_from_docx(filepath):
    """Extract full text from DOCX using python-docx."""
    if not DocxDocument:
        return "[Parser Error] python-docx not installed"
    
    try:
        doc = DocxDocument(filepath)
        return "\n".join([para.text for para in doc.paragraphs])
    except Exception as e:
        logger.error(f"DOCX Extraction Error: {e}")
        return f"[Parser Error] Failed to read DOCX: {str(e)}"

def extract_text_from_excel(filepath):
    """Extract text from all cells in an Excel file."""
    if not openpyxl:
        return "[Parser Error] openpyxl not installed"
    
    content = []
    try:
        wb = openpyxl.load_workbook(filepath, data_only=True)
        for sheet in wb.worksheets:
            content.append(f"--- Sheet: {sheet.title} ---")
            for row in sheet.iter_rows(values_only=True):
                row_str = " | ".join([str(val) if val is not None else "" for val in row])
                if row_str.strip():
                    content.append(row_str)
        return "\n".join(content)
    except Exception as e:
        # Fallback to simple pandas if possible, or return error
        return f"[Parser Error] Failed to read Excel: {str(e)}"

def extract_frames_from_video(filepath, interval_sec=2, max_frames=20):
    """Extract keyframes from a video file at specified intervals."""
    if not cv2:
        return []
    
    frames = []
    try:
        cap = cv2.VideoCapture(filepath)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0: fps = 25
        
        frame_interval = int(fps * interval_sec)
        count = 0
        extracted_count = 0
        
        while cap.isOpened() and extracted_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            if count % frame_interval == 0:
                # Store frame path or frame object
                # For FORAX, we'll return a list of (timestamp, frame_bgr)
                timestamp = round(count / fps, 2)
                frames.append((timestamp, frame))
                extracted_count += 1
            
            count += 1
        
        cap.release()
    except Exception as e:
        logger.error(f"Video Extraction Error: {e}")
    
    return frames

def get_parser_for_ext(ext):
    """Route file extension to appropriate parser."""
    ext = ext.lower().strip('.')
    if ext == 'pdf':
        return extract_text_from_pdf
    if ext in ('docx', 'docm'):
        return extract_text_from_docx
    if ext in ('xlsx', 'xlsm', 'xltx', 'xltm'):
        return extract_text_from_excel
    return None
