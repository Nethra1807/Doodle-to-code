import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import numpy as np
import os
import traceback

from utils.preprocess import preprocess_image
from utils.predictor import UIComponentPredictor
from utils.html_mapper import map_class_to_html

# ─── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Doodle to Code Generator",
    layout="wide",
    page_icon="🎨",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════════════════════
#  CUSTOM CSS  — Full SaaS dashboard theme
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

/* ── Reset ── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

/* ── Global Background ── */
html, body,
[data-testid="stAppViewContainer"],
[data-testid="stApp"] {
    font-family: 'Inter', sans-serif !important;
    background: #0d1117 !important;
    color: #e2e8f0 !important;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stDecoration"]  { display: none; }
[data-testid="stToolbar"]     { display: none; }

/* ── Main block ── */
[data-testid="stAppViewContainer"] > .main > .block-container {
    padding: 0 2rem 3rem !important;
    max-width: 1440px !important;
}

/* ── Remove top gap in the main content ── */
[data-testid="stVerticalBlock"] > [style*="flex-direction: column"] > [data-testid="stVerticalBlock"] {
    gap: 0 !important;
}

/* ═══ SIDEBAR ════════════════════════════════════════════════════════════════ */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0a0f1e 0%, #0d1b3e 50%, #0f2060 100%) !important;
    border-right: 1px solid rgba(99,179,237,0.12) !important;
    box-shadow: 4px 0 30px rgba(0,0,0,0.5) !important;
}
[data-testid="stSidebar"] * { color: #94a3b8 !important; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #f0f6ff !important;
    font-weight: 700 !important;
    letter-spacing: -0.3px;
}
[data-testid="stSidebar"] label {
    color: #7dd3fc !important;
    font-weight: 500 !important;
    font-size: 0.8rem !important;
    letter-spacing: 0.15px !important;
}
[data-testid="stSidebar"] [data-testid="stSelectbox"] > div:first-child,
[data-testid="stSidebar"] [data-testid="stSlider"],
[data-testid="stSidebar"] [data-testid="stColorPicker"] {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(99,179,237,0.15) !important;
    border-radius: 10px !important;
    padding: 4px 8px !important;
    margin-bottom: 6px !important;
}
[data-testid="stSidebar"] hr {
    border: none !important;
    border-top: 1px solid rgba(99,179,237,0.15) !important;
    margin: 18px 0 !important;
}
[data-testid="stSidebar"] .stButton > button {
    background: linear-gradient(135deg, rgba(239,68,68,0.2), rgba(220,38,38,0.25)) !important;
    color: #fca5a5 !important;
    border: 1px solid rgba(239,68,68,0.35) !important;
    border-radius: 10px !important;
    font-size: 0.85rem !important;
    font-weight: 600 !important;
    box-shadow: none !important;
    padding: 0.5rem 1rem !important;
    transition: all 0.2s ease !important;
    width: 100% !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: linear-gradient(135deg, rgba(239,68,68,0.35), rgba(220,38,38,0.4)) !important;
    border-color: rgba(239,68,68,0.6) !important;
    color: #fecaca !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 12px rgba(239,68,68,0.25) !important;
}

/* ═══ ANIMATED HERO ══════════════════════════════════════════════════════════ */
@keyframes gradientShift {
    0%   { background-position: 0% 50%; }
    50%  { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}
@keyframes float {
    0%, 100% { transform: translateY(0px); }
    50%       { transform: translateY(-6px); }
}
@keyframes pulse-glow {
    0%, 100% { box-shadow: 0 0 20px rgba(59,130,246,0.3), 0 0 60px rgba(59,130,246,0.1); }
    50%       { box-shadow: 0 0 40px rgba(59,130,246,0.5), 0 0 100px rgba(59,130,246,0.2); }
}
@keyframes shimmer {
    0%   { opacity: 0.6; }
    50%  { opacity: 1.0; }
    100% { opacity: 0.6; }
}
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(20px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes badgePop {
    0%   { transform: scale(0.8); opacity: 0; }
    70%  { transform: scale(1.05); }
    100% { transform: scale(1); opacity: 1; }
}
@keyframes barFill {
    from { width: 0%; }
}

.hero-card {
    background: linear-gradient(130deg, #050d2d, #0a1962, #1230a0, #0d3693, #050d2d);
    background-size: 300% 300%;
    animation: gradientShift 8s ease infinite;
    border-radius: 24px;
    padding: 3rem 3.5rem 2.5rem;
    margin: 1.2rem 0 1.8rem;
    position: relative;
    overflow: hidden;
    border: 1px solid rgba(99,179,237,0.15);
    animation: gradientShift 8s ease infinite, pulse-glow 4s ease-in-out infinite;
}
.hero-card::before {
    content: '';
    position: absolute; top: -80px; right: -80px;
    width: 320px; height: 320px; border-radius: 50%;
    background: radial-gradient(circle, rgba(59,130,246,0.18) 0%, transparent 70%);
    pointer-events: none;
}
.hero-card::after {
    content: '';
    position: absolute; bottom: -100px; left: -60px;
    width: 360px; height: 360px; border-radius: 50%;
    background: radial-gradient(circle, rgba(139,92,246,0.12) 0%, transparent 70%);
    pointer-events: none;
}
.hero-inner { position: relative; z-index: 2; text-align: center; animation: fadeInUp 0.6s ease both; }
.hero-pill {
    display: inline-block;
    background: rgba(99,179,237,0.12);
    border: 1px solid rgba(99,179,237,0.3);
    color: #7dd3fc !important;
    font-size: 0.7rem; font-weight: 700;
    letter-spacing: 2px; text-transform: uppercase;
    padding: 5px 18px; border-radius: 999px;
    margin-bottom: 1rem;
    animation: shimmer 3s ease-in-out infinite;
}
.hero-title {
    font-size: 3rem; font-weight: 900;
    background: linear-gradient(135deg, #ffffff 0%, #93c5fd 50%, #818cf8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: -1px;
    margin-bottom: 0.8rem;
    line-height: 1.1;
    animation: float 6s ease-in-out infinite;
}
.hero-subtitle {
    font-size: 1.05rem; font-weight: 400;
    color: rgba(148,163,184,0.9) !important;
    max-width: 600px; margin: 0 auto 1.6rem;
    line-height: 1.7;
}
.hero-stats {
    display: flex; justify-content: center; gap: 0;
    margin-top: 1.2rem;
    position: relative; z-index: 1;
    border-top: 1px solid rgba(99,179,237,0.15);
    padding-top: 1.4rem;
}
.hero-stat {
    text-align: center;
    padding: 0 2.5rem;
    border-right: 1px solid rgba(99,179,237,0.15);
}
.hero-stat:last-child { border-right: none; }
.hero-stat-value {
    font-size: 1.6rem; font-weight: 800;
    background: linear-gradient(135deg, #60a5fa, #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.hero-stat-label {
    font-size: 0.7rem; font-weight: 600;
    color: rgba(148,163,184,0.6) !important;
    text-transform: uppercase; letter-spacing: 1px;
    margin-top: 2px;
}

/* ═══ SECTION CARDS ══════════════════════════════════════════════════════════ */
.card {
    background: linear-gradient(145deg, #131d35, #0e1729);
    border-radius: 20px;
    padding: 1.5rem 1.75rem 1.2rem;
    margin-bottom: 1.4rem;
    border: 1px solid rgba(99,179,237,0.1);
    box-shadow: 0 4px 30px rgba(0,0,0,0.3), inset 0 1px 0 rgba(255,255,255,0.04);
    transition: box-shadow 0.3s ease, transform 0.25s ease, border-color 0.3s ease;
    position: relative;
    overflow: hidden;
}
.card::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(99,179,237,0.3), transparent);
}
.card:hover {
    box-shadow: 0 8px 40px rgba(59,130,246,0.15), 0 2px 8px rgba(0,0,0,0.3);
    transform: translateY(-2px);
    border-color: rgba(99,179,237,0.22);
}
.card-title {
    font-size: 0.95rem; font-weight: 700;
    color: #e2e8f0 !important;
    margin-bottom: 1.1rem;
    padding-bottom: 0.8rem;
    border-bottom: 1px solid rgba(99,179,237,0.12);
    display: flex; align-items: center; gap: 10px;
    letter-spacing: -0.2px;
}
.card-title-icon {
    width: 32px; height: 32px;
    background: linear-gradient(135deg, rgba(59,130,246,0.25), rgba(139,92,246,0.2));
    border: 1px solid rgba(99,179,237,0.2);
    border-radius: 9px;
    display: inline-flex; align-items: center; justify-content: center;
    font-size: 1rem;
    flex-shrink: 0;
}
.card-subtitle {
    font-size: 0.8rem;
    color: rgba(148,163,184,0.7) !important;
    margin-top: -0.6rem;
    margin-bottom: 1rem;
    line-height: 1.5;
}

/* ═══ SECTION LABELS ════════════════════════════════════════════════════════ */
.section-label {
    display: flex; align-items: center; gap: 10px;
    font-size: 0.7rem; font-weight: 700;
    letter-spacing: 2px; text-transform: uppercase;
    color: #60a5fa !important;
    margin-bottom: 0.9rem;
    padding-left: 2px;
}
.section-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, rgba(96,165,250,0.3), transparent);
}

/* ═══ BADGES ════════════════════════════════════════════════════════════════ */
.badges        { display: flex; flex-wrap: wrap; gap: 10px; margin: 0.4rem 0 1rem; }
.badge {
    display: inline-flex; align-items: center; gap: 7px;
    padding: 7px 16px; border-radius: 999px;
    font-size: 0.8rem; font-weight: 600;
    transition: transform 0.15s ease, box-shadow 0.15s ease;
    cursor: default;
    animation: badgePop 0.4s cubic-bezier(0.34,1.56,0.64,1) both;
}
.badge:nth-child(1) { animation-delay: 0.05s; }
.badge:nth-child(2) { animation-delay: 0.12s; }
.badge:nth-child(3) { animation-delay: 0.20s; }
.badge:hover { transform: translateY(-3px); }
.badge-blue   { background: rgba(37,99,235,0.15); color: #93c5fd !important; border: 1px solid rgba(37,99,235,0.35); }
.badge-blue:hover { box-shadow: 0 4px 16px rgba(37,99,235,0.35); }
.badge-green  { background: rgba(22,163,74,0.15); color: #86efac !important; border: 1px solid rgba(22,163,74,0.35); }
.badge-green:hover { box-shadow: 0 4px 16px rgba(22,163,74,0.35); }
.badge-purple { background: rgba(124,58,237,0.15); color: #c4b5fd !important; border: 1px solid rgba(124,58,237,0.35); }
.badge-purple:hover { box-shadow: 0 4px 16px rgba(124,58,237,0.35); }
.badge-amber  { background: rgba(180,83,9,0.18); color: #fcd34d !important; border: 1px solid rgba(245,158,11,0.35); }
.badge-amber:hover { box-shadow: 0 4px 16px rgba(245,158,11,0.3); }
.badge-rose   { background: rgba(190,18,60,0.18); color: #fda4af !important; border: 1px solid rgba(244,63,94,0.35); }
.badge-rose:hover { box-shadow: 0 4px 16px rgba(244,63,94,0.3); }

/* ═══ CONFIDENCE BAR ════════════════════════════════════════════════════════ */
.conf-wrap   { margin: 1.1rem 0 0.6rem; }
.conf-header {
    font-size: 0.8rem; color: #94a3b8 !important;
    margin-bottom: 8px; font-weight: 500;
    display: flex; justify-content: space-between; align-items: center;
}
.conf-pct { font-size: 0.95rem; font-weight: 700; color: #60a5fa !important; }
.conf-bar-bg {
    background: rgba(255,255,255,0.07);
    border: 1px solid rgba(99,179,237,0.12);
    border-radius: 99px; height: 10px; overflow: hidden;
}
.conf-bar {
    background: linear-gradient(90deg, #1d4ed8, #60a5fa, #93c5fd);
    height: 10px; border-radius: 99px;
    box-shadow: 0 0 12px rgba(96,165,250,0.5);
    animation: barFill 1.2s cubic-bezier(0.4,0,0.2,1) both;
    animation-delay: 0.3s;
}
.conf-note {
    margin-top: 1rem;
    padding: 12px 16px;
    background: rgba(30,58,138,0.2);
    border: 1px solid rgba(99,179,237,0.15);
    border-left: 3px solid #3b82f6;
    border-radius: 0 10px 10px 0;
    font-size: 0.83rem; color: #94a3b8 !important;
    line-height: 1.5;
}

/* ═══ MAIN BUTTONS ══════════════════════════════════════════════════════════ */
.stButton > button {
    background: linear-gradient(135deg, #1d4ed8 0%, #2563eb 50%, #3b82f6 100%) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.7rem 1.6rem !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    letter-spacing: 0.2px !important;
    box-shadow: 0 4px 20px rgba(37,99,235,0.45), 0 1px 0 rgba(255,255,255,0.1) inset !important;
    transition: all 0.25s cubic-bezier(.4,0,.2,1) !important;
    width: 100% !important;
    position: relative !important;
    overflow: hidden !important;
}
.stButton > button::before {
    content: ''; position: absolute; inset: 0;
    background: linear-gradient(135deg, rgba(255,255,255,0.1), transparent);
    border-radius: 12px;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #1e40af 0%, #1d4ed8 50%, #2563eb 100%) !important;
    box-shadow: 0 8px 28px rgba(37,99,235,0.6), 0 1px 0 rgba(255,255,255,0.15) inset !important;
    transform: translateY(-2px) !important;
}
.stButton > button:active { transform: translateY(0px) !important; }

/* Download button */
.stDownloadButton > button {
    background: rgba(255,255,255,0.06) !important;
    color: #93c5fd !important;
    border: 1px solid rgba(99,179,237,0.25) !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    box-shadow: none !important;
    transition: all 0.2s ease !important;
}
.stDownloadButton > button:hover {
    background: rgba(37,99,235,0.18) !important;
    border-color: rgba(96,165,250,0.5) !important;
    box-shadow: 0 4px 14px rgba(37,99,235,0.25) !important;
    transform: translateY(-1px) !important;
    color: #bfdbfe !important;
}

/* ═══ TABS ═══════════════════════════════════════════════════════════════════ */
[data-testid="stTabs"] {
    background: transparent !important;
}
[data-testid="stTabs"] [role="tablist"] {
    background: rgba(255,255,255,0.04) !important;
    border-radius: 12px !important;
    border: 1px solid rgba(99,179,237,0.1) !important;
    padding: 4px !important;
    gap: 4px !important;
}
[data-testid="stTabs"] [role="tab"] {
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    color: #64748b !important;
    border-radius: 9px !important;
    transition: all 0.2s ease !important;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    background: linear-gradient(135deg, #1d4ed8, #2563eb) !important;
    color: #ffffff !important;
    box-shadow: 0 4px 14px rgba(37,99,235,0.4) !important;
}

/* ═══ MISC ═══════════════════════════════════════════════════════════════════ */
[data-testid="stAlert"] { border-radius: 12px !important; }
[data-testid="stCode"]  {
    border-radius: 14px !important;
    border: 1px solid rgba(99,179,237,0.12) !important;
    background: #0d1117 !important;
}
[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.03) !important;
    border: 2px dashed rgba(99,179,237,0.25) !important;
    border-radius: 16px !important;
    transition: border-color 0.2s, background 0.2s !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: rgba(96,165,250,0.6) !important;
    background: rgba(37,99,235,0.05) !important;
}
[data-testid="stImage"] img {
    border-radius: 14px !important;
    border: 1px solid rgba(99,179,237,0.15) !important;
}
iframe {
    border-radius: 14px !important;
    border: 1px solid rgba(99,179,237,0.15) !important;
}

/* ── Divider ── */
.divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(99,179,237,0.2) 30%, rgba(99,179,237,0.2) 70%, transparent);
    margin: 1.6rem 0;
}

/* ── Canvas wrapper ── */
.canvas-wrapper {
    border-radius: 14px;
    overflow: hidden;
    border: 1px solid rgba(99,179,237,0.15);
    box-shadow: 0 0 30px rgba(0,0,0,0.3) inset;
}

/* ── Placeholder box ── */
.placeholder {
    display: flex; flex-direction: column;
    align-items: center; justify-content: center;
    height: 260px;
    border: 2px dashed rgba(99,179,237,0.2);
    border-radius: 14px;
    background: rgba(255,255,255,0.02);
}
.placeholder-icon { font-size: 3rem; margin-bottom: 0.8rem; opacity: 0.6; }
.placeholder-text { font-size: 0.85rem; color: #475569 !important; text-align: center; font-weight: 500; line-height: 1.6; }

/* ── React info box ── */
.react-info {
    background: rgba(124,58,237,0.12);
    border: 1px solid rgba(139,92,246,0.25);
    border-left: 3px solid #7c3aed;
    border-radius: 0 10px 10px 0;
    padding: 10px 14px;
    font-size: 0.8rem;
    color: #c4b5fd !important;
    margin-bottom: 1rem;
    line-height: 1.5;
}

/* ── Copy btn ── */
.copy-btn {
    display: inline-flex; align-items: center; gap: 6px;
    background: rgba(255,255,255,0.07);
    border: 1px solid rgba(99,179,237,0.2);
    color: #93c5fd !important;
    padding: 6px 14px; border-radius: 8px;
    font-size: 0.8rem; font-weight: 600;
    cursor: pointer;
    transition: all 0.2s ease;
    font-family: 'Inter', sans-serif;
}
.copy-btn:hover {
    background: rgba(37,99,235,0.2);
    border-color: rgba(96,165,250,0.4);
    color: #bfdbfe !important;
}

/* ── Live preview wrapper ── */
.preview-frame-wrapper {
    border-radius: 14px;
    overflow: hidden;
    border: 1px solid rgba(99,179,237,0.15);
    background: #ffffff;
    box-shadow: 0 4px 24px rgba(0,0,0,0.35);
}
.preview-browser-bar {
    background: linear-gradient(90deg, #1a2744, #1e2f5c);
    padding: 8px 16px;
    display: flex; align-items: center; gap: 8px;
    border-bottom: 1px solid rgba(99,179,237,0.1);
}
.preview-dot {
    width: 10px; height: 10px; border-radius: 50%;
}

/* ── Idle ready card ── */
.idle-card {
    background: linear-gradient(145deg, #0f1c35, #0c1526);
    border: 1px solid rgba(99,179,237,0.1);
    border-radius: 20px;
    padding: 3rem 2rem;
    text-align: center;
}
.idle-rocket {
    font-size: 3.5rem;
    display: block;
    margin-bottom: 1rem;
    animation: float 5s ease-in-out infinite;
}

/* ── Footer ── */
.footer {
    text-align: center;
    padding: 1.6rem;
    color: #475569 !important;
    font-size: 0.78rem;
    line-height: 1.8;
}
.footer-title { font-weight: 700; color: #60a5fa !important; font-size: 0.85rem; }
.footer-stack span {
    display: inline-block;
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(99,179,237,0.15);
    border-radius: 6px;
    padding: 2px 10px;
    margin: 2px;
    font-size: 0.75rem;
    color: #94a3b8 !important;
}
.footer-copy { color: #334155 !important; margin-top: 6px; font-size: 0.72rem; }

/* ── Spinner override ── */
[data-testid="stSpinner"] p { color: #60a5fa !important; }
[data-testid="stWarning"] { background: rgba(251,191,36,0.1) !important; border-color: rgba(251,191,36,0.3) !important; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════════════════════
BADGE_COLOR = {
    "Button":                "badge-blue",
    "Radio":                 "badge-purple",
    "Checkbox":              "badge-green",
    "Table":                 "badge-amber",
    "data_table":            "badge-amber",
    "checkbox_unchecked":    "badge-green",
    "radio_button_unchecked":"badge-purple",
    "radio_button_checked":  "badge-purple",
    "Form":                  "badge-blue",
    "alert":                 "badge-rose",
    "switch_enabled":        "badge-green",
    "switch_disabled":       "badge-amber",
    "text_area":             "badge-blue",
}
BADGE_ICON = {
    "Button":                "🔷",
    "Radio":                 "🔘",
    "Checkbox":              "☑️",
    "Table":                 "📊",
    "data_table":            "📊",
    "checkbox_unchecked":    "⬜",
    "radio_button_unchecked":"⭕",
    "radio_button_checked":  "🔵",
    "Form":                  "📝",
    "alert":                 "⚠️",
    "switch_enabled":        "🟢",
    "switch_disabled":       "⚫",
    "text_area":             "📄",
}

def html_to_react(label: str, html: str) -> str:
    """Generate a minimal React functional component for the detected element."""
    comp = label.replace(" ", "").replace("_", "").capitalize()
    return f"""import React from 'react';

// Auto-generated React component for: {label}
const {comp} = () => {{
  return (
    <div style={{{{ fontFamily: 'Inter, sans-serif', padding: '16px' }}}}>
      {{/* {label} component */}}
      <div dangerouslySetInnerHTML={{{{ __html: `{html.replace(chr(96), "'").replace(chr(10), " ")}` }}}} />
    </div>
  );
}};

export default {comp};
"""

def copy_button_js(code_id: str, btn_id: str) -> str:
    """Return an HTML/JS snippet for a clipboard copy button."""
    return f"""
<button class="copy-btn" id="{btn_id}" onclick="
  var text = document.getElementById('{code_id}') ? document.getElementById('{code_id}').innerText : '';
  if(!text) {{
    var pre = document.querySelectorAll('pre code');
    if(pre[0]) text = pre[0].innerText;
  }}
  navigator.clipboard.writeText(text).then(function() {{
    var btn = document.getElementById('{btn_id}');
    btn.innerHTML = '✅ Copied!';
    setTimeout(function(){{ btn.innerHTML = '📋 Copy Code'; }}, 2000);
  }});
" >📋 Copy Code</button>
"""


# ═══════════════════════════════════════════════════════════════════════════════
#  LOAD MODEL  (ML — unchanged)
# ═══════════════════════════════════════════════════════════════════════════════
@st.cache_resource
def load_predictor(model_path):
    return UIComponentPredictor(model_path)

try:
    predictor = load_predictor("model/ui_model.keras")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.text(traceback.format_exc())
    st.stop()


# ═══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
<div style='padding:1rem 0 0.5rem;'>
  <div style='font-size:1.4rem;font-weight:900;color:#f0f6ff;letter-spacing:-0.5px;'>
    🎨 Doodle to Code
  </div>
  <div style='font-size:0.72rem;color:#475569;font-weight:500;margin-top:3px;letter-spacing:0.5px;'>
    AI-POWERED UI GENERATOR
  </div>
</div>
""", unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    st.markdown("""
<div style='font-size:0.68rem;font-weight:700;letter-spacing:2px;color:#60a5fa;text-transform:uppercase;margin-bottom:12px;'>
  ✏️ Canvas Controls
</div>
""", unsafe_allow_html=True)

    drawing_mode  = st.selectbox("Drawing Tool", ("freedraw", "line", "rect", "circle", "transform"))
    stroke_width  = st.slider("Stroke Width", 1, 25, 3)
    stroke_color  = st.color_picker("Stroke Color", "#000000")
    bg_color      = st.color_picker("Background Color", "#ffffff")

    st.markdown("<hr>", unsafe_allow_html=True)

    st.markdown("""
<div style='font-size:0.68rem;font-weight:700;letter-spacing:2px;color:#f87171;text-transform:uppercase;margin-bottom:10px;'>
  🔄 Actions
</div>
""", unsafe_allow_html=True)

    if st.button("🗑️ Reset Canvas"):
        st.session_state["canvas_key"] = st.session_state.get("canvas_key", 0) + 1

    st.markdown("<hr>", unsafe_allow_html=True)

    st.markdown("""
<div style='font-size:0.68rem;font-weight:700;letter-spacing:2px;color:#a78bfa;text-transform:uppercase;margin-bottom:10px;'>
  💡 Quick Tips
</div>
<div style='font-size:0.78rem;line-height:1.8;color:#64748b;'>
  <div style='margin-bottom:5px;'>
    <span style='color:#93c5fd;font-weight:600;'>freedraw</span> — freehand sketch
  </div>
  <div style='margin-bottom:5px;'>
    <span style='color:#93c5fd;font-weight:600;'>rect / circle</span> — geometric
  </div>
  <div style='margin-bottom:5px;'>
    Use <span style='color:#fcd34d;font-weight:600;'>bold strokes</span> for better detection
  </div>
  <div>
    Upload a <span style='color:#86efac;font-weight:600;'>cleaner image</span> for best results
  </div>
</div>
""", unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    st.markdown("""
<div style='text-align:center;padding:0.5rem 0;'>
  <div style='font-size:0.72rem;font-weight:600;color:#334155;'>Doodle to Code</div>
  <div style='font-size:0.65rem;color:#1e293b;margin-top:2px;'>v2.0 · Powered by TensorFlow</div>
  <div style='margin-top:10px;display:flex;justify-content:center;gap:6px;flex-wrap:wrap;'>
    <span style='background:rgba(59,130,246,0.12);border:1px solid rgba(59,130,246,0.2);color:#60a5fa;border-radius:6px;padding:2px 8px;font-size:0.65rem;'>TensorFlow</span>
    <span style='background:rgba(16,185,129,0.12);border:1px solid rgba(16,185,129,0.2);color:#34d399;border-radius:6px;padding:2px 8px;font-size:0.65rem;'>OpenCV</span>
    <span style='background:rgba(239,68,68,0.12);border:1px solid rgba(239,68,68,0.2);color:#f87171;border-radius:6px;padding:2px 8px;font-size:0.65rem;'>Streamlit</span>
  </div>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  HERO HEADER
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero-card">
  <div class="hero-inner">
    <div class="hero-pill">✦ AI-Powered Design Tool</div>
    <div class="hero-title">🎨 Doodle to Code Generator</div>
    <div class="hero-subtitle">
      Transform hand-drawn UI sketches into production-ready code instantly.<br>
      Powered by deep learning — no design experience needed.
    </div>
    <div class="hero-stats">
      <div class="hero-stat">
        <div class="hero-stat-value">25+</div>
        <div class="hero-stat-label">UI Components</div>
      </div>
      <div class="hero-stat">
        <div class="hero-stat-value">HTML</div>
        <div class="hero-stat-label">Output Format</div>
      </div>
      <div class="hero-stat">
        <div class="hero-stat-value">React</div>
        <div class="hero-stat-label">Also Generated</div>
      </div>
      <div class="hero-stat">
        <div class="hero-stat-value">Live</div>
        <div class="hero-stat-label">Preview</div>
      </div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  INPUT ROW — Draw (left) | Upload (right)
# ═══════════════════════════════════════════════════════════════════════════════
canvas_key   = st.session_state.get("canvas_key", 0)
input_image  = None
upload_image = None

col_draw, col_upload = st.columns([1, 1], gap="large")

# ── LEFT: Draw ───────────────────────────────────────────────────────────────
with col_draw:
    st.markdown("""
<div class="card">
  <div class="card-title">
    <span class="card-title-icon">🎨</span>
    Draw Your UI
  </div>
  <div class="card-subtitle">Sketch a UI component using the tools in the sidebar. Use bold strokes for better recognition.</div>
</div>
""", unsafe_allow_html=True)

    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",
        stroke_width=stroke_width,
        stroke_color=stroke_color,
        background_color=bg_color,
        height=370,
        width=None,
        drawing_mode=drawing_mode,
        key=f"canvas_{canvas_key}",
    )
    if canvas_result.image_data is not None:
        input_image = canvas_result.image_data

# ── RIGHT: Upload ─────────────────────────────────────────────────────────────
with col_upload:
    st.markdown("""
<div class="card">
  <div class="card-title">
    <span class="card-title-icon">📤</span>
    Upload UI Sketch
  </div>
  <div class="card-subtitle">Upload a photo or scan of your hand-drawn UI wireframe (PNG, JPG, JPEG).</div>
</div>
""", unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Drop an image here",
        type=["png", "jpg", "jpeg"],
        label_visibility="collapsed",
    )
    if uploaded_file is not None:
        upload_image = Image.open(uploaded_file)
        st.image(upload_image, caption="📷 Uploaded Sketch", use_container_width=True)
    else:
        st.markdown("""
<div class="placeholder">
  <div class="placeholder-icon">🖼️</div>
  <div class="placeholder-text">
    Drag &amp; drop or click above<br>
    to upload your UI sketch
  </div>
</div>
""", unsafe_allow_html=True)


# ── Active image: prefer upload over canvas ───────────────────────────────────
active_image = upload_image if upload_image is not None else input_image

# ── Generate button (full-width, centered) ────────────────────────────────────
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
btn_col = st.columns([1, 2, 1])[1]
with btn_col:
    generate_button = st.button("🚀 Generate Code", type="primary")


# ═══════════════════════════════════════════════════════════════════════════════
#  PREDICTION  (ML — unchanged logic, session-state cached)
# ═══════════════════════════════════════════════════════════════════════════════
if generate_button:
    if active_image is None:
        st.warning("⚠️ Please draw something on the canvas or upload an image first.")
    else:
        with st.spinner("🧠 Analyzing your sketch…"):
            try:
                processed_img = preprocess_image(active_image)           # ← unchanged ML
                label, confidence = predictor.predict(processed_img)     # ← unchanged ML
                html_code = map_class_to_html(label)                     # ← unchanged ML
                st.session_state["last_label"]      = label
                st.session_state["last_confidence"] = confidence
                st.session_state["last_html"]       = html_code
            except Exception as ex:
                st.error(f"Prediction error: {ex}")
                st.text(traceback.format_exc())


# ═══════════════════════════════════════════════════════════════════════════════
#  RESULTS  (shown whenever session_state has a valid prediction)
# ═══════════════════════════════════════════════════════════════════════════════
if "last_label" in st.session_state:
    label      = st.session_state["last_label"]
    confidence = st.session_state["last_confidence"]
    html_code  = st.session_state["last_html"]
    react_code = html_to_react(label, html_code)
    pct        = int(confidence * 100)

    # ── 🧠 Detected UI Components ─────────────────────────────────────────────
    st.markdown("""
<div class="section-label">🧠 &nbsp;Analysis Results</div>
""", unsafe_allow_html=True)

    color_cls = BADGE_COLOR.get(label, "badge-blue")
    icon      = BADGE_ICON.get(label, "🧩")

    st.markdown(f"""
<div class="card">
  <div class="card-title">
    <span class="card-title-icon">🧠</span>
    Detected UI Components
  </div>
  <div class="badges">
    <span class="badge {color_cls}">{icon} {label}</span>
    <span class="badge badge-green">✔ Successfully Detected</span>
    <span class="badge badge-amber">⚡ {pct}% Confidence</span>
  </div>
  <div class="conf-wrap">
    <div class="conf-header">
      <span>Model Confidence Score</span>
      <span class="conf-pct">{pct}%</span>
    </div>
    <div class="conf-bar-bg">
      <div class="conf-bar" style="width:{pct}%;"></div>
    </div>
  </div>
  <div class="conf-note">
    The deep learning model identified <b style="color:#93c5fd;">{label}</b> as the most likely UI
    element in your sketch with <b style="color:#86efac;">{pct}%</b> confidence.
    Higher confidence scores indicate more distinct, clearly drawn shapes.
  </div>
</div>
""", unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # ── 💻 Generated Code (tabbed: HTML | React) ──────────────────────────────
    st.markdown("""
<div class="section-label">💻 &nbsp;Code Output</div>
""", unsafe_allow_html=True)

    st.markdown("""
<div class="card">
  <div class="card-title">
    <span class="card-title-icon">💻</span>
    Generated Website Code
  </div>
  <div class="card-subtitle">Production-ready code generated from your sketch. Copy or download below.</div>
</div>
""", unsafe_allow_html=True)

    tab_html, tab_react = st.tabs(["🌐  HTML", "⚛️  React / JSX"])

    with tab_html:
        # JS-powered copy button
        st.markdown(copy_button_js("html_code_block", "copy_html_btn"), unsafe_allow_html=True)
        st.code(html_code, language="html")
        dl1, spacer, dl2 = st.columns([2, 4, 2])
        with dl2:
            st.download_button(
                label="📥 Download HTML",
                data=html_code,
                file_name=f"{label.lower().replace(' ', '_')}_snippet.html",
                mime="text/html",
            )

    with tab_react:
        st.markdown("""
<div class="react-info">
  ⚛️ <b>Auto-generated React functional component</b> — paste into your project and import as needed.
  Requires React 16.8+ (hooks supported).
</div>
""", unsafe_allow_html=True)
        st.markdown(copy_button_js("react_code_block", "copy_react_btn"), unsafe_allow_html=True)
        st.code(react_code, language="jsx")
        dl3, spacer2, dl4 = st.columns([2, 4, 2])
        with dl4:
            st.download_button(
                label="📥 Download JSX",
                data=react_code,
                file_name=f"{label.lower().replace(' ', '_')}.jsx",
                mime="text/plain",
            )

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # ── 🌐 Live Preview ────────────────────────────────────────────────────────
    st.markdown("""
<div class="section-label">🌐 &nbsp;Live Preview</div>
""", unsafe_allow_html=True)

    st.markdown("""
<div class="card">
  <div class="card-title">
    <span class="card-title-icon">🌐</span>
    Live Website Preview
  </div>
  <div class="card-subtitle">
    Rendered output exactly as it would appear in a browser — powered by your generated HTML.
  </div>
</div>
""", unsafe_allow_html=True)

    # Browser chrome mockup bar
    st.markdown("""
<div class="preview-frame-wrapper">
  <div class="preview-browser-bar">
    <div class="preview-dot" style="background:#ef4444;"></div>
    <div class="preview-dot" style="background:#f59e0b;"></div>
    <div class="preview-dot" style="background:#22c55e;"></div>
    <div style="flex:1;background:rgba(255,255,255,0.06);border-radius:6px;padding:4px 12px;margin-left:12px;font-size:0.72rem;color:#64748b;font-family:'Inter',monospace;">
      doodle-to-code · live-preview
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

    st.components.v1.html(
        f"""
        <style>
          body {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            padding: 32px 40px;
            background: #ffffff;
            margin: 0;
            color: #1e293b;
            font-size: 15px;
          }}
        </style>
        {html_code}
        """,
        height=200,
        scrolling=True,
    )

else:
    # ── Idle state ────────────────────────────────────────────────────────────
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown("""
<div class="idle-card">
  <span class="idle-rocket">🚀</span>
  <div style="font-size:1.1rem;font-weight:800;color:#e2e8f0;margin-bottom:0.5rem;letter-spacing:-0.3px;">
    Ready to generate code
  </div>
  <div style="font-size:0.88rem;color:#475569;line-height:1.7;max-width:420px;margin:0 auto;">
    Draw a UI component on the canvas <b style="color:#93c5fd;">or</b> upload a sketch image,<br>
    then click <b style="color:#60a5fa;">Generate Code</b> to see the magic happen here.
  </div>
  <div style="display:flex;justify-content:center;gap:12px;margin-top:1.6rem;flex-wrap:wrap;">
    <div style="background:rgba(59,130,246,0.1);border:1px solid rgba(59,130,246,0.2);border-radius:10px;padding:10px 18px;font-size:0.78rem;color:#7dd3fc;">
      🧠 AI-Powered Detection
    </div>
    <div style="background:rgba(139,92,246,0.1);border:1px solid rgba(139,92,246,0.2);border-radius:10px;padding:10px 18px;font-size:0.78rem;color:#c4b5fd;">
      ⚛️ React + HTML Output
    </div>
    <div style="background:rgba(22,163,74,0.1);border:1px solid rgba(22,163,74,0.2);border-radius:10px;padding:10px 18px;font-size:0.78rem;color:#86efac;">
      🌐 Live Preview
    </div>
  </div>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  FOOTER
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.markdown("""
<div class="footer">
  <div class="footer-title">🎨 Doodle to Code Generator</div>
  <div style="color:#334155;margin:6px 0 10px;font-size:0.8rem;">
    Transform hand-drawn UI sketches into website code using AI and deep learning.
  </div>
  <div class="footer-stack">
    <span>Streamlit</span>
    <span>TensorFlow</span>
    <span>OpenCV</span>
    <span>PIL</span>
    <span>Python 3</span>
  </div>
  <div class="footer-copy">© 2024 Doodle to Code Project · All Rights Reserved</div>
</div>
""", unsafe_allow_html=True)
