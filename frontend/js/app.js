import { initCanvas, captureCanvas, undo, clearCanvas, setDrawingMode, uploadImage } from './canvas.js';
import { requireAuth, getToken, logout } from './auth.js';

const API_URL = 'http://127.0.0.1:5000';

document.addEventListener('DOMContentLoaded', async () => {
    // Auth check
    await requireAuth();

    // Set user name
    const nameEl = document.getElementById('userName');
    if (nameEl) {
        nameEl.innerText = localStorage.getItem('authName') || 'Developer';
    }

    // Init canvas
    initCanvas();

    // Bind Canvas Toolbar basic actions
    const btnUndo = document.getElementById('btnUndo');
    const btnClear = document.getElementById('btnClear');
    if (btnUndo) btnUndo.addEventListener('click', undo);
    if (btnClear) btnClear.addEventListener('click', clearCanvas);

    // Bind Generate Action
    const btnGenerate = document.getElementById('btnGenerate');
    if (btnGenerate) btnGenerate.addEventListener('click', handleGenerate);

    // Bind tool select
    const toolSelect = document.getElementById('toolSelect');
    if (toolSelect) {
        toolSelect.addEventListener('change', (e) => {
            setDrawingMode(e.target.value);
        });
    }

    // Bind upload
    const btnUpload = document.getElementById('btnUpload');
    if (btnUpload) {
        btnUpload.addEventListener('change', (e) => {
            if (e.target.files && e.target.files[0]) {
                const url = URL.createObjectURL(e.target.files[0]);
                uploadImage(url);
            }
        });
    }

    // Bind tabs
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.addEventListener('click', (e) => {
            const targetId = e.target.getAttribute('data-target');
            
            document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
            
            e.target.classList.add('active');
            document.getElementById(targetId).classList.add('active');
        });
    });

    // Setup Copy buttons
    setupCopyBtn('copyHtmlBtn', 'htmlCodeBlock');
    setupCopyBtn('copyReactBtn', 'reactCodeBlock');

    // Logout
    const btnLogout = document.getElementById('btnLogout');
    if (btnLogout) btnLogout.addEventListener('click', (e) => {
        e.preventDefault();
        logout();
    });
});

async function handleGenerate() {
    const base64Img = captureCanvas();
    if (!base64Img) return;

    // Show loading
    const overlay = document.getElementById('loadingOverlay');
    const resultsSection = document.getElementById('resultsSection');
    const idleState = document.getElementById('idleState');
    const errorBox = document.getElementById('errorBox');
    
    if (idleState) idleState.style.display = 'none';
    if (resultsSection) resultsSection.style.display = 'none';
    if (errorBox) errorBox.style.display = 'none';
    if (overlay) overlay.classList.add('visible');

    try {
        const res = await fetch(`${API_URL}/predict`, {
            method: 'POST',
            headers: { 
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${getToken()}`
            },
            body: JSON.stringify({ image: base64Img })
        });
        
        const data = await res.json();
        
        if (overlay) overlay.classList.remove('visible');

        if (!res.ok) {
            showError(data.error || 'Failed to generate code from server.');
            return;
        }

        renderResults(data);

    } catch (err) {
        if (overlay) overlay.classList.remove('visible');
        showError('Network error connecting to Backend.');
    }
}

function showError(msg) {
    const box = document.getElementById('errorBox');
    if (!box) return;
    box.style.display = 'flex';
    box.innerHTML = `<strong>Error:</strong> ${msg}`;
}

function renderResults(data) {
    const section = document.getElementById('resultsSection');
    if (!section) return;
    
    section.style.display = 'block';

    // 1. Badge + Conf
    document.getElementById('lblComponent').innerText = data.label;
    document.getElementById('lblConfidence').innerText = `${data.confidence}%`;
    document.getElementById('confBar').style.width = `${data.confidence}%`;
    
    let colorClass, iconHtml;
    // Map existing badge stuff
    if(data.label.toLowerCase().includes('text') || data.label.toLowerCase().includes('form')) {
        colorClass = 'badge-blue'; iconHtml = '📝';
    } else if (data.label.toLowerCase().includes('check') || data.label.toLowerCase().includes('enabled')) {
        colorClass = 'badge-green'; iconHtml = '✅';
    } else if (data.label.toLowerCase().includes('alert')) {
        colorClass = 'badge-rose'; iconHtml = '⚠️';
    } else {
        colorClass = 'badge-blue'; iconHtml = '🧩';
    }
    
    const b = document.getElementById('dynamicBadge');
    b.className = `badge ${colorClass}`;
    b.innerText = `${iconHtml} ${data.label}`;

    // 2. Code blocks
    document.getElementById('htmlCodeBlock').innerText = data.html_code;
    document.getElementById('reactCodeBlock').innerText = data.react_code;

    // 3. Iframe preview (render HTML inside Iframe)
    const iframe = document.getElementById('previewIframe');
    const c = iframe.contentWindow.document;
    c.open();
    c.write(`
        <style>
          body { font-family: 'Inter', -apple-system, sans-serif; padding: 20px; margin: 0;}
        </style>
        ${data.html_code}
    `);
    c.close();
}

function setupCopyBtn(btnId, targetId) {
    const btn = document.getElementById(btnId);
    if (!btn) return;
    btn.addEventListener('click', () => {
        const text = document.getElementById(targetId).innerText;
        navigator.clipboard.writeText(text).then(() => {
            const orig = btn.innerText;
            btn.innerText = '✅ Copied!';
            setTimeout(() => btn.innerText = orig, 2000);
        });
    });
}
