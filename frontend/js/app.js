import { initCanvas, captureCanvas, undo, clearCanvas, setDrawingMode, uploadImage } from './canvas.js';
import { requireAuth, getToken, logout } from './auth.js';

// ✅ TWO BACKENDS
const ML_API = "http://127.0.0.1:5000";       // TensorFlow

let currentMode = "draw";

document.addEventListener('DOMContentLoaded', async () => {
    await requireAuth();

    const nameEl = document.getElementById('userName');
    if (nameEl) {
        nameEl.innerText = localStorage.getItem('authName') || 'Developer';
    }

    initCanvas();

    const btnUndo = document.getElementById('btnUndo');
    const btnClear = document.getElementById('btnClear');
    if (btnUndo) btnUndo.addEventListener('click', undo);
    if (btnClear) btnClear.addEventListener('click', clearCanvas);

    const btnGenerate = document.getElementById('btnGenerate');
    if (btnGenerate) btnGenerate.addEventListener('click', handleGenerate);
    setupModeToggle();

    const toolSelect = document.getElementById('toolSelect');
    if (toolSelect) {
        toolSelect.addEventListener('change', (e) => {
            setDrawingMode(e.target.value);
        });
    }

    const btnUpload = document.getElementById('btnUpload');
    if (btnUpload) {
        btnUpload.addEventListener('change', (e) => {
            if (e.target.files && e.target.files[0]) {
                const url = URL.createObjectURL(e.target.files[0]);
                uploadImage(url);
            }
        });
    }

    setupCopyBtn('copyHtmlBtn', 'htmlCodeBlock');
    setupCopyBtn('copyReactBtn', 'reactCodeBlock');

    const btnLogout = document.getElementById('btnLogout');
    if (btnLogout) btnLogout.addEventListener('click', (e) => {
        e.preventDefault();
        logout();
    });
});


// =========================
// 🔥 MAIN GENERATE FUNCTION
// =========================
async function handleGenerate() {
    // Prompt mode has its own React UI and submit handler.
    if (currentMode !== "draw") {
        return;
    }

    // =========================
    // ✏️ DRAW MODE → ML (5000)
    // =========================
    const base64Img = captureCanvas();
    if (!base64Img) return;

    try {
        const res = await fetch(`${ML_API}/predict`, {
            method: 'POST',
            headers: { 
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${getToken()}`
            },
            body: JSON.stringify({ image: base64Img })
        });

        const data = await res.json();

        if (!res.ok) {
            showError(data.error || 'Prediction failed');
            return;
        }

        renderResults(data);

    } catch (err) {
        showError('Network error connecting to ML backend.');
    }
}


// =========================
// 🔹 ERROR HANDLER
// =========================
function showError(msg) {
    const box = document.getElementById('errorBox');
    if (!box) return;
    box.style.display = 'flex';
    box.innerHTML = `<strong>Error:</strong> ${msg}`;
}


// =========================
// 🔹 RENDER DRAW RESULTS
// =========================
function renderResults(data) {
    const section = document.getElementById('resultsSection');
    if (!section) return;

    const htmlCode = data?.html || data?.html_code || data?.response || "";
    const reactCode = data?.react || data?.react_code || htmlCode;

    section.style.display = 'block';

    const htmlCodeBlock = document.getElementById('htmlCodeBlock');
    const reactCodeBlock = document.getElementById('reactCodeBlock');
    if (htmlCodeBlock) htmlCodeBlock.innerText = htmlCode;
    if (reactCodeBlock) reactCodeBlock.innerText = reactCode;

    const iframe = document.getElementById('previewIframe');
    if (iframe && iframe.contentWindow) {
        const c = iframe.contentWindow.document;
        c.open();
        c.write(htmlCode);
        c.close();
    }
}


// =========================
// 🔹 COPY BUTTON
// =========================
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

function setupModeToggle() {
    const modeBtnDraw = document.getElementById('modeBtnDraw');
    const modeBtnPrompt = document.getElementById('modeBtnPrompt');
    const drawCard = document.getElementById('drawCard');
    const promptCard = document.getElementById('promptCard');
    const idleState = document.getElementById('idleState');
    const resultsSection = document.getElementById('resultsSection');
    const errorBox = document.getElementById('errorBox');

    if (!modeBtnDraw || !modeBtnPrompt || !drawCard || !promptCard) return;

    const applyMode = (mode) => {
        currentMode = mode;

        const isDraw = mode === "draw";
        drawCard.style.display = isDraw ? 'block' : 'none';
        promptCard.style.display = isDraw ? 'none' : 'block';

        modeBtnDraw.className = isDraw ? 'btn-primary' : 'btn-ghost';
        modeBtnPrompt.className = isDraw ? 'btn-ghost' : 'btn-primary';

        if (!isDraw) {
            if (idleState) idleState.style.display = 'none';
            if (resultsSection) resultsSection.style.display = 'none';
            if (errorBox) errorBox.style.display = 'none';
        }
    };

    modeBtnDraw.addEventListener('click', () => applyMode("draw"));
    modeBtnPrompt.addEventListener('click', () => applyMode("prompt"));
    applyMode("draw");
}