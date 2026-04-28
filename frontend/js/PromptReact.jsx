const { useState } = React;

const PromptReact = () => {
    const [prompt, setPrompt] = useState("");
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState(null);
    const [result, setResult] = useState(null);

    const handleGenerate = async () => {
        if (!prompt.trim()) return;

        setIsLoading(true);
        setError(null);
        setResult(null);

        // Hide any previously displayed Vanilla JS elements
        const idleState = document.getElementById('idleState');
        const resultsSection = document.getElementById('resultsSection');
        if (idleState) idleState.style.display = 'none';
        if (resultsSection) resultsSection.style.display = 'none';

        try {
            // "Send a request to a backend API endpoint (e.g., /generate)"
            const API_URL = 'http://127.0.0.1:5000';
            
            // Try to get token from existing auth module if possible
            const token = localStorage.getItem('authToken');

            const res = await fetch(`${API_URL}/generate-ui`, {
                method: 'POST',
                headers: { 
                    'Content-Type': 'application/json',
                    ...(token && { 'Authorization': `Bearer ${token}` })
                },
                body: JSON.stringify({ prompt })
            });

            const data = await res.json();

            if (!res.ok) {
                throw new Error(data.error || 'Failed to generate UI code from server.');
            }

            setResult(data);

            // ✅ Sync with Vanilla JS UI for preview
            const resultsSection = document.getElementById('resultsSection');
            const idleState = document.getElementById('idleState');
            if (resultsSection) resultsSection.style.display = 'block';
            if (idleState) idleState.style.display = 'none';

            const htmlCode = data.html_code || data.html || "";
            const reactCode = data.react_code || data.react || htmlCode;

            // Update blocks
            const htmlCodeBlock = document.getElementById('htmlCodeBlock');
            const reactCodeBlock = document.getElementById('reactCodeBlock');
            if (htmlCodeBlock) htmlCodeBlock.innerText = htmlCode;
            if (reactCodeBlock) reactCodeBlock.innerText = reactCode;

            // Update badge and confidence if available
            const dynamicBadge = document.getElementById('dynamicBadge');
            const lblConfidence = document.getElementById('lblConfidence');
            const confBar = document.getElementById('confBar');
            
            if (dynamicBadge) dynamicBadge.innerText = data.label || "Generated UI";
            if (lblConfidence) lblConfidence.innerText = `${data.confidence || 100}%`;
            if (confBar) confBar.style.width = `${data.confidence || 100}%`;

            // Update Preview Iframe
            const iframe = document.getElementById('previewIframe');
            if (iframe && iframe.contentWindow) {
                const doc = iframe.contentWindow.document;
                doc.open();
                doc.write(htmlCode);
                doc.close();
            }

        } catch (err) {
            setError(err.message || 'Network error connecting to Backend.');
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <div style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
            
            <div className="canvas-card-header" style={{ marginBottom: '1rem' }}>
                <div className="canvas-card-icon">💬</div>
                <div className="canvas-card-title">Prompt Your UI (React)</div>
            </div>
            
            <div style={{ flex: 1, display: 'flex', flexDirection: 'column' }}>
                {/* Input Section */}
                <textarea 
                    placeholder="Describe the UI you want to build (e.g. 'A sleek login form with email and password')" 
                    value={prompt}
                    onChange={(e) => setPrompt(e.target.value)}
                    style={{
                        flex: 1,
                        width: '100%',
                        minHeight: '150px',
                        borderRadius: 'var(--radius-sm)',
                        border: '1px solid var(--border)',
                        background: 'var(--bg-card)',
                        color: 'var(--text-color)',
                        padding: '1rem',
                        fontFamily: 'Inter',
                        fontSize: '1rem',
                        outline: 'none',
                        resize: 'none',
                        marginBottom: '1rem',
                        lineHeight: 1.5,
                        boxShadow: 'inset 0 2px 4px rgba(0,0,0,0.2)'
                    }}
                />
                
                <div style={{ marginBottom: '1rem' }}>
                    <div className="section-label" style={{ fontSize: '0.75rem', marginBottom: '0.5rem', textTransform: 'uppercase' }}>Sample prompts</div>
                    <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap' }}>
                        {['Login page', 'Dashboard with sidebar', 'E-commerce product card'].map(s => (
                            <span 
                                key={s} 
                                className="idle-chip" 
                                style={{ cursor: 'pointer', transition: 'all 0.2s' }} 
                                onClick={() => setPrompt(s)}
                            >
                                {s}
                            </span>
                        ))}
                    </div>
                </div>

                {/* Error State */}
                {error && (
                    <div className="alert-error" style={{ marginBottom: '1rem', display: 'flex', textAlign: 'left', padding: '1rem', backgroundColor: 'rgba(239,68,68,0.1)', color: '#fca5a5', border: '1px solid rgba(239,68,68,0.3)', borderRadius: '8px' }}>
                        <strong>Error:</strong>&nbsp;{error}
                    </div>
                )}
            </div>

            {/* Loading State */}
            {isLoading && (
                <div style={{ padding: '2rem', textAlign: 'center', backgroundColor: 'rgba(255,255,255,0.02)', borderRadius: '8px', border: '1px dashed rgba(99,179,237,0.2)', marginBottom: '1rem' }}>
                    <div className="spinner spinner-large" style={{ margin: '0 auto 1rem' }}></div>
                    <div className="loading-text" style={{ color: '#60a5fa', fontWeight: '500' }}>Generating UI code using OpenAI...</div>
                </div>
            )}

            {/* Result display */}
            {result && !isLoading && (
                <div style={{ marginTop: '1rem', padding: '1.5rem', borderTop: '1px solid rgba(99,179,237,0.15)', backgroundColor: 'var(--bg-card)', borderRadius: '12px' }}>
                     <div className="section-label" style={{ marginBottom: '1rem' }}>🎉 Generated Code Output</div>
                     
                     <div className="tabs" style={{ marginBottom: '0.5rem' }}>
                         <button className="tab-btn active">⚛️ React / JSX</button>
                     </div>
                     <div className="code-block" style={{ backgroundColor: '#0d1117', padding: '1.2rem', borderRadius: '10px', overflowX: 'auto', border: '1px solid rgba(99,179,237,0.15)' }}>
                         {/* We handle various response formats just in case */}
                         <pre style={{ margin: 0, color: '#e2e8f0', fontSize: '13px', lineHeight: 1.6 }}>
                             {result.react_code || result.generated_code || result.html_code || result.code || JSON.stringify(result, null, 2)}
                         </pre>
                     </div>
                     
                     <div style={{ display: 'flex', justifyContent: 'flex-end', marginTop: '10px' }}>
                        <button 
                            className="copy-btn" 
                            onClick={(e) => {
                                const text = result.react_code || result.generated_code || result.html_code || result.code;
                                navigator.clipboard.writeText(text);
                                const btn = e.target;
                                const originalText = btn.innerText;
                                btn.innerText = '✅ Copied!';
                                setTimeout(() => btn.innerText = originalText, 2000);
                            }}
                        >
                            📋 Copy React Snippet
                        </button>
                    </div>
                </div>
            )}

            {/* Submit Button */}
            {!isLoading && (
                <div className="canvas-toolbar" style={{ borderTop: 'none', paddingTop: '0', marginTop: '1rem' }}>
                    <div className="spacer"></div>
                    <button className="btn-primary" onClick={handleGenerate} style={{ padding: '0.75rem 2rem', fontSize: '1rem' }}>✨ Generate React UI</button>
                </div>
            )}
        </div>
    );
};

const rootNode = document.getElementById('react-prompt-root');
if (rootNode) {
    const root = ReactDOM.createRoot(rootNode);
    root.render(<PromptReact />);
}
