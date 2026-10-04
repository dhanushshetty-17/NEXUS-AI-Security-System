// Utility: Debounce high-frequency events
function debounce(func, wait = 300) {
    let timeout;
    return function (...args) {
        clearTimeout(timeout);
        timeout = setTimeout(() => func.apply(this, args), wait);
    };
}

// Update system time
function updateTime() {
    const timeElem = document.getElementById('sys-time');
    if (timeElem) {
        timeElem.textContent = new Date().toLocaleString();
    }
}
setInterval(updateTime, 1000);
updateTime();

// Security utility
function escapeHtml(unsafe) {
    return (unsafe || "").toString()
         .replace(/&/g, "&amp;")
         .replace(/</g, "&lt;")
         .replace(/>/g, "&gt;")
         .replace(/"/g, "&quot;")
         .replace(/'/g, "&#039;");
}

// Audio Context for beep
let audioCtx = null;
let webAlarmsEnabled = localStorage.getItem('webAlarmsEnabled') === 'true';

const audioBtn = document.getElementById('enable-audio-btn');
if (audioBtn && webAlarmsEnabled) {
    audioBtn.textContent = '🔊 Web Alarms Enabled';
    audioBtn.style.background = 'rgba(76, 175, 80, 0.2)';
    audioBtn.style.borderColor = '#4caf50';
}

audioBtn?.addEventListener('click', (e) => {
    if (webAlarmsEnabled) {
        webAlarmsEnabled = false;
        localStorage.setItem('webAlarmsEnabled', 'false');
        e.target.textContent = '🔇 Enable Web Alarms';
        e.target.style.background = 'rgba(255,255,255,0.1)';
        e.target.style.borderColor = 'var(--border-color)';
    } else {
        if (!audioCtx) {
            audioCtx = new (window.AudioContext || window.webkitAudioContext)();
        }
        if (audioCtx.state === 'suspended') {
            audioCtx.resume();
        }
        webAlarmsEnabled = true;
        localStorage.setItem('webAlarmsEnabled', 'true');
        e.target.textContent = '🔊 Web Alarms Enabled';
        e.target.style.background = 'rgba(76, 175, 80, 0.2)';
        e.target.style.borderColor = '#4caf50';
        playBeep(); // test beep
    }
});

function playBeep() {
    if (!webAlarmsEnabled) return;
    try {
        if (!audioCtx) {
            audioCtx = new (window.AudioContext || window.webkitAudioContext)();
        }
        if (audioCtx.state === 'suspended') {
            audioCtx.resume().catch(e => console.log(e));
        }
        const oscillator = audioCtx.createOscillator();
        const gainNode = audioCtx.createGain();
        oscillator.connect(gainNode);
        gainNode.connect(audioCtx.destination);
        
        oscillator.type = 'square';
        oscillator.frequency.setValueAtTime(880, audioCtx.currentTime); // A5
        gainNode.gain.setValueAtTime(0.1, audioCtx.currentTime);
        
        oscillator.start();
        gainNode.gain.exponentialRampToValueAtTime(0.00001, audioCtx.currentTime + 0.5);
        oscillator.stop(audioCtx.currentTime + 0.5);
    } catch (err) {
        console.warn("Audio playback error:", err);
    }
}

// Chart Instance
let threatChart = null;
let pollInFlight = false;
let reportsController = null;
let reportsRequestId = 0;

function initChart() {
    const ctx = document.getElementById('threatChart');
    if (!ctx) return;
    
    threatChart = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: ['Low', 'Medium', 'High', 'Critical'],
            datasets: [{
                data: [0, 0, 0, 0],
                backgroundColor: ['#2e7d32', '#a66f00', '#b3261e', '#7f0000'],
                borderWidth: 0
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { position: 'right', labels: { color: 'white' } }
            }
        }
    });
}
initChart();

// Set to keep track of seen events so we don't beep twice for the same event
const seenEvents = new Set();
let lastEventsSignature = "";

// Poll for events and status
async function pollData() {
    if (pollInFlight) return;
    pollInFlight = true;
    try {
        const response = await fetch('/api/events');
        if (response.status === 401) {
            window.location.href = '/login';
            return;
        }
        if (!response.ok) return;
        
        const data = await response.json();
        
        // Update camera statuses
        if (Array.isArray(data.statuses)) {
            data.statuses.forEach(status => {
                const ind = document.getElementById(`status-${status.camera_id}`);
                const fps = document.getElementById(`fps-${status.camera_id}`);
                const modelStatus = document.getElementById(`model-status-${status.camera_id}`);
                
                if (ind) {
                    if (status.connected || status.frame_count > 0) {
                        ind.classList.add('connected');
                        ind.setAttribute('aria-label', 'Connected');
                        const loading = document.getElementById(`loading-${status.camera_id}`);
                        if (loading) loading.hidden = true;
                    } else {
                        ind.classList.remove('connected');
                        ind.setAttribute('aria-label', 'Disconnected');
                        const loading = document.getElementById(`loading-${status.camera_id}`);
                        if (loading && !status.running) {
                            loading.hidden = false;
                            loading.querySelector('.loading-message')?.replaceChildren(
                                document.createTextNode('Camera unavailable. Retrying...')
                            );
                        }
                    }
                }
                if (modelStatus) {
                    modelStatus.textContent = status.models_ready ? 'AI READY' : 'AI LOADING';
                    modelStatus.classList.toggle('ready', Boolean(status.models_ready));
                }
                if (fps) {
                    fps.textContent = `FPS: ${status.fps.toFixed(1)}`;
                }
            });
        }
        
        // System Health
        if (data.sys_health) {
            const cpuEl = document.getElementById('cpu-usage');
            const ramEl = document.getElementById('ram-usage');
            if (cpuEl) cpuEl.textContent = data.sys_health.cpu;
            if (ramEl) ramEl.textContent = data.sys_health.ram;
        }
        
        // System Status
        if (data.current_threat_level) {
            const badge = document.getElementById('overall-threat');
            if (badge && badge.textContent !== data.current_threat_level) {
                badge.className = 'badge ' + data.current_threat_level;
                badge.textContent = data.current_threat_level;
            }
        }

        // Check if events actually changed before re-rendering DOM
        const currentSignature = (data.events || []).map(e => e.event_id).join("|");
        if (currentSignature !== lastEventsSignature && Array.isArray(data.events)) {
            lastEventsSignature = currentSignature;
            const tbody = document.getElementById('events-tbody');
            if (!tbody) return;

            if (data.events.length === 0) {
                tbody.innerHTML = '<tr><td colspan="4" style="text-align: center; color: var(--text-secondary);">No threat events detected.</td></tr>';
            } else {
                tbody.innerHTML = '';
                let counts = { LOW: 0, MEDIUM: 0, HIGH: 0, CRITICAL: 0 };
                
                data.events.forEach(evt => {
                    const tr = document.createElement('tr');
                    tr.style.cursor = 'pointer';
                    tr.className = 'event-row';
                    
                    tr.tabIndex = 0;
                    tr.setAttribute('role', 'button');
                    tr.setAttribute('aria-label', `View snapshot for ${evt.label || 'threat event'}`);
                    const openSnapshot = () => {
                        const modal = document.getElementById('snapshot-modal');
                        const modalImg = document.getElementById('modal-img');
                        const modalNoImg = document.getElementById('modal-no-img');
                        if (modal) {
                            modalReturnFocus = tr;
                            modal.style.display = 'flex';
                            modal.setAttribute('aria-hidden', 'false');
                            document.getElementById('modal-close-btn')?.focus();
                        }
                        
                        if (evt.snapshot_path) {
                            const filename = evt.snapshot_path.split(/[\/\\]/).pop();
                            if (modalImg) {
                                modalImg.src = '/snapshots/' + filename;
                                modalImg.style.display = 'block';
                            }
                            if (modalNoImg) modalNoImg.style.display = 'none';
                        } else {
                            if (modalImg) modalImg.style.display = 'none';
                            if (modalNoImg) modalNoImg.style.display = 'block';
                        }
                    };
                    tr.addEventListener('click', openSnapshot);
                    tr.addEventListener('keydown', (event) => {
                        if (event.key === 'Enter' || event.key === ' ') {
                            event.preventDefault();
                            openSnapshot();
                        }
                    });
                    
                    const date = new Date(evt.timestamp * 1000);
                    const lvlClass = evt.threat_level || 'LOW';
                    const badge = `<span class="badge ${lvlClass}">${lvlClass}</span>`;
                    
                    counts[lvlClass] = (counts[lvlClass] || 0) + 1;
                    
                    if ((lvlClass === 'HIGH' || lvlClass === 'CRITICAL') && !seenEvents.has(evt.event_id)) {
                        playBeep();
                    }
                    seenEvents.add(evt.event_id);
                    
                    tr.innerHTML = `
                        <td>${date.toLocaleTimeString()}</td>
                        <td>${escapeHtml(evt.camera_id)}</td>
                        <td>${escapeHtml(evt.label)}</td>
                        <td>${badge}</td>
                    `;
                    tbody.appendChild(tr);
                });
                
                // Update chart
                if (threatChart) {
                    threatChart.data.datasets[0].data = [counts.LOW, counts.MEDIUM, counts.HIGH, counts.CRITICAL];
                    threatChart.update();
                }
            }
        }
        
    } catch (e) {
        console.error("Polling error:", e);
    } finally {
        pollInFlight = false;
    }
}

// Fetch and render reports
async function fetchReports(query = "") {
    const requestId = ++reportsRequestId;
    reportsController?.abort();
    reportsController = new AbortController();
    const container = document.getElementById('reports-container');
    if (!container) return;
    container.setAttribute('aria-busy', 'true');
    try {
        const url = query ? `/api/search?query=${encodeURIComponent(query)}` : '/api/reports';
        const response = await fetch(url, { signal: reportsController.signal });
        if (response.status === 401 || !response.ok) return;
        
        const reports = await response.json();
        if (requestId !== reportsRequestId) return;
        
        if (!Array.isArray(reports) || reports.length === 0) {
            container.innerHTML = '<div style="color: var(--text-secondary);">No reports found.</div>';
            return;
        }
        
        container.innerHTML = '';
        reports.forEach(r => {
            const date = new Date(r.timestamp * 1000).toLocaleString();
            const card = document.createElement('div');
            card.style.cssText = 'background: rgba(255,255,255,0.05); padding: 1rem; border-radius: 6px; border: 1px solid var(--border-color);';
            
            let imgHtml = '';
            if (r.image_url) {
                imgHtml = `<img src="${r.image_url}" alt="Incident Snapshot" style="width: 100%; border-radius: 4px; margin-bottom: 0.5rem;" loading="lazy">`;
            }
            
            card.innerHTML = `
                ${imgHtml}
                <div style="font-size: 0.8rem; color: var(--text-secondary); margin-bottom: 0.5rem;">${date} • ${escapeHtml(r.camera_id)}</div>
                <h4 style="margin-bottom: 0.5rem; color: #ffeb3b;">${escapeHtml(r.threat_label)}</h4>
                <p style="font-size: 0.9rem; line-height: 1.4; color: #e2e8f0; white-space: pre-wrap;">${escapeHtml(r.ai_summary)}</p>
            `;
            container.appendChild(card);
        });
        
    } catch (e) {
        if (e.name !== 'AbortError') {
            container.innerHTML = '<div class="inline-error" role="alert">Reports are temporarily unavailable.</div>';
            console.error("Failed to fetch reports:", e);
        }
    } finally {
        if (requestId === reportsRequestId) container.setAttribute('aria-busy', 'false');
    }
}

// Set up search listener with 300ms debounce
const searchInput = document.getElementById('report-search');
if (searchInput) {
    searchInput.addEventListener('input', debounce((e) => {
        fetchReports(e.target.value);
    }, 300));
}

// Initial polling intervals
setInterval(pollData, 2000);
pollData();

// Fetch reports less frequently
setInterval(() => {
    const searchVal = document.getElementById('report-search')?.value || "";
    fetchReports(searchVal);
}, 10000);
fetchReports();

// Centralized Heatmap controls
const heatmapToggleBtn = document.getElementById('heatmap-toggle-btn');
if (heatmapToggleBtn) {
    heatmapToggleBtn.addEventListener('click', async (e) => {
        if (heatmapToggleBtn.disabled) return;
        heatmapToggleBtn.disabled = true;
        const originalText = heatmapToggleBtn.textContent;
        heatmapToggleBtn.textContent = 'Updating...';
        try {
            const res = await fetch('/api/heatmap/toggle', { method: 'POST' });
            if (!res.ok) throw new Error(`HTTP ${res.status}`);
            const data = await res.json();
            if (data.enabled) {
                heatmapToggleBtn.textContent = '🔥 Heatmap ON';
                e.target.style.background = 'rgba(59, 130, 246, 0.2)';
                e.target.style.borderColor = '#3b82f6';
            } else {
                heatmapToggleBtn.textContent = '⏸️ Heatmap OFF';
                heatmapToggleBtn.style.background = 'rgba(255,255,255,0.1)';
                heatmapToggleBtn.style.borderColor = 'var(--border-color)';
            }
        } catch (err) {
            console.error("Heatmap toggle failed", err);
            heatmapToggleBtn.textContent = originalText;
            alert('Unable to update the heatmap. Please try again.');
        } finally {
            heatmapToggleBtn.disabled = false;
        }
    });
}

const heatmapResetBtn = document.getElementById('heatmap-reset-btn');
if (heatmapResetBtn) {
    heatmapResetBtn.addEventListener('click', async () => {
        if (heatmapResetBtn.disabled) return;
        heatmapResetBtn.disabled = true;
        try {
            const res = await fetch('/api/heatmap/reset', { method: 'POST' });
            if (!res.ok) throw new Error(`HTTP ${res.status}`);
            const originalText = heatmapResetBtn.innerHTML;
            heatmapResetBtn.innerHTML = '✅ Cleared';
            setTimeout(() => { heatmapResetBtn.innerHTML = originalText; }, 1500);
        } catch (err) {
            console.error("Heatmap reset failed", err);
            alert('Unable to reset the heatmap. Please try again.');
        } finally {
            heatmapResetBtn.disabled = false;
        }
    });
}

// Keep destructive actions, modal state, and native form submissions keyboard-safe.
document.querySelectorAll('.stop-camera-btn').forEach((button) => {
    button.addEventListener('click', async () => {
        const cameraId = button.dataset.cameraId;
        if (!cameraId || button.disabled || !confirm(`Are you sure you want to stop and remove ${cameraId}?`)) return;
        button.disabled = true;
        button.textContent = 'Stopping...';
        try {
            const response = await fetch(`/api/remove_source/${encodeURIComponent(cameraId)}`, { method: 'POST' });
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            window.location.reload();
        } catch (error) {
            button.disabled = false;
            button.textContent = '✖ Stop';
            alert('Unable to stop this camera. Please try again.');
            console.error('Failed to stop camera', error);
        }
    });
});

document.querySelectorAll('[data-loading-id]').forEach((image) => {
    const loading = document.getElementById(image.dataset.loadingId);
    window.setTimeout(() => {
        if (loading && image.naturalWidth > 0) loading.hidden = true;
    }, 3000);
    image.addEventListener('load', () => {
        if (loading) loading.hidden = true;
    });
    image.addEventListener('error', () => {
        if (loading) loading.textContent = 'Live feed unavailable';
    });
});

const snapshotModal = document.getElementById('snapshot-modal');
const modalCloseButton = document.getElementById('modal-close-btn');
let modalReturnFocus = null;
function closeSnapshotModal() {
    if (!snapshotModal) return;
    snapshotModal.style.display = 'none';
    snapshotModal.setAttribute('aria-hidden', 'true');
    modalReturnFocus?.focus();
}
modalCloseButton?.addEventListener('click', closeSnapshotModal);
snapshotModal?.addEventListener('click', (event) => {
    if (event.target === snapshotModal) closeSnapshotModal();
});
document.addEventListener('keydown', (event) => {
    if (event.key === 'Escape' && snapshotModal?.style.display === 'flex') closeSnapshotModal();
});

document.querySelectorAll('.single-submit-form, form[action="/settings"], form[action="/login"]').forEach((form) => {
    form.addEventListener('submit', () => {
        const submit = form.querySelector('button[type="submit"]');
        if (!submit || submit.disabled) return;
        submit.disabled = true;
        submit.dataset.originalText = submit.textContent;
        submit.textContent = 'Working...';
    });
});
