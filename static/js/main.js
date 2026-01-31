// Global State
let currentSection = 'dashboard';
let regStream = null;
let verifyStream = null;
let threatStream = null;
let capturedRegFace = null;
let capturedThreatFace = null;

// Initialization
document.addEventListener('DOMContentLoaded', () => {
    fetchStats();
    initCharts();

    // Auto-refresh stats every 30s
    setInterval(fetchStats, 30000);
});

// Navigation
function switchSection(sectionId) {
    // Stop all streams when switching
    stopAllCameras();

    // Update UI
    document.querySelectorAll('section').forEach(s => s.classList.remove('active'));
    document.querySelectorAll('.sidebar nav li').forEach(li => li.classList.remove('active'));

    document.getElementById(sectionId).classList.add('active');
    document.querySelector(`li[onclick="switchSection('${sectionId}')"]`).classList.add('active');

    currentSection = sectionId;

    // Section-specific init
    if (sectionId === 'dashboard') fetchStats();
    if (sectionId === 'alerts') fetchIncidents();
    if (sectionId === 'database') fetchStudents();
}

function stopAllCameras() {
    [regStream, verifyStream, threatStream].forEach(stream => {
        if (stream) {
            stream.getTracks().forEach(track => track.stop());
        }
    });
    regStream = verifyStream = threatStream = null;
}

// Stats & Dashboard
async function fetchStats() {
    try {
        const response = await fetch('/api/stats');
        const stats = await response.json();

        document.getElementById('total-students').innerText = stats.total_students;
        document.getElementById('total-incidents').innerText = stats.total_incidents;
        document.getElementById('total-threats').innerText = stats.total_threats;
        document.getElementById('resolved-incidents').innerText = stats.resolved_incidents;
    } catch (e) {
        console.error("Failed to fetch stats", e);
    }
}

function initCharts() {
    const ctx = document.getElementById('incidentChart').getContext('2d');
    new Chart(ctx, {
        type: 'line',
        data: {
            labels: ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
            datasets: [{
                label: 'Incidents',
                data: [2, 5, 3, 8, 4, 6, 2],
                borderColor: '#3b82f6',
                tension: 0.4,
                fill: true,
                backgroundColor: 'rgba(59, 130, 246, 0.1)'
            }]
        },
        options: {
            responsive: true,
            plugins: { legend: { display: false } },
            scales: {
                y: { beginAtZero: true, grid: { color: '#334155' } },
                x: { grid: { color: '#334155' } }
            }
        }
    });
}

// --- Registration Logic ---
async function processID() {
    const fileInput = document.getElementById('id-card-input');
    if (!fileInput.files[0]) return alert("Please select a file");

    showLoading(true);
    const formData = new FormData();
    formData.append('id_card', fileInput.files[0]);

    try {
        const response = await fetch('/api/register/upload_id', { method: 'POST', body: formData });
        const result = await response.json();

        if (result.success) {
            document.getElementById('reg-name').value = result.name;
            document.getElementById('reg-roll').value = result.roll;
            document.getElementById('reg-dept').value = result.dept;

            // Move to Step 2
            goToRegStep(2);
        } else {
            alert("Error: " + result.error);
        }
    } catch (e) {
        alert("OCR failed");
    } finally {
        showLoading(false);
    }
}

function goToRegStep(step) {
    document.querySelectorAll('.reg-step').forEach(s => s.classList.remove('active'));
    document.querySelectorAll('.step-content').forEach(s => s.classList.remove('active'));

    document.getElementById(`step${step}-head`).classList.add('active');
    document.getElementById(`reg-step${step}`).classList.add('active');
}

async function startRegCamera() {
    try {
        regStream = await navigator.mediaDevices.getUserMedia({ video: true });
        document.getElementById('reg-video').srcObject = regStream;
        document.getElementById('reg-face-preview').classList.add('hidden');
    } catch (e) {
        alert("Camera access denied");
    }
}

function captureRegFace() {
    const video = document.getElementById('reg-video');
    const canvas = document.createElement('canvas');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext('2d').drawImage(video, 0, 0);

    capturedRegFace = canvas.toDataURL('image/jpeg');
    document.getElementById('reg-face-img').src = capturedRegFace;
    document.getElementById('reg-face-preview').classList.remove('hidden');

    goToRegStep(3);
}

async function completeRegistration() {
    const data = {
        name: document.getElementById('reg-name').value,
        roll: document.getElementById('reg-roll').value,
        dept: document.getElementById('reg-dept').value,
        face_image: capturedRegFace
    };

    showLoading(true);
    try {
        const res = await fetch('/api/register/complete', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        });
        const result = await res.json();

        if (result.success) {
            document.getElementById('qr-img').src = result.qr_path;
            document.getElementById('qr-result').classList.remove('hidden');
            alert("Registration successful!");
        } else {
            alert("Error: " + result.error);
        }
    } catch (e) {
        alert("Reg failed");
    } finally {
        showLoading(false);
    }
}

// --- Verification Logic ---
let isVerifying = false;
async function startVerification() {
    if (isVerifying) return;
    try {
        verifyStream = await navigator.mediaDevices.getUserMedia({ video: true });
        const video = document.getElementById('verify-video');
        video.srcObject = verifyStream;
        isVerifying = true;

        document.getElementById('verify-status').innerText = "SCANNING FOR QR...";

        // Loop verification poll
        verifyPollLoop();
    } catch (e) {
        alert("Camera access denied");
    }
}

async function verifyPollLoop() {
    if (!isVerifying || currentSection !== 'verification') return;

    const video = document.getElementById('verify-video');
    const canvas = document.createElement('canvas');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext('2d').drawImage(video, 0, 0);
    const frame = canvas.toDataURL('image/jpeg');

    // In a real app, you'd use a JS QR library if possible, 
    // but here we send to backend as the backend already has the logic.
    // However, the backend verify logic expects QR and Face separately.
    // For this prototype, we'll assume the frame contains BOTH or we mock it.
    // Let's implement a "Capture and Verify" manual trigger for better stability.

    // If we were auto-verifying:
    // const res = await fetch('/api/verify/check', ...);

    // Let's change the UI to a "Verify Now" button to avoid spamming the backend.
    document.getElementById('verify-status').innerHTML = '<button class="btn btn-primary" onclick="doVerify()">Verify Frame</button>';
}

async function doVerify() {
    const video = document.getElementById('verify-video');
    const canvas = document.createElement('canvas');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext('2d').drawImage(video, 0, 0);
    const frame = canvas.toDataURL('image/jpeg');

    showLoading(true);
    try {
        // Need to extract QR from frame on backend
        // We'll update the backend to handle single-frame verification
        const res = await fetch('/api/verify/check_frame', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image: frame })
        });
        const result = await res.json();

        if (result.success) {
            const s = result.student;
            document.getElementById('student-info-display').innerHTML = `
                <div class="info-content">
                    <img src="${s.face_path}" style="width: 100px; border-radius: 8px;">
                    <p><strong>Name:</strong> ${s.name}</p>
                    <p><strong>Roll:</strong> ${s.roll_no}</p>
                    <p><strong>Status:</strong> <span style="color: #10b981">VERIFIED</span></p>
                    <p><strong>Match:</strong> ${(result.similarity * 100).toFixed(1)}%</p>
                </div>
            `;
            document.getElementById('verify-status').innerText = "✅ VERIFIED!";
        } else {
            alert(result.error);
        }
    } catch (e) {
        alert("Verification failed");
    } finally {
        showLoading(false);
    }
}

// --- Threat Analysis Logic ---
function switchThreatInput(type) {
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.threat-input-sec').forEach(s => s.classList.remove('active'));

    event.target.classList.add('active');
    document.getElementById(`threat-${type}-area`).classList.add('active');

    if (type === 'camera') startThreatCamera();
    else stopThreatCamera();
}

async function startThreatCamera() {
    threatStream = await navigator.mediaDevices.getUserMedia({ video: true });
    document.getElementById('threat-video').srcObject = threatStream;
}

function stopThreatCamera() {
    if (threatStream) threatStream.getTracks().forEach(t => t.stop());
}

async function runThreatAnalysis() {
    showLoading(true);
    let payload = {};
    const location = document.getElementById('threat-location').value;

    if (document.getElementById('threat-upload-area').classList.contains('active')) {
        const file = document.getElementById('threat-file').files[0];
        if (!file) return alert("Select a file");
        const formData = new FormData();
        formData.append('image', file);
        formData.append('location', location);

        try {
            const res = await fetch('/api/threat/analyze', { method: 'POST', body: formData });
            const result = await res.json();
            displayThreatResult(result);
        } catch (e) { alert("Analysis failed"); }
        finally { showLoading(false); }
    } else {
        const video = document.getElementById('threat-video');
        const canvas = document.createElement('canvas');
        canvas.width = video.videoWidth; canvas.height = video.videoHeight;
        canvas.getContext('2d').drawImage(video, 0, 0);
        const image = canvas.toDataURL('image/jpeg');

        try {
            const res = await fetch('/api/threat/analyze', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ image, location })
            });
            const result = await res.json();
            displayThreatResult(result);
        } catch (e) { alert("Analysis failed"); }
        finally { showLoading(false); }
    }
}

function displayThreatResult(result) {
    if (!result.success) return alert(result.error);
    const i = result.incident;
    document.getElementById('threat-output').innerHTML = `
        <div class="threat-summary">
            <p><strong>Score:</strong> <span class="badge badge-${i.threat_level.toLowerCase()}">${i.threat_score}%</span></p>
            <p><strong>Level:</strong> ${i.threat_level}</p>
            <p><strong>Summary:</strong> ${i.summary}</p>
        </div>
    `;
}

// --- Tables ---
async function fetchIncidents() {
    const res = await fetch('/api/incidents');
    const data = await res.json();
    const tbody = document.querySelector('#incidents-table tbody');
    tbody.innerHTML = data.map(i => `
        <tr>
            <td>#${i.id}</td>
            <td>${i.time}</td>
            <td>${i.location}</td>
            <td><span class="badge badge-${i.threat_level.toLowerCase()}">${i.threat_level}</span></td>
            <td>${i.status}</td>
            <td><button class="btn btn-sm btn-primary" onclick="updateIncidentStatus(${i.id})">Update</button></td>
        </tr>
    `).join('');
}

async function fetchStudents() {
    const res = await fetch('/api/students');
    const data = await res.json();
    const tbody = document.querySelector('#students-table tbody');
    tbody.innerHTML = data.map(s => `
        <tr>
            <td>${s.roll_no}</td>
            <td>${s.name}</td>
            <td>${s.department}</td>
            <td>${s.created_at}</td>
            <td><a href="/get_qr/${s.roll_no}" target="_blank">QR</a></td>
        </tr>
    `).join('');
}

// UI Helpers
function showLoading(show) {
    document.getElementById('loading').classList.toggle('hidden', !show);
}
