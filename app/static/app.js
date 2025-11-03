const form = document.getElementById('form');
const input = document.getElementById('question');
const messages = document.getElementById('messages');
const sourcesEl = document.getElementById('sources');
const dropzone = document.getElementById('dropzone');
const fileInput = document.getElementById('file-input');
const uploadStatus = document.getElementById('upload-status');

// Allow running frontend on a different port/domain than the backend
// Usage: /index.html?api=http://127.0.0.1:8000
const API_BASE = new URLSearchParams(location.search).get('api') || (window.API_BASE || '').trim();
const api = (path) => `${API_BASE}${path}`;

const appendMsg = (text, who = 'bot') => {
  const div = document.createElement('div');
  div.className = `msg ${who}`;
  div.textContent = text;
  messages.appendChild(div);
  messages.scrollTop = messages.scrollHeight;
};

const setSources = (sources) => {
  sourcesEl.innerHTML = '';
  const unique = Array.from(new Set(sources || []));
  for (const src of unique) {
    const li = document.createElement('li');
    li.textContent = src;
    sourcesEl.appendChild(li);
  }
};

form.addEventListener('submit', async (e) => {
  e.preventDefault();
  const q = input.value.trim();
  if (!q) return;
  appendMsg(q, 'user');
  input.value = '';

  try {
    const res = await fetch(api('/api/v1/query'), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question: q, top_k: 5 })
    });
    if (!res.ok) {
      const txt = await res.text();
      appendMsg(`Error ${res.status}: ${txt}`);
      return;
    }
    const data = await res.json();
    appendMsg(data.answer || '(no answer)');
    setSources(data.sources);
  } catch (err) {
    appendMsg(`Request failed: ${err}`);
  }
});

// Warm up: add a greeting message
appendMsg('Hi! Ask me about the financial docs in this project.');

// Update footer links to target the API base if provided
try {
  const links = Array.from(document.querySelectorAll('footer a'));
  for (const a of links) {
    if (a.textContent.includes('API Docs')) a.href = api('/docs');
    if (a.textContent.includes('Health')) a.href = api('/api/v1/health');
  }
} catch {}

// --- Upload/drag-and-drop logic ---
function setUploadStatus(msg, ok = true) {
  uploadStatus.textContent = msg;
  uploadStatus.style.color = ok ? '#6ea8fe' : '#e57373';
}

function uploadFiles(files) {
  if (!files || !files.length) return;
  setUploadStatus('Uploading...');
  const form = new FormData();
  for (const f of files) form.append('files', f);
  fetch(api('/api/v1/upload'), {
    method: 'POST',
    body: form
  })
    .then(res => res.json())
    .then(data => {
      if (data.success) {
        setUploadStatus('Upload and ingestion complete!', true);
      } else {
        setUploadStatus(data.error || 'Upload failed', false);
      }
    })
    .catch(e => setUploadStatus('Upload error: ' + e, false));
}

dropzone.addEventListener('dragover', e => {
  e.preventDefault();
  dropzone.classList.add('dragover');
});
dropzone.addEventListener('dragleave', e => {
  dropzone.classList.remove('dragover');
});
dropzone.addEventListener('drop', e => {
  e.preventDefault();
  dropzone.classList.remove('dragover');
  uploadFiles(e.dataTransfer.files);
});
dropzone.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', e => uploadFiles(e.target.files));
