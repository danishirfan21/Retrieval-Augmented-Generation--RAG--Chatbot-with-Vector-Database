const form = document.getElementById('form');
const input = document.getElementById('question');
const messages = document.getElementById('messages');
const sourcesEl = document.getElementById('sources');

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
