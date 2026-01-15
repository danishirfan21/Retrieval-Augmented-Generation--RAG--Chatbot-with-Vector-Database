// Determine API base
const queryApi = new URLSearchParams(location.search).get('api');
let API_BASE = '';
if (queryApi) {
  API_BASE = queryApi.replace(/\/$/, '');
} else if (window.API_BASE && window.API_BASE.trim()) {
  API_BASE = window.API_BASE.trim().replace(/\/$/, '');
} else {
  API_BASE = window.location.origin;
}
const api = (path) => `${API_BASE}${path}`;

// DOM Elements
const queryInput = document.getElementById('query-input');
const responseSection = document.getElementById('response-section');
const emptyState = document.getElementById('empty-state');
const responseContainer = document.getElementById('response-container');
const sourcesList = document.getElementById('sources-list');
const conversationList = document.getElementById('conversation-list');
const newQueryBtn = document.getElementById('new-query-btn');

// Modal elements
const uploadModal = document.getElementById('upload-modal');
const statsModal = document.getElementById('stats-modal');
const uploadNav = document.getElementById('upload-nav');
const statsNav = document.getElementById('stats-nav');
const modalClose = document.getElementById('modal-close');
const statsModalClose = document.getElementById('stats-modal-close');
const dropzone = document.getElementById('dropzone');
const fileInput = document.getElementById('file-input');
const uploadStatus = document.getElementById('upload-status');

// State
let conversations = [];
let currentConversationId = null;

// Initialize
document.addEventListener('DOMContentLoaded', () => {
  loadConversations();
  loadStats();
});

// Handle query input
queryInput.addEventListener('keypress', async (e) => {
  if (e.key === 'Enter' && queryInput.value.trim()) {
    const question = queryInput.value.trim();
    queryInput.value = '';
    await handleQuery(question);
  }
});

// Handle new query button
newQueryBtn.addEventListener('click', () => {
  currentConversationId = null;
  responseContainer.innerHTML = '';
  responseContainer.style.display = 'none';
  emptyState.style.display = 'block';
  sourcesList.innerHTML =
    '<div class="empty-state" style="padding: 20px; text-align: center;"><div class="empty-state-text" style="font-size: 13px;">Sources will appear here after you ask a question</div></div>';
  updateConversationList();
});

// Handle query
async function handleQuery(question) {
  // Hide empty state
  emptyState.style.display = 'none';
  responseContainer.style.display = 'block';

  // Create conversation if new
  if (!currentConversationId) {
    currentConversationId = Date.now().toString();
    conversations.unshift({
      id: currentConversationId,
      title: question.substring(0, 50),
      queries: [],
    });
    updateConversationList();
  }

  // Add loading state
  const loadingCard = createLoadingCard(question);
  responseContainer.appendChild(loadingCard);

  try {
    const response = await fetch(api('/api/v1/query'), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question, top_k: 5 }),
    });

    if (!response.ok) {
      throw new Error(`Error ${response.status}: ${await response.text()}`);
    }

    const data = await response.json();

    // Remove loading card
    loadingCard.remove();

    // Add response card
    const responseCard = createResponseCard(data);
    responseContainer.appendChild(responseCard);

    // Update sources
    updateSources(data.retrieved_docs);

    // Save to conversation
    const conversation = conversations.find(
      (c) => c.id === currentConversationId
    );
    if (conversation) {
      conversation.queries.push(data);
      saveConversations();
    }
  } catch (err) {
    loadingCard.remove();
    const errorCard = createErrorCard(question, err.message);
    responseContainer.appendChild(errorCard);
  }

  // Scroll to bottom
  responseSection.scrollTop = responseSection.scrollHeight;
}

// Create response card
function createResponseCard(data) {
  const card = document.createElement('div');
  card.className = 'response-card';

  const questionLabel = document.createElement('div');
  questionLabel.className = 'question-label';
  questionLabel.textContent = 'Query';

  const questionText = document.createElement('div');
  questionText.className = 'question-text';
  questionText.textContent = data.question;

  const answerLabel = document.createElement('div');
  answerLabel.className = 'answer-label';
  answerLabel.textContent = 'Answer';

  const answerText = document.createElement('div');
  answerText.className = 'answer-text';

  // Process answer with citations
  const answerWithCitations = data.answer.replace(
    /\[(\d+)\]/g,
    (match, num) => {
      return `<span class="citation">[${num}]</span>`;
    }
  );

  answerText.innerHTML = `<p>${answerWithCitations}</p>`;

  card.appendChild(questionLabel);
  card.appendChild(questionText);
  card.appendChild(answerLabel);
  card.appendChild(answerText);

  return card;
}

// Create loading card
function createLoadingCard(question) {
  const card = document.createElement('div');
  card.className = 'response-card';
  card.style.opacity = '0.6';

  const questionLabel = document.createElement('div');
  questionLabel.className = 'question-label';
  questionLabel.textContent = 'Query';

  const questionText = document.createElement('div');
  questionText.className = 'question-text';
  questionText.textContent = question;

  const answerLabel = document.createElement('div');
  answerLabel.className = 'answer-label';
  answerLabel.textContent = 'Answer';

  const answerText = document.createElement('div');
  answerText.className = 'answer-text';
  answerText.textContent = 'Generating answer...';

  card.appendChild(questionLabel);
  card.appendChild(questionText);
  card.appendChild(answerLabel);
  card.appendChild(answerText);

  return card;
}

// Create error card
function createErrorCard(question, error) {
  const card = document.createElement('div');
  card.className = 'response-card';
  card.style.borderColor = '#e57373';

  const questionLabel = document.createElement('div');
  questionLabel.className = 'question-label';
  questionLabel.textContent = 'Query';

  const questionText = document.createElement('div');
  questionText.className = 'question-text';
  questionText.textContent = question;

  const answerLabel = document.createElement('div');
  answerLabel.className = 'answer-label';
  answerLabel.textContent = 'Error';
  answerLabel.style.color = '#e57373';

  const answerText = document.createElement('div');
  answerText.className = 'answer-text';
  answerText.textContent = error;
  answerText.style.color = '#e57373';

  card.appendChild(questionLabel);
  card.appendChild(questionText);
  card.appendChild(answerLabel);
  card.appendChild(answerText);

  return card;
}

// Update sources
function updateSources(docs) {
  sourcesList.innerHTML = '';

  if (!docs || docs.length === 0) {
    sourcesList.innerHTML =
      '<div class="empty-state" style="padding: 20px; text-align: center;"><div class="empty-state-text" style="font-size: 13px;">No sources found</div></div>';
    return;
  }

  docs.forEach((doc, index) => {
    const card = document.createElement('div');
    card.className = 'source-card';

    const filename = document.createElement('div');
    filename.className = 'source-filename';
    filename.textContent = doc.source || 'unknown';

    const preview = document.createElement('div');
    preview.className = 'source-preview';
    preview.textContent = doc.text;

    const metadata = document.createElement('div');
    metadata.className = 'source-metadata';

    const score = document.createElement('span');
    score.className = 'source-score';
    score.textContent = doc.score ? doc.score.toFixed(2) : '0.00';

    const chunkId = document.createElement('span');
    chunkId.className = 'source-chunk-id';
    chunkId.textContent = `chunk_${index + 1}`;

    metadata.appendChild(score);
    metadata.appendChild(chunkId);

    card.appendChild(filename);
    card.appendChild(preview);
    card.appendChild(metadata);

    sourcesList.appendChild(card);
  });
}

// Update conversation list
function updateConversationList() {
  conversationList.innerHTML = '';

  conversations.forEach((conv) => {
    const item = document.createElement('div');
    item.className = 'conversation-item';
    if (conv.id === currentConversationId) {
      item.classList.add('active');
    }
    item.textContent = conv.title;
    item.addEventListener('click', () => loadConversation(conv.id));
    conversationList.appendChild(item);
  });
}

// Load conversation
function loadConversation(id) {
  const conversation = conversations.find((c) => c.id === id);
  if (!conversation) return;

  currentConversationId = id;
  responseContainer.innerHTML = '';
  emptyState.style.display = 'none';
  responseContainer.style.display = 'block';

  conversation.queries.forEach((query) => {
    const card = createResponseCard(query);
    responseContainer.appendChild(card);
  });

  if (conversation.queries.length > 0) {
    updateSources(
      conversation.queries[conversation.queries.length - 1].retrieved_docs
    );
  }

  updateConversationList();
}

// Save conversations to localStorage
function saveConversations() {
  localStorage.setItem('rag_conversations', JSON.stringify(conversations));
}

// Load conversations from localStorage
function loadConversations() {
  const saved = localStorage.getItem('rag_conversations');
  if (saved) {
    conversations = JSON.parse(saved);
    updateConversationList();
  }
}

// Modal handlers
uploadNav.addEventListener('click', () => {
  uploadModal.classList.add('active');
});

statsNav.addEventListener('click', async () => {
  statsModal.classList.add('active');
  await loadStats();
});

modalClose.addEventListener('click', () => {
  uploadModal.classList.remove('active');
});

statsModalClose.addEventListener('click', () => {
  statsModal.classList.remove('active');
});

window.addEventListener('click', (e) => {
  if (e.target === uploadModal) {
    uploadModal.classList.remove('active');
  }
  if (e.target === statsModal) {
    statsModal.classList.remove('active');
  }
});

// Upload handlers
dropzone.addEventListener('click', () => fileInput.click());
dropzone.addEventListener('dragover', (e) => {
  e.preventDefault();
  dropzone.classList.add('dragover');
});
dropzone.addEventListener('dragleave', () => {
  dropzone.classList.remove('dragover');
});
dropzone.addEventListener('drop', (e) => {
  e.preventDefault();
  dropzone.classList.remove('dragover');
  uploadFiles(e.dataTransfer.files);
});
fileInput.addEventListener('change', (e) => uploadFiles(e.target.files));

async function uploadFiles(files) {
  if (!files || files.length === 0) return;

  uploadStatus.textContent = 'Uploading and indexing documents...';
  uploadStatus.style.color = 'var(--color-accent)';

  const formData = new FormData();
  for (const file of files) {
    formData.append('files', file);
  }

  try {
    const response = await fetch(api('/api/v1/upload'), {
      method: 'POST',
      body: formData,
    });

    const data = await response.json();

    if (data.success) {
      uploadStatus.textContent = `Successfully uploaded ${data.files.length} file(s)!`;
      uploadStatus.style.color = '#6ea8fe';
      setTimeout(() => {
        uploadModal.classList.remove('active');
        uploadStatus.textContent = '';
      }, 2000);
    } else {
      uploadStatus.textContent = `Error: ${data.error}`;
      uploadStatus.style.color = '#e57373';
    }
  } catch (err) {
    uploadStatus.textContent = `Upload failed: ${err.message}`;
    uploadStatus.style.color = '#e57373';
  }
}

// Load stats
async function loadStats() {
  try {
    const response = await fetch(api('/api/v1/stats'));
    const data = await response.json();

    document.getElementById('stat-docs').textContent =
      data.total_vector_count.toLocaleString();
    document.getElementById('stat-dim').textContent = data.dimension;
    document.getElementById('stat-fullness').textContent =
      (data.index_fullness * 100).toFixed(2) + '%';
  } catch (err) {
    console.error('Failed to load stats:', err);
  }
}
