// ============================================
// RAG System - Frontend JavaScript
// ============================================

// Global error handler
window.addEventListener('error', (event) => {
    console.error('Global error:', event.error);
    return false;
});

window.addEventListener('unhandledrejection', (event) => {
    console.error('Unhandled promise rejection:', event.reason);
});

// API Configuration
const API_BASE = window.location.origin;

// Global State
let currentSettings = {
    searchType: 'hybrid',
    useReranker: false,
    topK: 5,
    theme: 'dark'
};

let uploadedFiles = [];

// ============================================
// Initialization
// ============================================

document.addEventListener('DOMContentLoaded', () => {
    try {
        console.log('Initializing application...');
        initializeNavigation();
        initializeChat();
        initializeIndex();
        initializeSettings();
        loadSettings();
        checkSystemHealth();
        addPageLoadAnimation();
        enhanceDropdowns();
        console.log('Application initialized successfully');
    } catch (error) {
        console.error('Error during initialization:', error);
    }
});

// ============================================
// Navigation
// ============================================

function initializeNavigation() {
    const navBtns = document.querySelectorAll('.nav-btn[data-page]');
    
    navBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const page = btn.dataset.page;
            switchPage(page);
            
            // Update active state
            navBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
        });
    });
}

function switchPage(page) {
    const pages = document.querySelectorAll('.page');
    pages.forEach(p => p.classList.remove('active'));
    
    const targetPage = document.getElementById(`${page}-page`);
    if (targetPage) {
        targetPage.classList.add('active');
        
        // Load collections when switching to index page
        if (page === 'index') {
            loadCollections();
        }
    }
}

// ============================================
// Chat Functionality
// ============================================

function initializeChat() {
    const sendBtn = document.getElementById('send-btn');
    const userInput = document.getElementById('user-input');
    const searchType = document.getElementById('search-type');
    const useReranker = document.getElementById('use-reranker');
    const exampleBtns = document.querySelectorAll('.example-btn');
    
    // Send button
    sendBtn.addEventListener('click', () => sendMessage());
    
    // Enter key to send (Shift+Enter for new line)
    userInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });
    
    // Auto-resize textarea
    userInput.addEventListener('input', () => {
        userInput.style.height = 'auto';
        userInput.style.height = userInput.scrollHeight + 'px';
    });
    
    // Update settings when changed
    searchType.addEventListener('change', (e) => {
        currentSettings.searchType = e.target.value;
    });
    
    useReranker.addEventListener('change', (e) => {
        currentSettings.useReranker = e.target.checked;
    });
    
    // Initialize current settings from UI (in case loaded from localStorage)
    currentSettings.searchType = searchType.value;
    currentSettings.useReranker = useReranker.checked;
    
    // Example questions
    exampleBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const query = btn.dataset.query;
            userInput.value = query;
            sendMessage();
        });
    });
}

async function sendMessage() {
    const userInput = document.getElementById('user-input');
    const sendBtn = document.getElementById('send-btn');
    const query = userInput.value.trim();
    
    if (!query) return;
    
    // Clear input
    userInput.value = '';
    userInput.style.height = 'auto';
    
    // Disable send button
    sendBtn.disabled = true;
    
    // Remove welcome message if exists
    const welcomeMsg = document.querySelector('.welcome-message');
    if (welcomeMsg) {
        welcomeMsg.remove();
    }
    
    // Add user message
    addMessage('user', query);
    
    // Add typing indicator and prepare for streaming
    const typingIndicator = addTypingIndicator();
    let streamedContent = '';
    let sources = null;
    let assistantMessageDiv = null;
    let lastRenderTime = 0;
    const renderThrottle = 100; // Render markdown every 100ms
    
    try {
        // Make streaming API request
        const response = await fetch(`${API_BASE}/api/query/stream`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                query: query,
                search_type: currentSettings.searchType,
                use_reranker: currentSettings.useReranker,
                top_k: currentSettings.topK
            })
        });
        
        if (!response.ok) {
            throw new Error('Failed to get response');
        }
        
        // Remove typing indicator and create assistant message
        typingIndicator.remove();
        assistantMessageDiv = createStreamingMessage();
        
        // Read the stream
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';
        
        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            
            const chunk = decoder.decode(value, { stream: true });
            buffer += chunk;
            const lines = buffer.split('\n');
            
            // Keep last incomplete line in buffer
            buffer = lines.pop() || '';
            
            for (const line of lines) {
                if (line.startsWith('data: ')) {
                    const data = line.slice(6).trim();
                    if (!data) continue;
                    
                    try {
                        const parsed = JSON.parse(data);
                        
                        if (parsed.type === 'sources') {
                            sources = parsed.data;
                        } else if (parsed.type === 'token') {
                            streamedContent += parsed.data;
                            const now = Date.now();
                            // Throttle markdown rendering to prevent flickering
                            if (now - lastRenderTime > renderThrottle) {
                                updateStreamingMessage(assistantMessageDiv, streamedContent);
                                lastRenderTime = now;
                            }
                        } else if (parsed.type === 'done') {
                            // Finalize message with sources
                            finalizeStreamingMessage(assistantMessageDiv, streamedContent, sources);
                        } else if (parsed.type === 'error') {
                            throw new Error(parsed.data);
                        }
                    } catch (parseError) {
                        console.warn('Failed to parse SSE data:', data, parseError);
                    }
                }
            }
        }
        
    } catch (error) {
        console.error('Error:', error);
        if (typingIndicator.parentNode) {
            typingIndicator.remove();
        }
        if (assistantMessageDiv) {
            assistantMessageDiv.remove();
        }
        addMessage('assistant', 'Sorry, I encountered an error processing your request. Please try again.');
        showToast('Error processing query', 'error');
    } finally {
        sendBtn.disabled = false;
    }
}

// Simple markdown to HTML converter
function markdownToHtml(markdown) {
    let html = markdown;
    
    // Code blocks
    html = html.replace(/```([\s\S]*?)```/g, '<pre><code>$1</code></pre>');
    
    // Inline code
    html = html.replace(/`([^`]+)`/g, '<code>$1</code>');
    
    // Bold
    html = html.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
    html = html.replace(/__([^_]+)__/g, '<strong>$1</strong>');
    
    // Italic
    html = html.replace(/\*([^*]+)\*/g, '<em>$1</em>');
    html = html.replace(/_([^_]+)_/g, '<em>$1</em>');
    
    // Headers
    html = html.replace(/^### (.+)$/gm, '<h3>$1</h3>');
    html = html.replace(/^## (.+)$/gm, '<h2>$1</h2>');
    html = html.replace(/^# (.+)$/gm, '<h1>$1</h1>');
    
    // Lists
    html = html.replace(/^\* (.+)$/gm, '<li>$1</li>');
    html = html.replace(/^- (.+)$/gm, '<li>$1</li>');
    html = html.replace(/(<li>.*<\/li>)/s, '<ul>$1</ul>');
    
    // Links
    html = html.replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank">$1</a>');
    
    // Line breaks and paragraphs
    html = html.split('\n\n').map(para => {
        if (para.trim() && !para.startsWith('<')) {
            return '<p>' + para.replace(/\n/g, '<br>') + '</p>';
        }
        return para;
    }).join('\n');
    
    return html;
}

function addMessage(type, content, sources = null) {
    const chatMessages = document.getElementById('chat-messages');
    const messageDiv = document.createElement('div');
    messageDiv.className = `message message-${type}`;
    
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    
    // Convert markdown to HTML for assistant messages
    if (type === 'assistant') {
        contentDiv.innerHTML = markdownToHtml(content);
    } else {
        // For user messages, keep as plain text
        const paragraphs = content.split('\n\n').filter(p => p.trim());
        paragraphs.forEach(para => {
            const p = document.createElement('p');
            p.textContent = para;
            contentDiv.appendChild(p);
        });
    }
    
    messageDiv.appendChild(contentDiv);
    
    // Add sources if available
    if (sources && sources.length > 0) {
        const sourcesDiv = document.createElement('div');
        sourcesDiv.className = 'sources';
        
        const title = document.createElement('div');
        title.className = 'sources-title';
        title.textContent = '📚 Sources';
        sourcesDiv.appendChild(title);
        
        sources.forEach(source => {
            const sourceItem = document.createElement('div');
            sourceItem.className = 'source-item';
            
            const text = document.createElement('div');
            text.textContent = source.text;
            sourceItem.appendChild(text);
            
            const meta = document.createElement('div');
            meta.className = 'source-meta';
            meta.innerHTML = `
                <span><i class="fas fa-file"></i> ${source.source}</span>
                <span><i class="fas fa-book-open"></i> Page ${source.page_no}</span>
                <span><i class="fas fa-star"></i> Score: ${source.score}</span>
            `;
            sourceItem.appendChild(meta);
            
            sourcesDiv.appendChild(sourceItem);
        });
        
        messageDiv.appendChild(sourcesDiv);
    }
    
    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function addTypingIndicator() {
    const chatMessages = document.getElementById('chat-messages');
    const indicator = document.createElement('div');
    indicator.className = 'message message-assistant';
    indicator.innerHTML = `
        <div class="message-content">
            <div class="typing-indicator">
                <span></span>
                <span></span>
                <span></span>
            </div>
        </div>
    `;
    chatMessages.appendChild(indicator);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    return indicator;
}

function createStreamingMessage() {
    const chatMessages = document.getElementById('chat-messages');
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message message-assistant';
    
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    contentDiv.innerHTML = '';
    
    messageDiv.appendChild(contentDiv);
    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    
    return messageDiv;
}

function updateStreamingMessage(messageDiv, content) {
    const contentDiv = messageDiv.querySelector('.message-content');
    // Render markdown during streaming (throttled to prevent flickering)
    contentDiv.innerHTML = markdownToHtml(content);
    
    // Scroll to bottom
    const chatMessages = document.getElementById('chat-messages');
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function finalizeStreamingMessage(messageDiv, content, sources) {
    const contentDiv = messageDiv.querySelector('.message-content');
    contentDiv.innerHTML = markdownToHtml(content);
    
    // Add sources button if sources available
    if (sources && sources.length > 0) {
        const sourcesContainer = document.createElement('div');
        sourcesContainer.className = 'sources-container';
        
        // Create toggle button
        const toggleBtn = document.createElement('button');
        toggleBtn.className = 'sources-toggle-btn';
        toggleBtn.innerHTML = `
            <i class="fas fa-book"></i>
            <span>View Sources (${sources.length})</span>
            <i class="fas fa-chevron-down"></i>
        `;
        
        // Create sources content (hidden by default)
        const sourcesDiv = document.createElement('div');
        sourcesDiv.className = 'sources-content';
        sourcesDiv.style.display = 'none';
        
        sources.forEach((source, index) => {
            const sourceItem = document.createElement('div');
            sourceItem.className = 'source-item';
            
            const header = document.createElement('div');
            header.className = 'source-header';
            header.innerHTML = `
                <span class="source-number">#${index + 1}</span>
                <span><i class="fas fa-file"></i> ${source.source}</span>
                <span><i class="fas fa-book-open"></i> Page ${source.page_no}</span>
                <span><i class="fas fa-star"></i> ${source.score}</span>
            `;
            
            const text = document.createElement('div');
            text.className = 'source-text';
            text.textContent = source.text;
            
            sourceItem.appendChild(header);
            sourceItem.appendChild(text);
            sourcesDiv.appendChild(sourceItem);
        });
        
        // Toggle functionality
        toggleBtn.addEventListener('click', () => {
            const isVisible = sourcesDiv.style.display !== 'none';
            sourcesDiv.style.display = isVisible ? 'none' : 'block';
            toggleBtn.classList.toggle('active', !isVisible);
            const chevron = toggleBtn.querySelector('.fa-chevron-down');
            chevron.className = isVisible ? 'fas fa-chevron-down' : 'fas fa-chevron-up';
        });
        
        sourcesContainer.appendChild(toggleBtn);
        sourcesContainer.appendChild(sourcesDiv);
        messageDiv.appendChild(sourcesContainer);
    }
    
    // Scroll to bottom
    const chatMessages = document.getElementById('chat-messages');
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

// ============================================
// Index Functionality
// ============================================

function initializeIndex() {
    const fileInput = document.getElementById('file-input');
    const uploadArea = document.getElementById('upload-area');
    const startIndexBtn = document.getElementById('start-index-btn');
    const refreshBtn = document.getElementById('refresh-collections-btn');
    
    // File input change
    fileInput.addEventListener('change', (e) => {
        handleFiles(e.target.files);
    });
    
    // Drag and drop
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('dragover');
    });
    
    uploadArea.addEventListener('dragleave', () => {
        uploadArea.classList.remove('dragover');
    });
    
    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
        handleFiles(e.dataTransfer.files);
    });
    
    // Start indexing
    startIndexBtn.addEventListener('click', startIndexing);
    
    // Refresh collections
    if (refreshBtn) {
        refreshBtn.addEventListener('click', (e) => {
            try {
                e.preventDefault();
                e.stopPropagation();
                console.log('Refresh button clicked');
                loadCollections().catch(err => {
                    console.error('Error in loadCollections:', err);
                });
            } catch (error) {
                console.error('Error in click handler:', error);
            }
            return false;
        });
    } else {
        console.error('Refresh button not found');
    }
}

function handleFiles(files) {
    uploadedFiles = Array.from(files);
    displayFileList();
    uploadFiles();
}

function displayFileList() {
    const fileList = document.getElementById('file-list');
    fileList.innerHTML = '';
    
    if (uploadedFiles.length === 0) return;
    
    uploadedFiles.forEach((file, index) => {
        const fileItem = document.createElement('div');
        fileItem.className = 'file-item';
        fileItem.innerHTML = `
            <div class="file-info">
                <div class="file-icon"><i class="fas fa-file-pdf"></i></div>
                <div>
                    <div class="file-name">${file.name}</div>
                    <div class="file-size">${formatFileSize(file.size)}</div>
                </div>
            </div>
            <button class="file-remove" onclick="removeFile(${index})">
                <i class="fas fa-times"></i>
            </button>
        `;
        fileList.appendChild(fileItem);
    });
}

function removeFile(index) {
    uploadedFiles.splice(index, 1);
    displayFileList();
}

function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
}

async function uploadFiles() {
    showLoading();
    
    for (const file of uploadedFiles) {
        try {
            const formData = new FormData();
            formData.append('file', file);
            
            const response = await fetch(`${API_BASE}/api/upload`, {
                method: 'POST',
                body: formData
            });
            
            if (!response.ok) {
                throw new Error(`Failed to upload ${file.name}`);
            }
            
        } catch (error) {
            console.error('Upload error:', error);
            showToast(`Failed to upload ${file.name}`, 'error');
        }
    }
    
    hideLoading();
    showToast('Files uploaded successfully', 'success');
}

async function startIndexing() {
    const collectionName = document.getElementById('collection-name').value.trim();
    const dropExisting = document.getElementById('drop-existing').checked;
    
    if (!collectionName) {
        showToast('Please enter a collection name', 'error');
        return;
    }
    
    if (uploadedFiles.length === 0) {
        showToast('Please upload some files first', 'error');
        return;
    }
    
    try {
        const response = await fetch(`${API_BASE}/api/index`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                collection_name: collectionName,
                drop_existing: dropExisting
            })
        });
        
        if (!response.ok) {
            throw new Error('Failed to start indexing');
        }
        
        showToast('Indexing started', 'info');
        
        // Show status card and start polling
        document.getElementById('status-card').style.display = 'block';
        pollIndexingStatus();
        
    } catch (error) {
        console.error('Indexing error:', error);
        showToast('Failed to start indexing', 'error');
    }
}

async function pollIndexingStatus() {
    const statusCard = document.getElementById('status-card');
    const progressFill = document.getElementById('progress-fill');
    const statusMessage = document.getElementById('status-message');
    
    const interval = setInterval(async () => {
        try {
            const response = await fetch(`${API_BASE}/api/index/status`);
            const data = await response.json();
            
            progressFill.style.width = data.progress + '%';
            progressFill.textContent = data.progress + '%';
            statusMessage.textContent = data.message;
            
            if (data.status === 'completed') {
                clearInterval(interval);
                showToast('Indexing completed successfully!', 'success');
                setTimeout(() => {
                    statusCard.style.display = 'none';
                    uploadedFiles = [];
                    displayFileList();
                    loadCollections();
                }, 3000);
            } else if (data.status === 'error') {
                clearInterval(interval);
                showToast(data.message, 'error');
            }
            
        } catch (error) {
            console.error('Status polling error:', error);
            clearInterval(interval);
        }
    }, 2000);
}

async function loadCollections() {
    const collectionsList = document.getElementById('collections-list');
    
    if (!collectionsList) {
        console.error('collections-list element not found');
        return;
    }
    
    collectionsList.innerHTML = '<p class="loading">Loading collections...</p>';
    
    try {
        const response = await fetch(`${API_BASE}/api/collections`);
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        console.log('Collections data:', data);
        
        if (!data.collections || data.collections.length === 0) {
            collectionsList.innerHTML = '<p class="loading">No collections found</p>';
            return;
        }
        
        collectionsList.innerHTML = '';
        data.collections.forEach(collection => {
            const item = document.createElement('div');
            item.className = 'collection-item';
            item.innerHTML = `
                <div class="collection-icon"><i class="fas fa-database"></i></div>
                <div>
                    <strong>${collection}</strong>
                    <div style="font-size: 0.9rem; color: var(--text-secondary);">
                        Database: ${data.database || 'N/A'}
                    </div>
                </div>
            `;
            collectionsList.appendChild(item);
        });
        
        console.log('Collections loaded successfully');
        
    } catch (error) {
        console.error('Error loading collections:', error);
        collectionsList.innerHTML = `<p class="loading" style="color: var(--accent-danger);">Failed to load collections: ${error.message}</p>`;
    }
}

// ============================================
// Settings Functionality
// ============================================

function initializeSettings() {
    const saveBtn = document.getElementById('save-settings-btn');
    const themeBtns = document.querySelectorAll('.theme-btn');
    
    // Theme buttons
    themeBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const theme = btn.dataset.theme;
            setTheme(theme);
            themeBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
        });
    });
    
    // Save settings
    saveBtn.addEventListener('click', saveSettings);
    
    // Load system info
    loadSystemInfo();
}

function loadSettings() {
    const saved = localStorage.getItem('rag_settings');
    if (saved) {
        currentSettings = JSON.parse(saved);
        
        // Apply settings to UI
        document.getElementById('search-type').value = currentSettings.searchType;
        document.getElementById('use-reranker').checked = currentSettings.useReranker;
        document.getElementById('default-search-type').value = currentSettings.searchType;
        document.getElementById('top-k').value = currentSettings.topK;
        document.getElementById('default-reranker').checked = currentSettings.useReranker;
        
        setTheme(currentSettings.theme);
    }
}

function saveSettings() {
    currentSettings = {
        searchType: document.getElementById('default-search-type').value,
        useReranker: document.getElementById('default-reranker').checked,
        topK: parseInt(document.getElementById('top-k').value),
        theme: currentSettings.theme
    };
    
    localStorage.setItem('rag_settings', JSON.stringify(currentSettings));
    
    // Update chat page settings
    document.getElementById('search-type').value = currentSettings.searchType;
    document.getElementById('use-reranker').checked = currentSettings.useReranker;
    
    showToast('Settings saved successfully', 'success');
}

function setTheme(theme) {
    document.documentElement.setAttribute('data-theme', theme);
    currentSettings.theme = theme;
    localStorage.setItem('rag_settings', JSON.stringify(currentSettings));
}

async function loadSystemInfo() {
    const systemInfo = document.getElementById('system-info');
    
    try {
        const response = await fetch(`${API_BASE}/health`);
        const data = await response.json();
        
        systemInfo.innerHTML = `
            <p><strong>Status:</strong> ${data.status}</p>
            <p><strong>Milvus:</strong> ${data.milvus_initialized ? '✓ Connected' : '✗ Not Connected'}</p>
            <p><strong>Retriever:</strong> ${data.retriever_initialized ? '✓ Initialized' : '✗ Not Initialized'}</p>
            <p><strong>Ollama:</strong> ${data.ollama_initialized ? '✓ Connected' : '✗ Not Connected'}</p>
        `;
    } catch (error) {
        systemInfo.innerHTML = '<p>Failed to load system information</p>';
    }
}

async function checkSystemHealth() {
    try {
        const response = await fetch(`${API_BASE}/health`);
        const data = await response.json();
        
        if (!data.milvus_initialized || !data.retriever_initialized || !data.ollama_initialized) {
            showToast('Some system components are not initialized', 'error');
        }
    } catch (error) {
        console.error('Health check failed:', error);
    }
}

// ============================================
// Utility Functions
// ============================================

function showLoading() {
    document.getElementById('loading-overlay').classList.add('active');
}

function hideLoading() {
    document.getElementById('loading-overlay').classList.remove('active');
}

function showToast(message, type = 'info') {
    const toast = document.getElementById('toast');
    toast.textContent = message;
    toast.className = `toast ${type} show`;
    
    setTimeout(() => {
        toast.classList.remove('show');
    }, 3000);
}

// ============================================
// UI Enhancement Functions
// ============================================

function addPageLoadAnimation() {
    // Add fade-in animation to elements
    const elements = document.querySelectorAll('.card, .chat-container, .nav-btn');
    elements.forEach((el, index) => {
        el.style.opacity = '0';
        el.style.transform = 'translateY(20px)';
        setTimeout(() => {
            el.style.transition = 'all 0.6s cubic-bezier(0.4, 0, 0.2, 1)';
            el.style.opacity = '1';
            el.style.transform = 'translateY(0)';
        }, index * 50);
    });
}

function enhanceDropdowns() {
    // Add enhanced interaction for dropdowns
    const selects = document.querySelectorAll('.select-input');
    
    selects.forEach(select => {
        // Add visual feedback on change
        select.addEventListener('change', () => {
            select.style.transform = 'scale(0.98)';
            setTimeout(() => {
                select.style.transform = 'scale(1)';
            }, 100);
            
            // Add ripple effect
            createRipple(select);
        });
        
        // Add focus enhancement
        select.addEventListener('focus', () => {
            select.parentElement.classList.add('focused');
        });
        
        select.addEventListener('blur', () => {
            select.parentElement.classList.remove('focused');
        });
    });
    
    // Enhance checkbox animations
    const checkboxes = document.querySelectorAll('input[type="checkbox"]');
    checkboxes.forEach(checkbox => {
        checkbox.addEventListener('change', () => {
            if (checkbox.checked) {
                checkbox.parentElement.style.transform = 'scale(1.05)';
                setTimeout(() => {
                    checkbox.parentElement.style.transform = 'scale(1)';
                }, 200);
            }
        });
    });
}

function createRipple(element) {
    const ripple = document.createElement('span');
    const rect = element.getBoundingClientRect();
    const size = Math.max(rect.width, rect.height);
    const x = rect.width / 2;
    const y = rect.height / 2;
    
    ripple.style.width = ripple.style.height = size + 'px';
    ripple.style.left = x - size / 2 + 'px';
    ripple.style.top = y - size / 2 + 'px';
    ripple.style.position = 'absolute';
    ripple.style.borderRadius = '50%';
    ripple.style.background = 'rgba(102, 126, 234, 0.5)';
    ripple.style.transform = 'scale(0)';
    ripple.style.animation = 'ripple 0.6s ease-out';
    ripple.style.pointerEvents = 'none';
    
    element.style.position = 'relative';
    element.style.overflow = 'hidden';
    element.appendChild(ripple);
    
    setTimeout(() => ripple.remove(), 600);
}

// Add smooth scroll behavior
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        const target = document.querySelector(this.getAttribute('href'));
        if (target) {
            target.scrollIntoView({
                behavior: 'smooth',
                block: 'start'
            });
        }
    });
});
