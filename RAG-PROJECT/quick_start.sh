#!/bin/bash
# Quick start script for RAG System Web UI

echo "╔══════════════════════════════════════════════╗"
echo "║      RAG System Web UI - Quick Start         ║"
echo "╚══════════════════════════════════════════════╝"
echo ""

# Check dependencies
echo "Checking dependencies..."

# Check uv
if ! command -v uv &> /dev/null; then
    echo "❌ uv is not installed"
    echo "Install with: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi
echo "✅ uv found"

# Check Milvus
if command -v docker &> /dev/null; then
    if docker ps | grep -q milvus; then
        echo "✅ Milvus is running"
    else
        echo "⚠️  Milvus container not found (check if it's running)"
    fi
fi

# Check Ollama
OLLAMA_URL="http://localhost:11434"
if curl -s "${OLLAMA_URL}/api/tags" > /dev/null 2>&1; then
    echo "✅ Ollama is accessible"
else
    echo "⚠️  Ollama server not responding at ${OLLAMA_URL}"
fi

echo ""
echo "Starting web server..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "📱 Open in browser: http://localhost:8080"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Start server
cd "$(dirname "$0")"
uv run uvicorn web_app:app --host 0.0.0.0 --port 8080 --reload
