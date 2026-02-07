import sys
from pathlib import Path
from typing import List, Dict
from datetime import datetime

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

BASE_DIR = Path(__file__).parent
sys.path.append(str(BASE_DIR))

from src.milvus_store import MilvusStore
from src.hybrid_search import create_hybrid_retriever
from src.index import process_and_index_directory
from src.config import ConfigLoader
from ollama import Client

# Initialize FastAPI app
app = FastAPI(title="RAG System", description="Document Indexing and Q&A System")

# Mount static files
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load configuration
config = ConfigLoader(str(BASE_DIR / "config.yaml"))

# Global variables for storing system state
milvus_store = None
retriever = None
ollama_client = None
indexing_status = {"status": "idle", "message": "", "progress": 0}


# Pydantic models for request/response
class QueryRequest(BaseModel):
    query: str
    search_type: str = "hybrid"
    use_reranker: bool = False
    top_k: int = 5


class QueryResponse(BaseModel):
    query: str
    answer: str
    sources: List[Dict]
    search_method: str
    timestamp: str


class IndexRequest(BaseModel):
    collection_name: str = "collectionDemo"
    drop_existing: bool = False


class StatusResponse(BaseModel):
    status: str
    message: str
    progress: int


# Initialize system components
def initialize_system():
    """Initialize Milvus store and retriever"""
    global milvus_store, retriever, ollama_client

    try:
        milvus_store = MilvusStore(
            uri=config.get("database", "uri", default="http://localhost:19530"),
            db_name=config.get("database", "name", default="ragMultimodal"),
            collection_name=config.get(
                "database", "collection_name", default="collectionDemo"
            ),
            embed_model=config.get(
                "model", "embeddings", default="sentence-transformers/all-MiniLM-L6-v2"
            ),
            drop_old=False,
            namespace=config.get("database", "namespace", default="ragDemo2"),
        )

        retriever = create_hybrid_retriever(
            milvus_store=milvus_store,
            search_type="hybrid",
            enable_reranker=False,
            bm25_weight=0.5,
            semantic_weight=0.5,
            rrf_k=60,
        )

        # Initialize Ollama client
        ollama_url = config.get("ollama", "url", default="http://localhost:11434")
        ollama_client = Client(host=ollama_url)

        return True
    except Exception as e:
        print(f"Error initializing system: {e}")
        return False


# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize system on startup"""
    print("Initializing RAG system...")
    if initialize_system():
        print("✓ System initialized successfully")
    else:
        print("✗ System initialization failed")


# Root endpoint - serve HTML interface
@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main HTML interface"""
    html_file = BASE_DIR / "static" / "index.html"
    if html_file.exists():
        return html_file.read_text()
    else:
        return """
        <html>
            <body>
                <h1>RAG System</h1>
                <p>Frontend not found. Please create static/index.html</p>
                <p>API is running at <a href="/docs">/docs</a></p>
            </body>
        </html>
        """


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "milvus_initialized": milvus_store is not None,
        "retriever_initialized": retriever is not None,
        "ollama_initialized": ollama_client is not None,
    }


@app.post("/api/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """
    Query documents and get AI response
    """
    global retriever, ollama_client

    if not retriever or not ollama_client:
        raise HTTPException(status_code=500, detail="System not initialized")

    try:
        # Update retriever search type if needed
        if retriever.search_type != request.search_type:
            retriever = create_hybrid_retriever(
                milvus_store=milvus_store,
                search_type=request.search_type,
                enable_reranker=request.use_reranker,
                bm25_weight=0.5,
                semantic_weight=0.5,
                rrf_k=60,
            )

        # Perform search
        if request.use_reranker and hasattr(retriever, "search_and_rerank"):
            results = retriever.search_and_rerank(
                query=request.query,
                initial_k=request.top_k * 4,
                top_k=request.top_k,
            )
        else:
            results = retriever.search(query=request.query, top_k=request.top_k)

        if not results:
            return QueryResponse(
                query=request.query,
                answer="No relevant documents found for your query.",
                sources=[],
                search_method=request.search_type,
                timestamp=datetime.now().isoformat(),
            )

        # Format context
        context = "\n\n".join(
            [
                f"{i+1}. {result.text}\n   Source: {result.source}, Page: {result.page_no}"
                for i, result in enumerate(results)
            ]
        )

        # Generate response using Ollama
        system_prompt = "You are an AI assistant. Provide accurate answers based on the given context. The response should be perfect without adding based on context like that."
        user_prompt = f"""
Use the following information to answer the question.

Context:
{context}

Question: {request.query}

Answer:"""

        response = ollama_client.chat(
            model=config.get("ollama", "model", default="llama3.1:8b"),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            stream=False,
        )

        answer = response["message"]["content"]

        # Format sources
        sources = [
            {
                "text": (
                    result.text[:200] + "..." if len(result.text) > 200 else result.text
                ),
                "source": result.source,
                "page_no": result.page_no,
                "score": round(result.score, 4),
                "method": result.search_method,
            }
            for result in results
        ]

        return QueryResponse(
            query=request.query,
            answer=answer,
            sources=sources,
            search_method=request.search_type,
            timestamp=datetime.now().isoformat(),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


@app.post("/api/query/stream")
async def query_documents_stream(request: QueryRequest):
    """
    Query documents and stream AI response
    """
    import json
    import asyncio

    global retriever, ollama_client

    if not retriever or not ollama_client:
        raise HTTPException(status_code=500, detail="System not initialized")

    async def generate():
        try:
            # Update retriever search type if needed
            if retriever.search_type != request.search_type:
                new_retriever = create_hybrid_retriever(
                    milvus_store=milvus_store,
                    search_type=request.search_type,
                    enable_reranker=request.use_reranker,
                    bm25_weight=0.5,
                    semantic_weight=0.5,
                    rrf_k=60,
                )
            else:
                new_retriever = retriever

            # Perform search
            if request.use_reranker:
                new_retriever = create_hybrid_retriever(
                    milvus_store=milvus_store,
                    search_type=request.search_type,
                    enable_reranker=request.use_reranker,
                    bm25_weight=0.5,
                    semantic_weight=0.5,
                    rrf_k=60,
                )
                results = new_retriever.search_and_rerank(
                    query=request.query,
                    initial_k=request.top_k * 4,
                    top_k=request.top_k,
                )
            else:
                results = new_retriever.search(query=request.query, top_k=request.top_k)

            sources = [
                {
                    "text": result.text,
                    "source": result.source,
                    "page_no": result.page_no,
                    "score": round(result.score, 4),
                }
                for result in results
            ]

            yield f"data: {json.dumps({'type': 'sources', 'data': sources})}\n\n"

            if not results:
                yield f"data: {json.dumps({'type': 'token', 'data': 'No relevant documents found.'})}\n\n"
                yield f"data: {json.dumps({'type': 'done'})}\n\n"
                return

            # Format context
            context = "\n\n".join(
                [
                    f"{i+1}. {result.text}\n   Source: {result.source}, Page: {result.page_no}"
                    for i, result in enumerate(results)
                ]
            )

            # Stream response
            system_prompt = (
                "You are a knowledgeable assistant. "
                "Answer questions clearly and directly. "
                "Do not mention sources, context, or documents. "
                "If the information contains tables, summarize them in natural language; do not print raw tables. "
                "Provide answers in a human-readable way."
            )

            user_prompt = f"""
Use the following information to answer the question.

Context:
{context}

Question: {request.query}

Answer:"""

            stream = ollama_client.chat(
                model=config.get("ollama", "model", default="llama3.1:8b"),
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                stream=True,
            )

            for chunk in stream:
                content = chunk["message"]["content"]
                if content:
                    yield f"data: {json.dumps({'type': 'token', 'data': content})}\n\n"
                    await asyncio.sleep(0)

            yield f"data: {json.dumps({'type': 'done'})}\n\n"

        except Exception as e:
            error_msg = str(e).replace("\\", "\\\\").replace('"', '\\"')
            yield f"data: {json.dumps({'type': 'error', 'data': error_msg})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


def index_documents_background(
    input_dir: Path, collection_name: str, drop_existing: bool
):
    """Background task for indexing documents"""
    global indexing_status

    try:
        indexing_status = {
            "status": "running",
            "message": "Starting indexing...",
            "progress": 10,
        }

        # Set collection name
        config.set("database", "collection_name", collection_name)

        indexing_status = {
            "status": "running",
            "message": "Processing documents...",
            "progress": 30,
        }

        # Process and index
        process_and_index_directory(
            input_dir, drop_existing=drop_existing, config=config
        )

        indexing_status = {
            "status": "running",
            "message": "Rebuilding retriever...",
            "progress": 80,
        }

        # Reinitialize system with new collection
        initialize_system()

        indexing_status = {
            "status": "completed",
            "message": "Indexing completed successfully!",
            "progress": 100,
        }

    except Exception as e:
        indexing_status = {
            "status": "error",
            "message": f"Indexing failed: {str(e)}",
            "progress": 0,
        }


@app.post("/api/index")
async def index_documents(request: IndexRequest, background_tasks: BackgroundTasks):
    """
    Index documents from input_files directory
    """
    global indexing_status

    if indexing_status["status"] == "running":
        raise HTTPException(status_code=400, detail="Indexing is already in progress")

    input_dir = BASE_DIR / "input_files"
    if not input_dir.exists():
        raise HTTPException(status_code=404, detail="Input files directory not found")

    # Start indexing in background
    background_tasks.add_task(
        index_documents_background,
        input_dir,
        request.collection_name,
        request.drop_existing,
    )

    return {"message": "Indexing started", "status": "running"}


@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    Upload a document to input_files directory
    """
    try:
        input_dir = BASE_DIR / "input_files"
        input_dir.mkdir(exist_ok=True)

        file_path = input_dir / file.filename

        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)

        return {
            "message": f"File uploaded successfully: {file.filename}",
            "filename": file.filename,
            "size": len(content),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error uploading file: {str(e)}")


@app.get("/api/index/status", response_model=StatusResponse)
async def get_indexing_status():
    """Get current indexing status"""
    return indexing_status


@app.get("/api/collections")
async def list_collections():
    """List available collections"""
    try:
        from pymilvus import utility, db

        db_name = config.get("database", "name", default="ragMultimodal")
        db.using_database(db_name)

        collections = utility.list_collections()

        return {
            "database": db_name,
            "collections": collections,
            "count": len(collections),
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error listing collections: {str(e)}"
        )


if __name__ == "__main__":
    # Create static directory if it doesn't exist
    static_dir = BASE_DIR / "static"
    static_dir.mkdir(exist_ok=True)

    # Mount static files
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    # Run the server
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
