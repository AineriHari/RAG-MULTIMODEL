from typing import List, Literal
from pymilvus import Collection
from rank_bm25 import BM25Okapi
import numpy as np
from dataclasses import dataclass


@dataclass
class SearchResult:
    """Container for search results"""

    text: str
    source: str
    page_no: int
    score: float
    search_method: str


class HybridSearchRetriever:
    """
    Hybrid search retriever combining BM25 and semantic search.

    Features:
    - BM25 keyword search for exact matching
    - Semantic search for contextual understanding
    - Hybrid search with Reciprocal Rank Fusion
    - Configurable weights for combining results
    """

    def __init__(
        self,
        milvus_store,
        search_type: Literal["bm25", "semantic", "hybrid"] = "hybrid",
        bm25_weight: float = 0.5,
        semantic_weight: float = 0.5,
        rrf_k: int = 60,
    ):
        """
        Initialize HybridSearchRetriever.

        Args:
            milvus_store: MilvusStore instance
            search_type: Type of search to perform ("bm25", "semantic", or "hybrid")
            bm25_weight: Weight for BM25 scores in hybrid search (0-1)
            semantic_weight: Weight for semantic scores in hybrid search (0-1)
            rrf_k: Constant for Reciprocal Rank Fusion (default: 60)
        """
        self.milvus_store = milvus_store
        self.search_type = search_type
        self.bm25_weight = bm25_weight
        self.semantic_weight = semantic_weight
        self.rrf_k = rrf_k

        # Initialize BM25 index
        self.bm25_index = None
        self.corpus_texts = []
        self.corpus_metadata = []

        # Build BM25 index if needed
        if search_type in ["bm25", "hybrid"]:
            self._build_bm25_index()

    def _build_bm25_index(self) -> None:
        """Build BM25 index from Milvus collection."""
        print("Building BM25 index...")

        try:
            collection = Collection(name=self.milvus_store.collection_name)

            # Query all documents from collection
            collection.load()
            results = collection.query(
                expr="id >= 0",
                output_fields=["text", "source", "page_no"],
                limit=16384,
            )

            if not results:
                print("Warning: No documents found in collection for BM25 indexing")
                return

            # Store corpus
            self.corpus_texts = [doc["text"] for doc in results]
            self.corpus_metadata = [
                {"source": doc["source"], "page_no": doc["page_no"]} for doc in results
            ]

            # Tokenize and build BM25 index
            tokenized_corpus = [self._tokenize(text) for text in self.corpus_texts]
            self.bm25_index = BM25Okapi(tokenized_corpus)

            print(f"BM25 index built with {len(self.corpus_texts)} documents")

        except Exception as e:
            print(f"Error building BM25 index: {e}")

    def _tokenize(self, text: str) -> List[str]:
        """
        Simple tokenization for BM25.

        Args:
            text: Input text to tokenize

        Returns:
            List of tokens
        """
        return text.lower().split()

    def search_bm25(self, query: str, top_k: int = 10) -> List[SearchResult]:
        """
        Perform BM25 keyword search.

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            List of SearchResult objects
        """
        if self.bm25_index is None:
            print("BM25 index not initialized")
            return []

        # Tokenize query
        tokenized_query = self._tokenize(query)

        # Get BM25 scores
        scores = self.bm25_index.get_scores(tokenized_query)

        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:top_k]

        # Build results
        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append(
                    SearchResult(
                        text=self.corpus_texts[idx],
                        source=self.corpus_metadata[idx]["source"],
                        page_no=self.corpus_metadata[idx]["page_no"],
                        score=float(scores[idx]),
                        search_method="bm25",
                    )
                )

        return results

    def search_semantic(self, query: str, top_k: int = 10) -> List[SearchResult]:
        """
        Perform semantic vector search using Milvus.

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            List of SearchResult objects
        """
        from pymilvus import Collection

        query_vec = self.milvus_store.embeddings_model.embed_query(query)
        search_params = {"metric_type": "L2", "params": {}}
        collection = Collection(name=self.milvus_store.collection_name)

        results = collection.search(
            data=[query_vec],
            anns_field="vector",
            param=search_params,
            limit=top_k,
            output_fields=["text", "source", "page_no"],
        )

        # Convert to SearchResult objects
        search_results = []
        for result in results[0]:
            # Higher rank = higher score (inverse distance for L2)
            score = 1.0 / (result.get("distance") + 1)

            search_results.append(
                SearchResult(
                    text=result["entity"]["text"],
                    source=result["entity"]["source"],
                    page_no=result["entity"]["page_no"],
                    score=score,
                    search_method="semantic",
                )
            )

        return search_results

    def search_hybrid(self, query: str, top_k: int = 10) -> List[SearchResult]:
        """
        Perform hybrid search using Reciprocal Rank Fusion (RRF).

        RRF combines rankings from multiple search methods:
        RRF_score = sum(1 / (k + rank_i)) for each ranking i

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            List of SearchResult objects
        """
        bm25_results = self.search_bm25(query, top_k=top_k * 2)
        semantic_results = self.search_semantic(query, top_k=top_k * 2)

        # Create a dictionary to store combined scores
        combined_scores = {}

        # RRF(d) = Σ ( weight / (k + rank_d + 1) )
        # Add BM25 scores with RRF
        for rank, result in enumerate(bm25_results):
            key = (result.text, result.source, result.page_no)
            rrf_score = self.bm25_weight / (self.rrf_k + rank + 1)
            combined_scores[key] = combined_scores.get(key, 0) + rrf_score

        # Add semantic scores with RRF
        for rank, result in enumerate(semantic_results):
            key = (result.text, result.source, result.page_no)
            rrf_score = self.semantic_weight / (self.rrf_k + rank + 1)
            combined_scores[key] = combined_scores.get(key, 0) + rrf_score

        # Sort by combined score
        sorted_results = sorted(
            combined_scores.items(), key=lambda x: x[1], reverse=True
        )[:top_k]

        # Build final results
        results = []
        for (text, source, page_no), score in sorted_results:
            results.append(
                SearchResult(
                    text=text,
                    source=source,
                    page_no=page_no,
                    score=score,
                    search_method="hybrid",
                )
            )

        return results

    def search(self, query: str, top_k: int = 10) -> List[SearchResult]:
        """
        Perform search based on configured search_type.

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            List of SearchResult objects
        """
        if self.search_type == "bm25":
            return self.search_bm25(query, top_k)
        elif self.search_type == "semantic":
            return self.search_semantic(query, top_k)
        elif self.search_type == "hybrid":
            return self.search_hybrid(query, top_k)
        else:
            raise ValueError(f"Unknown search type: {self.search_type}")

    def refresh_bm25_index(self) -> None:
        """Refresh BM25 index with latest data from Milvus."""
        if self.search_type in ["bm25", "hybrid"]:
            self._build_bm25_index()


class RerankerMixin:
    """
    Mixin class for reranking search results using cross-encoder models.

    Cross-encoders provide better relevance scoring by jointly encoding
    query and document pairs, but are slower than bi-encoders.
    """

    def __init__(self, reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        Initialize reranker.

        Args:
            reranker_model: HuggingFace cross-encoder model name
        """
        try:
            from sentence_transformers import CrossEncoder

            self.reranker = CrossEncoder(reranker_model)
            self.has_reranker = True
            print(f"Initialized reranker: {reranker_model}")
        except ImportError:
            print("Warning: sentence-transformers not installed. Reranking disabled.")
            self.has_reranker = False

    def rerank(
        self, query: str, results: List[SearchResult], top_k: int = None
    ) -> List[SearchResult]:
        """
        Rerank search results using cross-encoder.

        Args:
            query: Original search query
            results: Initial search results
            top_k: Number of top results to return after reranking

        Returns:
            Reranked list of SearchResult objects
        """
        if not self.has_reranker or not results:
            return results

        # Prepare query-document pairs
        pairs = [(query, result.text) for result in results]

        # Get reranking scores
        scores = self.reranker.predict(pairs)

        # Update results with new scores
        reranked_results = []
        for result, score in zip(results, scores):
            reranked_results.append(
                SearchResult(
                    text=result.text,
                    source=result.source,
                    page_no=result.page_no,
                    score=float(score),
                    search_method=f"{result.search_method}_reranked",
                )
            )

        # Sort by new scores
        reranked_results.sort(key=lambda x: x.score, reverse=True)

        # Return top-k
        if top_k is not None:
            return reranked_results[:top_k]
        return reranked_results


class HybridSearchWithReranker(HybridSearchRetriever, RerankerMixin):
    """
    Hybrid search retriever with reranking capability.

    Combines BM25, semantic search, and cross-encoder reranking
    for optimal retrieval performance.

    Usage:
        ```python
        retriever = HybridSearchWithReranker(
            milvus_store=store,
            search_type="hybrid",
            reranker_model="cross-encoder/ms-marco-MiniLM-L-6-v2"
        )

        # Search and rerank
        results = retriever.search_and_rerank(
            query="What is RAG?",
            initial_k=20,  # Retrieve more candidates
            top_k=5  # Return top 5 after reranking
        )
        ```
    """

    def __init__(
        self,
        milvus_store,
        search_type: Literal["bm25", "semantic", "hybrid"] = "hybrid",
        bm25_weight: float = 0.5,
        semantic_weight: float = 0.5,
        rrf_k: int = 60,
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        enable_reranker: bool = True,
    ):
        """
        Initialize HybridSearchWithReranker.

        Args:
            milvus_store: MilvusStore instance
            search_type: Type of search ("bm25", "semantic", or "hybrid")
            bm25_weight: Weight for BM25 in hybrid search
            semantic_weight: Weight for semantic search in hybrid search
            rrf_k: Constant for Reciprocal Rank Fusion
            reranker_model: Cross-encoder model for reranking
            enable_reranker: Whether to enable reranking
        """
        HybridSearchRetriever.__init__(
            self,
            milvus_store=milvus_store,
            search_type=search_type,
            bm25_weight=bm25_weight,
            semantic_weight=semantic_weight,
            rrf_k=rrf_k,
        )

        if enable_reranker:
            RerankerMixin.__init__(self, reranker_model=reranker_model)
        else:
            self.has_reranker = False

    def search_and_rerank(
        self, query: str, initial_k: int = 20, top_k: int = 5
    ) -> List[SearchResult]:
        """
        Perform search followed by reranking.

        This is the recommended method for best retrieval quality.
        It retrieves more candidates initially and then reranks them
        for better precision.

        Args:
            query: Search query
            initial_k: Number of candidates to retrieve initially
            top_k: Number of results to return after reranking

        Returns:
            Reranked list of SearchResult objects
        """
        # Get initial results
        results = self.search(query, top_k=initial_k)

        # Rerank if enabled
        if self.has_reranker:
            results = self.rerank(query, results, top_k=top_k)
        else:
            results = results[:top_k]

        return results


# Utility function for easy usage
def create_hybrid_retriever(
    milvus_store, search_type: str = "hybrid", enable_reranker: bool = False, **kwargs
) -> HybridSearchRetriever:
    """
    Factory function to create appropriate retriever.

    Args:
        milvus_store: MilvusStore instance
        search_type: Type of search ("bm25", "semantic", or "hybrid")
        enable_reranker: Whether to enable reranking
        **kwargs: Additional arguments for retriever configuration

    Returns:
        HybridSearchRetriever or HybridSearchWithReranker instance
    """
    if enable_reranker:
        return HybridSearchWithReranker(
            milvus_store=milvus_store,
            search_type=search_type,
            enable_reranker=enable_reranker,
            **kwargs,
        )
    else:
        return HybridSearchRetriever(
            milvus_store=milvus_store, search_type=search_type, **kwargs
        )
