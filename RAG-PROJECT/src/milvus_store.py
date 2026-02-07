# Import configuration
from src.config import config

# Import vector store libraries
from langchain_huggingface import HuggingFaceEmbeddings
from pymilvus import Collection, MilvusException, connections, db, utility
from pymilvus import (
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
    utility,
)


def drop_collection(collection_name: str, db_name: str) -> bool:
    """
    Drop a collection from the specified database.

    Args:
        collection_name: Name of the collection to drop
        db_name: Name of the database containing the collection

    Returns:
        bool: True if collection was successfully dropped, False otherwise
    """
    try:
        # Check if the database exists
        existing_databases = db.list_database()
        if db_name not in existing_databases:
            print(f"Database '{db_name}' does not exist.")
            return False

        # Switch to the specified database
        db.using_database(db_name)

        # Check if the collection exists
        collections = utility.list_collections()
        if collection_name not in collections:
            print(
                f"Collection '{collection_name}' does not exist in database '{db_name}'."
            )
            return False

        # Drop the collection
        collection = Collection(name=collection_name)
        collection.drop()
        print(
            f"Collection '{collection_name}' has been dropped from database '{db_name}'."
        )
        return True

    except MilvusException as e:
        print(f"Error dropping collection: {e}")
        return False


def drop_all_collections(db_name: str, confirm: bool = False) -> bool:
    """
    Drop all collections in a database.

    Args:
        db_name: Name of the database containing the collections
        confirm: Set to True to confirm the operation (defaults to False)

    Returns:
        bool: True if all collections were successfully dropped, False otherwise
    """
    if not confirm:
        print(f"WARNING: You are about to drop all collections in database '{db_name}'")
        print("This operation is irreversible. Set confirm=True to proceed.")
        return False

    try:
        # Check if the database exists
        existing_databases = db.list_database()
        if db_name not in existing_databases:
            print(f"Database '{db_name}' does not exist.")
            return False

        # Switch to the specified database
        db.using_database(db_name)

        # Get all collections in the database
        collections = utility.list_collections()
        print(f"Found {len(collections)} collections in database '{db_name}'")

        # Drop each collection
        for collection_name in collections:
            print(f"Dropping collection '{collection_name}'...")
            success = drop_collection(collection_name=collection_name, db_name=db_name)
            if not success:
                print(f"Failed to drop collection '{collection_name}'")
                return False
            print(f"Successfully dropped collection '{collection_name}'")

        return True

    except MilvusException as e:
        print(f"Error dropping collections: {e}")
        return False


def drop_database(db_name: str, confirm: bool = False) -> bool:
    """
    Drop a database and all its collections.

    Args:
        db_name: Name of the database to drop
        confirm: Set to True to confirm the operation (defaults to False)

    Returns:
        bool: True if database was successfully dropped, False otherwise
    """
    if not confirm:
        print(f"WARNING: You are about to drop database '{db_name}'")
        print("This operation is irreversible. Set confirm=True to proceed.")
        return False

    try:
        # Check if the database exists
        existing_databases = db.list_database()
        if db_name not in existing_databases:
            print(f"Database '{db_name}' does not exist.")
            return False

        # First drop all collections in the database
        if not drop_all_collections(db_name, confirm=True):
            print(f"Failed to drop all collections in database '{db_name}'")
            return False

        # Now drop the database
        db.drop_database(db_name)
        print(f"Database '{db_name}' has been dropped.")
        return True

    except MilvusException as e:
        print(f"Error dropping database: {e}")
        return False


class MilvusStore:
    """
    A class to manage interactions with Milvus vector database.

    This class provides functionality to interact with Milvus vector database for storing and retrieving embeddings.
    It includes comprehensive printging capabilities that can be configured using the set_print_level method.

    printging Features:
    - Console printging (default at INFO level)
    - File printging (default to 'prints/milvus_store.print')
    - Configurable print levels (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    - Formatted print messages with timestamps

    Example:
        ```python
        # Set custom debug level with custom file path
        MilvusStore.set_print_level(printging.DEBUG, 'custom/path/milvus.print')

        # Create store with default printging (INFO level)
        store = MilvusStore()
        ```
    """

    def __init__(
        self,
        uri: str = None,
        db_name: str = None,
        collection_name: str = None,
        embed_model: str = None,
        drop_old: bool = False,
        namespace: str = None,
    ):
        """
        Initialize the MilvusStore.

        Args:
            uri: URI for Milvus connection (defaults to config.get("database", "uri"))
            db_name: Name of the database (defaults to config.get("database", "name"))
            collection_name: Name of the collection (defaults to config.get("database", "collection_name"))
            embed_model: Embedding model to use (defaults to config.get("model", "embeddings"))
            drop_old: Whether to drop the existing collection if it exists
            namespace: Default namespace to use for documents (defaults to config.get("database", "namespace"))
        """
        self.uri = uri or config.get(
            "database", "uri", default="http://localhost:19530"
        )
        self.db_name = db_name or config.get(
            "database", "name", default="rag_multimodal"
        )
        self.collection_name = collection_name or config.get(
            "database", "collection_name", default="collection_demo"
        )
        self.embed_model = embed_model or config.get(
            "model", "embeddings", default="nomic-embed-text"
        )
        self.namespace = namespace or config.get(
            "database", "namespace", default="RAGDemo"
        )

        # Connect to Milvus
        self._connect_to_milvus()

        # Initialize the database
        self._initialize_vector_store(drop_old=drop_old)

        # Create embeddings model
        self.embeddings_model = HuggingFaceEmbeddings(model_name=self.embed_model)

        # create collection
        self._create_collection(self.collection_name)

    def _connect_to_milvus(self) -> None:
        """
        Connect to Milvus server.
        """
        host = self.uri.split("://")[1].split(":")[0]
        port = int(self.uri.split(":")[-1])
        connections.connect(host=host, port=port)

    def _initialize_vector_store(self, drop_old: bool = False) -> None:
        """
        Initialize the vector store database.

        Args:
            drop_old: Whether to drop the existing database if it exists
        """
        try:
            existing_databases = db.list_database()
            if self.db_name in existing_databases:
                print(f"Database '{self.db_name}' already exists.")

                # Use the database context
                db.using_database(self.db_name)

                if drop_old:
                    # Drop the collection if it exists
                    collections = utility.list_collections()
                    if self.collection_name in collections:
                        drop_collection(self.collection_name, self.db_name)
            else:
                print(f"Database '{self.db_name}' does not exist.")
                db.create_database(self.db_name)
                print(f"Database '{self.db_name}' created successfully.")

                # Use the database context
                db.using_database(self.db_name)
        except MilvusException as e:
            print(f"An error occurred: {e}")

    def _create_collection(self, collection_name: str = None) -> None:
        """
        Create Milvus collection with schema and index if it does not exist.
        """
        try:
            name = collection_name or self.collection_name

            if utility.has_collection(name):
                print(f"Collection '{name}' already exists.")

                # Load collection
                collection = Collection(name=name)
                collection.load()
                print("Collection loaded successfully.")

                return

            # Determine embedding dimension dynamically
            test_vector = self.embeddings_model.embed_query("dimension_probe")
            dim = len(test_vector)

            # Define tokenizer parameters for text analysis
            analyzer_params = {"tokenizer": "standard", "filter": ["lowercase"]}

            # Define schema
            fields = [
                FieldSchema(
                    name="id",
                    dtype=DataType.INT64,
                    is_primary=True,
                    auto_id=True,
                ),
                FieldSchema(
                    name="vector",
                    dtype=DataType.FLOAT_VECTOR,
                    dim=dim,
                ),
                FieldSchema(
                    name="text",
                    dtype=DataType.VARCHAR,
                    max_length=65535,
                    analyzer_params=analyzer_params,
                ),
                FieldSchema(
                    name="source",
                    dtype=DataType.VARCHAR,
                    max_length=512,
                ),
                FieldSchema(
                    name="page_no",
                    dtype=DataType.INT64,
                ),
                FieldSchema(
                    name="namespace",
                    dtype=DataType.VARCHAR,
                    max_length=128,
                ),
            ]

            schema = CollectionSchema(
                fields=fields,
                description="RAG multimodal document chunks",
            )

            # Create collection
            collection = Collection(
                name=name,
                schema=schema,
            )

            print(f"Collection '{name}' created successfully.")

            # Create index
            index_params = {
                "index_type": "HNSW",
                "metric_type": "L2",
                "params": {
                    "M": 8,
                    "efConstruction": 64,
                },
            }

            collection.create_index(
                field_name="vector",
                index_params=index_params,
            )

            print("Index created successfully.")

            # Load collection
            collection.load()
            print("Collection loaded successfully.")

        except MilvusException as e:
            print(f"Failed to create collection '{name}': {e}")

    def retriever(
        self,
        query: str,
        top_k: int = 3,
        search_type: str = "hybrid",
        bm25_weight: float = 0.5,
        semantic_weight: float = 0.5,
        enable_reranker: bool = False,
        initial_k: int = None,
    ) -> list[dict[str, str]]:
        """
        Retrieve documents using hybrid search (BM25 + Semantic + optional Reranking).

        Args:
            query: Search query
            collection_name: Name of collection to search (defaults to self.collection_name)
            top_k: Number of results to return
            search_type: Type of search - "bm25", "semantic", or "hybrid" (default: "hybrid")
            bm25_weight: Weight for BM25 scores in hybrid search (0-1, default: 0.5)
            semantic_weight: Weight for semantic scores in hybrid search (0-1, default: 0.5)
            enable_reranker: Whether to use cross-encoder reranking (default: False)
            initial_k: Number of candidates to retrieve before reranking (default: top_k * 4)

        Returns:
            List of document dictionaries with keys: context, source, page_no
        """
        from src.hybrid_search import create_hybrid_retriever

        # Create hybrid retriever
        hybrid_retriever = create_hybrid_retriever(
            milvus_store=self,
            search_type=search_type,
            enable_reranker=enable_reranker,
            bm25_weight=bm25_weight,
            semantic_weight=semantic_weight,
        )

        # Perform search
        if enable_reranker and hasattr(hybrid_retriever, "search_and_rerank"):
            if initial_k is None:
                initial_k = top_k * 4  # Retrieve more candidates for reranking
            results = hybrid_retriever.search_and_rerank(
                query=query, initial_k=initial_k, top_k=top_k
            )
        else:
            results = hybrid_retriever.search(query=query, top_k=top_k)

        # Convert SearchResult objects to dictionary format for backward compatibility
        return [
            {
                "context": result.text,
                "source": result.source,
                "page_no": result.page_no,
            }
            for result in results
        ]

    def add_documents(self, documents: list[dict]) -> list[int]:
        """
        Add processed documents to Milvus.

        Args:
            documents: List of processed document dicts

        Returns:
            List[int]: Inserted primary keys
        """
        if not documents:
            print("No documents provided for insertion.")
            return []

        try:
            collection = Collection(name=self.collection_name)

            texts = [doc.page_content for doc in documents]
            sources = [doc.metadata.get("source") for doc in documents]
            page_nos = [doc.metadata.get("page_no", -1) for doc in documents]
            namespaces = [
                doc.metadata.get("namespace", self.namespace) for doc in documents
            ]

            # Generate embeddings
            embeddings = self.embeddings_model.embed_documents(texts)

            # Prepare data in schema order (excluding auto_id)
            data = [
                embeddings,  # vector
                texts,  # text
                sources,  # source
                page_nos,  # page_no
                namespaces,  # namespace
            ]

            insert_result = collection.insert(data)
            collection.flush()

            print(f"Inserted {len(insert_result.primary_keys)} documents into Milvus.")

            return insert_result.primary_keys

        except MilvusException as e:
            print(f"Error inserting documents into Milvus: {e}")
            return []

    def drop_collection(self, collection_name: str = None, db_name: str = None) -> bool:
        """
        Drop a collection from the specified database.

        Args:
            collection_name: Name of the collection to drop (defaults to self.collection_name)
            db_name: Name of the database containing the collection (defaults to self.db_name)

        Returns:
            bool: True if collection was successfully dropped, False otherwise
        """
        collection_name = collection_name or self.collection_name
        db_name = db_name or self.db_name

        return drop_collection(collection_name, db_name)

    def drop_all_collections(self, db_name: str = None, confirm: bool = False) -> bool:
        """
        Drop all collections in a database.

        Args:
            db_name: Name of the database containing the collections (defaults to self.db_name)
            confirm: Set to True to confirm the operation (defaults to False)

        Returns:
            bool: True if all collections were successfully dropped, False otherwise
        """
        db_name = db_name or self.db_name

        return drop_all_collections(db_name, confirm)

    def drop_database(self, db_name: str = None, confirm: bool = False) -> bool:
        """
        Drop a database and all its collections.

        Args:
            db_name: Name of the database to drop (defaults to self.db_name)
            confirm: Set to True to confirm the operation (defaults to False)

        Returns:
            bool: True if database was successfully dropped, False otherwise
        """
        db_name = db_name or self.db_name

        return drop_database(db_name, confirm)


if __name__ == "__main__":
    ms = MilvusStore()
    print(ms)
