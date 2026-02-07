from pathlib import Path
from typing import List, Optional, Union

# Import configuration
from src.config import config, ConfigLoader

# Import document processing libraries
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from docling_core.transforms.chunker.hierarchical_chunker import (
    ChunkingDocSerializer,
    ChunkingSerializerProvider,
)
from docling_core.transforms.serializer.markdown import (
    MarkdownTableSerializer,
    MarkdownParams,
)
from docling_core.types.doc import ImageRefMode
from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.chunking import HybridChunker
from docling.datamodel.pipeline_options import PdfPipelineOptions
from langchain_docling import DoclingLoader
from langchain_docling.loader import ExportType
from langchain_core.documents import Document
from transformers import AutoTokenizer
from docling.document_converter import DocumentConverter

# Import our Milvus store module
from src.milvus_store import MilvusStore


def get_document_converter(
    pdf_pipeline_options: Optional[PdfPipelineOptions] = None,
) -> DocumentConverter:
    """
    Create and configure a document converter.

    Returns:
        DocumentConverter: Configured document converter
    """
    pdf_pipeline_options = pdf_pipeline_options or config.get_pdf_pipeline_options()

    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_pipeline_options)
        }
    )


def get_chunker(
    tokenizer_model_id: Optional[str] = None,
    max_tokens: Optional[int] = None,
    image_mode: Optional[ImageRefMode] = None,
    image_placeholder: Optional[str] = None,
    mark_annotations: Optional[bool] = None,
    include_annotations: Optional[bool] = None,
    config: Optional["ConfigLoader"] = None,
) -> HybridChunker:
    """
    Create and configure a document chunker with optional configuration overrides.

    Args:
        tokenizer_model_id: Model ID for tokenizer (defaults to tokenizer from config)
        max_tokens: Maximum tokens per chunk (defaults to config.get("document", "max_tokens"))
        image_mode: Image reference mode (defaults to ImageRefMode.PLACEHOLDER)
        image_placeholder: Placeholder text for images (defaults to "")
        mark_annotations: Whether to mark annotations (defaults to True)
        include_annotations: Whether to include annotations (defaults to True)
        config: Optional ConfigLoader instance to use instead of global Config

    Returns:
        HybridChunker: Configured document chunker
    """
    # Use provided values or fall back to config defaults
    config_to_use = config if config else globals()["config"]
    default_tokenizer_model_id = config_to_use.get(
        "model", "tokenizer", default="sentence-transformers/all-MiniLM-L6-v2"
    )
    default_max_tokens = config_to_use.get("document", "max_tokens", default=512)

    model_id = tokenizer_model_id or default_tokenizer_model_id
    tokens = max_tokens or default_max_tokens
    img_mode = image_mode if image_mode is not None else ImageRefMode.PLACEHOLDER
    img_placeholder = image_placeholder if image_placeholder is not None else ""
    mark_annot = mark_annotations if mark_annotations is not None else True
    include_annot = include_annotations if include_annotations is not None else True

    tokenizer = HuggingFaceTokenizer(
        tokenizer=AutoTokenizer.from_pretrained(model_id),
        max_tokens=tokens,
    )

    class CustomMDSerializerProvider(ChunkingSerializerProvider):
        def get_serializer(self, doc):
            return ChunkingDocSerializer(
                doc=doc,
                table_serializer=MarkdownTableSerializer(),
                params=MarkdownParams(
                    image_mode=img_mode,
                    image_placeholder=img_placeholder,
                    mark_annotations=mark_annot,
                    include_annotations=include_annot,
                ),
            )

    return HybridChunker(
        tokenizer=tokenizer,
        serializer_provider=CustomMDSerializerProvider(),
    )


def process_file(
    file_path: Union[str, Path],
    converter: DocumentConverter,
    chunker: HybridChunker,
    namespace: str,
    output_dir: Union[str, Path] = "outputs",
) -> List[Document]:
    """
    Process a single file and prepare documents for indexing.

    Args:
        file_path: Path to the file to process
        converter: Document converter to use
        chunker: Document chunker to use
        namespace: Namespace to use for the documents
        output_dir: Root folder for saving Markdown + images

    Returns:
        List[Document]: List of processed documents
    """
    file_path = Path(file_path) if isinstance(file_path, str) else file_path

    # Create document loader
    loader = DoclingLoader(
        file_path=file_path,
        converter=converter,
        chunker=chunker,
        export_type=ExportType.DOC_CHUNKS,
    )

    # Load and process documents
    docs = loader.load()

    # Prepare documents for indexing
    processed_docs = []
    for doc in docs:
        metadata = doc.metadata
        _metadata = dict()
        _metadata["source"] = str(metadata["source"])
        _metadata["page_no"] = metadata["dl_meta"]["doc_items"][0]["prov"][0]["page_no"]
        _metadata["namespace"] = namespace

        processed_doc = Document(page_content=doc.page_content, metadata=_metadata)
        processed_docs.append(processed_doc)

    return processed_docs


def process_and_index_directory(
    directory_path: Union[str, Path],
    drop_existing: bool = False,
    namespace: str = None,
    file_extensions: List[str] = None,
    config: Optional["ConfigLoader"] = None,
) -> None:
    """
    Process all files in a directory and index them to a vector store.

    Args:
        directory_path: Path to the directory containing files to process
        drop_existing: Whether to drop the existing collection if it exists
        namespace: Namespace to use for the documents (defaults to config.get("database", "namespace"))
        file_extensions: List of file extensions to process (defaults to config.get("document", "supported_file_types"))
        config: Optional ConfigLoader instance to use instead of global Config
    """
    directory_path = (
        Path(directory_path) if isinstance(directory_path, str) else directory_path
    )

    # Use provided config or fall back to global config
    config_to_use = config if config else globals()["config"]
    namespace = namespace or config_to_use.get(
        "database", "namespace", default="ragMultimodal"
    )
    file_extensions = file_extensions or config_to_use.get(
        "document", "supported_file_types", default=[".pdf"]
    )

    if not directory_path.exists():
        raise FileNotFoundError(f"Directory {directory_path} does not exist")

    # Get all files in the directory with specified extensions
    files = []
    for ext in file_extensions:
        files.extend(directory_path.glob(f"*{ext}"))

    if not files:
        print(f"No files with extensions {file_extensions} found in {directory_path}")
        return

    print(f"Found {len(files)} files to process")

    # Create document converter and chunker
    pdf_pipeline_options = config.get_pdf_pipeline_options()
    converter = get_document_converter(pdf_pipeline_options=pdf_pipeline_options)
    chunker = get_chunker(config=config)

    # Extract config values for MilvusStore
    uri = config_to_use.get("database", "uri", default="http://localhost:19530")
    db_name = config_to_use.get("database", "name", default="ragMultimodal")
    collection_name = config_to_use.get(
        "database", "collection_name", default="collectionDemo"
    )
    embed_model = config_to_use.get(
        "model", "embeddings", default="sentence-transformers/all-mpnet-base-v2"
    )

    # Create vector store with explicit parameters
    milvus_store = MilvusStore(
        uri=uri,
        db_name=db_name,
        collection_name=collection_name,
        embed_model=embed_model,
        drop_old=drop_existing,
        namespace=namespace,
    )

    # Process and index each file
    all_docs = []
    for file in files:
        if file.name == ".DS_Store":
            continue

        try:
            docs = process_file(file, converter, chunker, namespace)
            all_docs.extend(docs)
        except Exception as e:
            print(f"Error processing {file}: {e}")

    # Index all documents
    if all_docs:
        print(f"Indexing {len(all_docs)} documents...")
        ids = milvus_store.add_documents(documents=all_docs)
        if not ids:
            print("Failed to index documents")
    else:
        print("No documents to index")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Process and index files in a directory"
    )
    parser.add_argument("directory", help="Directory containing files to process")
    parser.add_argument(
        "--drop", action="store_true", help="Drop existing collection if it exists"
    )
    parser.add_argument("--namespace", help="Namespace to use for the documents")

    args = parser.parse_args()

    process_and_index_directory(
        directory_path=args.directory, drop_existing=args.drop, namespace=args.namespace
    )
