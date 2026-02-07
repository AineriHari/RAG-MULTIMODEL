import sys
from pathlib import Path

BASE_DIR = Path(__file__).parent
sys.path.append(str(BASE_DIR))

DOC_DIR = BASE_DIR / "input_files"

import docling

docling.allow_external_plugins = True
docling.enable_remote_services = True


# Indexing
from src.index import process_and_index_directory, get_chunker, get_document_converter
from src.config import ConfigLoader

config = ConfigLoader(str(BASE_DIR) / "config.yaml")
chunker = get_chunker(config=config)
pdf_pipeline_options = config.get_pdf_pipeline_options()
converter = get_document_converter(pdf_pipeline_options)

# processing documents
process_and_index_directory(DOC_DIR, drop_existing=True, config=config)
