"""Initialize the vector_store module."""

from .arxiv_loader import load_and_preprocess_data
from .embedder import generate_embeddings, get_embedding_model, EMBEDDING_DIM, MODEL_NAME
from .indexer import connect_to_milvus, create_collection_if_not_exists, create_index, COLLECTION_NAME, VECTOR_FIELD 