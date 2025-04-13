"""Tool for performing local RAG search against the arXiv Milvus collection."""

import time
from pymilvus import Collection, connections, utility, DataType
from sentence_transformers import SentenceTransformer

# Import relevant configs/functions from the vector_store module
from einstein.vector_store import (
    connect_to_milvus,
    get_embedding_model,
    COLLECTION_NAME,
    MODEL_NAME,
    VECTOR_FIELD,
    EMBEDDING_DIM
)

# --- Search Configuration ---
DEFAULT_TOP_K = 5 # Default number of results to return
DEFAULT_OUTPUT_FIELDS = ["id", "title", "abstract", "authors", "categories", "doi"] # Fields to retrieve
# Search parameters (adjust ef based on desired speed/accuracy trade-off)
# ef should be >= top_k
DEFAULT_SEARCH_PARAMS = {"metric_type": "L2", "params": {"ef": 64}}
# --- End Search Configuration ---

# Global variables to hold the model and collection connection
# This avoids reloading the model or reconnecting for every search call
_model = None
_collection = None
_connected = False

def _initialize_search_components(force_reload: bool = False):
    """Initializes Milvus connection and loads embedding model if not already done."""
    global _model, _collection, _connected
    
    # Connect to Milvus if not connected
    if not _connected or force_reload:
        try:
            # Ensure any previous connection is cleaned up if forcing reload
            if _connected:
                connections.disconnect("default")
                _connected = False
                _collection = None
            
            connect_to_milvus() # Uses defaults from indexer config
            _connected = True
            
            # Check if collection exists
            if not utility.has_collection(COLLECTION_NAME):
                print(f"Error: Collection '{COLLECTION_NAME}' does not exist in Milvus.")
                print("Please run the indexer script first.")
                _connected = False # Mark as not usable
                return False
                
            _collection = Collection(COLLECTION_NAME)
            # Load collection into memory for searching
            # Check loading status first
            if utility.loading_progress(COLLECTION_NAME).get('loading_progress', '0%') != '100%':
                print(f"Loading collection '{COLLECTION_NAME}' into memory for search...")
                _collection.load()
                utility.wait_for_loading_complete(COLLECTION_NAME)
                print("Collection loaded.")
            else:
                print(f"Collection '{COLLECTION_NAME}' already loaded.")
                
        except Exception as e:
            print(f"Error initializing Milvus connection or loading collection: {e}")
            _connected = False
            _collection = None
            return False
            
    # Load embedding model if not loaded
    if _model is None or force_reload:
        try:
            _model = get_embedding_model(model_name=MODEL_NAME) # Uses defaults
            if _model is None:
                 print("Error: Failed to load the embedding model.")
                 return False
        except Exception as e:
            print(f"Error loading embedding model: {e}")
            _model = None
            return False
            
    return _connected and _model is not None and _collection is not None

def local_arxiv_search(query: str, top_k: int = DEFAULT_TOP_K) -> list[dict]:
    """
    Performs a similarity search for relevant arXiv papers in the local Milvus DB.

    Args:
        query: The search query string (e.g., a question, topic, paper abstract).
        top_k: The maximum number of results to return.

    Returns:
        A list of dictionaries, where each dictionary represents a relevant paper
        and contains the fields specified in DEFAULT_OUTPUT_FIELDS, plus a 'distance' score.
        Returns an empty list if the search fails or no results are found.
    """
    print(f"\n--- Received Local arXiv Search Query ---")
    print(f"Query: '{query[:100]}...'")
    print(f"Top K: {top_k}")
    
    start_time = time.time()
    
    # 1. Initialize components (model, Milvus connection, collection)
    if not _initialize_search_components():
        print("Search initialization failed. Cannot proceed.")
        return []
        
    # Ensure _model and _collection are not None after initialization check
    if _model is None or _collection is None:
         print("Search components (_model or _collection) are None after initialization. Cannot proceed.")
         return []
         
    # 2. Embed the query
    print("Embedding the query...")
    try:
        query_embedding = _model.encode([query]) # Pass query as a list
        # query_embedding = [query_embedding[0].tolist()] # Ensure it's list of lists for Milvus
    except Exception as e:
        print(f"Error embedding query: {e}")
        return []
        
    # 3. Perform the search in Milvus
    print(f"Searching collection '{COLLECTION_NAME}'...")
    search_results = []
    try:
        results = _collection.search(
            data=query_embedding, 
            anns_field=VECTOR_FIELD, 
            param=DEFAULT_SEARCH_PARAMS, 
            limit=top_k, 
            output_fields=DEFAULT_OUTPUT_FIELDS
        )
        
        # Process results
        # Results is a list (one per query vector), get the first element's hits
        hits = results[0]
        print(f"Found {len(hits)} potential results.")
        for hit in hits:
            result_dict = {
                "id": hit.id,
                "distance": hit.distance # Lower distance means more similar for L2
            }
            # Add other requested fields from the entity
            for field in DEFAULT_OUTPUT_FIELDS:
                if field != "id": # ID is already included
                    try:
                         result_dict[field] = hit.entity.get(field)
                    except Exception as e:
                         print(f"Warning: Could not retrieve field '{field}' for hit {hit.id}: {e}")
                         result_dict[field] = None
            search_results.append(result_dict)
            
    except Exception as e:
        print(f"Error during Milvus search: {e}")
        return [] # Return empty list on search error
        
    end_time = time.time()
    print(f"Search completed in {end_time - start_time:.2f} seconds.")
    
    # Sort results by distance (ascending for L2)
    search_results.sort(key=lambda x: x['distance'])
    
    print(f"Returning {len(search_results)} results.")
    return search_results

# --- Example Usage --- 
if __name__ == '__main__':
    print("--- Testing Local arXiv Search Tool ---")
    
    # Ensure Milvus is running and indexer has been run!
    
    test_query_1 = "Contrastive learning for sentence embeddings" # Example query
    test_query_2 = "Using large language models for code generation" 
    test_query_3 = "Applications of graph neural networks in drug discovery"
    
    queries = [test_query_1, test_query_2, test_query_3]
    
    for i, query in enumerate(queries):
        print(f"\n--- Running Test Query {i+1} --- ")
        results = local_arxiv_search(query, top_k=3)
        
        if results:
            print("\nSearch Results:")
            for idx, res in enumerate(results):
                print(f"  Result {idx + 1}:")
                print(f"    ID: {res.get('id')}")
                print(f"    Distance: {res.get('distance'):.4f}")
                print(f"    Title: {res.get('title', '').strip()[:100]}...")
                # print(f"    Abstract: {res.get('abstract', '').strip()[:150]}...")
                print(f"    Categories: {res.get('categories')}")
        else:
            print("\nNo results found or search failed.")
            
    # Clean up connection (optional, depends if script exits)
    if _connected:
        print("\nDisconnecting from Milvus after tests...")
        connections.disconnect("default")
        _connected = False
        _collection = None
        _model = None # Clear model reference
        
    print("\n--- Search Tool Test Finished ---") 