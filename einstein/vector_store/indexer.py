"""Module for indexing arXiv paper embeddings into Milvus (using Pandas)."""

import os
import time
# import dask.dataframe as dd # No longer using Dask
import pandas as pd 
from pymilvus import (
    connections,
    utility,
    FieldSchema, CollectionSchema, DataType,
    Collection
)

# Assuming embedder.py is in the same directory or path is configured
# Need to import the *new* parquet path from the refactored embedder
from .embedder import EMBEDDINGS_PARQUET_PATH, EMBEDDING_DIM 

# --- Milvus Configuration ---
HOS = 'localhost' # Milvus server host (default for local Docker)
PORT = '19530'      # Milvus server port (default for local Docker)
COLLECTION_NAME = 'arxiv_papers' # Name for the Milvus collection

# Schema Definition
# Ensure field names match columns in the Parquet file 
# and choose appropriate Milvus data types.
# Make VARCHAR lengths generous, especially for abstract/title.
ID_FIELD = "id" # Primary key field name (must match Parquet column)
VECTOR_FIELD = "embedding" # Vector field name (must match Parquet column)

fields = [
    FieldSchema(name=ID_FIELD, dtype=DataType.VARCHAR, is_primary=True, auto_id=False, max_length=50), # Explicitly set auto_id=False
    FieldSchema(name=VECTOR_FIELD, dtype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM),
    FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=1024), # Increased length
    FieldSchema(name="abstract", dtype=DataType.VARCHAR, max_length=65535), # Max length for VARCHAR
    FieldSchema(name="authors", dtype=DataType.VARCHAR, max_length=2048), # Increased length
    FieldSchema(name="categories", dtype=DataType.VARCHAR, max_length=256),
    FieldSchema(name="v1_timestamp", dtype=DataType.INT64), # Assuming non-null after processing
    FieldSchema(name="doi", dtype=DataType.VARCHAR, max_length=256)
]
schema = CollectionSchema(fields, description="arXiv Paper Embeddings Collection", primary_field=ID_FIELD)

# Index parameters for the vector field
INDEX_PARAMS = {
    "metric_type": "L2",    # Or "IP" (Inner Product) - depends on the embedding model
    "index_type": "HNSW",   # Hierarchical Navigable Small World - good balance
    "params": {"M": 16, "efConstruction": 128} # Adjust based on dataset size/query speed needs
}

# --- End Milvus Configuration ---

def connect_to_milvus(alias="default", host=HOS, port=PORT):
    """Establishes connection to the Milvus server."""
    print(f"Connecting to Milvus server at {host}:{port}...")
    try:
        connections.connect(alias=alias, host=host, port=port)
        print("Successfully connected to Milvus.")
    except Exception as e:
        print(f"Failed to connect to Milvus: {e}")
        raise

def create_collection_if_not_exists(collection_name=COLLECTION_NAME, schema_to_use=schema):
    """Creates the Milvus collection if it doesn't already exist."""
    # Check connection status first
    try:
        if not connections.has_connection("default"):
            print("Milvus connection lost, attempting to reconnect...")
            connect_to_milvus()
    except Exception as e:
        print(f"Failed to check/re-establish Milvus connection: {e}")
        raise # Re-raise to signal failure
        
    has_collection = utility.has_collection(collection_name)
    
    if has_collection:
        print(f"Collection '{collection_name}' already exists.")
        # Option to drop: Consider adding a flag or prompt if needed
        # print(f"Dropping existing collection '{collection_name}'...")
        # utility.drop_collection(collection_name)
        # time.sleep(1)
        # print("Recreating collection...")
        # collection = Collection(name=collection_name, schema=schema_to_use)
        collection = Collection(name=collection_name)
    else:
        print(f"Creating collection '{collection_name}'...")
        collection = Collection(name=collection_name, schema=schema_to_use)
        print(f"Collection '{collection_name}' created.")
    return collection

def create_index(collection: Collection, field_name=VECTOR_FIELD, index_params=INDEX_PARAMS):
    """Creates the index on the vector field if it doesn't exist."""
    if not collection.has_index(index_name=f"{field_name}_index"): # Check by potential index name
        print(f"Creating index '{field_name}_index' for field '{field_name}'...")
        try:
            # Ensure collection is loaded before creating index
            is_loaded = utility.load_state(collection.name) == "Loaded"
            if collection.num_entities > 0 and not is_loaded:
                 print("Loading collection before creating index...")
                 collection.load()
                 utility.wait_for_loading_complete(collection.name)
                 print("Collection loaded.")
                 
            collection.create_index(
                 field_name=field_name, 
                 index_params=index_params,
                 index_name=f"{field_name}_index" # Explicitly name index
            )
            print("Index creation initiated successfully.")
            print("Waiting for index to build (this can take time)...")
            utility.wait_for_index_building_complete(collection.name)
            print("Index building complete.")
        except Exception as e:
            print(f"Failed to create index: {e}")
            # Consider cleanup if index creation fails partially
            raise
    else:
        print(f"Index '{field_name}_index' already exists on field '{field_name}'.")
        # Optionally check params and recreate
        # current_index = collection.index(index_name=f"{field_name}_index")
        # if current_index.params != index_params:
        #    ... recreate logic ...

def load_data_from_parquet(parquet_path=EMBEDDINGS_PARQUET_PATH) -> pd.DataFrame:
    """Loads the Pandas DataFrame from the Parquet file."""
    if not os.path.exists(parquet_path):
        print(f"Error: Embeddings Parquet file not found at {parquet_path}")
        print("Please run the embedder script first (API mode).")
        return None
    print(f"Loading data from {parquet_path}...")
    try:
        df = pd.read_parquet(parquet_path)
        print(f"Data loaded successfully. Rows: {len(df)}")
        # Ensure required columns exist
        required_cols = [f.name for f in schema.fields]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"Error: Parquet file is missing required columns for schema: {missing_cols}")
            return None
        # Ensure embedding column is list type if needed by Milvus client
        if VECTOR_FIELD in df.columns and not isinstance(df[VECTOR_FIELD].iloc[0], list):
             print(f"Converting '{VECTOR_FIELD}' column to list of lists...")
             df[VECTOR_FIELD] = df[VECTOR_FIELD].apply(lambda x: list(x) if x is not None else None)
             
        return df
    except Exception as e:
        print(f"Error loading Parquet file: {e}")
        return None

def insert_data(collection: Collection, df: pd.DataFrame):
    """
    Inserts data from the Pandas DataFrame into the Milvus collection.
    Handles potential nulls and prepares data in the correct format.
    """
    print(f"Starting data insertion into '{collection.name}'...")
    start_time = time.time()

    if df.empty:
        print("Input DataFrame is empty. No data to insert.")
        return 0

    required_columns = [field.name for field in collection.schema.fields]
    
    # --- Data Preparation ---
    print(f"Preparing {len(df)} rows for insertion...")
    # Make a copy to avoid modifying the original DataFrame if loaded elsewhere
    pdf = df.copy()
    
    # Drop rows with null primary key or vector (critical fields)
    initial_rows = len(pdf)
    pdf = pdf.dropna(subset=[ID_FIELD, VECTOR_FIELD])
    if len(pdf) < initial_rows:
        print(f"Dropped {initial_rows - len(pdf)} rows with null ID or vector.")

    if pdf.empty:
        print("DataFrame became empty after dropping critical nulls. No data to insert.")
        return 0

    # Ensure vector dimension matches
    initial_rows = len(pdf)
    pdf[VECTOR_FIELD] = pdf[VECTOR_FIELD].apply(lambda x: x if isinstance(x, list) and len(x) == EMBEDDING_DIM else None)
    pdf = pdf.dropna(subset=[VECTOR_FIELD])
    if len(pdf) < initial_rows:
        print(f"Dropped {initial_rows - len(pdf)} rows with mismatched vector dimension.")

    if pdf.empty:
        print("DataFrame became empty after dropping mismatched vectors. No data to insert.")
        return 0

    # Fillna for other fields according to schema requirements
    pdf['title'] = pdf['title'].fillna('').astype(str)
    pdf['abstract'] = pdf['abstract'].fillna('').astype(str)
    pdf['authors'] = pdf['authors'].fillna('').astype(str)
    pdf['categories'] = pdf['categories'].fillna('').astype(str)
    # Ensure timestamp is int64, handle potential Pandas nullable Int64
    if pd.api.types.is_integer_dtype(pdf['v1_timestamp']) and pdf['v1_timestamp'].isnull().any():
         pdf['v1_timestamp'] = pdf['v1_timestamp'].fillna(-1) # Fill NA before converting
    pdf['v1_timestamp'] = pdf['v1_timestamp'].astype('int64')
    pdf['doi'] = pdf['doi'].fillna('').astype(str)
    
    # Select columns in the order defined by the schema
    try:
        # Convert DataFrame columns to lists for Milvus insertion
        data_to_insert = [pdf[col_name].tolist() for col_name in required_columns]
    except KeyError as e:
        print(f"  Error preparing data: Missing column {e} after processing.")
        return 0
        
    # --- Insertion ---    
    total_inserted = 0
    try:
        print(f"Inserting {len(pdf)} prepared rows into Milvus...")
        # Milvus insert can handle the full list of lists directly
        insert_result = collection.insert(data_to_insert)
        inserted_count = len(insert_result.primary_keys)
        total_inserted += inserted_count
        print(f"Successfully inserted {inserted_count} rows.")
        
        # Flush after insertion
        print("Flushing collection...")
        collection.flush() 
        print("Flush complete.")
             
    except Exception as e:
        print(f"Error inserting data: {e}")
        # Example: print first few problematic IDs if error occurs during bulk insert
        print(f"  Problematic IDs (first 5): {pdf[ID_FIELD].head().tolist()}")

    end_time = time.time()
    print(f"\nTotal time for data insertion and flush: {end_time - start_time:.2f} seconds.")
    print(f"Total rows successfully inserted in this run: {total_inserted}")
    try:
        # Get updated entity count after flush
        collection.load() # Ensure counts are up-to-date
        utility.wait_for_loading_complete(collection.name)
        final_count = collection.num_entities
        print(f"Total entities now in collection '{collection.name}': {final_count}")
    except Exception as e:
        print(f"Could not retrieve final entity count: {e}")
        
    return total_inserted


if __name__ == '__main__':
    print("--- Starting Milvus Indexing Script (API Mode) ---")

    # 1. Connect to Milvus
    try:
        connect_to_milvus()
    except Exception as e:
        print(f"Exiting script due to Milvus connection failure: {e}")
        exit(1)

    # 2. Create or get collection
    try:
        collection = create_collection_if_not_exists()
    except Exception as e:
        print(f"Exiting script due to collection creation/retrieval failure: {e}")
        connections.disconnect("default")
        exit(1)

    # 3. Load data from Parquet
    # Ensure we use the correct path for the API-fetched embeddings
    papers_df = load_data_from_parquet(parquet_path=EMBEDDINGS_PARQUET_PATH)

    if papers_df is None or papers_df.empty:
        print("Failed to load data or data is empty. Exiting indexer script.")
        connections.disconnect("default")
        exit(1)

    # 4. Insert data
    should_insert = True
    try:
        collection.load() # Load to check entities
        utility.wait_for_loading_complete(collection.name)
        current_entities = collection.num_entities
        print(f"Collection '{collection.name}' currently contains {current_entities} entities.")
        # Ask user whether to add or replace (by dropping first)
        user_choice = input("Add data (a), replace existing data (r), or skip (s)? [a/r/s]: ").lower()
        if user_choice == 'r':
             print("Dropping existing collection to replace data...")
             utility.drop_collection(collection.name)
             time.sleep(1)
             collection = create_collection_if_not_exists() # Recreate it empty
             print("Proceeding with insertion into fresh collection.")
        elif user_choice == 's':
             print("Skipping data insertion.")
             should_insert = False
        else: # Default to add
             print("Proceeding to add data (duplicates with same ID will be ignored by Milvus)." )
             
    except Exception as e:
        print(f"Could not check entity count or handle replace logic: {e}. Proceeding to add data.")

    if should_insert:
        try:
            insert_data(collection, papers_df)
        except Exception as e:
             print(f"An error occurred during the insertion process: {e}")
             # Potentially exit or just proceed to indexing
    
    # 5. Create index (if needed)
    try:
        # No need to load again if inserted, but load ensures it is ready for index
        print("Ensuring collection is loaded before index creation...")
        collection.load()
        utility.wait_for_loading_complete(collection.name)
        print("Collection loaded.")
        create_index(collection)
    except Exception as e:
        print(f"An error occurred during index creation: {e}")

    # 6. Disconnect
    print("Disconnecting from Milvus...")
    try:
        connections.disconnect("default")
    except Exception as e:
        print(f"Error during disconnection: {e}")

    print("--- Milvus Indexing Script Finished ---")
