"""Module for generating embeddings for arXiv papers (using Pandas)."""

import os
import time
import torch
import numpy as np
import pandas as pd
# import dask.dataframe as dd # No longer using Dask
from sentence_transformers import SentenceTransformer

# Assuming arxiv_loader.py is in the same directory
# from .arxiv_loader import load_and_preprocess_data, BASE_DIR # Old Dask loader
from .arxiv_loader import fetch_arxiv_papers, BASE_DIR # New arxiv API loader

# --- Configuration ---
# Model recommended for scientific text (requires separate installation? Check docs)
# MODEL_NAME = 'allenai/specter2_base' 
# General purpose, faster model:
MODEL_NAME = 'all-MiniLM-L6-v2' 
EMBEDDING_DIM = 384 # Dimension for all-MiniLM-L6-v2. Adjust if changing model (SPECTER is 768)
# Define where to save the embeddings
EMBEDDINGS_DIR = os.path.join(BASE_DIR, 'data', 'arxiv_embeddings')
# Changed filename slightly to indicate it's from the API fetch
EMBEDDINGS_PARQUET_PATH = os.path.join(EMBEDDINGS_DIR, 'arxiv_api_embeddings.parquet') 
# --- End Configuration ---

os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

def get_device():
    """Gets the best available device for torch (MPS, CUDA, or CPU)."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

def get_embedding_model(model_name: str = MODEL_NAME, device=None):
    """Loads the Sentence Transformer model."""
    if device is None:
        device = get_device()
    print(f"Loading Sentence Transformer model '{model_name}' onto device '{device}'...")
    model = SentenceTransformer(model_name, device=str(device))
    print(f"Model loaded. Max sequence length: {model.max_seq_length}")
    return model

def generate_embeddings_pandas(
    df: pd.DataFrame, # Takes Pandas DataFrame
    model: SentenceTransformer,
    output_path: str = EMBEDDINGS_PARQUET_PATH,
    batch_size: int = 64,
    recompute: bool = False
) -> pd.DataFrame:
    """
    Generates embeddings for a Pandas DataFrame and saves the result to Parquet.

    Args:
        df: Input Pandas DataFrame with a 'text' column.
        model: Loaded Sentence Transformer model.
        output_path: Path to save the output Parquet file/directory.
        batch_size: Batch size for embedding generation.
        recompute: If True, recompute even if output file exists.

    Returns:
        Pandas DataFrame with added 'embedding' column, or None if error.
    """
    if os.path.exists(output_path) and not recompute:
        print(f"Embeddings already exist at {output_path}. Skipping generation.")
        print("Loading existing embeddings from Parquet...")
        try:
            df_with_embeddings = pd.read_parquet(output_path)
            print("Existing embeddings loaded successfully.")
            return df_with_embeddings
        except Exception as e:
            print(f"Error loading existing embeddings: {e}. Will recompute.")
    
    if df.empty or 'text' not in df.columns:
         print("Input DataFrame is empty or missing 'text' column. Cannot generate embeddings.")
         return df # Return original empty/invalid df
         
    print(f"Generating embeddings for {len(df)} rows...")
    start_time = time.time()
    
    # Ensure text column is string and handle potential NaNs/Nones
    texts = df['text'].fillna('').astype(str).tolist()
    
    # Compute embeddings in batches
    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True) # Can show progress bar now
    
    # Add embeddings as a new column (list of floats)
    df['embedding'] = [emb.tolist() for emb in embeddings]

    print("Embeddings generated. Now saving to Parquet...")
    try:
        df.to_parquet(
            output_path,
            engine='pyarrow',
            index=False # Don't save pandas index
        )
        # No need to manually overwrite, pandas `to_parquet` handles it.
        end_time = time.time()
        print(f"Embeddings saved to {output_path}. Total time: {end_time - start_time:.2f} seconds.")
        return df
    except Exception as e:
        print(f"Error saving embeddings to Parquet: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == '__main__':
    print("--- Starting Embedding Generation Script (API Mode) ---")

    # --- Configuration for Loading Data ---
    # Match the configuration used in the new arxiv_loader main block
    ml_query = "cat:cs.LG OR cat:stat.ML OR cat:cs.AI OR cat:cs.CV OR cat:cs.CL OR cat:cs.NE"
    num_papers_to_fetch = 100 # Fetch 100 papers for testing
    # --- End Loading Configuration ---

    # 1. Load data using the new fetcher
    print("\nStep 1: Fetching data from arXiv API...")
    papers_df = fetch_arxiv_papers(
        query=ml_query,
        max_results=num_papers_to_fetch
    )

    if papers_df is None or papers_df.empty:
        print("Failed to fetch data or no papers found. Exiting embedder script.")
        exit(1)
    
    # 2. Load embedding model
    print("\nStep 2: Loading embedding model...")
    device = get_device()
    embedding_model = get_embedding_model(model_name=MODEL_NAME, device=device)

    # 3. Generate and save embeddings (using the Pandas version)
    print("\nStep 3: Generating and saving embeddings...")
    # Set recompute=True to force regeneration
    df_embedded = generate_embeddings_pandas(
        papers_df,
        embedding_model,
        recompute=True, # Force overwrite of existing Parquet file
        batch_size=128 # Adjust batch size based on GPU/CPU memory
    )

    # 4. Verification (Optional)
    if df_embedded is not None and not df_embedded.empty:
        print("\nStep 4: Verifying generated embeddings (displaying head)...")
        try:
            print(f"\n--- Sample Embedded DataFrame Head ---")
            # Display head, potentially excluding the long embedding vector for readability
            print(df_embedded.head().drop(columns=['embedding'], errors='ignore'))
            if 'embedding' in df_embedded.columns:
                print("\nEmbedding column sample (first row):")
                first_embedding = df_embedded['embedding'].iloc[0]
                print(f"Type: {type(first_embedding)}, Length: {len(first_embedding) if first_embedding is not None else 'N/A'}")
                # print(f"Values (first 5): {first_embedding[:5]}...")
            print("--------------------------------------------------------")

        except Exception as e:
            print(f"\nAn error occurred during verification: {e}")
            import traceback
            traceback.print_exc()
    elif df_embedded is not None and df_embedded.empty:
         print("\nEmbedding generation resulted in an empty DataFrame.")
    else:
        print("\nEmbedding generation failed.")

    print("\n--- Embedding Generation Script Finished ---") 