"""Module for loading and processing arXiv data using the arxiv API."""

import os
import time
import json # Still used for printing examples
import pandas as pd
import arxiv # Import the new library
from datetime import datetime

# Define paths (adjust if needed, DATA_DIR might not be used the same way)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Project root
# DATA_DIR = os.path.join(BASE_DIR, 'data', 'arxiv') # No longer downloading bulk data
MILVUS_DIR = os.path.join(BASE_DIR, 'milvus_data') # Still potentially useful
# ARXIV_JSON_PATH = os.path.join(DATA_DIR, 'arxiv-metadata-oai-snapshot.json') # No longer used

# Ensure Milvus directory exists if needed elsewhere
os.makedirs(MILVUS_DIR, exist_ok=True)

# --- Helper Functions --- 
def process_arxiv_result(result: arxiv.Result) -> dict:
    """Processes a single arxiv.Result object into a dictionary."""
    try:
        # Extract basic fields
        paper_data = {
            "id": result.entry_id.split('/')[-1], # Get the short ID
            "title": result.title.strip().replace('\n', ' '),
            "abstract": result.summary.strip().replace('\n', ' '),
            "authors": ", ".join([str(a) for a in result.authors]),
            "categories": " ".join(result.categories), 
            "published_datetime": result.published, # Keep datetime object for now
            "updated_datetime": result.updated,
            "doi": result.doi,
            "pdf_url": result.pdf_url,
        }

        # Add combined text field (handle potential None title/abstract)
        title = paper_data.get('title', '')
        abstract = paper_data.get('abstract', '')
        paper_data['text'] = f"{title}[SEP]{abstract}"
        
        # Add v1_timestamp (using published date as approximation)
        if result.published:
             paper_data['v1_timestamp'] = int(result.published.timestamp())
        else:
             paper_data['v1_timestamp'] = None
             
        return paper_data

    except Exception as e:
        print(f"Warning: Failed to process result {getattr(result, 'entry_id', 'N/A')}: {e}")
        return None

# --- Main Fetching Function --- 
def fetch_arxiv_papers(
    query: str = "cat:cs.LG OR cat:stat.ML OR cat:cs.AI OR cat.CV OR cat.CL OR cat.NE",
    max_results: int = 100, # Fetch 100 recent papers matching the query
    sort_by = arxiv.SortCriterion.SubmittedDate # Get the most recent ones
) -> pd.DataFrame:
    """
    Fetches recent papers from arXiv based on a query and returns a Pandas DataFrame.

    Args:
        query: The arXiv query string (see arXiv API documentation).
        max_results: The maximum number of papers to fetch.
        sort_by: Criterion to sort results by (e.g., SubmittedDate, Relevance).

    Returns:
        A Pandas DataFrame containing processed paper data, or None if fetching fails.
    """
    print(f"Fetching up to {max_results} papers from arXiv with query: '{query}'")
    print(f"Sorting by: {sort_by}")
    start_time = time.time()
    
    try:
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=sort_by
        )
        
        results = list(search.results()) # Execute the search and get results
        
        if not results:
            print("No results found for the query.")
            return pd.DataFrame() # Return empty DataFrame
            
        print(f"Fetched {len(results)} paper results. Processing...")
        
        # Process results into a list of dictionaries
        processed_data = [process_arxiv_result(res) for res in results]
        # Filter out any results that failed processing
        processed_data = [p for p in processed_data if p is not None]
        
        if not processed_data:
             print("No papers could be successfully processed.")
             return pd.DataFrame()
             
        # Create Pandas DataFrame
        papers_df = pd.DataFrame(processed_data)
        
        # Select and order columns relevant for embedding/indexing
        # Ensure columns match the Milvus schema expected later
        final_columns = [
            'id', 'title', 'abstract', 'authors', 'categories', 
            'v1_timestamp', 'text', 'doi' # Match indexer schema
            # Add others if needed: 'published_datetime', 'updated_datetime', 'pdf_url'
        ]
        # Select only columns that actually exist in the DataFrame
        existing_columns = [col for col in final_columns if col in papers_df.columns]
        papers_df = papers_df[existing_columns]

        # Optional: Handle missing values if necessary (e.g., fill doi Nones)
        # papers_df['doi'] = papers_df['doi'].fillna('')
        # papers_df['v1_timestamp'] = papers_df['v1_timestamp'].fillna(-1) # If nulls occurred
        
        end_time = time.time()
        print(f"Data fetching and processing took {end_time - start_time:.2f} seconds.")
        print(f"Created DataFrame with {len(papers_df)} papers and columns: {list(papers_df.columns)}")
        
        return papers_df

    except Exception as e:
        print(f"An error occurred during arXiv search or processing: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == '__main__':
    print("--- Running arXiv API Loader Script ---")
    
    # --- Configuration ---
    # Query for recent Machine Learning papers
    ml_query = "cat:cs.LG OR cat:stat.ML OR cat:cs.AI OR cat:cs.CV OR cat:cs.CL OR cat:cs.NE"
    num_papers_to_fetch = 100 # Fetch 100 papers for testing
    # --- End Configuration ---

    papers_dataframe = fetch_arxiv_papers(
        query=ml_query,
        max_results=num_papers_to_fetch
    )

    if papers_dataframe is not None and not papers_dataframe.empty:
        print("\n--- Sample DataFrame Head ---")
        print(papers_dataframe.head())
        print("---------------------------")
    elif papers_dataframe is not None and papers_dataframe.empty:
         print("\nFetching returned an empty DataFrame (no results or processing failed)." )
    else:
        print("\nFailed to fetch or process data.")

    print("\n--- Script Finished ---") 