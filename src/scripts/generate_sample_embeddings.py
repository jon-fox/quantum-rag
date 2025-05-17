#!/usr/bin/env python
"""
Generate sample watch embeddings and save them to a file
for use with the FastAPI embeddings endpoints.

This script uses the sample watch data from example_embedding.py
and saves the embeddings to the data/embeddings directory.

Usage:
    python src/scripts/generate_sample_embeddings.py
"""
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.scripts.example_embedding import SAMPLE_WATCHES
from src.embeddings.embed_utils import (
    get_embedding_provider,
    embed_watch_items,
    save_embeddings
)

# Default path for embeddings storage
EMBEDDINGS_PATH = Path(os.environ.get("EMBEDDINGS_PATH", "data/embeddings"))

def main():
    """Main function to generate and save sample embeddings"""
    print("Generating sample watch embeddings...")
    
    # Ensure embeddings directory exists
    os.makedirs(EMBEDDINGS_PATH, exist_ok=True)
    
    # Get embedding provider
    provider = get_embedding_provider()
    print(f"Using embedding provider: {provider.__class__.__name__}")
    
    # Generate embeddings for sample watches
    processed_items, embeddings = embed_watch_items(SAMPLE_WATCHES, provider)
    
    # Save to file
    filename = str(EMBEDDINGS_PATH / "watch_embeddings.json")
    save_embeddings(processed_items, embeddings, filename)
    print(f"Embeddings saved to {filename}")
    
    print(f"\nYou can now test the API endpoints by running:")
    print(f"  uvicorn app:app --reload")
    print(f"\nThen visit http://localhost:8000/docs in your browser.")

if __name__ == "__main__":
    main()