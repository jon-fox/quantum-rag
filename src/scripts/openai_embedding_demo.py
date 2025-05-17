#!/usr/bin/env python
"""
Demo script showing how to use OpenAI embeddings specifically
for the watch arbitrage project.

Usage:
    python src/scripts/openai_embedding_demo.py

Make sure to set your OPENAI_API_KEY environment variable before running:
    export OPENAI_API_KEY='your-api-key-here'
"""
import os
import sys
from dotenv import load_dotenv
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.embeddings.embed_utils import (
    OpenAIEmbedding,
    create_watch_embedding_text,
    price_aware_embedding,
    save_embeddings
)
from src.scripts.example_embedding import SAMPLE_WATCHES

# Load environment variables from .env file if it exists
load_dotenv()

def check_api_key():
    """Check if OpenAI API key is set"""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY environment variable not set.")
        print("Please set it before running this script:")
        print("  export OPENAI_API_KEY='your-api-key-here'")
        return False
    return True

def main():
    """Main function demonstrating OpenAI embeddings"""
    if not check_api_key():
        sys.exit(1)
    
    print("Watch Arbitrage - OpenAI Embedding Demo")
    print("=======================================")
    
    # Initialize OpenAI embedding provider
    try:
        embedding_provider = OpenAIEmbedding()
        print(f"Using embedding model: {embedding_provider.model_name}")
        print(f"Embedding dimension: {embedding_provider.embedding_dim}")
    except Exception as e:
        print(f"ERROR: Failed to initialize OpenAI embedding provider: {e}")
        sys.exit(1)
    
    # Process a single watch first as demonstration
    sample_watch = SAMPLE_WATCHES[0]
    print(f"\nGenerating embedding for {sample_watch['brand']} {sample_watch['model']}")
    
    # Create embedding text for a watch
    embedding_text = create_watch_embedding_text(sample_watch)
    print(f"\nEmbedding text: {embedding_text[:100]}...")
    
    # Generate embedding
    try:
        embedding = embedding_provider.get_embeddings([embedding_text])[0]
        print(f"Embedding shape: {embedding.shape}")
        print(f"First few values: {embedding[:5]}...")
    except Exception as e:
        print(f"ERROR: Failed to generate embedding: {e}")
        sys.exit(1)
    
    # Generate price-aware embedding
    print("\nGenerating price-aware embedding...")
    try:
        price_aware_emb = price_aware_embedding(
            description=embedding_text,
            price=sample_watch["price"],
            provider=embedding_provider
        )
        print(f"Price-aware embedding shape: {price_aware_emb.shape}")
    except Exception as e:
        print(f"ERROR: Failed to generate price-aware embedding: {e}")
        sys.exit(1)
    
    # Process all sample watches
    print("\nProcessing all sample watches...")
    embeddings_batch = []
    processed_watches = []
    
    for watch in SAMPLE_WATCHES:
        watch_copy = watch.copy()  # Create a copy to avoid modifying the original
        # Create text representation
        text = create_watch_embedding_text(watch_copy)
        watch_copy["embedding_text"] = text
        
        try:
            # Add price-aware embedding
            embedding = price_aware_embedding(
                description=text,
                price=watch_copy["price"],
                provider=embedding_provider
            )
            embeddings_batch.append(embedding)
            processed_watches.append(watch_copy)
        except Exception as e:
            print(f"WARNING: Failed to process watch {watch_copy.get('id')}: {e}")
    
    embeddings_array = np.array(embeddings_batch)
    
    # Save to file
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 
                             "data", "embeddings")
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, "openai_watch_embeddings.json")
    try:
        save_embeddings(processed_watches, embeddings_array, output_file)
        print(f"\nEmbeddings saved to {output_file}")
    except Exception as e:
        print(f"ERROR: Failed to save embeddings: {e}")
    
    print("\nTo use these embeddings with the API:")
    print("1. Start the FastAPI server: uvicorn app:app --reload")
    print("2. Visit http://localhost:8000/docs to use the API")

if __name__ == "__main__":
    main()