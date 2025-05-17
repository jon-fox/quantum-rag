#!/usr/bin/env python
"""
Example script demonstrating how to use watch embeddings
for semantic search of luxury watch descriptions and prices.

Usage:
    python src/scripts/example_embedding.py
"""
import os
import sys
import json
from typing import List, Dict, Any
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.embeddings.embed_utils import (
    get_embedding_provider,
    embed_watch_items,
    price_aware_embedding,
    embed_query,
    find_similar_watches,
    save_embeddings,
    load_embeddings
)

# Sample watch data for demonstration
SAMPLE_WATCHES = [
    {
        "id": "watch_1",
        "brand": "Rolex",
        "model": "Submariner",
        "reference": "116610LN",
        "description": "Black dial, ceramic bezel, stainless steel Oyster bracelet, date function",
        "condition": "Excellent",
        "year": "2019",
        "price": 12500.00,
        "box_papers": True
    },
    {
        "id": "watch_2",
        "brand": "Rolex",
        "model": "Daytona",
        "reference": "116500LN",
        "description": "White dial, ceramic bezel, stainless steel Oyster bracelet, chronograph",
        "condition": "New",
        "year": "2022",
        "price": 32000.00,
        "box_papers": True
    },
    {
        "id": "watch_3",
        "brand": "Omega",
        "model": "Speedmaster Professional",
        "reference": "311.30.42.30.01.005",
        "description": "Moonwatch, black dial, hesalite crystal, manual-winding chronograph",
        "condition": "Very Good",
        "year": "2018",
        "price": 6500.00,
        "box_papers": True
    },
    {
        "id": "watch_4",
        "brand": "Patek Philippe",
        "model": "Nautilus",
        "reference": "5711/1A-010",
        "description": "Blue dial, stainless steel bracelet, automatic movement",
        "condition": "Excellent",
        "year": "2020",
        "price": 135000.00,
        "box_papers": True
    },
    {
        "id": "watch_5", 
        "brand": "Omega",
        "model": "Seamaster",
        "reference": "210.30.42.20.01.001",
        "description": "Black dial, ceramic bezel, stainless steel bracelet, date function, co-axial chronometer",
        "condition": "Good",
        "year": "2021",
        "price": 4800.00,
        "box_papers": False
    }
]


def demonstrate_basic_embeddings():
    """Demonstrate basic embedding functionality"""
    print("\n=== Basic Embeddings Example ===")
    
    # Get a provider - this will use OpenAI if API key is set, otherwise fall back to sentence-transformer
    provider = get_embedding_provider()
    print(f"Using embedding provider: {provider.__class__.__name__}")
    
    # Generate embeddings for all watches
    processed_items, embeddings = embed_watch_items(SAMPLE_WATCHES, provider)
    
    # Print embedding info
    print(f"Generated {len(embeddings)} embeddings with dimension {embeddings.shape[1]}")
    
    # Show sample embedding text for first watch
    print(f"\nSample embedding text for {processed_items[0]['brand']} {processed_items[0]['model']}:")
    print(processed_items[0]['embedding_text'])
    
    # Print first few values of first embedding
    print(f"\nFirst few values of embedding: {embeddings[0][:5]}...")
    
    return processed_items, embeddings


def demonstrate_semantic_search(processed_items, embeddings):
    """Demonstrate semantic search functionality"""
    print("\n=== Semantic Search Example ===")
    
    # Create a query
    search_query = "stainless steel diving watch with date function"
    print(f"Query: '{search_query}'")
    
    # Get embedding for query
    query_embedding = embed_query(search_query)
    
    # Find similar watches
    results = find_similar_watches(
        query_embedding=query_embedding,
        all_embeddings=embeddings,
        all_items=processed_items,
        top_k=3,
        min_similarity=0.5
    )
    
    # Display results
    print("\nResults:")
    for i, result in enumerate(results, 1):
        watch = result['item']
        similarity = result['similarity']
        print(f"{i}. {watch['brand']} {watch['model']} (Ref: {watch['reference']})")
        print(f"   Similarity: {similarity:.4f}")
        print(f"   Price: ${watch['price']:.2f}")
        print(f"   Description: {watch['description'][:80]}{'...' if len(watch['description']) > 80 else ''}")
        print()


def demonstrate_price_aware_search(processed_items):
    """Demonstrate price-aware search functionality"""
    print("\n=== Price-Aware Search Example ===")
    
    # Get provider
    provider = get_embedding_provider()
    
    # Generate price-aware embeddings
    price_aware_embeddings = []
    for item in processed_items:
        embedding = price_aware_embedding(
            description=item['embedding_text'],
            price=item['price'],
            provider=provider
        )
        price_aware_embeddings.append(embedding)
    
    price_aware_embeddings = np.array(price_aware_embeddings)
    
    # Create a price-aware query
    search_query = "affordable Omega under 5000 dollars"
    print(f"Query: '{search_query}'")
    
    # Embed the query
    query_embedding = embed_query(search_query, provider)
    
    # Find similar watches
    results = find_similar_watches(
        query_embedding=query_embedding,
        all_embeddings=price_aware_embeddings,
        all_items=processed_items,
        top_k=2,
        min_similarity=0.5
    )
    
    # Display results
    print("\nResults:")
    for i, result in enumerate(results, 1):
        watch = result['item']
        similarity = result['similarity']
        print(f"{i}. {watch['brand']} {watch['model']} (Ref: {watch['reference']})")
        print(f"   Similarity: {similarity:.4f}")
        print(f"   Price: ${watch['price']:.2f}")
        print(f"   Description: {watch['description'][:80]}{'...' if len(watch['description']) > 80 else ''}")
        print()


def demonstrate_saving_loading():
    """Demonstrate saving and loading embeddings"""
    print("\n=== Save & Load Embeddings Example ===")
    
    # Get embeddings
    items, embeddings = embed_watch_items(SAMPLE_WATCHES)
    
    # Save to file
    filename = "watch_embeddings_demo.json"
    save_embeddings(items, embeddings, filename)
    
    # Load from file
    loaded_items, loaded_embeddings = load_embeddings(filename)
    
    print(f"Loaded {len(loaded_items)} items with embeddings of shape {loaded_embeddings.shape}")
    
    # Check that embeddings are the same
    if np.allclose(embeddings, loaded_embeddings):
        print("✓ Loaded embeddings match the original ones")
    else:
        print("✗ Loaded embeddings don't match")


def main():
    """Main function to run all demonstrations"""
    print("Watch Embedding Demonstration")
    print("============================")
    
    # Basic embedding demo
    processed_items, embeddings = demonstrate_basic_embeddings()
    
    # Semantic search demo
    demonstrate_semantic_search(processed_items, embeddings)
    
    # Price-aware search demo
    demonstrate_price_aware_search(processed_items)
    
    # Save/load demo
    demonstrate_saving_loading()


if __name__ == "__main__":
    main()