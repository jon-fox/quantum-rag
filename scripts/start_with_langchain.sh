#!/bin/bash

# Script to install LangChain dependencies and restart the server

echo "Installing LangChain and required dependencies..."
pip install langchain~=0.0.267 langchain-community~=0.0.10 faiss-cpu~=1.7.4 sentence-transformers~=2.2.2

echo "Checking if OpenAI is properly installed..."
pip install --upgrade openai~=1.0.0

echo "Creating required directories..."
mkdir -p data/vectors

echo "Setting up environment..."
export PYTHONPATH=/mnt/c/Developer_Workspace/quantum_work:$PYTHONPATH

echo "Starting the server with LangChain integration..."
python app/main.py
