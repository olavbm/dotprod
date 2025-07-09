# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This repository contains experiments with dot products and natural language processing, focusing on:
- Basic dot product implementations (NumPy vs custom implementation)
- RAG (Retrieval-Augmented Generation) question-answering systems
- Natural language processing with transformers
- Jupyter notebooks for interactive development

## Key Files and Structure

- `run.py` - Main Python script with dot product implementations and testing
- `rag_llamaindex.ipynb` - RAG implementation using LlamaIndex with Ollama
- `rag_qa.ipynb` - RAG implementation using Haystack framework
- `07_question_answering_v2.ipynb` - Comprehensive question-answering notebook from NLP transformers book
- `data/` - Contains text data for RAG experiments:
  - `hdir/` - Norwegian text about health services for severely ill children
  - `graham.txt` - Paul Graham essay text
  - `mg.txt` - Additional text data
- `assets/` - Contains images (matmul.png)

## Development Environment

The project uses a conda environment called `dotprod` with Python 3.11. Key dependencies include:
- NumPy for mathematical operations
- LlamaIndex for RAG implementation
- Haystack for question-answering pipelines
- Transformers for NLP models
- Jupyter for interactive development

## Running the Code

### Basic dot product testing:
```bash
python run.py
```

### Jupyter notebooks:
```bash
jupyter lab
# Open any of the .ipynb files
```

### RAG with LlamaIndex:
The `rag_llamaindex.ipynb` notebook demonstrates:
- Loading documents from the `data/hdir` directory
- Using BGE embeddings and Mistral model via Ollama
- Querying Norwegian text about health services

### Question-Answering with Haystack:
The `rag_qa.ipynb` notebook shows:
- Setting up document stores with Elasticsearch
- Using BM25 and dense retrieval
- Training extractive QA models

## Architecture Notes

### RAG Implementation:
- Uses local embeddings (BGE-large-en-v1.5) for document indexing
- Integrates with Ollama for local LLM inference
- Supports both Norwegian and English text processing

### Question-Answering Pipeline:
- Retriever-reader architecture
- Supports both sparse (BM25) and dense retrieval
- Uses pre-trained models from Hugging Face

### Data Processing:
- Text files are loaded as documents for RAG
- Supports multiple document formats and languages
- Includes preprocessing for question-answering datasets

## Testing

The `run.py` file includes a comprehensive test function `test_dotprods()` that validates the consistency between NumPy and custom dot product implementations across various input ranges.

## Model Dependencies

- Ollama with Mistral model for text generation
- BGE embeddings for document retrieval
- RoBERTa-based models for question answering
- Elasticsearch for document storage (in some configurations)