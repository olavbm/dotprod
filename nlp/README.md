# NLP and RAG Experiments

This directory contains Natural Language Processing and Retrieval-Augmented Generation experiments.

## Contents

### Notebooks
- **07_question_answering_v2.ipynb**: Comprehensive question-answering notebook from NLP transformers book
- **rag_llamaindex.ipynb**: RAG implementation using LlamaIndex with Ollama
- **rag_qa.ipynb**: RAG implementation using Haystack framework

### Data
- **graham.txt**: Paul Graham essay text
- **mg.txt**: Additional text data  
- **hdir/**: Norwegian text about health services for severely ill children
  - `hdir.txt`: Norwegian text
  - `hdir_eng.txt`: English translation

## Setup

The project uses a conda environment called `dotprod` with Python 3.11. Key dependencies:

```bash
conda activate dotprod
pip install llama-index haystack transformers torch
```

## Usage

### RAG with LlamaIndex
```python
# Load documents from data/hdir directory
# Uses BGE embeddings and Mistral model via Ollama
# Supports Norwegian and English text processing
```

### Question-Answering with Haystack
```python
# Document stores with Elasticsearch
# BM25 and dense retrieval
# Extractive QA models from Hugging Face
```

## Key Features

- **Local LLM inference** with Ollama
- **Multi-language support** (Norwegian/English)
- **Hybrid retrieval** (sparse + dense)
- **Pre-trained models** from Hugging Face
- **Document preprocessing** for various formats

## Models Used

- **Embeddings**: BGE-large-en-v1.5
- **LLM**: Mistral via Ollama
- **QA**: RoBERTa-based models
- **Search**: Elasticsearch (optional)

Start with the notebooks to explore the implementations!