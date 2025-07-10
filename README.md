# Matrix Multiplication & NLP Experiments

This repository contains two main areas of exploration:

1. **Matrix Multiplication Benchmarks**: Cross-language performance comparisons
2. **NLP/RAG Experiments**: Natural Language Processing and Retrieval-Augmented Generation

⚠️ **Disclaimer**: The implementations in this repository are experimental and may not be optimized for production use. Results may vary across different hardware configurations and should not be considered definitive performance benchmarks.

## Quick Start

### Matrix Multiplication Benchmarks
```bash
cd benchmarks/
make compare-languages          # Compare Python, C++, Rust
make run-rust DTYPE=float32    # Test fastest data type
```

### NLP Experiments  
```bash
cd nlp/
jupyter lab                    # Open notebooks
```

## Repository Structure

```
dotprod/
├── benchmarks/           # Matrix multiplication benchmarks
│   ├── src/             # Source code (Python, C++, Rust)
│   ├── bin/             # Compiled binaries
│   ├── results/         # Performance results
│   └── README.md        # Benchmark documentation
├── nlp/                 # NLP and RAG experiments
│   ├── notebooks/       # Jupyter notebooks
│   ├── data/           # Text data for experiments
│   └── README.md       # NLP documentation
├── legacy/             # Legacy code (run.py)
└── assets/            # Images and assets
```

## Key Findings

### Matrix Multiplication Performance
- **float32 is fastest**: 36.1 GFLOPS with NumPy
- **NumPy dominates** floating-point operations
- **Rust excels** with integer types and flat arrays
- **Manual implementations** are ~1000x slower

### NLP Capabilities
- **Multi-language RAG** (Norwegian/English)
- **Hybrid retrieval** (sparse + dense)
- **Local LLM inference** with Ollama
- **Question-answering** with transformers

![Matrix Multiplication Visualization](assets/matmul.png)

## Getting Started

1. **For Benchmarks**: See [benchmarks/README.md](benchmarks/README.md)
2. **For NLP**: See [nlp/README.md](nlp/README.md)
3. **Legacy Code**: See [legacy/](legacy/) for original experiments

Each directory contains detailed documentation and usage examples.
