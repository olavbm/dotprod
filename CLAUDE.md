# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This repository contains four main areas of computational research:
1. **Cross-language matrix multiplication benchmarks** (Python, C++, Rust)
2. **Custom automatic differentiation library** with neural networks
3. **NLP/RAG experiments** using modern transformers and retrieval systems
4. **Reinforcement learning experiments** with gymnasium environments

## Architecture and Structure

### Multi-language Benchmark System
- **Standardized methodology**: All implementations follow `benchmarks/BENCHMARK_SPEC.md`
- **Cross-language comparison**: Python (NumPy), C++ (manual/Eigen), Rust (ndarray/nalgebra)
- **Data type flexibility**: float32/64, int32/8 with configurable matrix sizes
- **Performance measurement**: GFLOPS calculation with statistical analysis

### Custom Autograd System (`autograd/`)
- **Scalar automatic differentiation**: `Value` class with gradient tracking
- **Neural network primitives**: `Neuron` → `Layer` → `MLP` hierarchy
- **Activation functions**: ReLU, sigmoid, tanh, with backward passes
- **Loss functions**: MSE and cross-entropy with accuracy metrics

### NLP Pipeline (`nlp/`)
- **Local-first RAG**: LlamaIndex with Ollama integration for offline inference
- **Hybrid retrieval**: BM25 sparse + BGE dense embeddings
- **Multi-language support**: Norwegian and English text processing
- **Question-answering**: Haystack framework with transformer models

### Reinforcement Learning System
- **Environment integration**: Gymnasium for standard RL environments
- **CartPole visualization**: Interactive demos with pygame rendering
- **Custom autograd integration**: RL algorithms using the Value class
- **Policy learning**: Gradient-based policy optimization methods
- **Distributed training**: Ray cluster support for multi-node RL experiments

## Development Commands

### Environment Setup
**Note**: This repository uses standard Python dependency management, not UV package manager.

```bash
# Install core dependencies
pip install numpy matplotlib requests

# For RL experiments (when needed)
pip install "ray[rllib,tune]" gymnasium pygame

# For development tools (when needed)  
pip install pytest black ruff
```

### Matrix Multiplication Benchmarks
```bash
cd benchmarks/
make compare-languages              # Compare Python, C++, Rust performance
make standardized SIZE=500 DTYPE=float32  # Custom size/type benchmark
make run-rust DTYPE=int32          # Test specific language and data type
make clean                         # Clean compiled binaries and results
```

### Autograd System Testing
```bash
python -m pytest autograd/         # Run all autograd tests (if pytest installed)
python -m pytest autograd/test_value.py -v  # Test scalar differentiation
python minimal_train.py            # Demo neural network training
python mnist_classifier.py         # MNIST digit classification
```

### NLP Experiments
```bash
cd nlp/
jupyter lab                        # Open interactive notebooks (requires jupyter installation)
# Key notebooks:
# - rag_llamaindex.ipynb: RAG with Ollama/Mistral
# - rag_qa.ipynb: Question-answering with Haystack
# - 07_question_answering_v2.ipynb: Transformers QA pipeline
```

### Reinforcement Learning
**Current RL Files**: `cartpole_viz.py`, `cartpole_tune.py`, `minimal_ray_test.py`

```bash
cd rl/
python cartpole_viz.py             # CartPole environment visualization
python cartpole_tune.py            # Ray Tune with hoppetusse cluster (requires Ray + SSH setup)
python minimal_ray_test.py         # Test Ray Client connection to remote cluster
```

### Ray Client with SSH Tunneling (Working Setup)
**Tested Configuration**: Ray Client connection to hoppetusse cluster (16 CPUs, 1 GPU)

```bash
# 1. On remote machine (hoppetusse):
ray start --head --ray-client-server-port=10001

# 2. SSH config (~/.ssh/config) with port forwarding:
Host hoppetusse
    LocalForward 10001 localhost:10001

# 3. Connect via SSH (keep connection open):
ssh hoppetusse

# 4. Test connection:
python rl/minimal_ray_test.py

# 5. Run Ray Tune on cluster:
python rl/cartpole_tune.py
```

### Code Quality
```bash
black . --line-length 88          # Format Python code (if black installed)
ruff check .                       # Lint Python code (if ruff installed)
ruff check --fix .                 # Auto-fix linting issues
```

## Key Performance Findings

### Matrix Multiplication (100x100)
- **Best float32**: 36.1 GFLOPS (NumPy @ operator)
- **Best float64**: 27.6 GFLOPS (NumPy @ operator)  
- **Best int32**: 6.91 GFLOPS (Rust flat arrays)
- **Pattern**: NumPy dominates floating-point, Rust excels with integers

### Architecture Insights
- **Manual implementations** are ~1000x slower than optimized libraries
- **float32 outperforms float64** due to better SIMD utilization
- **Rust consistency** across data types with predictable performance
- **Library optimization** is crucial for production performance

## Component Integration

### Autograd + Benchmarks
The custom `Value` class could integrate with benchmark results for gradient-based optimization of matrix operations or hyperparameter tuning.

### NLP + Autograd  
The neural network components (`MLP`, `Layer`) are designed to work with NLP embeddings and could be extended for custom transformer implementations.

### Multi-language Learning
Benchmark results inform data type choices for autograd computations, while NLP experiments demonstrate real-world applications of the neural network primitives.

### RL + Autograd Integration
The reinforcement learning experiments leverage the custom autograd system for gradient-based policy optimization, demonstrating practical applications of automatic differentiation in sequential decision-making.

## File Organization Logic

```
benchmarks/src/           # Language-specific implementations
benchmarks/results/       # JSON performance data by language/type
autograd/                 # Self-contained differentiation library
nlp/notebooks/           # Interactive experimentation
nlp/data/                # Multi-language text corpora
rl/                      # Reinforcement learning experiments
legacy/                  # Original dot product experiments (run.py)
```

## Development Environment Notes

- **Python 3.11+** required for modern type hints and performance
- **Standard pip/conda** for dependency management (not UV package manager)
- **GNU Make** orchestrates multi-language build process
- **Jupyter Lab** for interactive NLP experimentation (requires separate installation)
- **Git** configured to ignore build artifacts (`bin/`, `target/`, results)

## Testing Strategy

- **Unit tests**: `python -m pytest autograd/` for differentiation correctness (requires pytest)
- **Integration tests**: `make test` for benchmark validation  
- **Performance verification**: Cross-language result comparison
- **Manual validation**: `python minimal_train.py` demonstrates learning convergence
- **RL demos**: `python rl/cartpole_viz.py` for environment interaction testing
- **Ray cluster tests**: `python rl/minimal_ray_test.py` for distributed computing verification

## Important Usage Notes

**Use standard Python commands** - this repository uses standard Python dependency management:
- ✅ `python rl/cartpole_viz.py`
- ✅ `python script.py` 
- ✅ `python -m pytest tests/`
- ⚠️ Ensure dependencies are installed: `pip install numpy matplotlib requests`

## Ray Cluster Configuration

### Current Working Setup
The repository includes a tested Ray Client setup with SSH tunneling to a remote machine (hoppetusse) with 16 CPUs and 1 GPU. This configuration works for distributed RL experiments.

### Ray Client Architecture (Tested Working)
Ray Client allows connecting to remote clusters through firewalls/NAT using only SSH port forwarding:

```bash
# On remote machine (hoppetusse):
ray start --head --ray-client-server-port=10001

# SSH config (~/.ssh/config):  
Host hoppetusse
    LocalForward 10001 localhost:10001

# Connect and test:
ssh hoppetusse
python rl/minimal_ray_test.py
```

### Network Requirements
- **Port 6379**: Ray cluster communication (head ↔ worker nodes)
- **Port 8265**: Ray Dashboard (web interface)  
- **Port 10001**: Ray Client server (remote connections via SSH tunnel)

### Current Working Files for Ray Cluster
- **`rl/cartpole_tune.py`**: Ray Tune PPO with Ray Client connection to hoppetusse cluster
- **`rl/minimal_ray_test.py`**: Comprehensive Ray Client connection testing and validation
- **`rl/cartpole_viz.py`**: Basic CartPole environment visualization

### Ray Tune Integration Pattern
The `cartpole_tune.py` demonstrates the working pattern:
```python
# Connect to cluster via Ray Client
ray.init("ray://localhost:10001")
cluster_resources = ray.cluster_resources()
total_cpus = int(cluster_resources.get('CPU', 0))

# Use cluster CPUs for parallel hyperparameter tuning
tune_config=tune.TuneConfig(
    max_concurrent_trials=min(total_cpus, num_trials)
)
```