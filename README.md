# 🦙 SimpleLlama

A benchmarking tool for self-hosted LLMs running on Ollama, using the public SimpleBench benchmark.

## Overview

SimpleLlama evaluates language models installed through Ollama using the 10 publicly released questions from SimpleBench.

While this provides a basic performance assessment on consumer hardware, the limited question set constrains its resolution - models can only score in 10-percentage-point increments. For context, `gpt-4o-mini`, which is speculated to be comparable in size to locally-runnable models, achieved only 10.7% on SimpleBench's complete 200+ question evaluation set. SimpleLlama's 10-question subset will not provide a precise view of a model's performance given this constraint.

Hopefully local models reasoning abilities will improve over time, and maybe the question set of this benchmark will grow over time.

## Prerequisites

- Python 3.12.7 or higher
- Ollama installed and running
  - Required LLM models downloaded in Ollama

## Installation

1. Install [pyenv](https://github.com/pyenv/pyenv) for Python version management:

   ```bash
   # macOS
   brew install pyenv

   # Linux (modify your shell profile too)
   curl https://pyenv.run | bash
   ```

2. Set up the Python environment:

   ```bash
   # Install Python 3.12.7
   pyenv install 3.12.7

   # Set the local Python version
   pyenv local 3.12.7

   # Create and activate the virtual environment
   python -m venv .venv
   source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`

   # Install the dependencies
   pip install -r requirements.txt
   ```

## Usage

1. Ensure Ollama is running:

   ```bash
   ollama serve
   ```

2. Run the benchmark:

   ```bash
   python simplellama.py --model_name=llama3.1:8b-instruct-fp16
   ```

   Additional options:

   ```bash
   python simplellama.py --model_name=gemma2:27b --benchmark_path=benchmarks/custom_bench.json --log_level=INFO --silence_http=false --num_responses=3 --temperature=0.5 --top_p=0.9 --max_tokens=4096 --max_retries=1
   ```

### Output

Benchmark results are stored in the `logs/` directory and are printed upon completion. Each run generates a timestamped log file containing:

- Model information
- Question responses
- Correct answers
- Parsed model responses
- Benchmark result
