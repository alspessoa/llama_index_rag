#!/bin/bash

pip uninstall llama-index  
pip install -U llama-index --upgrade --no-cache-dir --force-reinstall

pip install llama-parse
pip install llama-index-core
pip install llama-index-readers-file
pip install python-dotenv
pip install llama-index-llms-ollama
pip install llama-index-embeddings-huggingface

