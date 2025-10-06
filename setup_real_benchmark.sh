#!/bin/bash

# Install required packages for real Tier-1 benchmark
echo "Installing packages for real Tier-1 benchmark..."

# Core IR libraries
pip install beir
pip install pyserini
pip install pytrec-eval

# Sentence transformers for baselines
pip install sentence-transformers

# Statistical testing
pip install scipy

# Additional utilities
pip install datasets
pip install faiss-cpu  # For efficient similarity search

echo "Package installation complete!"

# Download MS MARCO dataset
echo "Downloading MS MARCO dev dataset..."
mkdir -p datasets/msmarco/dev
cd datasets/msmarco/dev

# Download queries, qrels, and collection
wget https://msmarco.blob.core.windows.net/msmarcoranking/queries.dev.small.tsv
wget https://msmarco.blob.core.windows.net/msmarcoranking/qrels.dev.small.tsv  
wget https://msmarco.blob.core.windows.net/msmarcoranking/collection.tsv

echo "MS MARCO download complete!"

# Pre-download some BEIR datasets
echo "Pre-downloading key BEIR datasets..."
cd ../../../

python -c "
from beir import util
datasets = ['nfcorpus', 'scifact', 'nq', 'hotpotqa'] 
for dataset in datasets:
    print(f'Downloading {dataset}...')
    url = f'https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip'
    util.download_and_unzip(url, 'datasets/beir')
    print(f'{dataset} downloaded!')
"

echo "Setup complete! Ready to run real Tier-1 benchmark."