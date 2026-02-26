#!/bin/bash
# usage: ./deploy.sh

echo ">>> 1. Deploying decoder service (GPU)..."
modal deploy services/decoder.py

echo ">>> 2. Deploying encoder upserter service (CPU)..."
modal deploy services/encoder_cpu_upserter.py

echo ">>> 3. Deploying encoder retriever service (CPU)..."
modal deploy services/encoder_cpu_retriever.py

echo ">>> 4. Deploying encoder upserter service (GPU)..."
modal deploy services/encoder_gpu_upserter.py

echo ">>> 5. Deploying encoder retriever service (GPU)..."
modal deploy services/encoder_gpu_retriever.py

echo ">>> 6. Deploying crawler agent (yearly cron)..."
modal deploy services/crawler_agent.py

echo ">>> 7. Deploying email agent (daily cron)..."
modal deploy services/email_agent.py

echo ">>> Done! Full fleet is live."
