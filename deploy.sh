#!/bin/bash
# usage: ./deploy.sh

echo ">>> 1. Deploying decoder service (GPU)..."
modal deploy services/decoder.py

echo ">>> 2. Deploying encoder service (GPU)..."
modal deploy services/encoder.py

echo ">>> 3. Deploying crawler agent (yearly cron)..."
modal deploy services/crawler_agent.py

echo ">>> 4. Deploying email agent (daily cron)..."
modal deploy services/email_agent.py

echo ">>> Done! Full fleet is live."