#!/bin/bash
cd "$(dirname "$0")"
source .venv/bin/activate
echo "🚀 Launching NeuroStock..."
open "http://127.0.0.1:5000"
python3 app.py