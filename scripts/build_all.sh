#!/bin/bash
# Build all aviation knowledge base collections
set -e

cd "$(dirname "$0")/.."

echo "=== Building Aviation RAG Knowledge Bases ==="

# Build aircraft structure collection
if [ -d "data/aircraft_structure" ] && [ "$(ls -A data/aircraft_structure 2>/dev/null)" ]; then
    echo ""
    echo "[1/3] Building aircraft_structure..."
    python -m storage.populate --collection aircraft_structure --source-dir data/aircraft_structure
else
    echo "[1/3] Skipping aircraft_structure (no data)"
fi

# Build flight ops collection
if [ -d "data/flight_ops" ] && [ "$(ls -A data/flight_ops 2>/dev/null)" ]; then
    echo ""
    echo "[2/3] Building flight_ops..."
    python -m storage.populate --collection flight_ops --source-dir data/flight_ops
else
    echo "[2/3] Skipping flight_ops (no data)"
fi

# Build regulations collection
if [ -d "data/regulations" ] && [ "$(ls -A data/regulations 2>/dev/null)" ]; then
    echo ""
    echo "[3/3] Building regulations..."
    python -m storage.populate --collection regulations --source-dir data/regulations
else
    echo "[3/3] Skipping regulations (no data)"
fi

echo ""
echo "=== Build complete ==="
