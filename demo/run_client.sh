#!/bin/bash

# Example script to run the WebSocket ASR client
# Edit the parameters below as needed

SERVER="ws://localhost:8765"
DURATION=5
SAMPLE_RATE=16000

echo "Starting WebSocket ASR Client..."
echo "Server: $SERVER"
echo "Duration: $DURATION seconds"
echo "Sample Rate: $SAMPLE_RATE Hz"
echo ""

python demo/client.py \
    --server "$SERVER" \
    --duration "$DURATION" \
    --sample-rate "$SAMPLE_RATE"
