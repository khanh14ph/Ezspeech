#!/bin/bash

# Example script to run the WebSocket ASR server
# Edit the paths below to match your setup

CHECKPOINT="/Users/khanh/dev/asr_dev/ckpt.ckpt"
TOKENIZER="/Users/khanh/dev/asr_dev/tokenizer.model"
HOST="localhost"
PORT=8765

echo "Starting WebSocket ASR Server..."
echo "Checkpoint: $CHECKPOINT"
echo "Tokenizer: $TOKENIZER"
echo "Server: ws://$HOST:$PORT"
echo ""

python demo/server.py \
    --checkpoint "$CHECKPOINT" \
    --tokenizer "$TOKENIZER" \
    --host "$HOST" \
    --port "$PORT"
