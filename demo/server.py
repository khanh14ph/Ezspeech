#!/usr/bin/env python3
"""
WebSocket Server for Real-time ASR
Receives audio streams from clients, performs inference, and returns transcription results.
"""

import argparse
import asyncio
import json
import logging
import os
import tempfile
from pathlib import Path

import torch
import torchaudio
import websockets
from websockets.server import WebSocketServerProtocol

from ezspeech.models.ctc import ASR_ctc_inference

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ASRServer:
    def __init__(self, checkpoint_path: str, tokenizer_path: str):
        """Initialize ASR server with model"""
        self.checkpoint_path = checkpoint_path
        self.tokenizer_path = tokenizer_path
        self.model = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def load_model(self):
        """Load ASR model"""
        logger.info(f"Loading model on device: {self.device}")
        logger.info(f"Checkpoint: {self.checkpoint_path}")
        logger.info(f"Tokenizer: {self.tokenizer_path}")

        self.model = ASR_ctc_inference(
            filepath=self.checkpoint_path,
            device=self.device,
            tokenizer_path=self.tokenizer_path
        )
        logger.info("Model loaded successfully")

    async def handle_client(self, websocket: WebSocketServerProtocol):
        """Handle a client connection"""
        client_id = f"{websocket.remote_address[0]}:{websocket.remote_address[1]}"
        logger.info(f"Client connected: {client_id}")

        try:
            # Send ready signal
            await websocket.send(json.dumps({"status": "ready", "message": "Server ready to receive audio"}))

            # Collect audio chunks
            audio_chunks = []
            sample_rate = None

            async for message in websocket:
                try:
                    data = json.loads(message)

                    if data.get("type") == "audio_chunk":
                        # Receive audio chunk
                        chunk_data = data.get("data")
                        if sample_rate is None:
                            sample_rate = data.get("sample_rate", 16000)
                            logger.info(f"Receiving audio at {sample_rate} Hz")

                        audio_chunks.append(chunk_data)
                        logger.debug(f"Received chunk {len(audio_chunks)}, size: {len(chunk_data)}")

                    elif data.get("type") == "audio_end":
                        # Client finished sending audio
                        logger.info(f"Client finished sending audio. Total chunks: {len(audio_chunks)}")

                        if not audio_chunks:
                            await websocket.send(json.dumps({
                                "status": "error",
                                "message": "No audio data received"
                            }))
                            continue

                        # Process audio and get transcription
                        try:
                            transcription = await self.process_audio(audio_chunks, sample_rate)

                            # Send result
                            await websocket.send(json.dumps({
                                "status": "success",
                                "transcription": transcription,
                                "message": f"Processed {len(audio_chunks)} chunks"
                            }))
                            logger.info(f"Transcription: {transcription}")

                        except Exception as e:
                            logger.error(f"Error processing audio: {e}", exc_info=True)
                            await websocket.send(json.dumps({
                                "status": "error",
                                "message": f"Error processing audio: {str(e)}"
                            }))

                        # Reset for next recording
                        audio_chunks = []
                        sample_rate = None

                except json.JSONDecodeError as e:
                    logger.error(f"Invalid JSON received: {e}")
                    await websocket.send(json.dumps({
                        "status": "error",
                        "message": "Invalid JSON format"
                    }))

        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Client disconnected: {client_id}")
        except Exception as e:
            logger.error(f"Error handling client {client_id}: {e}", exc_info=True)

    async def process_audio(self, audio_chunks: list, sample_rate: int) -> str:
        """Process audio chunks and return transcription"""
        # Combine all chunks into a single audio array
        import numpy as np

        # Flatten all chunks
        audio_data = []
        for chunk in audio_chunks:
            audio_data.extend(chunk)

        # Convert to numpy array
        audio_array = np.array(audio_data, dtype=np.float32)

        # Save to temporary WAV file
        # with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
        #     tmp_path = tmp_file.name
        tmp_path="/Users/khanh/dev/temp/1.wav"
        try:
            # Convert to torch tensor and save
            audio_tensor = torch.from_numpy(audio_array).unsqueeze(0)  # Add channel dimension

            # Resample to 16kHz if needed (most ASR models expect 16kHz)
            target_sample_rate = 16000
            if sample_rate != target_sample_rate:
                logger.info(f"Resampling audio from {sample_rate}Hz to {target_sample_rate}Hz")
                resampler = torchaudio.transforms.Resample(
                    orig_freq=sample_rate,
                    new_freq=target_sample_rate
                )
                audio_tensor = resampler(audio_tensor)
                sample_rate = target_sample_rate

            torchaudio.save(tmp_path, audio_tensor, sample_rate)
            print(audio_tensor.shape)
            logger.info(f"Saved audio to {tmp_path}, duration: {len(audio_tensor)/sample_rate:.2f}s")

            # Run inference
            transcriptions = self.model.transcribe([tmp_path])
            transcription = transcriptions[0] if transcriptions else ""

            return transcription

        finally:
            # Clean up temp file
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    async def start(self, host: str, port: int):
        """Start the WebSocket server"""
        # Load model first
        self.load_model()

        # Start server
        logger.info(f"Starting server on {host}:{port}")
        async with websockets.serve(self.handle_client, host, port):
            logger.info(f"Server running on ws://{host}:{port}")
            logger.info("Press Ctrl+C to stop")
            await asyncio.Future()  # Run forever


def main():
    parser = argparse.ArgumentParser(description='WebSocket ASR Server')

    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint file'
    )

    parser.add_argument(
        '--tokenizer',
        type=str,
        required=True,
        help='Path to tokenizer model file'
    )

    parser.add_argument(
        '--host',
        type=str,
        default='0.0.0.0',
        help='Server host (default: localhost)'
    )

    parser.add_argument(
        '--port',
        type=int,
        default=8765,
        help='Server port (default: 8765)'
    )

    args = parser.parse_args()

    # Create and start server
    server = ASRServer(args.checkpoint, args.tokenizer)

    try:
        asyncio.run(server.start(args.host, args.port))
    except KeyboardInterrupt:
        logger.info("Server stopped by user")


if __name__ == "__main__":
    main()
