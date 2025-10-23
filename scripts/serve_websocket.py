#!/usr/bin/env python3
"""
WebSocket server for real-time ASR inference using EzSpeech models.
"""

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Dict, Optional
from aiohttp import web

import torch
import torchaudio
import websockets
from hydra import compose, initialize
from omegaconf import DictConfig

from Ezspeech.ezspeech.models.ctc import ASR_ctc_training

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ASRWebSocketServer:
    """WebSocket server for real-time ASR."""

    def __init__(self, model_path: str, config_path: str, config_name: str = "ctc_sc"):
        """Initialize the ASR WebSocket server."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        # Load configuration
        with initialize(version_base=None, config_path=config_path):
            self.config = compose(config_name=config_name)

        # Load model
        logger.info(f"Loading model from: {model_path}")
        self.model = ASR_ctc_training.load_from_checkpoint(
            model_path, config=self.config
        )
        self.model.eval()
        self.model.to(self.device)

        # Model settings
        self.sample_rate = 16000
        self.chunk_duration = 1.0  # Process 1 second chunks
        self.chunk_samples = int(self.sample_rate * self.chunk_duration)

        # Health status
        self.is_ready = True
        self.start_time = time.time()
        self.request_count = 0

        logger.info("ASR WebSocket server initialized successfully")

    async def handle_client(self, websocket, path):
        """Handle WebSocket client connection."""
        client_id = f"{websocket.remote_address[0]}:{websocket.remote_address[1]}"
        logger.info(f"Client connected: {client_id}")

        try:
            # Send welcome message
            await websocket.send(json.dumps({
                "type": "connection",
                "status": "connected",
                "message": "ASR WebSocket server ready",
                "config": {
                    "sample_rate": self.sample_rate,
                    "chunk_duration": self.chunk_duration,
                    "device": str(self.device)
                }
            }))

            async for message in websocket:
                try:
                    await self.process_message(websocket, message, client_id)
                except Exception as e:
                    logger.error(f"Error processing message from {client_id}: {e}")
                    await websocket.send(json.dumps({
                        "type": "error",
                        "message": str(e)
                    }))

        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Client disconnected: {client_id}")
        except Exception as e:
            logger.error(f"Error handling client {client_id}: {e}")

    async def process_message(self, websocket, message, client_id: str):
        """Process incoming WebSocket message."""
        try:
            data = json.loads(message)
            msg_type = data.get("type")

            if msg_type == "audio_chunk":
                await self.process_audio_chunk(websocket, data, client_id)
            elif msg_type == "audio_file":
                await self.process_audio_file(websocket, data, client_id)
            elif msg_type == "config":
                await self.update_config(websocket, data, client_id)
            else:
                await websocket.send(json.dumps({
                    "type": "error",
                    "message": f"Unknown message type: {msg_type}"
                }))

        except json.JSONDecodeError:
            await websocket.send(json.dumps({
                "type": "error",
                "message": "Invalid JSON format"
            }))

    async def process_audio_chunk(self, websocket, data: Dict, client_id: str):
        """Process real-time audio chunk."""
        start_time = time.time()
        self.request_count += 1

        try:
            # Extract audio data (assuming base64 encoded)
            import base64
            audio_bytes = base64.b64decode(data["audio_data"])

            # Convert to tensor
            # Assuming 16-bit PCM audio
            import numpy as np
            audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
            audio_tensor = torch.from_numpy(audio_array).float() / 32768.0
            audio_tensor = audio_tensor.unsqueeze(0)  # Add batch dimension

            # Resample if necessary
            if data.get("sample_rate", self.sample_rate) != self.sample_rate:
                resampler = torchaudio.transforms.Resample(
                    data["sample_rate"], self.sample_rate
                )
                audio_tensor = resampler(audio_tensor)

            # Move to device
            audio_tensor = audio_tensor.to(self.device)

            # Get prediction
            with torch.no_grad():
                prediction = await self.predict_audio(audio_tensor)

            processing_time = time.time() - start_time

            # Send response
            await websocket.send(json.dumps({
                "type": "transcription",
                "chunk_id": data.get("chunk_id"),
                "text": prediction,
                "confidence": 0.95,  # Placeholder - implement confidence scoring
                "processing_time_ms": round(processing_time * 1000, 2),
                "is_final": data.get("is_final", False)
            }))

        except Exception as e:
            logger.error(f"Error processing audio chunk from {client_id}: {e}")
            await websocket.send(json.dumps({
                "type": "error",
                "message": f"Audio processing error: {str(e)}"
            }))

    async def process_audio_file(self, websocket, data: Dict, client_id: str):
        """Process complete audio file."""
        start_time = time.time()

        try:
            # Load audio file
            audio_path = data["file_path"]
            if not Path(audio_path).exists():
                raise FileNotFoundError(f"Audio file not found: {audio_path}")

            waveform, sample_rate = torchaudio.load(audio_path)

            # Resample if necessary
            if sample_rate != self.sample_rate:
                resampler = torchaudio.transforms.Resample(sample_rate, self.sample_rate)
                waveform = resampler(waveform)

            # Move to device
            waveform = waveform.to(self.device)

            # Get prediction
            with torch.no_grad():
                prediction = await self.predict_audio(waveform)

            processing_time = time.time() - start_time

            # Send response
            await websocket.send(json.dumps({
                "type": "file_transcription",
                "file_path": audio_path,
                "text": prediction,
                "duration_seconds": waveform.shape[1] / self.sample_rate,
                "processing_time_ms": round(processing_time * 1000, 2)
            }))

        except Exception as e:
            logger.error(f"Error processing audio file from {client_id}: {e}")
            await websocket.send(json.dumps({
                "type": "error",
                "message": f"File processing error: {str(e)}"
            }))

    async def predict_audio(self, waveform: torch.Tensor) -> str:
        """Predict transcription for audio waveform."""
        # Preprocess audio
        if hasattr(self.model, 'preprocessor'):
            features = self.model.preprocessor(waveform)
        else:
            # Basic mel spectrogram if no preprocessor
            mel_transform = torchaudio.transforms.MelSpectrogram(
                sample_rate=self.sample_rate,
                n_mels=80,
                n_fft=512,
                win_length=int(0.025 * self.sample_rate),
                hop_length=int(0.01 * self.sample_rate)
            ).to(self.device)
            features = mel_transform(waveform)

        # Add batch dimension if needed
        if len(features.shape) == 2:
            features = features.unsqueeze(0)

        # Forward pass through encoder
        encoded, encoded_len = self.model.encoder(features, torch.tensor([features.shape[-1]]))

        # CTC decoding
        logits = self.model.ctc_decoder(encoded)
        log_probs = torch.log_softmax(logits, dim=-1)

        # Greedy decoding
        predicted_ids = torch.argmax(log_probs, dim=-1)
        predicted_ids = predicted_ids.squeeze(0)

        # Remove blanks and consecutive duplicates
        blank_id = self.model.ctc_decoder.num_classes - 1
        prediction = []
        prev_id = None

        for token_id in predicted_ids:
            token_id = token_id.item()
            if token_id != blank_id and token_id != prev_id:
                prediction.append(token_id)
            prev_id = token_id

        # Decode tokens to text
        if hasattr(self.model, 'tokenizer_grapheme'):
            text = self.model.tokenizer_grapheme.decode(prediction)
        else:
            # Fallback - join token IDs as string
            text = " ".join(map(str, prediction))

        return text

    async def update_config(self, websocket, data: Dict, client_id: str):
        """Update server configuration."""
        try:
            if "chunk_duration" in data:
                self.chunk_duration = data["chunk_duration"]
                self.chunk_samples = int(self.sample_rate * self.chunk_duration)

            await websocket.send(json.dumps({
                "type": "config_updated",
                "config": {
                    "chunk_duration": self.chunk_duration,
                    "sample_rate": self.sample_rate
                }
            }))

        except Exception as e:
            await websocket.send(json.dumps({
                "type": "error",
                "message": f"Config update error: {str(e)}"
            }))


async def health_check(request):
    """Health check endpoint for AWS ECS/ALB."""
    return web.json_response({
        "status": "healthy",
        "timestamp": time.time()
    })


async def readiness_check(request):
    """Readiness check endpoint for AWS ECS/ALB."""
    server = request.app['asr_server']

    if not server.is_ready:
        return web.json_response({
            "status": "not_ready",
            "message": "Model is still loading"
        }, status=503)

    return web.json_response({
        "status": "ready",
        "uptime_seconds": time.time() - server.start_time,
        "device": str(server.device),
        "request_count": server.request_count
    })


async def metrics(request):
    """Metrics endpoint for monitoring."""
    server = request.app['asr_server']

    return web.json_response({
        "uptime_seconds": time.time() - server.start_time,
        "device": str(server.device),
        "request_count": server.request_count,
        "sample_rate": server.sample_rate,
        "chunk_duration": server.chunk_duration,
        "model_loaded": server.model is not None
    })


async def start_health_server(asr_server, health_port=8080):
    """Start HTTP server for health checks."""
    app = web.Application()
    app['asr_server'] = asr_server

    app.router.add_get('/health', health_check)
    app.router.add_get('/ready', readiness_check)
    app.router.add_get('/metrics', metrics)

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, '0.0.0.0', health_port)
    await site.start()

    logger.info(f"Health check server started on port {health_port}")
    return runner


async def main():
    """Main function to start the WebSocket server."""
    import argparse
    import os

    parser = argparse.ArgumentParser(description="ASR WebSocket Server")
    parser.add_argument("--model-path",
                       default=os.getenv("MODEL_PATH", ""),
                       help="Path to model checkpoint")
    parser.add_argument("--config-path",
                       default=os.getenv("CONFIG_PATH", "../config"),
                       help="Path to config directory")
    parser.add_argument("--config-name",
                       default=os.getenv("CONFIG_NAME", "ctc_sc"),
                       help="Config file name")
    parser.add_argument("--host",
                       default=os.getenv("SERVER_HOST", "0.0.0.0"),
                       help="Server host")
    parser.add_argument("--port",
                       type=int,
                       default=int(os.getenv("SERVER_PORT", "8765")),
                       help="WebSocket server port")
    parser.add_argument("--health-port",
                       type=int,
                       default=int(os.getenv("HEALTH_PORT", "8080")),
                       help="Health check server port")

    args = parser.parse_args()

    if not args.model_path:
        logger.error("Model path is required. Use --model-path or set MODEL_PATH env variable")
        return

    # Initialize ASR server
    server = ASRWebSocketServer(
        model_path=args.model_path,
        config_path=args.config_path,
        config_name=args.config_name
    )

    # Start health check server
    health_runner = await start_health_server(server, args.health_port)

    # Start WebSocket server
    logger.info(f"Starting ASR WebSocket server on {args.host}:{args.port}")
    async with websockets.serve(server.handle_client, args.host, args.port):
        logger.info("Server is running. Press Ctrl+C to stop.")
        await asyncio.Future()  # Run forever


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}")