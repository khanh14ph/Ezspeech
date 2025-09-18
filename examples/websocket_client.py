#!/usr/bin/env python3
"""
Example WebSocket client for EzSpeech ASR service.
Demonstrates real-time audio streaming and file-based inference.
"""

import asyncio
import base64
import json
import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pyaudio
import websockets

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ASRWebSocketClient:
    """WebSocket client for ASR service."""

    def __init__(self, server_url: str = "ws://localhost:8765"):
        self.server_url = server_url
        self.websocket = None
        self.is_connected = False

        # Audio settings
        self.sample_rate = 16000
        self.chunk_duration = 1.0
        self.chunk_size = int(self.sample_rate * self.chunk_duration)
        self.audio_format = pyaudio.paInt16

    async def connect(self):
        """Connect to the WebSocket server."""
        try:
            self.websocket = await websockets.connect(self.server_url)
            self.is_connected = True
            logger.info(f"Connected to ASR server at {self.server_url}")

            # Wait for welcome message
            welcome_msg = await self.websocket.recv()
            welcome_data = json.loads(welcome_msg)
            logger.info(f"Server response: {welcome_data}")

        except Exception as e:
            logger.error(f"Failed to connect to server: {e}")
            self.is_connected = False

    async def disconnect(self):
        """Disconnect from the WebSocket server."""
        if self.websocket:
            await self.websocket.close()
            self.is_connected = False
            logger.info("Disconnected from server")

    async def transcribe_file(self, audio_file_path: str) -> Optional[str]:
        """Transcribe an audio file."""
        if not self.is_connected:
            logger.error("Not connected to server")
            return None

        message = {
            "type": "audio_file",
            "file_path": audio_file_path
        }

        try:
            await self.websocket.send(json.dumps(message))
            response = await self.websocket.recv()
            response_data = json.loads(response)

            if response_data["type"] == "file_transcription":
                return response_data["text"]
            elif response_data["type"] == "error":
                logger.error(f"Server error: {response_data['message']}")
                return None

        except Exception as e:
            logger.error(f"Error transcribing file: {e}")
            return None

    async def stream_audio_from_microphone(self, duration_seconds: float = 10.0):
        """Stream audio from microphone and get real-time transcriptions."""
        if not self.is_connected:
            logger.error("Not connected to server")
            return

        # Initialize PyAudio
        audio = pyaudio.PyAudio()

        try:
            # Open microphone stream
            stream = audio.open(
                format=self.audio_format,
                channels=1,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size
            )

            logger.info(f"Recording for {duration_seconds} seconds...")
            start_time = time.time()
            chunk_id = 0

            while time.time() - start_time < duration_seconds:
                # Read audio chunk
                audio_data = stream.read(self.chunk_size, exception_on_overflow=False)

                # Convert to base64
                audio_b64 = base64.b64encode(audio_data).decode('utf-8')

                # Send to server
                message = {
                    "type": "audio_chunk",
                    "chunk_id": chunk_id,
                    "audio_data": audio_b64,
                    "sample_rate": self.sample_rate,
                    "is_final": False
                }

                await self.websocket.send(json.dumps(message))

                # Try to receive response (non-blocking)
                try:
                    response = await asyncio.wait_for(
                        self.websocket.recv(), timeout=0.1
                    )
                    response_data = json.loads(response)

                    if response_data["type"] == "transcription":
                        print(f"Chunk {chunk_id}: {response_data['text']}")

                except asyncio.TimeoutError:
                    pass  # No response yet, continue

                chunk_id += 1

            # Send final chunk
            final_message = {
                "type": "audio_chunk",
                "chunk_id": chunk_id,
                "audio_data": "",
                "is_final": True
            }
            await self.websocket.send(json.dumps(final_message))

        except Exception as e:
            logger.error(f"Error during audio streaming: {e}")

        finally:
            # Clean up
            if 'stream' in locals():
                stream.stop_stream()
                stream.close()
            audio.terminate()

    async def send_audio_file_as_chunks(self, audio_file_path: str, chunk_duration: float = 1.0):
        """Send an audio file as chunks to simulate streaming."""
        if not self.is_connected:
            logger.error("Not connected to server")
            return

        try:
            import torchaudio

            # Load audio file
            waveform, sample_rate = torchaudio.load(audio_file_path)

            # Resample if necessary
            if sample_rate != self.sample_rate:
                resampler = torchaudio.transforms.Resample(sample_rate, self.sample_rate)
                waveform = resampler(waveform)

            # Convert to numpy and ensure mono
            audio_np = waveform.numpy()
            if audio_np.shape[0] > 1:
                audio_np = np.mean(audio_np, axis=0)
            else:
                audio_np = audio_np[0]

            # Convert to int16
            audio_int16 = (audio_np * 32767).astype(np.int16)

            # Split into chunks
            chunk_samples = int(self.sample_rate * chunk_duration)
            num_chunks = len(audio_int16) // chunk_samples

            logger.info(f"Sending {num_chunks} chunks from {audio_file_path}")

            for i in range(num_chunks):
                start_idx = i * chunk_samples
                end_idx = start_idx + chunk_samples
                chunk_data = audio_int16[start_idx:end_idx]

                # Convert to bytes and base64
                chunk_bytes = chunk_data.tobytes()
                chunk_b64 = base64.b64encode(chunk_bytes).decode('utf-8')

                # Send chunk
                message = {
                    "type": "audio_chunk",
                    "chunk_id": i,
                    "audio_data": chunk_b64,
                    "sample_rate": self.sample_rate,
                    "is_final": i == num_chunks - 1
                }

                await self.websocket.send(json.dumps(message))

                # Wait for response
                response = await self.websocket.recv()
                response_data = json.loads(response)

                if response_data["type"] == "transcription":
                    print(f"Chunk {i}: {response_data['text']}")
                elif response_data["type"] == "error":
                    logger.error(f"Server error: {response_data['message']}")

                # Small delay to simulate real-time
                await asyncio.sleep(chunk_duration * 0.8)

        except Exception as e:
            logger.error(f"Error sending file as chunks: {e}")


async def demo_file_transcription():
    """Demo: transcribe a single audio file."""
    client = ASRWebSocketClient()
    await client.connect()

    if client.is_connected:
        # Replace with your audio file path
        audio_file = "/path/to/your/audio.wav"

        if Path(audio_file).exists():
            logger.info(f"Transcribing file: {audio_file}")
            transcription = await client.transcribe_file(audio_file)
            if transcription:
                print(f"Transcription: {transcription}")
        else:
            logger.warning(f"Audio file not found: {audio_file}")

    await client.disconnect()


async def demo_real_time_microphone():
    """Demo: real-time microphone transcription."""
    client = ASRWebSocketClient()
    await client.connect()

    if client.is_connected:
        logger.info("Starting real-time microphone transcription...")
        await client.stream_audio_from_microphone(duration_seconds=10.0)

    await client.disconnect()


async def demo_file_streaming():
    """Demo: stream an audio file as chunks."""
    client = ASRWebSocketClient()
    await client.connect()

    if client.is_connected:
        # Replace with your audio file path
        audio_file = "/path/to/your/audio.wav"

        if Path(audio_file).exists():
            logger.info(f"Streaming file as chunks: {audio_file}")
            await client.send_audio_file_as_chunks(audio_file)
        else:
            logger.warning(f"Audio file not found: {audio_file}")

    await client.disconnect()


def main():
    """Main function with demo selection."""
    import argparse

    parser = argparse.ArgumentParser(description="ASR WebSocket Client Demo")
    parser.add_argument("--mode", choices=["file", "microphone", "streaming"],
                       default="file", help="Demo mode to run")
    parser.add_argument("--server", default="ws://localhost:8765",
                       help="WebSocket server URL")
    parser.add_argument("--audio-file", help="Audio file path (for file mode)")

    args = parser.parse_args()

    # Update client server URL
    global client
    client = ASRWebSocketClient(args.server)

    try:
        if args.mode == "file":
            asyncio.run(demo_file_transcription())
        elif args.mode == "microphone":
            asyncio.run(demo_real_time_microphone())
        elif args.mode == "streaming":
            asyncio.run(demo_file_streaming())

    except KeyboardInterrupt:
        logger.info("Demo stopped by user")
    except Exception as e:
        logger.error(f"Demo error: {e}")


if __name__ == "__main__":
    main()