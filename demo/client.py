#!/usr/bin/env python3
"""
WebSocket Client for Real-time ASR
Records audio from microphone and streams it to the server for transcription.
"""

import argparse
import asyncio
import json
import logging
import sys
import time

import numpy as np
import pyaudio
import websockets

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ASRClient:
    def __init__(self, server_url: str, duration: int = 5, sample_rate: int = 16000):
        """Initialize ASR client"""
        self.server_url = server_url
        self.duration = duration
        self.sample_rate = sample_rate
        self.chunk_size = 1024  # Number of frames per chunk
        self.format = pyaudio.paFloat32
        self.channels = 1

    async def record_and_stream(self, websocket):
        """Record audio from microphone and stream to server"""
        # Initialize PyAudio
        p = pyaudio.PyAudio()

        # Open stream
        try:
            stream = p.open(
                format=self.format,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size
            )

            logger.info(f"Recording for {self.duration} seconds...")
            print(f"\n🎤 Speak now! Recording for {self.duration} seconds...")

            # Calculate number of chunks to record
            num_chunks = int(self.sample_rate / self.chunk_size * self.duration)

            # Record and stream
            start_time = time.time()
            for i in range(num_chunks):
                # Read audio data
                data = stream.read(self.chunk_size, exception_on_overflow=False)

                # Convert to numpy array
                audio_chunk = np.frombuffer(data, dtype=np.float32).tolist()

                # Send to server
                message = json.dumps({
                    "type": "audio_chunk",
                    "data": audio_chunk,
                    "sample_rate": self.sample_rate
                })
                await websocket.send(message)

                # Show progress
                elapsed = time.time() - start_time
                remaining = self.duration - elapsed
                if i % 10 == 0:  # Update every 10 chunks
                    print(f"\rRecording... {remaining:.1f}s remaining", end='', flush=True)

            print("\n✓ Recording complete! Processing...")

            # Signal end of audio
            await websocket.send(json.dumps({"type": "audio_end"}))

        finally:
            # Clean up
            stream.stop_stream()
            stream.close()
            p.terminate()

    async def connect_and_record(self):
        """Connect to server and record audio"""
        try:
            async with websockets.connect(self.server_url) as websocket:
                logger.info(f"Connected to server: {self.server_url}")

                # Wait for ready signal
                ready_msg = await websocket.recv()
                ready_data = json.loads(ready_msg)

                if ready_data.get("status") == "ready":
                    logger.info("Server is ready")
                    print(f"\n✓ Connected to server at {self.server_url}")

                    # Record and stream audio
                    await self.record_and_stream(websocket)

                    # Wait for result
                    logger.info("Waiting for transcription result...")
                    print("\n⏳ Waiting for transcription...")

                    result_msg = await websocket.recv()
                    result_data = json.loads(result_msg)

                    # Display result
                    if result_data.get("status") == "success":
                        transcription = result_data.get("transcription", "")
                        print(f"\n{'='*60}")
                        print(f"📝 Transcription: {transcription}")
                        print(f"{'='*60}\n")
                        logger.info(f"Transcription: {transcription}")
                    else:
                        error_msg = result_data.get("message", "Unknown error")
                        print(f"\n❌ Error: {error_msg}\n")
                        logger.error(f"Server error: {error_msg}")

                else:
                    logger.error("Server not ready")
                    print(f"\n❌ Server not ready: {ready_data.get('message')}\n")

        except websockets.exceptions.ConnectionRefused:
            logger.error(f"Connection refused. Is the server running at {self.server_url}?")
            print(f"\n❌ Cannot connect to server at {self.server_url}")
            print("   Make sure the server is running.\n")
        except Exception as e:
            logger.error(f"Error: {e}", exc_info=True)
            print(f"\n❌ Error: {e}\n")

    def run(self):
        """Run the client"""
        try:
            asyncio.run(self.connect_and_record())
        except KeyboardInterrupt:
            logger.info("Client stopped by user")
            print("\n\n⚠️  Recording cancelled by user\n")


def main():
    parser = argparse.ArgumentParser(
        description='WebSocket ASR Client - Record and transcribe audio'
    )

    parser.add_argument(
        '--server',
        type=str,
        default='ws://192.168.0.192:8765',
        help='WebSocket server URL (default: ws://localhost:8765)'
    )

    parser.add_argument(
        '--duration',
        type=int,
        default=5,
        help='Recording duration in seconds (default: 5)'
    )

    parser.add_argument(
        '--sample-rate',
        type=int,
        default=16000,
        help='Audio sample rate (default: 16000)'
    )

    args = parser.parse_args()

    # Check if PyAudio is available
    try:
        import pyaudio
    except ImportError:
        print("\n❌ Error: PyAudio is not installed")
        print("   Install it with: pip install pyaudio")
        print("   On macOS, you may need: brew install portaudio && pip install pyaudio\n")
        sys.exit(1)

    # Create and run client
    print("\n" + "="*60)
    print("  WebSocket ASR Client")
    print("="*60)

    client = ASRClient(
        server_url=args.server,
        duration=args.duration,
        sample_rate=args.sample_rate
    )
    client.run()


if __name__ == "__main__":
    main()
