# WebSocket ASR Demo

A real-time speech recognition demo using WebSocket for client-server communication. The client records audio from the microphone for 5 seconds and continuously streams it to the server, which performs ASR inference and returns the transcription.

## Architecture

```
┌─────────────┐                    ┌─────────────┐
│   Client    │  WebSocket Stream  │   Server    │
│             │ ──────────────────>│             │
│  Microphone │    Audio Chunks    │  ASR Model  │
│  Recording  │                    │  Inference  │
│             │<────────────────── │             │
│   Display   │   Transcription    │   Result    │
└─────────────┘                    └─────────────┘
```

## Features

- Real-time audio streaming via WebSocket
- Continuous audio capture from microphone
- 5-second recording duration (configurable)
- Server-side ASR inference
- Clean CLI interface with progress indicators

## Requirements

Install the required dependencies:

```bash
pip install pyaudio websockets numpy torch torchaudio
```

For macOS users, you may need to install portaudio first:

```bash
brew install portaudio
pip install pyaudio
```

## Usage

### 1. Start the Server

First, start the WebSocket server with your trained model:

```bash
python demo/server.py \
    --checkpoint /path/to/checkpoint.ckpt \
    --tokenizer /path/to/tokenizer.model \
    --host localhost \
    --port 8765
```

**Server Arguments:**
- `--checkpoint`: Path to your ASR model checkpoint (required)
- `--tokenizer`: Path to your tokenizer model file (required)
- `--host`: Server host address (default: localhost)
- `--port`: Server port (default: 8765)

Example:
```bash
python demo/server.py \
    --checkpoint /Users/khanh/dev/asr_dev/ckpt.ckpt \
    --tokenizer /Users/khanh/dev/asr_dev/tokenizer.model
```

### 2. Run the Client

In a new terminal, run the client to start recording:

```bash
python demo/client.py \
    --server ws://localhost:8765 \
    --duration 5 \
    --sample-rate 16000
```

**Client Arguments:**
- `--server`: WebSocket server URL (default: ws://localhost:8765)
- `--duration`: Recording duration in seconds (default: 5)
- `--sample-rate`: Audio sample rate in Hz (default: 16000)

Example:
```bash
python demo/client.py --duration 10
```

### 3. Speak and Get Results

Once the client is running:

1. You'll see: "🎤 Speak now! Recording for 5 seconds..."
2. Speak clearly into your microphone
3. The audio is streamed in real-time to the server
4. After 5 seconds, the server processes the audio
5. The transcription result is displayed

Example output:
```
============================================================
  WebSocket ASR Client
============================================================

✓ Connected to server at ws://localhost:8765

🎤 Speak now! Recording for 5 seconds...
Recording... 0.0s remaining
✓ Recording complete! Processing...

⏳ Waiting for transcription...

============================================================
📝 Transcription: xin chào tôi là trợ lý ảo
============================================================
```

## How It Works

### Client Side
1. Connects to WebSocket server
2. Opens microphone stream using PyAudio
3. Records audio in chunks (1024 frames per chunk)
4. Continuously sends audio chunks to server as JSON messages
5. After recording duration, sends "audio_end" signal
6. Waits for and displays transcription result

### Server Side
1. Loads ASR model on startup
2. Accepts WebSocket connections from clients
3. Accumulates audio chunks from client
4. When "audio_end" is received:
   - Combines all chunks into single audio array
   - Saves to temporary WAV file
   - Runs ASR inference using the model
   - Returns transcription to client
5. Cleans up temporary files

## Message Protocol

### Client → Server

**Audio Chunk:**
```json
{
  "type": "audio_chunk",
  "data": [0.1, -0.2, ...],
  "sample_rate": 16000
}
```

**End of Audio:**
```json
{
  "type": "audio_end"
}
```

### Server → Client

**Ready Signal:**
```json
{
  "status": "ready",
  "message": "Server ready to receive audio"
}
```

**Success Response:**
```json
{
  "status": "success",
  "transcription": "xin chào",
  "message": "Processed 80 chunks"
}
```

**Error Response:**
```json
{
  "status": "error",
  "message": "Error description"
}
```

## Troubleshooting

### PyAudio Installation Issues

**macOS:**
```bash
brew install portaudio
pip install pyaudio
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get install portaudio19-dev python3-pyaudio
pip install pyaudio
```

**Windows:**
```bash
pip install pipwin
pipwin install pyaudio
```

### Connection Refused

Make sure the server is running before starting the client. Check that:
- Server started successfully without errors
- Host and port match between server and client
- No firewall blocking the connection

### No Audio Input

Check your microphone:
- Verify microphone is connected and working
- Check system audio input settings
- Grant microphone permission to terminal/Python

### Model Loading Errors

Ensure:
- Checkpoint file exists and is valid
- Tokenizer file exists and matches the model
- Model was trained with compatible version

## Customization

### Change Recording Duration

```bash
python demo/client.py --duration 10  # Record for 10 seconds
```

### Use Different Port

Server:
```bash
python demo/server.py --checkpoint ... --tokenizer ... --port 9000
```

Client:
```bash
python demo/client.py --server ws://localhost:9000
```

### Change Sample Rate

```bash
python demo/client.py --sample-rate 22050
```

Note: Make sure your model was trained with the same sample rate!

## Performance Tips

- Use GPU for faster inference (server will automatically use CUDA if available)
- Adjust chunk size in client.py for different latency/bandwidth tradeoffs
- For production, consider adding authentication and SSL/TLS encryption

## License

This demo is part of the Ezspeech project.
