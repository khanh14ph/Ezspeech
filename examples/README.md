# EzSpeech Examples

This directory contains example scripts and usage demonstrations for the EzSpeech ASR toolkit.

## WebSocket Client Examples

### Prerequisites

Install the required dependencies:

```bash
pip install websockets pyaudio torchaudio
```

### Running the WebSocket Server

First, start the ASR WebSocket server:

```bash
cd scripts
python serve_websocket.py --model-path /path/to/your/model.ckpt --host 0.0.0.0 --port 8765
```

### Client Examples

#### 1. File Transcription

Transcribe a single audio file:

```bash
cd examples
python websocket_client.py --mode file --audio-file /path/to/audio.wav
```

#### 2. Real-time Microphone

Stream audio from your microphone for real-time transcription:

```bash
python websocket_client.py --mode microphone
```

This will record for 10 seconds and provide real-time transcriptions.

#### 3. File Streaming

Stream an audio file as chunks to simulate real-time processing:

```bash
python websocket_client.py --mode streaming --audio-file /path/to/audio.wav
```

### WebSocket API

The WebSocket server accepts the following message types:

#### Audio Chunk (Real-time streaming)
```json
{
  "type": "audio_chunk",
  "chunk_id": 0,
  "audio_data": "base64_encoded_audio",
  "sample_rate": 16000,
  "is_final": false
}
```

#### Audio File (Complete file processing)
```json
{
  "type": "audio_file",
  "file_path": "/path/to/audio.wav"
}
```

#### Configuration Update
```json
{
  "type": "config",
  "chunk_duration": 1.0
}
```

### Response Format

The server responds with:

#### Transcription Response
```json
{
  "type": "transcription",
  "chunk_id": 0,
  "text": "transcribed text",
  "confidence": 0.95,
  "processing_time_ms": 45.2,
  "is_final": false
}
```

#### File Transcription Response
```json
{
  "type": "file_transcription",
  "file_path": "/path/to/audio.wav",
  "text": "complete transcription",
  "duration_seconds": 5.2,
  "processing_time_ms": 120.5
}
```

#### Error Response
```json
{
  "type": "error",
  "message": "Error description"
}
```

## Usage Tips

1. **Audio Format**: The server expects 16kHz, mono, 16-bit PCM audio
2. **Chunk Size**: Default chunk duration is 1 second, adjustable via configuration
3. **Latency**: Processing time depends on model size and hardware
4. **Error Handling**: Always check response type for errors

## Customization

You can modify the client examples to:

- Add custom audio preprocessing
- Implement different streaming strategies
- Add confidence thresholding
- Save transcriptions to files
- Integrate with other applications

## Troubleshooting

### Common Issues

1. **Connection Refused**: Make sure the WebSocket server is running
2. **Audio Device Error**: Check microphone permissions and PyAudio installation
3. **Model Loading Error**: Verify model checkpoint path and configuration
4. **Performance Issues**: Consider using GPU acceleration and adjusting batch sizes