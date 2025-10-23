# EzSpeech Deployment Guide

This document provides quick deployment instructions for EzSpeech ASR server.

## Deployment Options

### 1. AWS Deployment (Recommended for Production)

Deploy to AWS using ECS Fargate with auto-scaling, load balancing, and monitoring.

**Full documentation**: [aws/README.md](aws/README.md)

**Quick start:**

```bash
# 1. Configure infrastructure
cd aws/terraform
cp terraform.tfvars.example terraform.tfvars
# Edit terraform.tfvars with your settings

# 2. Deploy infrastructure
cd ../..
./aws/scripts/deploy.sh

# 3. Build and push Docker image
./aws/scripts/build-and-push.sh

# 4. Upload your model
./aws/scripts/upload-model.sh /path/to/model.ckpt

# 5. Update ECS service
./aws/scripts/update-service.sh
```

**Architecture:**
- ECS Fargate for containerized deployment
- Application Load Balancer for WebSocket support
- Auto-scaling based on CPU/memory
- CloudWatch for logging and monitoring
- S3 for model storage
- VPC with public/private subnets

**Cost**: ~$400-1,500/month depending on usage

### 2. Docker Compose (Local/Development)

Run locally using Docker Compose for development and testing.

**Prerequisites:**
- Docker and Docker Compose installed
- Trained model checkpoint

**Start services:**

```bash
# Ensure you have a model file
mkdir -p models
cp /path/to/your/model.ckpt models/

# Start the server
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f

# Test
curl http://localhost:8080/health
```

**With Nginx reverse proxy:**

```bash
docker-compose --profile with-nginx up -d
```

**Stop services:**

```bash
docker-compose down
```

### 3. Manual Docker Deployment

Build and run the Docker container manually.

**Build image:**

```bash
docker build -t ezspeech:latest -f Dockerfile .
```

**Run container:**

```bash
docker run -d \
  --name ezspeech-server \
  -p 8765:8765 \
  -p 8080:8080 \
  -v $(pwd)/models:/app/models:ro \
  -v $(pwd)/config:/app/config:ro \
  -e MODEL_PATH=/app/models/model.ckpt \
  -e CONFIG_NAME=ctc_sc \
  ezspeech:latest
```

**For GPU support:**

```bash
docker run -d \
  --name ezspeech-server \
  --gpus all \
  -p 8765:8765 \
  -p 8080:8080 \
  -v $(pwd)/models:/app/models:ro \
  -v $(pwd)/config:/app/config:ro \
  -e MODEL_PATH=/app/models/model.ckpt \
  -e CONFIG_NAME=ctc_sc \
  ezspeech:latest
```

### 4. Kubernetes Deployment (Advanced)

For Kubernetes deployments, example manifests:

**deployment.yaml:**

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ezspeech-server
spec:
  replicas: 2
  selector:
    matchLabels:
      app: ezspeech
  template:
    metadata:
      labels:
        app: ezspeech
    spec:
      containers:
      - name: ezspeech
        image: your-registry/ezspeech:latest
        ports:
        - containerPort: 8765
          name: websocket
        - containerPort: 8080
          name: health
        env:
        - name: MODEL_PATH
          value: /app/models/model.ckpt
        - name: CONFIG_NAME
          value: ctc_sc
        volumeMounts:
        - name: models
          mountPath: /app/models
          readOnly: true
        resources:
          requests:
            memory: "4Gi"
            cpu: "2"
          limits:
            memory: "8Gi"
            cpu: "4"
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 60
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /ready
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
      volumes:
      - name: models
        persistentVolumeClaim:
          claimName: ezspeech-models
```

**service.yaml:**

```yaml
apiVersion: v1
kind: Service
metadata:
  name: ezspeech-service
spec:
  type: LoadBalancer
  selector:
    app: ezspeech
  ports:
  - name: websocket
    port: 8765
    targetPort: 8765
  - name: health
    port: 8080
    targetPort: 8080
```

### 5. Direct Python Deployment (Not Recommended for Production)

Run directly with Python for development:

```bash
# Install dependencies
pip install -r requirements.txt

# Install package
pip install -e .

# Run server
python scripts/serve_websocket.py \
  --model-path /path/to/model.ckpt \
  --config-path ./config \
  --config-name ctc_sc \
  --host 0.0.0.0 \
  --port 8765
```

## Health Check Endpoints

All deployments expose these endpoints on port 8080:

- **GET /health**: Basic health check (returns 200 if alive)
- **GET /ready**: Readiness check (returns 200 if model loaded)
- **GET /metrics**: Metrics endpoint (JSON with server stats)

**Example:**

```bash
# Health check
curl http://localhost:8080/health

# Readiness check
curl http://localhost:8080/ready

# Metrics
curl http://localhost:8080/metrics
```

## Environment Variables

Configure the server using environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | - | Path to model checkpoint (required) |
| `CONFIG_PATH` | `../config` | Path to configuration directory |
| `CONFIG_NAME` | `ctc_sc` | Configuration file name |
| `SERVER_HOST` | `0.0.0.0` | Server bind address |
| `SERVER_PORT` | `8765` | WebSocket server port |
| `HEALTH_PORT` | `8080` | Health check server port |
| `LOG_LEVEL` | `INFO` | Logging level (DEBUG, INFO, WARNING, ERROR) |

## Testing the Deployment

### 1. Test Health Endpoint

```bash
curl http://localhost:8080/health
# Expected: {"status": "healthy", "timestamp": 1234567890.123}
```

### 2. Test WebSocket Connection

Use the example client:

```bash
cd examples
python websocket_client.py --server ws://localhost:8765
```

Or use `websocat`:

```bash
# Install websocat
# brew install websocat  # macOS
# sudo apt install websocat  # Ubuntu

# Connect
websocat ws://localhost:8765
```

### 3. Send Test Audio

```bash
# Prepare test audio (16kHz, mono, WAV)
ffmpeg -i input.mp3 -ar 16000 -ac 1 test.wav

# Send to server (using example client)
python examples/websocket_client.py \
  --server ws://localhost:8765 \
  --audio-file test.wav
```

## Monitoring

### Docker Compose

```bash
# View logs
docker-compose logs -f ezspeech-server

# Check container stats
docker stats ezspeech-server
```

### AWS ECS

```bash
# View logs
aws logs tail /ecs/ezspeech-asr-server --follow

# Check service status
aws ecs describe-services \
  --cluster ezspeech-cluster \
  --services ezspeech-service

# View metrics
aws cloudwatch get-metric-statistics \
  --namespace AWS/ECS \
  --metric-name CPUUtilization \
  --dimensions Name=ServiceName,Value=ezspeech-service
```

## Troubleshooting

### Container won't start

```bash
# Check logs
docker logs ezspeech-server

# Common issues:
# 1. Model file not found
#    Solution: Ensure MODEL_PATH is correct and file exists
# 2. Out of memory
#    Solution: Increase container memory limit
# 3. Port already in use
#    Solution: Change port mapping or stop conflicting service
```

### High latency

```bash
# Check if GPU is available
docker exec ezspeech-server python -c "import torch; print(torch.cuda.is_available())"

# If False:
# - Enable GPU support with --gpus all
# - Or accept CPU-only performance
# - Or use smaller model
```

### WebSocket connection refused

```bash
# Check if server is listening
docker exec ezspeech-server netstat -tuln | grep 8765

# Test from within container
docker exec ezspeech-server curl http://localhost:8080/health

# Check firewall rules
# AWS: Check security groups
# Local: Check iptables/firewall
```

## Security Considerations

For production deployments:

1. **Use HTTPS/WSS**: Deploy behind a reverse proxy with SSL
2. **Authentication**: Add API key or JWT authentication
3. **Rate Limiting**: Implement rate limiting to prevent abuse
4. **Network Isolation**: Use private networks where possible
5. **Secrets Management**: Use AWS Secrets Manager or similar
6. **Regular Updates**: Keep dependencies and base images updated
7. **Monitoring**: Set up alerts for unusual activity

## Performance Optimization

### For CPU-only inference:
- Use ONNX Runtime for faster inference
- Enable PyTorch JIT compilation
- Use quantized models (INT8)
- Batch requests when possible

### For GPU inference:
- Use TensorRT for optimization
- Enable mixed precision (FP16)
- Optimize batch size
- Use GPU with sufficient VRAM (8GB+)

### General:
- Use async/await for I/O operations
- Implement connection pooling
- Cache frequently used data
- Monitor and optimize based on metrics

## Cost Optimization (AWS)

- Use Fargate Spot for non-critical workloads (70% savings)
- Enable auto-scaling to match demand
- Use CloudWatch Logs filtering to reduce storage
- Implement intelligent model caching
- Use reserved capacity for predictable workloads
- Clean up old ECR images regularly

## Next Steps

After successful deployment:

1. **Set up monitoring**: Configure CloudWatch alarms or Prometheus
2. **Implement CI/CD**: Automate builds and deployments
3. **Add authentication**: Secure your endpoints
4. **Load testing**: Test with expected traffic patterns
5. **Documentation**: Document your specific configuration
6. **Backup**: Set up model and configuration backups
7. **Disaster recovery**: Plan for failures and recovery

## Support

- **AWS Deployment**: See [aws/README.md](aws/README.md)
- **Issues**: Report at [GitHub Issues](https://github.com/your-repo/issues)
- **Documentation**: See [docs/](docs/)

## License

See main project LICENSE file.
