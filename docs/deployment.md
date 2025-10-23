---
layout: page
title: Deployment Guide
permalink: /deployment/
---

# EzSpeech Deployment Guide

Deploy EzSpeech ASR server to production with multiple deployment options.

## Quick Navigation

- [AWS Deployment](#aws-deployment) - Recommended for production
- [Docker Compose](#docker-compose) - Local development
- [Kubernetes](#kubernetes) - Advanced orchestration
- [Manual Deployment](#manual-deployment) - Direct installation

---

## AWS Deployment

Deploy to AWS with auto-scaling, load balancing, and monitoring using ECS Fargate.

### Architecture

```
Internet
   ↓
Application Load Balancer (ALB)
   ↓
ECS Fargate Cluster (Auto-scaling 1-10 tasks)
   ↓
S3 (Model Storage) + CloudWatch (Monitoring)
```

### Features

- ✅ Auto-scaling based on CPU/memory
- ✅ Load balancing with WebSocket support
- ✅ Health checks and graceful shutdowns
- ✅ CloudWatch logging and monitoring
- ✅ Infrastructure as Code (Terraform)
- ✅ Cost-optimized (~$400-1,500/month)

### Prerequisites

- AWS Account with appropriate permissions
- AWS CLI configured (`aws configure`)
- Terraform installed (v1.0+)
- Docker installed
- Trained ASR model checkpoint

### Quick Start

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

### Configuration

Edit `aws/terraform/terraform.tfvars`:

```hcl
# AWS Configuration
aws_region  = "us-east-1"
environment = "prod"

# Compute Resources
task_cpu    = "2048"  # 2 vCPU
task_memory = "4096"  # 4 GB

# Auto-scaling
desired_count = 2
min_capacity  = 1
max_capacity  = 10

# Scaling thresholds
cpu_target_value    = 70  # %
memory_target_value = 70  # %
```

### Access Your Deployment

```bash
# Get ALB DNS name
cd aws/terraform
terraform output alb_dns_name

# Test health endpoint
curl http://YOUR_ALB_DNS/health

# Test WebSocket
wscat -c ws://YOUR_ALB_DNS:80
```

### Monitoring

**View Logs:**
```bash
aws logs tail /ecs/ezspeech-asr-server --follow
```

**Check Service Status:**
```bash
aws ecs describe-services \
  --cluster ezspeech-cluster \
  --services ezspeech-service
```

**View Metrics:**
```bash
# In AWS Console: CloudWatch > Dashboards
# Metrics: CPU, Memory, Request Count, Latency
```

### Cost Optimization

- Use **Fargate Spot** for 70% savings (non-critical workloads)
- Enable **auto-scaling** to match demand
- Use **Compute Savings Plans** for predictable workloads
- Clean up old **ECR images** regularly

**Full AWS Guide**: [aws/README.md](https://github.com/khanh14ph/Ezspeech/blob/main/aws/README.md)

---

## Docker Compose

Quick local deployment for development and testing.

### Prerequisites

- Docker and Docker Compose installed
- Trained model checkpoint

### Quick Start

```bash
# 1. Prepare model
mkdir -p models
cp /path/to/your/model.ckpt models/

# 2. Start services
docker-compose up -d

# 3. Check status
docker-compose ps

# 4. View logs
docker-compose logs -f
```

### Test Deployment

```bash
# Health check
curl http://localhost:8080/health
# Response: {"status": "healthy", "timestamp": ...}

# WebSocket connection
wscat -c ws://localhost:8765

# Or use Python client
cd examples
python websocket_client.py --server ws://localhost:8765
```

### With Nginx Reverse Proxy

```bash
docker-compose --profile with-nginx up -d
```

Access via: `http://localhost` (port 80)

### Configuration

Edit `docker-compose.yml`:

```yaml
environment:
  - MODEL_PATH=/app/models/model.ckpt
  - CONFIG_NAME=ctc_sc
  - SERVER_PORT=8765
  - LOG_LEVEL=INFO
```

### GPU Support

Uncomment GPU section in `docker-compose.yml`:

```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: 1
          capabilities: [gpu]
```

Requires: **nvidia-docker** installed

---

## Kubernetes

Advanced deployment with full orchestration capabilities.

### Prerequisites

- Kubernetes cluster (EKS, GKE, AKS, or local)
- kubectl configured
- Container registry (ECR, Docker Hub, etc.)

### Deployment Manifests

**1. Deployment:**

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ezspeech-server
spec:
  replicas: 3
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
```

**2. Service:**

```yaml
# k8s/service.yaml
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

**3. Horizontal Pod Autoscaler:**

```yaml
# k8s/hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: ezspeech-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: ezspeech-server
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 70
```

### Deploy

```bash
# Apply manifests
kubectl apply -f k8s/

# Check status
kubectl get pods
kubectl get svc

# Get external IP
kubectl get svc ezspeech-service

# View logs
kubectl logs -f deployment/ezspeech-server
```

### With Ingress

```yaml
# k8s/ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: ezspeech-ingress
  annotations:
    kubernetes.io/ingress.class: nginx
    nginx.ingress.kubernetes.io/websocket-services: ezspeech-service
spec:
  rules:
  - host: asr.yourdomain.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: ezspeech-service
            port:
              number: 8765
```

---

## Manual Deployment

Direct deployment without containers.

### Prerequisites

- Python 3.11+
- PyTorch 2.0+
- CUDA 11.8+ (for GPU support)

### Installation

```bash
# 1. Clone repository
git clone https://github.com/khanh14ph/Ezspeech.git
cd Ezspeech

# 2. Install dependencies
pip install -r requirements.txt

# 3. Install package
pip install -e .
```

### Run Server

```bash
python scripts/serve_websocket.py \
  --model-path /path/to/model.ckpt \
  --config-path ./config \
  --config-name ctc_sc \
  --host 0.0.0.0 \
  --port 8765 \
  --health-port 8080
```

### Run as systemd Service (Linux)

**Create service file:**

```bash
sudo nano /etc/systemd/system/ezspeech.service
```

```ini
[Unit]
Description=EzSpeech ASR Server
After=network.target

[Service]
Type=simple
User=www-data
WorkingDirectory=/opt/ezspeech
Environment="MODEL_PATH=/opt/ezspeech/models/model.ckpt"
Environment="CONFIG_PATH=/opt/ezspeech/config"
Environment="CONFIG_NAME=ctc_sc"
ExecStart=/usr/bin/python3 /opt/ezspeech/scripts/serve_websocket.py \
  --model-path ${MODEL_PATH} \
  --config-path ${CONFIG_PATH} \
  --config-name ${CONFIG_NAME} \
  --host 0.0.0.0 \
  --port 8765
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

**Enable and start:**

```bash
sudo systemctl enable ezspeech
sudo systemctl start ezspeech
sudo systemctl status ezspeech
```

---

## Health Check Endpoints

All deployments expose health endpoints on port 8080:

### GET /health

Basic liveness check.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": 1234567890.123
}
```

### GET /ready

Readiness check (model loaded).

**Response:**
```json
{
  "status": "ready",
  "uptime_seconds": 3600,
  "device": "cuda:0",
  "request_count": 150
}
```

### GET /metrics

Server metrics.

**Response:**
```json
{
  "uptime_seconds": 3600,
  "device": "cuda:0",
  "request_count": 150,
  "sample_rate": 16000,
  "chunk_duration": 1.0,
  "model_loaded": true
}
```

---

## Environment Variables

Configure server behavior:

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | - | Path to model checkpoint (required) |
| `CONFIG_PATH` | `../config` | Path to configuration directory |
| `CONFIG_NAME` | `ctc_sc` | Configuration file name |
| `SERVER_HOST` | `0.0.0.0` | Server bind address |
| `SERVER_PORT` | `8765` | WebSocket server port |
| `HEALTH_PORT` | `8080` | Health check server port |
| `LOG_LEVEL` | `INFO` | Logging level (DEBUG, INFO, WARNING, ERROR) |

---

## Troubleshooting

### Container won't start

```bash
# Check logs
docker logs ezspeech-server

# Common issues:
# 1. Model file not found → Verify MODEL_PATH
# 2. Out of memory → Increase container memory
# 3. Port conflict → Change port mapping
```

### High latency

```bash
# Check GPU availability
docker exec ezspeech-server python -c "import torch; print(torch.cuda.is_available())"

# Solutions:
# - Enable GPU support (--gpus all)
# - Use smaller model
# - Optimize with ONNX/TensorRT
```

### WebSocket connection refused

```bash
# Check if server is listening
netstat -tuln | grep 8765

# Check firewall
# AWS: Security groups
# Linux: iptables/ufw
```

---

## Performance Optimization

### CPU Inference
- Use ONNX Runtime for 10-30% speedup
- Enable PyTorch JIT compilation
- Use quantized models (INT8)
- Batch requests when possible

### GPU Inference
- Use TensorRT for optimization
- Enable mixed precision (FP16)
- Optimize batch size
- Use GPU with 8GB+ VRAM

### General
- Implement connection pooling
- Cache frequently used data
- Monitor and optimize based on metrics
- Use async/await for I/O

---

## Security Best Practices

1. **Use HTTPS/WSS** - Deploy behind reverse proxy with SSL
2. **Authentication** - Add API key or JWT authentication
3. **Rate Limiting** - Prevent abuse
4. **Network Isolation** - Use private networks
5. **Secrets Management** - AWS Secrets Manager, HashiCorp Vault
6. **Regular Updates** - Keep dependencies updated
7. **Monitoring** - Set up alerts for unusual activity

---

## Next Steps

- [AWS Detailed Guide](https://github.com/khanh14ph/Ezspeech/blob/main/aws/README.md)
- [Full Deployment Options](https://github.com/khanh14ph/Ezspeech/blob/main/DEPLOYMENT.md)
- [Training Guide](/training/)
- [API Reference](/api/)

---

## Support

- **GitHub Issues**: [Report Issue](https://github.com/khanh14ph/Ezspeech/issues)
- **Documentation**: [Main Docs](/)
- **Examples**: [Code Samples](https://github.com/khanh14ph/Ezspeech/tree/main/examples)
