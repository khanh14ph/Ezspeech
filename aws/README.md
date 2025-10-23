# AWS Deployment Guide for EzSpeech

This directory contains all the necessary configuration and scripts to deploy EzSpeech ASR (Automatic Speech Recognition) server to AWS using ECS Fargate and Terraform.

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Detailed Deployment Steps](#detailed-deployment-steps)
- [Configuration](#configuration)
- [Monitoring and Logging](#monitoring-and-logging)
- [Scaling](#scaling)
- [Cost Estimation](#cost-estimation)
- [Troubleshooting](#troubleshooting)

## Architecture Overview

The deployment uses the following AWS services:

```
┌─────────────────────────────────────────────────────────────┐
│                       Internet                               │
└────────────────┬────────────────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────────────────┐
│          Application Load Balancer (ALB)                     │
│              (HTTPS/WebSocket Support)                       │
└────────────────┬────────────────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────────────────┐
│               ECS Fargate Cluster                            │
│  ┌─────────────────────┐  ┌─────────────────────┐          │
│  │   ECS Task 1        │  │   ECS Task 2        │ ...      │
│  │  (ASR Server)       │  │  (ASR Server)       │          │
│  └─────────────────────┘  └─────────────────────┘          │
└────────────────┬────────────────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────────────────┐
│                    S3 Bucket                                 │
│              (Model Checkpoints)                             │
└─────────────────────────────────────────────────────────────┘
```

**Key Components:**
- **ECR (Elastic Container Registry)**: Stores Docker images
- **ECS Fargate**: Runs containerized ASR servers (auto-scaling)
- **ALB**: Load balances WebSocket connections with health checks
- **S3**: Stores model checkpoints and configuration
- **CloudWatch**: Logs and metrics monitoring
- **IAM**: Role-based access control
- **VPC**: Network isolation with public/private subnets

## Prerequisites

Before deploying, ensure you have:

1. **AWS Account** with appropriate permissions
2. **AWS CLI** installed and configured:
   ```bash
   aws --version
   aws configure
   ```

3. **Terraform** installed (v1.0+):
   ```bash
   terraform --version
   ```

4. **Docker** installed:
   ```bash
   docker --version
   ```

5. **Trained ASR Model**: Model checkpoint file (`.ckpt`)

6. **Required IAM Permissions**:
   - ECR: Full access
   - ECS: Full access
   - VPC: Create/modify
   - IAM: Create roles
   - CloudWatch: Logs and metrics
   - S3: Create buckets and upload objects

## Quick Start

### 1. Configure Terraform Variables

```bash
cd aws/terraform
cp terraform.tfvars.example terraform.tfvars
# Edit terraform.tfvars with your configuration
```

### 2. Deploy Infrastructure

```bash
# From the project root
./aws/scripts/deploy.sh
```

### 3. Build and Push Docker Image

```bash
./aws/scripts/build-and-push.sh
```

### 4. Upload Model to S3

```bash
./aws/scripts/upload-model.sh /path/to/your/model.ckpt
```

### 5. Update ECS Service

```bash
./aws/scripts/update-service.sh
```

### 6. Test the Deployment

```bash
# Get ALB DNS name
cd aws/terraform
ALB_DNS=$(terraform output -raw alb_dns_name)

# Test health endpoint
curl http://$ALB_DNS/health

# Test WebSocket connection (using the examples/websocket_client.py)
cd ../../examples
python websocket_client.py --server ws://$ALB_DNS:80
```

## Detailed Deployment Steps

### Step 1: Infrastructure Setup with Terraform

The Terraform configuration creates:
- VPC with public and private subnets across 3 availability zones
- Application Load Balancer with health checks
- ECS Cluster with Fargate launch type
- ECR repository for Docker images
- S3 bucket for model storage
- CloudWatch log groups
- IAM roles and policies
- Auto-scaling policies

**Deploy Infrastructure:**

```bash
cd aws/terraform

# Initialize Terraform
terraform init

# Review the plan
terraform plan

# Apply configuration
terraform apply
```

**Important Outputs:**
```bash
# After deployment, note these values:
terraform output ecr_repository_url    # For Docker push
terraform output alb_dns_name          # For accessing the service
terraform output s3_models_bucket      # For uploading models
```

### Step 2: Build and Push Docker Image

The optimized Dockerfile includes:
- Multi-stage build for smaller image size
- PyTorch with CUDA support
- Health check endpoints
- Non-root user for security

**Build and Push:**

```bash
# Automated script
./aws/scripts/build-and-push.sh

# Or manually:
export AWS_REGION=us-east-1
export AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)

# Login to ECR
aws ecr get-login-password --region $AWS_REGION | \
  docker login --username AWS --password-stdin \
  ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com

# Build
docker build -t ezspeech:latest -f Dockerfile .

# Tag
docker tag ezspeech:latest \
  ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/ezspeech:latest

# Push
docker push ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/ezspeech:latest
```

### Step 3: Upload Model Checkpoint

```bash
# Using the script
./aws/scripts/upload-model.sh outputs/model.ckpt models/model.ckpt

# Or manually
aws s3 cp outputs/model.ckpt \
  s3://ezspeech-models-${AWS_ACCOUNT_ID}/models/model.ckpt
```

**Note:** For production, you may want to:
1. Upload model during container build (includes in image)
2. Download from S3 during container startup (recommended for large models)
3. Use EFS (Elastic File System) for shared model access

### Step 4: Deploy ECS Service

The ECS service is created by Terraform, but you can force a new deployment:

```bash
./aws/scripts/update-service.sh

# Monitor deployment
aws ecs describe-services \
  --cluster ezspeech-cluster \
  --services ezspeech-service
```

### Step 5: Configure DNS (Optional)

For production, set up a custom domain:

1. **Get SSL Certificate from ACM:**
   ```bash
   aws acm request-certificate \
     --domain-name asr.yourdomain.com \
     --validation-method DNS
   ```

2. **Update terraform.tfvars:**
   ```hcl
   acm_certificate_arn = "arn:aws:acm:us-east-1:xxx:certificate/xxx"
   ```

3. **Create Route53 Record:**
   ```bash
   aws route53 change-resource-record-sets \
     --hosted-zone-id YOUR_ZONE_ID \
     --change-batch file://dns-change.json
   ```

## Configuration

### Environment Variables

Configure in `terraform/main.tf` or ECS task definition:

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | `/app/models/model.ckpt` | Path to model checkpoint |
| `CONFIG_PATH` | `/app/config` | Path to config directory |
| `CONFIG_NAME` | `ctc_sc` | Configuration file name |
| `SERVER_HOST` | `0.0.0.0` | Server bind address |
| `SERVER_PORT` | `8765` | WebSocket server port |
| `HEALTH_PORT` | `8080` | Health check port |
| `LOG_LEVEL` | `INFO` | Logging level |

### Terraform Variables

Key variables in `terraform.tfvars`:

```hcl
# Compute Resources
task_cpu    = "2048"  # 1024 = 1 vCPU
task_memory = "4096"  # MB

# Auto-scaling
desired_count = 2
min_capacity  = 1
max_capacity  = 10
cpu_target_value    = 70  # %
memory_target_value = 70  # %

# Environment
environment = "dev"  # dev, staging, prod
aws_region  = "us-east-1"
```

### Auto-scaling Policies

The deployment includes two auto-scaling policies:

1. **CPU-based**: Scales when average CPU > 70%
2. **Memory-based**: Scales when average memory > 70%

**Modify in `terraform/main.tf`:**
```hcl
variable "cpu_target_value" {
  default = 70  # Adjust threshold
}
```

## Monitoring and Logging

### CloudWatch Logs

View logs:
```bash
# Stream logs in real-time
aws logs tail /ecs/ezspeech-asr-server --follow

# Filter by error
aws logs tail /ecs/ezspeech-asr-server --filter-pattern "ERROR"

# Specific time range
aws logs tail /ecs/ezspeech-asr-server \
  --since 1h \
  --format short
```

### CloudWatch Metrics

Key metrics to monitor:
- **CPUUtilization**: ECS task CPU usage
- **MemoryUtilization**: ECS task memory usage
- **TargetResponseTime**: ALB response time
- **RequestCount**: Number of requests
- **HealthyHostCount**: Number of healthy tasks

**View metrics:**
```bash
# CPU utilization
aws cloudwatch get-metric-statistics \
  --namespace AWS/ECS \
  --metric-name CPUUtilization \
  --dimensions Name=ServiceName,Value=ezspeech-service \
                Name=ClusterName,Value=ezspeech-cluster \
  --start-time $(date -u -d '1 hour ago' +%Y-%m-%dT%H:%M:%S) \
  --end-time $(date -u +%Y-%m-%dT%H:%M:%S) \
  --period 300 \
  --statistics Average
```

### Custom Metrics

The ASR server exposes a `/metrics` endpoint:

```bash
curl http://ALB_DNS/metrics
```

Response:
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

### CloudWatch Alarms

Pre-configured alarms:
- High CPU (>80% for 10 minutes)
- High Memory (>80% for 10 minutes)

**Add custom alarms:**
```bash
aws cloudwatch put-metric-alarm \
  --alarm-name ezspeech-high-error-rate \
  --alarm-description "High error rate" \
  --metric-name HTTPCode_Target_5XX_Count \
  --namespace AWS/ApplicationELB \
  --statistic Sum \
  --period 300 \
  --threshold 10 \
  --comparison-operator GreaterThanThreshold \
  --evaluation-periods 2
```

## Scaling

### Manual Scaling

```bash
# Scale to 5 tasks
aws ecs update-service \
  --cluster ezspeech-cluster \
  --service ezspeech-service \
  --desired-count 5
```

### Auto-scaling Configuration

Modify `terraform/main.tf`:

```hcl
# Increase maximum capacity
variable "max_capacity" {
  default = 20  # Up to 20 tasks
}

# More aggressive scaling
resource "aws_appautoscaling_policy" "ecs_cpu" {
  target_tracking_scaling_policy_configuration {
    target_value       = 60  # Scale at 60% CPU
    scale_out_cooldown = 30  # Scale out faster
    scale_in_cooldown  = 300
  }
}
```

### Cost Optimization

For cost savings:

1. **Use Fargate Spot** (in `terraform/main.tf`):
   ```hcl
   capacity_provider_strategy {
     capacity_provider = "FARGATE_SPOT"
     weight           = 1
     base             = 0
   }
   ```

2. **Scheduled Scaling** (for predictable traffic):
   ```bash
   # Scale down at night
   aws application-autoscaling put-scheduled-action \
     --service-namespace ecs \
     --scalable-dimension ecs:service:DesiredCount \
     --resource-id service/ezspeech-cluster/ezspeech-service \
     --scheduled-action-name scale-down-night \
     --schedule "cron(0 22 * * ? *)" \
     --scalable-target-action MinCapacity=1,MaxCapacity=2
   ```

3. **Use Savings Plans**: Reserve capacity for consistent workloads

## Cost Estimation

**Monthly costs (us-east-1):**

### Development Environment
- **ECS Fargate** (2 tasks, 2 vCPU, 4GB): ~$300/month
- **ALB**: ~$32/month
- **Data Transfer**: ~$50/month
- **CloudWatch Logs**: ~$10/month
- **S3 Storage** (10GB): ~$0.23/month
- **ECR Storage** (5GB): ~$0.50/month
- **Total**: ~$400/month

### Production Environment
- **ECS Fargate** (4-10 tasks avg): ~$1,000/month
- **ALB**: ~$32/month
- **Data Transfer**: ~$200/month
- **CloudWatch**: ~$50/month
- **S3/ECR**: ~$2/month
- **Total**: ~$1,300/month

**Cost Optimization Tips:**
- Use Fargate Spot (70% savings)
- Enable S3 Intelligent-Tiering
- Use Compute Savings Plans (up to 66% off)
- Clean up old ECR images
- Reduce CloudWatch log retention

## Troubleshooting

### Common Issues

#### 1. ECS Tasks Failing Health Checks

**Symptoms:** Tasks start but are marked unhealthy

**Solutions:**
```bash
# Check logs
aws logs tail /ecs/ezspeech-asr-server --follow

# Common issues:
# - Model file not found: Upload model to S3 or include in image
# - Port mismatch: Ensure HEALTH_PORT=8080
# - Startup time too long: Increase startPeriod in health check
```

**Fix startup time:**
```hcl
# In terraform/main.tf
healthCheck = {
  startPeriod = 120  # Increase to 2 minutes
}
```

#### 2. Out of Memory Errors

**Symptoms:** Tasks crash with OOM errors

**Solutions:**
```bash
# Increase task memory
# In terraform.tfvars:
task_memory = "8192"  # 8GB instead of 4GB

# Or optimize model:
# - Use model quantization
# - Export to ONNX
# - Use smaller model variant
```

#### 3. Slow Inference

**Symptoms:** High latency, slow transcription

**Solutions:**
```bash
# Check if GPU is being used
aws ecs execute-command \
  --cluster ezspeech-cluster \
  --task TASK_ID \
  --command "nvidia-smi"

# If no GPU:
# 1. Use GPU-enabled Fargate (if available in region)
# 2. Switch to EC2 with GPU (g4dn instances)
# 3. Optimize model for CPU inference
```

#### 4. WebSocket Connection Drops

**Symptoms:** Connections close unexpectedly

**Solutions:**
```hcl
# Increase ALB timeout
# In terraform/main.tf:
resource "aws_lb_target_group" "websocket" {
  deregistration_delay = 60  # Increase delay
}

# Configure target group attributes
resource "aws_lb_target_group_attachment" "websocket" {
  target_group_arn = aws_lb_target_group.websocket.arn

  # Add stickiness
  stickiness {
    enabled = true
    type    = "lb_cookie"
    duration = 3600
  }
}
```

### Debugging Commands

```bash
# List running tasks
aws ecs list-tasks --cluster ezspeech-cluster

# Describe specific task
aws ecs describe-tasks \
  --cluster ezspeech-cluster \
  --tasks TASK_ARN

# Get task logs
aws logs get-log-events \
  --log-group-name /ecs/ezspeech-asr-server \
  --log-stream-name ecs/ezspeech-server/TASK_ID

# Check ALB target health
aws elbv2 describe-target-health \
  --target-group-arn TARGET_GROUP_ARN

# Execute command in running task (requires ECS Exec)
aws ecs execute-command \
  --cluster ezspeech-cluster \
  --task TASK_ID \
  --container ezspeech-server \
  --interactive \
  --command "/bin/bash"
```

## Local Testing with Docker Compose

Before deploying to AWS, test locally:

```bash
# Start services
docker-compose up -d

# Check health
curl http://localhost:8080/health

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

**With Nginx reverse proxy:**
```bash
docker-compose --profile with-nginx up -d
```

## Security Best Practices

1. **Use Private Subnets**: ECS tasks run in private subnets
2. **Enable VPC Endpoints**: Reduce NAT gateway costs and improve security
3. **Rotate Credentials**: Use AWS Secrets Manager for sensitive data
4. **Enable Container Insights**: Monitor security metrics
5. **Scan Images**: Enable ECR image scanning
6. **Use IAM Roles**: Avoid hardcoded credentials
7. **Enable ALB Access Logs**: Track all requests
8. **Configure WAF**: Add AWS WAF for additional protection

## Next Steps

- [ ] Set up CI/CD pipeline with GitHub Actions
- [ ] Implement model versioning and A/B testing
- [ ] Add distributed tracing with X-Ray
- [ ] Set up backup and disaster recovery
- [ ] Implement rate limiting
- [ ] Add authentication (API Gateway + Cognito)
- [ ] Configure CloudFront for global distribution

## Support

For issues and questions:
- GitHub Issues: [Report Issue](https://github.com/your-repo/issues)
- Documentation: [Project Docs](../docs/)
- AWS Support: [AWS Console](https://console.aws.amazon.com/support/)

## License

See the main project LICENSE file.
