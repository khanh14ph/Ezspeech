---
layout: page
title: AWS Deployment Tutorial
permalink: /aws-deployment/
---

# AWS Deployment Tutorial

Step-by-step guide to deploying EzSpeech ASR server on AWS using ECS Fargate.

## Overview

This tutorial will guide you through deploying a production-ready ASR server on AWS with:

- ⚡ Auto-scaling (1-10 tasks based on demand)
- 🌐 Load balancing with WebSocket support
- 📊 CloudWatch monitoring and logging
- 🔒 VPC isolation and security
- 💰 Cost-optimized configuration (~$400-1,500/month)

**Estimated Time**: 30-45 minutes

---

## Prerequisites

Before starting, ensure you have:

- [ ] AWS Account with admin permissions
- [ ] AWS CLI installed and configured
- [ ] Terraform installed (v1.0+)
- [ ] Docker installed
- [ ] Trained ASR model checkpoint (`.ckpt` file)
- [ ] Basic knowledge of AWS services

### Install Prerequisites

**AWS CLI:**
```bash
# macOS
brew install awscli

# Linux
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install

# Configure
aws configure
# Enter: Access Key ID, Secret Access Key, Region (e.g., us-east-1), Output format (json)
```

**Terraform:**
```bash
# macOS
brew install terraform

# Linux
wget https://releases.hashicorp.com/terraform/1.6.0/terraform_1.6.0_linux_amd64.zip
unzip terraform_1.6.0_linux_amd64.zip
sudo mv terraform /usr/local/bin/

# Verify
terraform --version
```

**Docker:**
```bash
# Follow official Docker installation guide
# https://docs.docker.com/get-docker/

# Verify
docker --version
```

---

## Step 1: Clone Repository

```bash
git clone https://github.com/khanh14ph/Ezspeech.git
cd Ezspeech
```

---

## Step 2: Configure Infrastructure

### 2.1 Copy Configuration Template

```bash
cd aws/terraform
cp terraform.tfvars.example terraform.tfvars
```

### 2.2 Edit Configuration

Open `terraform.tfvars` in your editor:

```hcl
# AWS Configuration
aws_region  = "us-east-1"      # Change to your preferred region
environment = "prod"            # or "dev" for development

# Project Configuration
project_name = "ezspeech"

# VPC Configuration (use defaults unless you have specific requirements)
vpc_cidr             = "10.0.0.0/16"
public_subnet_cidrs  = ["10.0.1.0/24", "10.0.2.0/24", "10.0.3.0/24"]
private_subnet_cidrs = ["10.0.11.0/24", "10.0.12.0/24", "10.0.13.0/24"]

# ECS Task Configuration
task_cpu    = "2048"  # 2 vCPU (1024 = 1 vCPU)
task_memory = "4096"  # 4 GB RAM

# For larger models, use:
# task_cpu    = "4096"  # 4 vCPU
# task_memory = "8192"  # 8 GB RAM

# Auto-scaling Configuration
desired_count = 2     # Initial number of tasks
min_capacity  = 1     # Minimum tasks (cost optimization)
max_capacity  = 10    # Maximum tasks (handle traffic spikes)

# Auto-scaling thresholds (percentage)
cpu_target_value    = 70  # Scale when CPU > 70%
memory_target_value = 70  # Scale when memory > 70%

# Logging
log_retention_days = 7  # Increase for production (e.g., 30, 90)

# Model Configuration
model_s3_key = "models/model.ckpt"
config_name  = "ctc_sc"  # or "ctc" for standard CTC

# Additional Tags
tags = {
  Owner      = "YourName"
  Team       = "ML"
  CostCenter = "Engineering"
}
```

**Key Decisions:**

- **task_cpu/memory**: Increase for larger models or GPU needs
- **desired_count**: Start with 2 for high availability
- **min/max_capacity**: Adjust based on expected traffic
- **log_retention_days**: Balance cost vs. debugging needs

---

## Step 3: Deploy Infrastructure

### 3.1 Initialize Terraform

```bash
cd aws/terraform
terraform init
```

This will:
- Download required providers (AWS)
- Initialize backend configuration
- Set up modules

### 3.2 Review Deployment Plan

```bash
terraform plan
```

Review the resources that will be created:
- ✅ VPC with public/private subnets
- ✅ Application Load Balancer
- ✅ ECS Cluster and Task Definition
- ✅ ECR Repository
- ✅ S3 Bucket for models
- ✅ IAM Roles and Security Groups
- ✅ CloudWatch Log Groups
- ✅ Auto-scaling policies

**Expected resources**: ~35-40 resources

### 3.3 Apply Configuration

```bash
terraform apply
```

Type `yes` when prompted.

**Estimated time**: 5-10 minutes

### 3.4 Save Outputs

```bash
# Display outputs
terraform output

# Save important values
ECR_REPO=$(terraform output -raw ecr_repository_url)
ALB_DNS=$(terraform output -raw alb_dns_name)
S3_BUCKET=$(terraform output -raw s3_models_bucket)

echo "ECR Repository: $ECR_REPO"
echo "ALB DNS: $ALB_DNS"
echo "S3 Bucket: $S3_BUCKET"
```

---

## Step 4: Build and Push Docker Image

### 4.1 Automated Script (Recommended)

```bash
cd ../..  # Back to project root
./aws/scripts/build-and-push.sh
```

This script will:
1. Create ECR repository (if needed)
2. Login to ECR
3. Build Docker image
4. Tag for ECR
5. Push to ECR

**Estimated time**: 10-15 minutes (first build)

### 4.2 Manual Process (Alternative)

```bash
# Get AWS account ID
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
AWS_REGION=us-east-1

# Login to ECR
aws ecr get-login-password --region $AWS_REGION | \
  docker login --username AWS --password-stdin \
  ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com

# Build image
docker build -t ezspeech:latest -f Dockerfile .

# Tag for ECR
docker tag ezspeech:latest \
  ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/ezspeech:latest

# Push to ECR
docker push ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/ezspeech:latest
```

---

## Step 5: Upload Model to S3

### 5.1 Automated Script (Recommended)

```bash
./aws/scripts/upload-model.sh /path/to/your/model.ckpt
```

### 5.2 Manual Upload

```bash
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
S3_BUCKET="ezspeech-models-${AWS_ACCOUNT_ID}"

aws s3 cp /path/to/your/model.ckpt \
  s3://${S3_BUCKET}/models/model.ckpt
```

**Note**: For large models (>1GB), consider:
- Using multipart upload
- Including model in Docker image
- Using EFS for shared storage

---

## Step 6: Update ECS Task Definition

The model needs to be accessible to ECS tasks. You have two options:

### Option A: Download from S3 during startup (Recommended)

Add to task definition environment:
```json
{
  "name": "MODEL_S3_BUCKET",
  "value": "ezspeech-models-ACCOUNT_ID"
},
{
  "name": "MODEL_S3_KEY",
  "value": "models/model.ckpt"
}
```

And add startup script to download model.

### Option B: Include in Docker image

Add to Dockerfile:
```dockerfile
COPY models/model.ckpt /app/models/model.ckpt
```

Rebuild and push image.

---

## Step 7: Deploy ECS Service

### 7.1 Update Service

```bash
./aws/scripts/update-service.sh
```

This will:
1. Force new deployment with latest image
2. Monitor deployment progress
3. Wait for tasks to become healthy

**Estimated time**: 2-5 minutes

### 7.2 Verify Deployment

```bash
# Check service status
aws ecs describe-services \
  --cluster ezspeech-cluster \
  --services ezspeech-service \
  --region us-east-1

# List running tasks
aws ecs list-tasks \
  --cluster ezspeech-cluster \
  --service-name ezspeech-service \
  --region us-east-1
```

---

## Step 8: Test Deployment

### 8.1 Test Health Endpoint

```bash
# Get ALB DNS
cd aws/terraform
ALB_DNS=$(terraform output -raw alb_dns_name)

# Test health
curl http://$ALB_DNS/health
```

**Expected response:**
```json
{
  "status": "healthy",
  "timestamp": 1234567890.123
}
```

### 8.2 Test Readiness

```bash
curl http://$ALB_DNS/ready
```

**Expected response:**
```json
{
  "status": "ready",
  "uptime_seconds": 120,
  "device": "cuda:0",
  "request_count": 0
}
```

### 8.3 Test WebSocket Connection

**Using Python client:**

```bash
cd ../../examples
python websocket_client.py --server ws://$ALB_DNS:80
```

**Using wscat:**

```bash
# Install wscat
npm install -g wscat

# Connect
wscat -c ws://$ALB_DNS:80
```

### 8.4 Test Audio Transcription

```bash
# Prepare test audio
ffmpeg -i test.mp3 -ar 16000 -ac 1 test.wav

# Transcribe
python examples/websocket_client.py \
  --server ws://$ALB_DNS:80 \
  --audio-file test.wav
```

---

## Step 9: Set Up Monitoring

### 9.1 View CloudWatch Logs

```bash
# Stream logs in real-time
aws logs tail /ecs/ezspeech-asr-server --follow --region us-east-1

# Filter by keyword
aws logs tail /ecs/ezspeech-asr-server --follow --filter-pattern "ERROR"

# View specific time range
aws logs tail /ecs/ezspeech-asr-server \
  --since 1h \
  --format short
```

### 9.2 Set Up CloudWatch Alarms

**High CPU Alarm:**

```bash
aws cloudwatch put-metric-alarm \
  --alarm-name ezspeech-high-cpu \
  --alarm-description "Alert when CPU > 80%" \
  --metric-name CPUUtilization \
  --namespace AWS/ECS \
  --statistic Average \
  --period 300 \
  --threshold 80 \
  --comparison-operator GreaterThanThreshold \
  --evaluation-periods 2 \
  --dimensions Name=ClusterName,Value=ezspeech-cluster Name=ServiceName,Value=ezspeech-service
```

**High Error Rate Alarm:**

```bash
aws cloudwatch put-metric-alarm \
  --alarm-name ezspeech-high-errors \
  --alarm-description "Alert on high 5XX errors" \
  --metric-name HTTPCode_Target_5XX_Count \
  --namespace AWS/ApplicationELB \
  --statistic Sum \
  --period 300 \
  --threshold 10 \
  --comparison-operator GreaterThanThreshold \
  --evaluation-periods 1
```

### 9.3 View Metrics Dashboard

1. Open AWS Console → CloudWatch → Dashboards
2. Create new dashboard: "EzSpeech-Monitoring"
3. Add widgets for:
   - ECS CPU/Memory Utilization
   - ALB Request Count
   - ALB Target Response Time
   - ALB Healthy Host Count
   - CloudWatch Logs Insights

---

## Step 10: Configure Custom Domain (Optional)

### 10.1 Request SSL Certificate

```bash
# Request certificate from ACM
aws acm request-certificate \
  --domain-name asr.yourdomain.com \
  --validation-method DNS \
  --region us-east-1
```

### 10.2 Validate Certificate

Follow the DNS validation process in ACM console.

### 10.3 Update Terraform Configuration

Edit `terraform.tfvars`:

```hcl
acm_certificate_arn = "arn:aws:acm:us-east-1:ACCOUNT:certificate/CERT_ID"
```

Uncomment HTTPS listener in `main.tf` and apply:

```bash
terraform apply
```

### 10.4 Create Route53 Record

```bash
# Get ALB Zone ID
ALB_ZONE_ID=$(cd aws/terraform && terraform output -raw alb_zone_id)
ALB_DNS=$(cd aws/terraform && terraform output -raw alb_dns_name)

# Create A record (alias to ALB)
aws route53 change-resource-record-sets \
  --hosted-zone-id YOUR_ZONE_ID \
  --change-batch '{
    "Changes": [{
      "Action": "CREATE",
      "ResourceRecordSet": {
        "Name": "asr.yourdomain.com",
        "Type": "A",
        "AliasTarget": {
          "HostedZoneId": "'"$ALB_ZONE_ID"'",
          "DNSName": "'"$ALB_DNS"'",
          "EvaluateTargetHealth": true
        }
      }
    }]
  }'
```

---

## Cost Management

### Estimated Monthly Costs

**Development Environment:**
- ECS Fargate (2 tasks, 2 vCPU, 4GB): ~$300/month
- ALB: ~$32/month
- Data Transfer: ~$50/month
- CloudWatch Logs: ~$10/month
- S3/ECR: ~$2/month
- **Total**: ~$400/month

**Production Environment:**
- ECS Fargate (4-10 tasks avg): ~$1,000/month
- ALB: ~$32/month
- Data Transfer: ~$200/month
- CloudWatch: ~$50/month
- **Total**: ~$1,300/month

### Cost Optimization Tips

1. **Use Fargate Spot** (70% savings):
   ```hcl
   # In main.tf
   capacity_provider_strategy {
     capacity_provider = "FARGATE_SPOT"
     weight           = 1
   }
   ```

2. **Reduce log retention**:
   ```hcl
   log_retention_days = 7  # Instead of 30
   ```

3. **Use Compute Savings Plans**:
   - Commit to 1 or 3 years
   - Save up to 66% on compute

4. **Enable auto-scaling**:
   - Scale down during off-peak hours
   - Set appropriate min/max values

5. **Clean up old resources**:
   ```bash
   # Delete old ECR images
   aws ecr batch-delete-image \
     --repository-name ezspeech \
     --image-ids imageTag=old-tag
   ```

### View Current Costs

```bash
# AWS Cost Explorer (requires AWS Console)
# Or use AWS CLI
aws ce get-cost-and-usage \
  --time-period Start=2024-01-01,End=2024-01-31 \
  --granularity MONTHLY \
  --metrics BlendedCost \
  --group-by Type=SERVICE
```

---

## Troubleshooting

### Tasks Failing Health Checks

**Check logs:**
```bash
aws logs tail /ecs/ezspeech-asr-server --follow
```

**Common issues:**
1. **Model not found**: Verify S3 upload and permissions
2. **Out of memory**: Increase `task_memory`
3. **Port mismatch**: Ensure `HEALTH_PORT=8080`
4. **Startup too slow**: Increase `startPeriod` in health check

### High Latency

**Check GPU availability:**
```bash
# Get task ID
TASK_ID=$(aws ecs list-tasks --cluster ezspeech-cluster --service-name ezspeech-service --query 'taskArns[0]' --output text)

# Execute command (requires ECS Exec enabled)
aws ecs execute-command \
  --cluster ezspeech-cluster \
  --task $TASK_ID \
  --command "nvidia-smi" \
  --interactive
```

**Solutions:**
- Use GPU-enabled Fargate (if available)
- Switch to EC2 with GPU (g4dn instances)
- Optimize model (quantization, ONNX)

### Auto-scaling Not Working

**Verify policies:**
```bash
aws application-autoscaling describe-scaling-policies \
  --service-namespace ecs \
  --resource-id service/ezspeech-cluster/ezspeech-service
```

**Check metrics:**
```bash
aws cloudwatch get-metric-statistics \
  --namespace AWS/ECS \
  --metric-name CPUUtilization \
  --dimensions Name=ServiceName,Value=ezspeech-service Name=ClusterName,Value=ezspeech-cluster \
  --start-time $(date -u -d '1 hour ago' +%Y-%m-%dT%H:%M:%S) \
  --end-time $(date -u +%Y-%m-%dT%H:%M:%S) \
  --period 300 \
  --statistics Average
```

---

## Updating the Deployment

### Update Docker Image

```bash
# Build and push new image
./aws/scripts/build-and-push.sh

# Force new deployment
./aws/scripts/update-service.sh
```

### Update Infrastructure

```bash
# Edit terraform.tfvars
cd aws/terraform
vim terraform.tfvars

# Apply changes
terraform plan
terraform apply
```

### Update Model

```bash
# Upload new model
./aws/scripts/upload-model.sh /path/to/new/model.ckpt

# Force ECS service update
./aws/scripts/update-service.sh
```

---

## Cleanup

To avoid ongoing costs, destroy all resources:

```bash
cd aws/terraform
terraform destroy
```

Type `yes` when prompted.

**Note**: This will delete:
- ECS cluster and tasks
- Load balancer
- VPC and networking
- ECR repository
- S3 bucket (if empty)
- All associated resources

**Estimated time**: 5-10 minutes

---

## Next Steps

- ✅ Set up CI/CD pipeline with GitHub Actions
- ✅ Implement A/B testing for models
- ✅ Add authentication (API Gateway + Cognito)
- ✅ Configure CloudFront for global distribution
- ✅ Set up backup and disaster recovery
- ✅ Implement rate limiting and throttling

---

## Additional Resources

- [Full AWS Guide](https://github.com/khanh14ph/Ezspeech/blob/main/aws/README.md)
- [Terraform Documentation](https://registry.terraform.io/providers/hashicorp/aws/latest/docs)
- [ECS Best Practices](https://docs.aws.amazon.com/AmazonECS/latest/bestpracticesguide/)
- [CloudWatch Logs Insights](https://docs.aws.amazon.com/AmazonCloudWatch/latest/logs/AnalyzingLogData.html)

---

## Support

- **Issues**: [GitHub Issues](https://github.com/khanh14ph/Ezspeech/issues)
- **Documentation**: [Main Docs](/)
- **AWS Support**: [AWS Console](https://console.aws.amazon.com/support/)
