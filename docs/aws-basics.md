# AWS Deployment Basics for EzSpeech

A beginner-friendly guide to deploying your WebSocket ASR server on AWS.

---

## Part 1: AWS Compute Options Explained

### Option A: EC2 (Elastic Compute Cloud)
**What it is**: Virtual servers (like renting a computer in the cloud)

```
┌─────────────────────────────┐
│   EC2 Instance (Ubuntu)     │
│  ┌──────────────────────┐   │
│  │  Your Server Code    │   │
│  │  (Python + PyTorch)  │   │
│  └──────────────────────┘   │
│  ┌──────────────────────┐   │
│  │  Model Files         │   │
│  └──────────────────────┘   │
└─────────────────────────────┘
         ↓ Port 8765
    [Internet Users]
```

**How it works**:
1. You choose instance type (CPU/RAM size)
2. AWS gives you a virtual machine
3. You install everything manually (Docker, Python, etc.)
4. You manage OS updates, security patches

**Pros**:
- Simple to understand
- Full control over the server
- Can SSH in and debug
- Good for learning

**Cons**:
- You manage everything (updates, scaling, backups)
- Manually restart if server crashes
- Need to handle scaling yourself

**Best for**: Development, learning, small-scale production

**Cost**: ~$30-200/month depending on instance size

**Example instance types**:
- `t3.medium`: 2 vCPU, 4GB RAM (~$30/month) - for testing
- `t3.xlarge`: 4 vCPU, 16GB RAM (~$120/month) - for production
- `g4dn.xlarge`: 4 vCPU, 16GB RAM + GPU (~$400/month) - for GPU inference

---

### Option B: ECS Fargate (Elastic Container Service)
**What it is**: Managed container service (you provide Docker image, AWS runs it)

```
                [Application Load Balancer]
                         ↓
        ┌────────────────┼────────────────┐
        ↓                ↓                ↓
   [Container 1]    [Container 2]    [Container 3]
   ┌─────────┐      ┌─────────┐      ┌─────────┐
   │ Docker  │      │ Docker  │      │ Docker  │
   │ Image   │      │ Image   │      │ Image   │
   └─────────┘      └─────────┘      └─────────┘

   Auto-scales based on CPU/Memory
```

**How it works**:
1. You build a Docker image with your code
2. Push to AWS ECR (Elastic Container Registry)
3. Tell ECS how much CPU/RAM each container needs
4. AWS automatically runs and manages containers

**Pros**:
- AWS handles server management
- Auto-scaling built-in
- Auto-restart if container crashes
- Load balancing included
- Only pay for containers running

**Cons**:
- More complex setup (requires Terraform or AWS CLI)
- Harder to debug (no direct SSH)
- Need to learn Docker + AWS concepts

**Best for**: Production applications, auto-scaling needs

**Cost**: ~$400-1,500/month for production setup with auto-scaling

---

### Option C: AWS Lightsail
**What it is**: Simplified EC2 with easier management

**How it works**:
- Simplified version of EC2 with fixed pricing
- Includes basic load balancing and storage
- More user-friendly interface

**Pros**:
- Simpler than EC2
- Predictable pricing
- Good for small projects

**Cons**:
- Less flexibility than EC2
- Limited scaling options
- Not suitable for high-traffic apps

**Best for**: Side projects, demos, MVP

**Cost**: $10-80/month (fixed pricing)

---

## Part 2: Supporting AWS Services

### S3 (Simple Storage Service)
**Purpose**: Store your model files

```
┌───────────────────┐
│   S3 Bucket       │
│  my-models/       │
│  ├── model.ckpt   │  ← Your ASR model
│  └── tokenizer    │  ← Your tokenizer
└───────────────────┘
         ↓
    Downloaded by
    containers on startup
```

**Why use it**:
- Don't need to include huge model files in Docker image
- Can update models without rebuilding image
- Cheap storage (~$0.023/GB/month)
- High availability

**Alternative**: Include model in Docker image (simpler but larger image)

---

### Application Load Balancer (ALB)
**Purpose**: Distribute traffic across multiple containers

```
         [Users]
            ↓
    [Load Balancer]  ← Single public URL
            ↓
    ┌───────┼───────┐
    ↓       ↓       ↓
 [Task1] [Task2] [Task3]

 - Health checks each task
 - Routes to healthy tasks only
 - Supports WebSocket (sticky sessions)
```

**Why use it**:
- Single entry point for users
- Distributes load across multiple servers
- Removes unhealthy servers automatically
- Supports WebSocket connections

**Cost**: ~$22/month base + $0.008/GB transferred

---

### VPC (Virtual Private Cloud)
**Purpose**: Isolated network for your resources

```
┌─────────────────── VPC ───────────────────┐
│                                            │
│  ┌─── Public Subnet ───┐                  │
│  │  [Load Balancer]    │                  │
│  └─────────────────────┘                  │
│            ↓                               │
│  ┌─── Private Subnet ──┐                  │
│  │  [Your Containers]  │  ← Can't be      │
│  │  - Safe from direct │     accessed     │
│  │    internet access  │     directly     │
│  └─────────────────────┘                  │
│                                            │
└────────────────────────────────────────────┘
```

**Why use it**:
- Security (containers not directly exposed to internet)
- Control network access
- Required for ECS Fargate

---

### ECR (Elastic Container Registry)
**Purpose**: Store Docker images (like Docker Hub, but private)

```
Local Computer                      AWS ECR
┌──────────────┐                 ┌──────────────┐
│ docker build │  ─── push ───→  │ Your Images: │
│              │                 │ - v1.0       │
└──────────────┘                 │ - v1.1       │
                                 │ - latest     │
                                 └──────────────┘
                                        ↓
                                   [ECS pulls
                                    from here]
```

**Why use it**:
- Private storage for your Docker images
- Integrated with ECS
- Scans for security vulnerabilities

**Cost**: ~$0.10/GB/month

---

## Part 3: Deployment Architectures

### Architecture 1: Simple EC2 Setup

```
                    [Internet]
                        ↓
                [Elastic IP: 52.1.2.3]
                        ↓
            ┌─────────────────────┐
            │   EC2 Instance      │
            │                     │
            │  Docker Container   │
            │  ┌───────────────┐  │
            │  │ Your Server   │  │
            │  │ Port 8765     │  │
            │  └───────────────┘  │
            │                     │
            │  /models/           │
            │  └─ model.ckpt      │
            └─────────────────────┘
```

**Setup steps**:
1. Launch EC2 instance
2. Install Docker
3. Copy your code + model
4. Run: `docker-compose up -d`
5. Configure security group (allow port 8765)

**Pros**: Simple, cheap, easy to debug
**Cons**: Manual management, no auto-scaling, single point of failure

---

### Architecture 2: ECS Fargate with Auto-scaling

```
                         [Internet]
                             ↓
                    [Route 53: asr.yourdomain.com]
                             ↓
                  [Application Load Balancer]
                             ↓
              ┌──────────────┼──────────────┐
              ↓              ↓              ↓
         [Container 1]  [Container 2]  [Container 3]
              ↓              ↓              ↓
         [Downloads]    [Downloads]    [Downloads]
              ↓              ↓              ↓
         ┌────────────────────────────────────┐
         │          S3 Bucket                 │
         │      model.ckpt (shared)           │
         └────────────────────────────────────┘

         Auto-scales: 1-10 containers based on CPU
```

**Setup steps**:
1. Create Docker image
2. Push to ECR
3. Upload model to S3
4. Deploy Terraform infrastructure (VPC, ECS, ALB)
5. ECS automatically runs containers

**Pros**: Production-ready, auto-scaling, high availability
**Cons**: Complex setup, higher cost

---

## Part 4: Cost Breakdown

### Scenario 1: Development/Testing (EC2)

| Service | Cost/Month |
|---------|-----------|
| EC2 t3.medium (2 vCPU, 4GB) | $30 |
| Storage (30GB) | $3 |
| **Total** | **~$33/month** |

---

### Scenario 2: Small Production (EC2)

| Service | Cost/Month |
|---------|-----------|
| EC2 t3.xlarge (4 vCPU, 16GB) | $120 |
| Elastic IP | $0 (if attached) |
| Storage (100GB) | $10 |
| Data transfer (500GB) | $45 |
| **Total** | **~$175/month** |

---

### Scenario 3: Full Production (ECS Fargate)

| Service | Cost/Month |
|---------|-----------|
| Fargate (2 tasks, 2 vCPU, 4GB each) | $300 |
| Application Load Balancer | $22 |
| Data transfer (1TB) | $90 |
| ECR storage | $5 |
| S3 storage | $2 |
| CloudWatch Logs | $10 |
| **Total** | **~$429/month** |

**With auto-scaling** (4-10 tasks during peak): ~$800-1,500/month

---

## Part 5: WebSocket-Specific Considerations

Your server uses WebSocket, which has special requirements:

### 1. Sticky Sessions (Session Affinity)
**Problem**: WebSocket = long-lived connection to one server
**Solution**: Load balancer must route user to same container

```
User connects ──→ [LB] ──→ Container 2
                    ↓
User sends message  ─────→ Container 2 (same!)
```

**How to enable**:
- ALB: Enable "stickiness" with target group attributes
- nginx: Use `ip_hash` for load balancing

### 2. Idle Timeout
**Problem**: Load balancers may disconnect idle WebSocket connections
**Solution**:
- Increase ALB idle timeout to 3600s (1 hour)
- Implement ping/pong keep-alive in your server
- Client should reconnect on disconnect

### 3. Connection Draining
**Problem**: When scaling down, active connections get killed
**Solution**:
- Enable connection draining (300s)
- Containers finish current requests before shutdown
- Client should handle reconnection gracefully

---

## Part 6: Security Basics

### Security Groups (Firewall Rules)

```
┌─────────────────────────────┐
│  Security Group             │
│                             │
│  Inbound Rules:             │
│  - Port 8765 from 0.0.0.0/0 │ ← Allow WebSocket from anywhere
│  - Port 8080 from LB only   │ ← Allow health checks from LB
│  - Port 22 from YOUR_IP     │ ← SSH only from your IP
│                             │
│  Outbound Rules:            │
│  - All traffic allowed      │ ← Can download models, etc.
│                             │
└─────────────────────────────┘
```

**Best practices**:
- Never allow SSH (port 22) from 0.0.0.0/0
- Use separate security groups for different layers
- Restrict health check port to load balancer only

### IAM Roles (Permissions)

```
┌──────────────────┐
│  ECS Task Role   │  ← What your container can do
│                  │
│  Permissions:    │
│  - Read from S3  │  ← Download model files
│  - Write logs    │  ← Send logs to CloudWatch
│                  │
└──────────────────┘
```

**Why needed**:
- Container needs to download model from S3
- Container needs to write logs to CloudWatch
- Never put AWS credentials in code!

---

## Part 7: Deployment Workflow

### For EC2:
```bash
# 1. One-time setup
aws ec2 run-instances --image-id ami-xxxxx --instance-type t3.medium
ssh ubuntu@<ip-address>
sudo apt update && sudo apt install docker.io docker-compose

# 2. Deploy/update
scp -r demo/ ubuntu@<ip>:/home/ubuntu/
ssh ubuntu@<ip>
cd /home/ubuntu/demo
docker-compose up -d
```

### For ECS Fargate:
```bash
# 1. One-time infrastructure setup
cd terraform/
terraform init
terraform apply

# 2. Deploy/update application
docker build -t ezspeech .
docker tag ezspeech:latest <account>.dkr.ecr.us-east-1.amazonaws.com/ezspeech:latest
docker push <account>.dkr.ecr.us-east-1.amazonaws.com/ezspeech:latest

aws ecs update-service --cluster ezspeech --service ezspeech --force-new-deployment
```

---

## Part 8: Monitoring & Debugging

### CloudWatch Logs
**What it does**: Centralized logging

```
All containers send logs ──→ CloudWatch Logs
                                    ↓
                            View in AWS Console
                            or stream with CLI:

aws logs tail /ecs/ezspeech --follow
```

**Key metrics to monitor**:
- Error count
- Request latency
- Active WebSocket connections
- CPU/Memory usage

### CloudWatch Alarms
**What it does**: Alert when things go wrong

```
If CPU > 80% for 5 minutes ──→ Send SNS notification ──→ Email you
If 5XX errors > 10         ──→ Send SNS notification ──→ Email you
```

---

## Part 9: Next Steps & Learning Path

### Path 1: Start Simple (Recommended for learning)
1. **Week 1**: Deploy on EC2, learn Docker basics
2. **Week 2**: Add HTTPS with Let's Encrypt
3. **Week 3**: Try ECS Fargate with 1 container
4. **Week 4**: Add auto-scaling and monitoring

### Path 2: Jump to Production
1. Use Terraform to deploy full ECS setup
2. Requires learning: Docker, Terraform, AWS networking
3. Best if you have DevOps experience

---

## Common Questions

### Q: Do I need a GPU for inference?
**A**: Depends on your model size and latency requirements
- **No GPU**: CPU inference on t3.xlarge (~2-5s per audio file)
- **With GPU**: g4dn.xlarge (~0.5-1s per audio file)
- **Cost difference**: ~$120/month vs ~$400/month

### Q: How do I update my model without downtime?
**A**:
1. Upload new model to S3 with different name
2. Update environment variable to point to new model
3. Deploy new task definition
4. ECS does rolling update (old tasks stay until new tasks are healthy)

### Q: What if my model file is huge (10GB+)?
**A**: Options:
1. Include in Docker image (slow builds, large image)
2. Download from S3 on startup (recommended, but increases startup time)
3. Use EFS (Elastic File System) - shared persistent storage

### Q: How do I handle multiple users simultaneously?
**A**:
- Each WebSocket connection is handled by asyncio
- Your server already supports concurrent connections
- For more users: scale horizontally (more containers)

---

## Summary: Which Option Should You Choose?

| Scenario | Recommendation | Monthly Cost |
|----------|---------------|--------------|
| Learning/Testing | EC2 t3.medium | ~$30 |
| MVP/Demo | EC2 t3.xlarge or Lightsail | ~$50-120 |
| Small Production (<100 users) | EC2 t3.xlarge | ~$175 |
| Production (auto-scaling) | ECS Fargate | ~$400-1,500 |
| High-performance (GPU) | EC2 g4dn.xlarge or ECS with GPU | ~$400+ |

---

## Glossary

- **Container**: Lightweight package with your code + dependencies
- **Docker Image**: Blueprint for creating containers
- **Task**: Running instance of a container in ECS
- **Cluster**: Group of compute resources (EC2 or Fargate)
- **Service**: Ensures desired number of tasks are running
- **Target Group**: Group of tasks that receive traffic from load balancer
- **Auto-scaling**: Automatically add/remove resources based on load
- **Terraform**: Infrastructure-as-code tool (write AWS config as code)
- **IAM**: Identity and Access Management (permissions system)
- **VPC**: Isolated network in AWS
- **Security Group**: Virtual firewall for controlling traffic

---

## Additional Resources

- [AWS Free Tier](https://aws.amazon.com/free/) - 12 months free for EC2
- [Docker Documentation](https://docs.docker.com/)
- [ECS Best Practices](https://docs.aws.amazon.com/AmazonECS/latest/bestpracticesguide/)
- [AWS Calculator](https://calculator.aws/) - Estimate costs

---

**Next**: Ready to deploy? Ask me about:
1. "Set up EC2 deployment" - Simple EC2 setup
2. "Set up ECS deployment" - Full production setup
3. "Update Dockerfile" - Fix Dockerfile for demo server
