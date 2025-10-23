#!/bin/bash
# Build and push Docker image to AWS ECR

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
AWS_REGION=${AWS_REGION:-us-east-1}
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ECR_REPOSITORY="ezspeech"
IMAGE_TAG=${IMAGE_TAG:-latest}

echo -e "${GREEN}Building and pushing EzSpeech Docker image${NC}"
echo "AWS Region: $AWS_REGION"
echo "AWS Account: $AWS_ACCOUNT_ID"
echo "ECR Repository: $ECR_REPOSITORY"
echo "Image Tag: $IMAGE_TAG"
echo ""

# Navigate to project root
cd "$(dirname "$0")/../.."

# Create ECR repository if it doesn't exist
echo -e "${YELLOW}Checking ECR repository...${NC}"
if ! aws ecr describe-repositories --repository-names "$ECR_REPOSITORY" --region "$AWS_REGION" >/dev/null 2>&1; then
    echo -e "${YELLOW}Creating ECR repository...${NC}"
    aws ecr create-repository \
        --repository-name "$ECR_REPOSITORY" \
        --region "$AWS_REGION" \
        --image-scanning-configuration scanOnPush=true \
        --encryption-configuration encryptionType=AES256
    echo -e "${GREEN}ECR repository created${NC}"
else
    echo -e "${GREEN}ECR repository already exists${NC}"
fi

# Login to ECR
echo -e "${YELLOW}Logging in to ECR...${NC}"
aws ecr get-login-password --region "$AWS_REGION" | \
    docker login --username AWS --password-stdin \
    "${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"

# Build Docker image
echo -e "${YELLOW}Building Docker image...${NC}"
docker build \
    -t "${ECR_REPOSITORY}:${IMAGE_TAG}" \
    -f Dockerfile \
    .

# Tag image for ECR
ECR_IMAGE_URI="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPOSITORY}:${IMAGE_TAG}"
echo -e "${YELLOW}Tagging image as ${ECR_IMAGE_URI}${NC}"
docker tag "${ECR_REPOSITORY}:${IMAGE_TAG}" "$ECR_IMAGE_URI"

# Push to ECR
echo -e "${YELLOW}Pushing image to ECR...${NC}"
docker push "$ECR_IMAGE_URI"

echo -e "${GREEN}Successfully pushed image to ECR!${NC}"
echo -e "Image URI: ${GREEN}${ECR_IMAGE_URI}${NC}"
echo ""
echo "Next steps:"
echo "  1. Update ECS service to use new image:"
echo "     aws ecs update-service --cluster ezspeech-cluster --service ezspeech-service --force-new-deployment"
echo "  2. Monitor deployment:"
echo "     aws ecs describe-services --cluster ezspeech-cluster --services ezspeech-service"
