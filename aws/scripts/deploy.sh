#!/bin/bash
# Deploy EzSpeech to AWS using Terraform

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
ENVIRONMENT=${ENVIRONMENT:-dev}
AWS_REGION=${AWS_REGION:-us-east-1}

echo -e "${GREEN}Deploying EzSpeech to AWS${NC}"
echo "Environment: $ENVIRONMENT"
echo "AWS Region: $AWS_REGION"
echo ""

# Navigate to terraform directory
cd "$(dirname "$0")/../terraform"

# Check if terraform.tfvars exists
if [ ! -f terraform.tfvars ]; then
    echo -e "${YELLOW}terraform.tfvars not found. Creating from example...${NC}"
    if [ -f terraform.tfvars.example ]; then
        cp terraform.tfvars.example terraform.tfvars
        echo -e "${RED}Please edit terraform.tfvars with your configuration before running this script again.${NC}"
        exit 1
    else
        echo -e "${RED}terraform.tfvars.example not found!${NC}"
        exit 1
    fi
fi

# Initialize Terraform
echo -e "${YELLOW}Initializing Terraform...${NC}"
terraform init

# Validate configuration
echo -e "${YELLOW}Validating Terraform configuration...${NC}"
terraform validate

# Plan deployment
echo -e "${YELLOW}Planning deployment...${NC}"
terraform plan -var="environment=${ENVIRONMENT}" -var="aws_region=${AWS_REGION}" -out=tfplan

# Ask for confirmation
echo ""
read -p "Do you want to apply this plan? (yes/no): " -r
if [[ ! $REPLY =~ ^[Yy]es$ ]]; then
    echo -e "${YELLOW}Deployment cancelled.${NC}"
    rm -f tfplan
    exit 0
fi

# Apply configuration
echo -e "${YELLOW}Applying Terraform configuration...${NC}"
terraform apply tfplan

# Clean up plan file
rm -f tfplan

echo ""
echo -e "${GREEN}Deployment complete!${NC}"
echo ""
echo "Outputs:"
terraform output

echo ""
echo "Next steps:"
echo "  1. Build and push Docker image:"
echo "     cd ../../ && ./aws/scripts/build-and-push.sh"
echo "  2. Update ECS service:"
echo "     aws ecs update-service --cluster \$(terraform output -raw ecs_cluster_name) --service \$(terraform output -raw ecs_service_name) --force-new-deployment"
echo "  3. Monitor logs:"
echo "     aws logs tail \$(terraform output -raw cloudwatch_log_group) --follow"
