#!/bin/bash
# Upload model checkpoint to S3

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check arguments
if [ $# -lt 1 ]; then
    echo "Usage: $0 <model_checkpoint_path> [s3_key]"
    echo ""
    echo "Example:"
    echo "  $0 outputs/model.ckpt models/model.ckpt"
    exit 1
fi

MODEL_PATH=$1
S3_KEY=${2:-models/model.ckpt}
AWS_REGION=${AWS_REGION:-us-east-1}
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
S3_BUCKET="ezspeech-models-${AWS_ACCOUNT_ID}"

echo -e "${GREEN}Uploading model to S3${NC}"
echo "Model: $MODEL_PATH"
echo "S3 Bucket: $S3_BUCKET"
echo "S3 Key: $S3_KEY"
echo ""

# Check if model file exists
if [ ! -f "$MODEL_PATH" ]; then
    echo -e "${RED}Error: Model file not found: $MODEL_PATH${NC}"
    exit 1
fi

# Upload to S3
echo -e "${YELLOW}Uploading...${NC}"
aws s3 cp "$MODEL_PATH" "s3://${S3_BUCKET}/${S3_KEY}" --region "$AWS_REGION"

echo -e "${GREEN}Model uploaded successfully!${NC}"
echo ""
echo "S3 URI: s3://${S3_BUCKET}/${S3_KEY}"
echo ""
echo "To use this model in ECS, update your task definition environment variables:"
echo "  MODEL_S3_BUCKET=${S3_BUCKET}"
echo "  MODEL_S3_KEY=${S3_KEY}"
echo ""
echo "Or download it during container startup with:"
echo "  aws s3 cp s3://${S3_BUCKET}/${S3_KEY} /app/models/model.ckpt"
