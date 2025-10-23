#!/bin/bash
# Update ECS service with new Docker image

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
AWS_REGION=${AWS_REGION:-us-east-1}
CLUSTER_NAME=${CLUSTER_NAME:-ezspeech-cluster}
SERVICE_NAME=${SERVICE_NAME:-ezspeech-service}

echo -e "${GREEN}Updating ECS service${NC}"
echo "Cluster: $CLUSTER_NAME"
echo "Service: $SERVICE_NAME"
echo "Region: $AWS_REGION"
echo ""

# Force new deployment
echo -e "${YELLOW}Triggering service update...${NC}"
aws ecs update-service \
    --cluster "$CLUSTER_NAME" \
    --service "$SERVICE_NAME" \
    --force-new-deployment \
    --region "$AWS_REGION" \
    >/dev/null

echo -e "${GREEN}Service update triggered!${NC}"
echo ""

# Monitor deployment
echo -e "${YELLOW}Monitoring deployment status...${NC}"
echo "Press Ctrl+C to stop monitoring (deployment will continue)"
echo ""

while true; do
    # Get service status
    STATUS=$(aws ecs describe-services \
        --cluster "$CLUSTER_NAME" \
        --services "$SERVICE_NAME" \
        --region "$AWS_REGION" \
        --query 'services[0].deployments[0]' \
        --output json)

    RUNNING_COUNT=$(echo "$STATUS" | jq -r '.runningCount')
    DESIRED_COUNT=$(echo "$STATUS" | jq -r '.desiredCount')
    STATUS_TEXT=$(echo "$STATUS" | jq -r '.rolloutState // "IN_PROGRESS"')

    echo -e "Status: ${YELLOW}${STATUS_TEXT}${NC} | Running: ${RUNNING_COUNT}/${DESIRED_COUNT}"

    if [ "$STATUS_TEXT" = "COMPLETED" ]; then
        echo -e "${GREEN}Deployment completed successfully!${NC}"
        break
    fi

    sleep 10
done

echo ""
echo "View logs with:"
echo "  aws logs tail /ecs/${SERVICE_NAME} --follow --region ${AWS_REGION}"
