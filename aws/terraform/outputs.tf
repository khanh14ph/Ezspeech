output "ecr_repository_url" {
  description = "URL of the ECR repository"
  value       = aws_ecr_repository.ezspeech.repository_url
}

output "alb_dns_name" {
  description = "DNS name of the Application Load Balancer"
  value       = aws_lb.main.dns_name
}

output "alb_zone_id" {
  description = "Zone ID of the Application Load Balancer"
  value       = aws_lb.main.zone_id
}

output "ecs_cluster_name" {
  description = "Name of the ECS cluster"
  value       = aws_ecs_cluster.main.name
}

output "ecs_service_name" {
  description = "Name of the ECS service"
  value       = aws_ecs_service.app.name
}

output "cloudwatch_log_group" {
  description = "Name of the CloudWatch log group"
  value       = aws_cloudwatch_log_group.ecs.name
}

output "s3_models_bucket" {
  description = "Name of the S3 bucket for models"
  value       = aws_s3_bucket.models.id
}

output "vpc_id" {
  description = "ID of the VPC"
  value       = module.vpc.vpc_id
}

output "private_subnet_ids" {
  description = "IDs of private subnets"
  value       = module.vpc.private_subnets
}

output "public_subnet_ids" {
  description = "IDs of public subnets"
  value       = module.vpc.public_subnets
}

output "security_group_alb_id" {
  description = "ID of the ALB security group"
  value       = aws_security_group.alb.id
}

output "security_group_ecs_tasks_id" {
  description = "ID of the ECS tasks security group"
  value       = aws_security_group.ecs_tasks.id
}

output "task_execution_role_arn" {
  description = "ARN of the ECS task execution role"
  value       = aws_iam_role.ecs_task_execution.arn
}

output "task_role_arn" {
  description = "ARN of the ECS task role"
  value       = aws_iam_role.ecs_task.arn
}

# Useful commands
output "useful_commands" {
  description = "Useful commands for managing the deployment"
  value = <<-EOT
    # View ECS service status
    aws ecs describe-services --cluster ${aws_ecs_cluster.main.name} --services ${aws_ecs_service.app.name}

    # View ECS tasks
    aws ecs list-tasks --cluster ${aws_ecs_cluster.main.name} --service-name ${aws_ecs_service.app.name}

    # View CloudWatch logs
    aws logs tail ${aws_cloudwatch_log_group.ecs.name} --follow

    # Update service to use new image
    aws ecs update-service --cluster ${aws_ecs_cluster.main.name} --service ${aws_ecs_service.app.name} --force-new-deployment

    # Access the service
    WebSocket: ws://${aws_lb.main.dns_name}:80
    Health: http://${aws_lb.main.dns_name}/health

    # Upload model to S3
    aws s3 cp model.ckpt s3://${aws_s3_bucket.models.id}/models/
  EOT
}
