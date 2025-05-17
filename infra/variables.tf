variable "region" {
  description = "The AWS region to deploy the infrastructure."
  type        = string
  default     = "us-east-1"
}

variable "environment" {
  description = "Deployment environment (e.g., dev, staging, prod)"
  type        = string
  default     = "prod"
}

variable "vpc_id" {
  description = "The ID of the VPC where the RDS instance will be deployed"
  type        = string
  default     = "vpc-0ae37bc4e75d574ad"
}

variable "database_subnet_ids" {
  description = "List of subnet IDs for the RDS subnet group"
  type        = list(string)
  default     = ["subnet-0d35ca21bbec62faa", "subnet-00751115f579e639c"]
}