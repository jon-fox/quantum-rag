provider "aws" {
  region = var.region
}

provider "random" {}

terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      # version = "~> 4.0"
    }
    random = {
      source  = "hashicorp/random"
      # version = "~> 3.0"
    }
  }
  
  backend "s3" {
    key    = "terraform/watch_arb/terraform.tfstate"  # Path inside the bucket to store the state
    region = "us-east-1"  # AWS region, e.g., us-west-2
  }
}

resource "aws_dynamodb_table" "ebay_watch_listings" {
  name           = "ebay_watch_listings"
  billing_mode   = "PAY_PER_REQUEST"
  hash_key       = "item_id"

  # Primary key attribute
  attribute {
    name = "item_id"
    type = "S"
  }
  
  # Price attribute for GSI
  attribute {
    name = "price"
    type = "N"
  }
  
  # Last updated attribute for GSI - more reliable than creation_date which may be missing
  attribute {
    name = "last_updated"
    type = "S"
  }

  # Price-based GSI for finding deals within price ranges
  global_secondary_index {
    name               = "PriceIndex"
    hash_key           = "price"
    projection_type    = "INCLUDE"
    non_key_attributes = ["title", "item_url", "image_url", "condition", "seller_username"]
  }
  
  # Date-based GSI for finding recent listings - using last_updated which will always exist
  global_secondary_index {
    name               = "DateIndex"
    hash_key           = "last_updated"
    projection_type    = "INCLUDE"
    non_key_attributes = ["title", "price", "item_url", "image_url", "item_id"]
  }
  
  # TTL is optional but configured for future use if needed
#   ttl {
#     attribute_name = "expiry_time"
#     enabled        = true
#   }
  
  # Enable point-in-time recovery for data protection
  point_in_time_recovery {
    enabled = true
  }

  tags = {
    Name        = "ebay_watch_listings"
    Environment = var.environment
    Project     = "watch-arbitrage"
    ManagedBy   = "terraform"
    Purpose     = "watch-deal-analysis"
    CreatedDate = "2025-05-06"
  }
}
