# Generate a random password for the RDS instance
resource "random_password" "db_password" {
  length           = 16
  special          = true
  override_special = "!#$%&*()-_=+[]{}<>:?"
}

# Create a security group for the RDS instance
resource "aws_security_group" "rds_sg" {
  name        = "rds-security-group"
  description = "Security group for RDS instance"
  vpc_id      = var.vpc_id

  # Allow PostgreSQL connections from anywhere (for testing only)
  # In production, restrict this to specific IP addresses or security groups
  ingress {
    from_port   = 5432
    to_port     = 5432
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]  # Allow from any IP address
    description = "Allow PostgreSQL access from anywhere (for testing)"
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "watch-arb-rds-sg"
  }
}

# Create a subnet group for the RDS instance
resource "aws_db_subnet_group" "rds_subnet_group" {
  name       = "watch-arb-rds-subnet-group"
  subnet_ids = var.database_subnet_ids

  tags = {
    Name = "watch-arb-rds-subnet-group"
  }
}

# Create a parameter group for PostgreSQL with pgvector extension
resource "aws_db_parameter_group" "postgres_pgvector" {
  name        = "postgres-pgvector"
  family      = "postgres17"
  description = "PostgreSQL parameter group with pgvector extension"

  # Update the allowed extensions parameter with correct syntax
  # In RDS, multiple extensions need to be comma-separated
  parameter {
    apply_method = "pending-reboot"
    name         = "rds.allowed_extensions"
    value        = "vector,pg_stat_statements,pgcrypto"  # Include common extensions along with vector
  }
}

# Create the RDS instance with PostgreSQL and pgvector
resource "aws_db_instance" "watch_arb_db" {
  identifier             = "watch-arb-db"
  engine                 = "postgres"
  engine_version         = "17.2"  # Use a version that supports pgvector
  instance_class         = "db.t3.small"
  allocated_storage      = 20
  max_allocated_storage  = 100
  storage_type           = "gp2"
  storage_encrypted      = true
  
  db_name                = "watch_arb"
  username               = "watchadmin"
  password               = random_password.db_password.result
  
  parameter_group_name   = aws_db_parameter_group.postgres_pgvector.name
  db_subnet_group_name   = aws_db_subnet_group.rds_subnet_group.name
  vpc_security_group_ids = [aws_security_group.rds_sg.id]
  
  # Add publicly_accessible option to allow direct connections for testing
  # For a production environment, consider removing this and using a bastion host
  publicly_accessible    = true
  
  skip_final_snapshot    = true  # Set to false for production
  final_snapshot_identifier = "watch-arb-db-final-snapshot"
  deletion_protection    = false  # Set to true for production
  backup_retention_period = 7
  backup_window          = "03:00-04:00"
  maintenance_window     = "Mon:04:00-Mon:05:00"
  
  # Enable minor version upgrade
  auto_minor_version_upgrade = true
  
  tags = {
    Name        = "watch-arb-db"
    Environment = var.environment
    Project     = "watch-arb"
  }
}

# Store RDS connection details in SSM Parameter Store
resource "aws_ssm_parameter" "db_endpoint" {
  name        = "/${var.environment}/watch-arb/db/endpoint"
  description = "RDS instance endpoint"
  type        = "String"
  value       = aws_db_instance.watch_arb_db.endpoint
  tags = {
    Environment = var.environment
    Project     = "watch-arb"
  }
}

resource "aws_ssm_parameter" "db_address" {
  name        = "/${var.environment}/watch-arb/db/address"
  description = "RDS instance address"
  type        = "String"
  value       = aws_db_instance.watch_arb_db.address
  tags = {
    Environment = var.environment
    Project     = "watch-arb"
  }
}

resource "aws_ssm_parameter" "db_port" {
  name        = "/${var.environment}/watch-arb/db/port"
  description = "RDS instance port"
  type        = "String"
  value       = aws_db_instance.watch_arb_db.port
  tags = {
    Environment = var.environment
    Project     = "watch-arb"
  }
}

resource "aws_ssm_parameter" "db_name" {
  name        = "/${var.environment}/watch-arb/db/name"
  description = "RDS database name"
  type        = "String"
  value       = "watch_arb"
  tags = {
    Environment = var.environment
    Project     = "watch-arb"
  }
}

resource "aws_ssm_parameter" "db_username" {
  name        = "/${var.environment}/watch-arb/db/username"
  description = "RDS username"
  type        = "String"
  value       = "watchadmin"
  tags = {
    Environment = var.environment
    Project     = "watch-arb"
  }
}

resource "aws_ssm_parameter" "db_password" {
  name        = "/${var.environment}/watch-arb/db/password"
  description = "RDS password"
  type        = "SecureString"
  value       = random_password.db_password.result
  tags = {
    Environment = var.environment
    Project     = "watch-arb"
  }
}

# Output the RDS endpoint
output "rds_endpoint" {
  value = aws_db_instance.watch_arb_db.endpoint
  description = "The endpoint of the RDS instance"
}

output "rds_address" {
  value = aws_db_instance.watch_arb_db.address
  description = "The hostname of the RDS instance"
}

output "rds_port" {
  value = aws_db_instance.watch_arb_db.port
  description = "The port of the RDS instance"
}

# Add note about pgvector extension installation
output "pgvector_setup_note" {
  value = "After the RDS instance is created, connect to it and run: CREATE EXTENSION IF NOT EXISTS vector;"
  description = "Note about setting up pgvector extension manually"
}