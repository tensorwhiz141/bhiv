# BHIV Core Deployment Guide

This document provides comprehensive instructions for deploying BHIV Core in production environments, including Docker and AWS deployment options.

## Overview

BHIV Core can be deployed in several configurations:
- **Local Development**: Single machine with all services
- **Docker Compose**: Containerized deployment for development/testing
- **Docker Production**: Multi-container production deployment
- **AWS Cloud**: Scalable cloud deployment with managed services

## Prerequisites

### System Requirements

- **CPU**: 4+ cores recommended (8+ for production)
- **RAM**: 8GB minimum (16GB+ recommended for production)
- **Storage**: 20GB minimum (SSD recommended)
- **Network**: Stable internet connection for API calls

### Software Dependencies

- Docker 20.10+
- Docker Compose 2.0+
- Python 3.11+ (for local development)
- MongoDB 5.0+ (or MongoDB Atlas for cloud)
- Git

## Local Development Deployment

### 1. Environment Setup

```bash
# Clone repository
git clone <repository-url>
cd BHIV-Second-Installment

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install NLP models
python -m spacy download en_core_web_sm
```

### 2. Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit .env file with your configuration
nano .env
```

Required environment variables:
```env
# API Keys
GROQ_API_KEY=your_groq_api_key_here

# MongoDB Configuration
MONGO_URI=mongodb://localhost:27017
MONGO_DATABASE=bhiv_core
MONGO_COLLECTION=task_logs

# Service Ports
MCP_BRIDGE_PORT=8002
WEB_INTERFACE_PORT=8003
SIMPLE_API_PORT=8001

# Security
SECRET_KEY=your_secret_key_here
ALLOWED_HOSTS=localhost,127.0.0.1

# Logging
LOG_LEVEL=INFO
LOG_FILE=logs/bhiv_core.log
```

### 3. Start Services

```bash
# Start MongoDB
sudo systemctl start mongod

# Start MCP Bridge
python mcp_bridge.py &

# Start Web Interface
python integration/web_interface.py &

# Optional: Start Simple API
python simple_api.py &
```

## Docker Deployment

### 1. Docker Compose (Development)

Create `docker-compose.yml`:

```yaml
version: '3.8'

services:
  mongodb:
    image: mongo:5.0
    container_name: bhiv-mongodb
    restart: unless-stopped
    ports:
      - "27017:27017"
    volumes:
      - mongodb_data:/data/db
    environment:
      MONGO_INITDB_ROOT_USERNAME: admin
      MONGO_INITDB_ROOT_PASSWORD: password123

  mcp-bridge:
    build: .
    container_name: bhiv-mcp-bridge
    restart: unless-stopped
    ports:
      - "8002:8002"
    depends_on:
      - mongodb
    environment:
      - MONGO_URI=mongodb://admin:password123@mongodb:27017/bhiv_core?authSource=admin
      - GROQ_API_KEY=${GROQ_API_KEY}
    volumes:
      - ./logs:/app/logs
      - ./temp:/app/temp
    command: python mcp_bridge.py

  web-interface:
    build: .
    container_name: bhiv-web-interface
    restart: unless-stopped
    ports:
      - "8003:8003"
    depends_on:
      - mongodb
      - mcp-bridge
    environment:
      - MONGO_URI=mongodb://admin:password123@mongodb:27017/bhiv_core?authSource=admin
    volumes:
      - ./logs:/app/logs
      - ./temp:/app/temp
      - ./templates:/app/templates
      - ./static:/app/static
    command: python integration/web_interface.py

volumes:
  mongodb_data:
```

Create `Dockerfile`:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install NLP models
RUN python -m spacy download en_core_web_sm

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p logs temp static templates

# Expose ports
EXPOSE 8001 8002 8003

# Default command
CMD ["python", "mcp_bridge.py"]
```

### 2. Build and Run

```bash
# Build and start services
docker-compose up --build -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

## Production Docker Deployment

### 1. Multi-Stage Dockerfile

Create `Dockerfile.prod`:

```dockerfile
# Build stage
FROM python:3.11-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libsndfile1-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Production stage
FROM python:3.11-slim

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Install NLP models
RUN python -m spacy download en_core_web_sm

# Create non-root user
RUN useradd --create-home --shell /bin/bash bhiv
USER bhiv

# Copy application code
COPY --chown=bhiv:bhiv . .

# Create necessary directories
RUN mkdir -p logs temp static templates

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8002/health || exit 1

EXPOSE 8002

CMD ["python", "mcp_bridge.py"]
```

### 2. Production Docker Compose

Create `docker-compose.prod.yml`:

```yaml
version: '3.8'

services:
  mongodb:
    image: mongo:5.0
    container_name: bhiv-mongodb-prod
    restart: always
    ports:
      - "27017:27017"
    volumes:
      - mongodb_data:/data/db
      - ./mongo-init:/docker-entrypoint-initdb.d
    environment:
      MONGO_INITDB_ROOT_USERNAME: ${MONGO_ROOT_USER}
      MONGO_INITDB_ROOT_PASSWORD: ${MONGO_ROOT_PASSWORD}
      MONGO_INITDB_DATABASE: bhiv_core
    networks:
      - bhiv-network

  mcp-bridge:
    build:
      context: .
      dockerfile: Dockerfile.prod
    container_name: bhiv-mcp-bridge-prod
    restart: always
    ports:
      - "8002:8002"
    depends_on:
      - mongodb
    environment:
      - MONGO_URI=mongodb://${MONGO_ROOT_USER}:${MONGO_ROOT_PASSWORD}@mongodb:27017/bhiv_core?authSource=admin
      - GROQ_API_KEY=${GROQ_API_KEY}
      - LOG_LEVEL=INFO
    volumes:
      - ./logs:/app/logs
      - ./temp:/app/temp
    networks:
      - bhiv-network
    deploy:
      resources:
        limits:
          memory: 2G
        reservations:
          memory: 1G

  web-interface:
    build:
      context: .
      dockerfile: Dockerfile.prod
    container_name: bhiv-web-interface-prod
    restart: always
    ports:
      - "8003:8003"
    depends_on:
      - mongodb
      - mcp-bridge
    environment:
      - MONGO_URI=mongodb://${MONGO_ROOT_USER}:${MONGO_ROOT_PASSWORD}@mongodb:27017/bhiv_core?authSource=admin
    volumes:
      - ./logs:/app/logs
      - ./temp:/app/temp
      - ./templates:/app/templates
      - ./static:/app/static
    networks:
      - bhiv-network
    command: python integration/web_interface.py
    deploy:
      resources:
        limits:
          memory: 1G
        reservations:
          memory: 512M

  nginx:
    image: nginx:alpine
    container_name: bhiv-nginx
    restart: always
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - web-interface
      - mcp-bridge
    networks:
      - bhiv-network

volumes:
  mongodb_data:

networks:
  bhiv-network:
    driver: bridge
```

### 3. Nginx Configuration

Create `nginx.conf`:

```nginx
events {
    worker_connections 1024;
}

http {
    upstream web-interface {
        server web-interface:8003;
    }
    
    upstream mcp-bridge {
        server mcp-bridge:8002;
    }
    
    server {
        listen 80;
        server_name your-domain.com;
        
        # Redirect HTTP to HTTPS
        return 301 https://$server_name$request_uri;
    }
    
    server {
        listen 443 ssl;
        server_name your-domain.com;
        
        ssl_certificate /etc/nginx/ssl/cert.pem;
        ssl_certificate_key /etc/nginx/ssl/key.pem;
        
        # Web interface
        location / {
            proxy_pass http://web-interface;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }
        
        # API endpoints
        location /api/ {
            proxy_pass http://mcp-bridge/;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }
        
        # Health checks
        location /health {
            proxy_pass http://mcp-bridge/health;
        }
    }
}
```

## AWS Cloud Deployment

### 1. AWS Architecture

**Recommended AWS Services:**
- **ECS Fargate**: Container orchestration
- **Application Load Balancer**: Traffic distribution
- **DocumentDB**: MongoDB-compatible database
- **S3**: File storage
- **CloudWatch**: Monitoring and logging
- **Route 53**: DNS management
- **Certificate Manager**: SSL certificates

### 2. ECS Task Definitions

Create `task-definition.json`:

```json
{
  "family": "bhiv-core",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "1024",
  "memory": "2048",
  "executionRoleArn": "arn:aws:iam::ACCOUNT:role/ecsTaskExecutionRole",
  "taskRoleArn": "arn:aws:iam::ACCOUNT:role/ecsTaskRole",
  "containerDefinitions": [
    {
      "name": "mcp-bridge",
      "image": "your-account.dkr.ecr.region.amazonaws.com/bhiv-core:latest",
      "portMappings": [
        {
          "containerPort": 8002,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "MONGO_URI",
          "value": "mongodb://username:password@docdb-cluster.cluster-xxx.region.docdb.amazonaws.com:27017/bhiv_core?ssl=true&replicaSet=rs0&readPreference=secondaryPreferred&retryWrites=false"
        }
      ],
      "secrets": [
        {
          "name": "GROQ_API_KEY",
          "valueFrom": "arn:aws:secretsmanager:region:account:secret:bhiv-core/groq-api-key"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/bhiv-core",
          "awslogs-region": "us-west-2",
          "awslogs-stream-prefix": "ecs"
        }
      },
      "healthCheck": {
        "command": ["CMD-SHELL", "curl -f http://localhost:8002/health || exit 1"],
        "interval": 30,
        "timeout": 5,
        "retries": 3
      }
    }
  ]
}
```

### 3. Terraform Configuration

Create `main.tf`:

```hcl
provider "aws" {
  region = var.aws_region
}

# VPC and Networking
resource "aws_vpc" "bhiv_vpc" {
  cidr_block           = "10.0.0.0/16"
  enable_dns_hostnames = true
  enable_dns_support   = true

  tags = {
    Name = "bhiv-vpc"
  }
}

resource "aws_subnet" "bhiv_subnet" {
  count             = 2
  vpc_id            = aws_vpc.bhiv_vpc.id
  cidr_block        = "10.0.${count.index + 1}.0/24"
  availability_zone = data.aws_availability_zones.available.names[count.index]

  tags = {
    Name = "bhiv-subnet-${count.index + 1}"
  }
}

# DocumentDB Cluster
resource "aws_docdb_cluster" "bhiv_docdb" {
  cluster_identifier      = "bhiv-docdb-cluster"
  engine                  = "docdb"
  master_username         = var.docdb_username
  master_password         = var.docdb_password
  backup_retention_period = 7
  preferred_backup_window = "07:00-09:00"
  skip_final_snapshot     = true

  vpc_security_group_ids = [aws_security_group.docdb_sg.id]
  db_subnet_group_name   = aws_docdb_subnet_group.bhiv_docdb_subnet_group.name
}

# ECS Cluster
resource "aws_ecs_cluster" "bhiv_cluster" {
  name = "bhiv-cluster"

  setting {
    name  = "containerInsights"
    value = "enabled"
  }
}

# Application Load Balancer
resource "aws_lb" "bhiv_alb" {
  name               = "bhiv-alb"
  internal           = false
  load_balancer_type = "application"
  security_groups    = [aws_security_group.alb_sg.id]
  subnets            = aws_subnet.bhiv_subnet[*].id

  enable_deletion_protection = false
}
```

### 4. Deployment Scripts

Create `deploy.sh`:

```bash
#!/bin/bash

set -e

# Configuration
AWS_REGION="us-west-2"
ECR_REPOSITORY="bhiv-core"
CLUSTER_NAME="bhiv-cluster"
SERVICE_NAME="bhiv-service"

# Build and push Docker image
echo "Building Docker image..."
docker build -f Dockerfile.prod -t $ECR_REPOSITORY:latest .

# Get ECR login token
aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com

# Tag and push image
docker tag $ECR_REPOSITORY:latest $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPOSITORY:latest
docker push $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPOSITORY:latest

# Update ECS service
echo "Updating ECS service..."
aws ecs update-service --cluster $CLUSTER_NAME --service $SERVICE_NAME --force-new-deployment

echo "Deployment completed!"
```

## Monitoring and Maintenance

### 1. Health Checks

Set up monitoring endpoints:

```bash
# Service health
curl http://your-domain.com/health

# Detailed metrics
curl http://your-domain.com/metrics

# Database connectivity
curl http://your-domain.com/api/nlos?limit=1
```

### 2. Logging

Configure centralized logging:

```yaml
# docker-compose.yml logging section
logging:
  driver: "json-file"
  options:
    max-size: "10m"
    max-file: "3"
```

### 3. Backup Strategy

```bash
# MongoDB backup script
#!/bin/bash
DATE=$(date +%Y%m%d_%H%M%S)
mongodump --uri="$MONGO_URI" --out="/backups/bhiv_backup_$DATE"
tar -czf "/backups/bhiv_backup_$DATE.tar.gz" "/backups/bhiv_backup_$DATE"
rm -rf "/backups/bhiv_backup_$DATE"

# Upload to S3 (if using AWS)
aws s3 cp "/backups/bhiv_backup_$DATE.tar.gz" "s3://your-backup-bucket/"
```

### 4. Scaling

Configure auto-scaling:

```yaml
# docker-compose.yml with scaling
deploy:
  replicas: 3
  update_config:
    parallelism: 1
    delay: 10s
  restart_policy:
    condition: on-failure
```

## Security Considerations

### 1. Environment Variables

Never commit sensitive data:

```bash
# Use secrets management
docker secret create groq_api_key groq_api_key.txt
docker secret create mongo_password mongo_password.txt
```

### 2. Network Security

```yaml
# Restrict network access
networks:
  bhiv-internal:
    driver: bridge
    internal: true
  bhiv-external:
    driver: bridge
```

### 3. SSL/TLS

Always use HTTPS in production:

```bash
# Generate SSL certificates
certbot --nginx -d your-domain.com
```

## Troubleshooting

### Common Issues

1. **Container startup failures**: Check logs with `docker logs container-name`
2. **Database connection issues**: Verify MongoDB connectivity and credentials
3. **Memory issues**: Increase container memory limits
4. **API rate limits**: Implement proper rate limiting and caching

### Performance Optimization

1. **Use multi-stage builds** to reduce image size
2. **Implement caching** for frequently accessed data
3. **Use connection pooling** for database connections
4. **Monitor resource usage** and scale accordingly

## Support

For deployment issues:
1. Check service logs and health endpoints
2. Verify all environment variables are set correctly
3. Ensure all required services are running
4. Contact support with deployment logs and configuration details
```
