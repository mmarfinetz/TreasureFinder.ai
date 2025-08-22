# 🚀 TreasureHunter Production Deployment Guide

This guide covers multiple deployment options for the TreasureHunter satellite analysis system.

## 📋 Table of Contents
- [Prerequisites](#prerequisites)
- [Quick Start (Docker)](#quick-start-docker)
- [Cloud Deployments](#cloud-deployments)
  - [AWS Deployment](#aws-deployment)
  - [Google Cloud Platform](#google-cloud-platform)
  - [Azure Deployment](#azure-deployment)
  - [DigitalOcean](#digitalocean)
  - [Heroku](#heroku)
- [Manual VPS Deployment](#manual-vps-deployment)
- [SSL/HTTPS Setup](#sslhttps-setup)
- [Monitoring & Maintenance](#monitoring--maintenance)
- [Troubleshooting](#troubleshooting)

## Prerequisites

### Required API Keys
At minimum, you need one satellite data provider:
- **Mapbox Token** (easiest): Get from [mapbox.com](https://www.mapbox.com/)
- **Google Earth Engine** (advanced): [earthengine.google.com](https://earthengine.google.com/)

### System Requirements
- **Minimum**: 2 CPU cores, 4GB RAM, 20GB storage
- **Recommended**: 4 CPU cores, 8GB RAM, 50GB storage
- **GPU (for training)**: NVIDIA GPU with 8GB+ VRAM

## Quick Start (Docker)

### 1. Clone and Configure
```bash
# Clone repository
git clone <your-repo-url>
cd Compare_Satellite_scripts

# Create environment file
cp .env.example .env
# Edit .env with your API keys
nano .env
```

### 2. Build and Run with Docker Compose
```bash
# Build and start all services
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f treasurehunter
```

Your app is now running at http://localhost

### 3. Stop Services
```bash
docker-compose down
```

## Cloud Deployments

### AWS Deployment

#### Option 1: EC2 with Docker (Recommended)
```bash
# Make script executable
chmod +x deploy_aws.sh

# Launch new instance
./deploy_aws.sh launch

# Deploy updates to existing instance
./deploy_aws.sh deploy <instance-ip>
```

#### Option 2: Elastic Beanstalk
1. Install EB CLI: `pip install awsebcli`
2. Initialize: `eb init -p docker treasurehunter`
3. Create environment: `eb create treasurehunter-env`
4. Deploy: `eb deploy`
5. Open: `eb open`

#### Option 3: ECS (Container Service)
```bash
# Build and push to ECR
aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin <account-id>.dkr.ecr.us-west-2.amazonaws.com
docker build -t treasurehunter .
docker tag treasurehunter:latest <account-id>.dkr.ecr.us-west-2.amazonaws.com/treasurehunter:latest
docker push <account-id>.dkr.ecr.us-west-2.amazonaws.com/treasurehunter:latest

# Create ECS task and service via console or CLI
```

### Google Cloud Platform

#### Option 1: Cloud Run (Serverless - Recommended)
```bash
# Set project
export GCP_PROJECT_ID=your-project-id

# Deploy
chmod +x deploy_gcp.sh
./deploy_gcp.sh cloudrun
```

#### Option 2: Compute Engine (VM)
```bash
./deploy_gcp.sh compute
```

#### Option 3: Google Kubernetes Engine
```bash
./deploy_gcp.sh gke
```

### Azure Deployment

#### Option 1: Azure Container Instances
```bash
# Login to Azure
az login

# Create resource group
az group create --name treasurehunter-rg --location westus2

# Create container registry
az acr create --resource-group treasurehunter-rg --name treasurehunteracr --sku Basic

# Build and push image
az acr build --registry treasurehunteracr --image treasurehunter:latest .

# Deploy container
az container create \
  --resource-group treasurehunter-rg \
  --name treasurehunter \
  --image treasurehunteracr.azurecr.io/treasurehunter:latest \
  --cpu 2 --memory 4 \
  --registry-login-server treasurehunteracr.azurecr.io \
  --ip-address Public \
  --ports 80 443 \
  --environment-variables PRODUCTION_MODE=true
```

#### Option 2: Azure App Service
```bash
# Create App Service plan
az appservice plan create --name treasurehunter-plan --resource-group treasurehunter-rg --sku B2 --is-linux

# Create web app
az webapp create --resource-group treasurehunter-rg --plan treasurehunter-plan --name treasurehunter-app --deployment-container-image-name treasurehunter:latest
```

### DigitalOcean

#### App Platform (Easiest)
1. Push code to GitHub
2. Go to [DigitalOcean App Platform](https://cloud.digitalocean.com/apps)
3. Create App → GitHub → Select repo
4. Choose "Dockerfile" as source
5. Set environment variables
6. Deploy

#### Droplet (VM)
```bash
# Create droplet via CLI
doctl compute droplet create treasurehunter \
  --image docker-20-04 \
  --size s-2vcpu-4gb \
  --region nyc1 \
  --ssh-keys <your-ssh-key-id>

# SSH and deploy
ssh root@<droplet-ip>
git clone <your-repo>
cd Compare_Satellite_scripts
docker-compose up -d
```

### Heroku

#### Create Heroku app
```bash
# Install Heroku CLI
# Create app
heroku create treasurehunter-app

# Set buildpack
heroku buildpacks:set heroku/python

# Set environment variables
heroku config:set PRODUCTION_MODE=true
heroku config:set MAPBOX_ACCESS_TOKEN=your-token

# Deploy
git push heroku main

# Scale
heroku ps:scale web=1
```

Create `Procfile`:
```
web: gunicorn treasure_api:app --workers 4 --threads 2
```

## Manual VPS Deployment

For any VPS (Linode, Vultr, OVH, etc.):

### 1. Server Setup
```bash
# SSH into server
ssh root@your-server-ip

# Update system
apt update && apt upgrade -y

# Install Docker
curl -fsSL https://get.docker.com | sh

# Install Docker Compose
curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
chmod +x /usr/local/bin/docker-compose

# Install git
apt install git -y
```

### 2. Deploy Application
```bash
# Clone repository
cd /opt
git clone <your-repo-url> treasurehunter
cd treasurehunter

# Configure environment
cp .env.example .env
nano .env  # Add your API keys

# Start services
docker-compose up -d

# Set up firewall
ufw allow 22/tcp
ufw allow 80/tcp
ufw allow 443/tcp
ufw enable
```

## SSL/HTTPS Setup

### Using Let's Encrypt with Certbot
```bash
# Install Certbot
apt install certbot python3-certbot-nginx -y

# Get certificate
certbot --nginx -d yourdomain.com -d www.yourdomain.com

# Auto-renewal
certbot renew --dry-run
```

### Using Cloudflare (Recommended)
1. Add your domain to Cloudflare
2. Update DNS to point to your server
3. Enable "Full SSL/TLS" mode
4. Enable "Always Use HTTPS"

## Monitoring & Maintenance

### Health Checks
```bash
# Check service status
curl http://your-domain/api/status

# Check Docker containers
docker-compose ps

# View logs
docker-compose logs -f --tail=100
```

### Backup Strategy
```bash
# Backup models and data
tar -czf backup-$(date +%Y%m%d).tar.gz saved_models/ data/

# Backup to S3
aws s3 cp backup-*.tar.gz s3://your-backup-bucket/
```

### Updates
```bash
# Pull latest code
git pull origin main

# Rebuild and restart
docker-compose down
docker-compose build
docker-compose up -d
```

### Monitoring Tools
- **Uptime**: [UptimeRobot](https://uptimerobot.com/) or [Pingdom](https://www.pingdom.com/)
- **Logs**: [Papertrail](https://papertrailapp.com/) or [Loggly](https://www.loggly.com/)
- **Metrics**: [New Relic](https://newrelic.com/) or [Datadog](https://www.datadoghq.com/)

## Performance Optimization

### 1. Enable Redis Caching
```yaml
# Add to docker-compose.yml (already included)
redis:
  image: redis:7-alpine
  ports:
    - "6379:6379"
```

### 2. Use CDN for Static Files
- Cloudflare (free tier available)
- AWS CloudFront
- Fastly

### 3. Database Optimization
If using external database:
```sql
-- Create indexes
CREATE INDEX idx_analysis_location ON analyses(lat, lon);
CREATE INDEX idx_models_date ON models(trained_date);
```

## Troubleshooting

### Common Issues

#### Port Already in Use
```bash
# Find process using port
lsof -i :5000
# Kill process
kill -9 <PID>
```

#### Docker Permission Denied
```bash
# Add user to docker group
sudo usermod -aG docker $USER
# Re-login
newgrp docker
```

#### Out of Memory
```bash
# Check memory usage
docker stats

# Increase swap
fallocate -l 4G /swapfile
chmod 600 /swapfile
mkswap /swapfile
swapon /swapfile
```

#### SSL Certificate Issues
```bash
# Renew certificate
certbot renew --force-renewal

# Check certificate
openssl s_client -connect yourdomain.com:443
```

### Debug Mode
```bash
# Enable debug mode
export FLASK_DEBUG=true
docker-compose restart treasurehunter

# View detailed logs
docker-compose logs -f treasurehunter
```

## Security Best Practices

1. **Environment Variables**
   - Never commit `.env` file
   - Use secrets management (AWS Secrets Manager, etc.)

2. **Firewall Rules**
   - Only open necessary ports
   - Restrict SSH to specific IPs

3. **Updates**
   - Regular security updates: `apt update && apt upgrade`
   - Monitor dependencies: `pip list --outdated`

4. **Rate Limiting**
   - Configured in nginx.conf
   - Adjust limits based on usage

5. **HTTPS Only**
   - Force SSL redirect
   - Use strong cipher suites

## Cost Optimization

### Estimated Monthly Costs
- **AWS EC2 t3.medium**: ~$30/month
- **GCP Cloud Run**: ~$10-50/month (pay per use)
- **DigitalOcean Droplet**: $24/month (4GB)
- **Azure Container**: ~$40/month
- **Heroku**: Free-$25/month

### Tips to Reduce Costs
1. Use spot/preemptible instances
2. Enable auto-scaling
3. Use serverless for sporadic traffic
4. Compress Docker images
5. Cache aggressively

## Support

For deployment issues:
1. Check logs: `docker-compose logs`
2. Review environment variables
3. Verify API keys are correct
4. Check firewall rules
5. Monitor resource usage

## Quick Commands Reference

```bash
# Start services
docker-compose up -d

# Stop services
docker-compose down

# View logs
docker-compose logs -f

# Rebuild after changes
docker-compose build

# Enter container shell
docker exec -it treasurehunter-app bash

# Check health
curl http://localhost:5000/api/status

# Backup data
tar -czf backup.tar.gz saved_models/ data/

# Update code
git pull && docker-compose restart
```

---

## 🎯 Next Steps After Deployment

1. **Configure DNS**: Point your domain to server IP
2. **Set up SSL**: Use Let's Encrypt or Cloudflare
3. **Configure backups**: Set up automated backups
4. **Monitor uptime**: Use monitoring service
5. **Test thoroughly**: Run all features in production
6. **Document API keys**: Store securely
7. **Set up alerts**: For errors and downtime

Good luck with your deployment! 🚀