#!/bin/bash

# AWS EC2 Deployment Script for TreasureHunter
# This script deploys the application to AWS EC2 with Docker

set -e

echo "🚀 TreasureHunter AWS Deployment Script"
echo "========================================"

# Configuration
AWS_REGION=${AWS_REGION:-"us-west-2"}
INSTANCE_TYPE=${INSTANCE_TYPE:-"t3.medium"}  # t3.medium for GPU: g4dn.xlarge
KEY_NAME=${KEY_NAME:-"treasurehunter-key"}
SECURITY_GROUP=${SECURITY_GROUP:-"treasurehunter-sg"}
APP_NAME="treasurehunter"

# Check AWS CLI is installed
if ! command -v aws &> /dev/null; then
    echo "❌ AWS CLI not found. Please install it first."
    exit 1
fi

# Function to create security group
create_security_group() {
    echo "📦 Creating security group..."
    
    # Check if security group exists
    if aws ec2 describe-security-groups --group-names $SECURITY_GROUP --region $AWS_REGION 2>/dev/null; then
        echo "Security group already exists"
    else
        # Create security group
        aws ec2 create-security-group \
            --group-name $SECURITY_GROUP \
            --description "Security group for TreasureHunter app" \
            --region $AWS_REGION
        
        # Add rules
        aws ec2 authorize-security-group-ingress \
            --group-name $SECURITY_GROUP \
            --protocol tcp --port 22 --cidr 0.0.0.0/0 \
            --region $AWS_REGION
        
        aws ec2 authorize-security-group-ingress \
            --group-name $SECURITY_GROUP \
            --protocol tcp --port 80 --cidr 0.0.0.0/0 \
            --region $AWS_REGION
        
        aws ec2 authorize-security-group-ingress \
            --group-name $SECURITY_GROUP \
            --protocol tcp --port 443 --cidr 0.0.0.0/0 \
            --region $AWS_REGION
    fi
}

# Function to launch EC2 instance
launch_instance() {
    echo "🖥️ Launching EC2 instance..."
    
    # Get latest Amazon Linux 2 AMI
    AMI_ID=$(aws ec2 describe-images \
        --owners amazon \
        --filters "Name=name,Values=amzn2-ami-hvm-*-x86_64-gp2" \
        --query 'sort_by(Images, &CreationDate)[-1].ImageId' \
        --output text \
        --region $AWS_REGION)
    
    echo "Using AMI: $AMI_ID"
    
    # Create user data script
    cat > user_data.sh << 'EOF'
#!/bin/bash
# Update system
yum update -y

# Install Docker
amazon-linux-extras install docker -y
service docker start
usermod -a -G docker ec2-user

# Install Docker Compose
curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
chmod +x /usr/local/bin/docker-compose

# Install Git
yum install git -y

# Clone repository (update with your repo URL)
cd /home/ec2-user
git clone https://github.com/yourusername/treasurehunter.git
cd treasurehunter

# Create .env file (you'll need to add your keys)
cp .env.example .env

# Start application
docker-compose up -d

# Install certbot for SSL
amazon-linux-extras install epel -y
yum install certbot -y
EOF
    
    # Launch instance
    INSTANCE_ID=$(aws ec2 run-instances \
        --image-id $AMI_ID \
        --instance-type $INSTANCE_TYPE \
        --key-name $KEY_NAME \
        --security-groups $SECURITY_GROUP \
        --user-data file://user_data.sh \
        --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=$APP_NAME}]" \
        --query 'Instances[0].InstanceId' \
        --output text \
        --region $AWS_REGION)
    
    echo "Instance launched: $INSTANCE_ID"
    
    # Wait for instance to be running
    echo "Waiting for instance to start..."
    aws ec2 wait instance-running --instance-ids $INSTANCE_ID --region $AWS_REGION
    
    # Get public IP
    PUBLIC_IP=$(aws ec2 describe-instances \
        --instance-ids $INSTANCE_ID \
        --query 'Reservations[0].Instances[0].PublicIpAddress' \
        --output text \
        --region $AWS_REGION)
    
    echo "✅ Instance is running!"
    echo "Public IP: $PUBLIC_IP"
    echo ""
    echo "Next steps:"
    echo "1. SSH into the instance: ssh -i ${KEY_NAME}.pem ec2-user@${PUBLIC_IP}"
    echo "2. Configure environment variables in /home/ec2-user/treasurehunter/.env"
    echo "3. Restart the application: cd treasurehunter && docker-compose restart"
    echo "4. Set up SSL certificate with: sudo certbot certonly --standalone -d yourdomain.com"
    
    # Clean up
    rm -f user_data.sh
}

# Function to deploy updates
deploy_update() {
    if [ -z "$1" ]; then
        echo "Usage: $0 deploy <instance-ip>"
        exit 1
    fi
    
    INSTANCE_IP=$1
    
    echo "📤 Deploying to $INSTANCE_IP..."
    
    # Create deployment package
    tar -czf treasurehunter.tar.gz \
        --exclude='.git' \
        --exclude='*.pyc' \
        --exclude='__pycache__' \
        --exclude='venv' \
        --exclude='.env' \
        --exclude='saved_models/*' \
        --exclude='logs/*' \
        .
    
    # Copy to server
    scp -i ${KEY_NAME}.pem treasurehunter.tar.gz ec2-user@${INSTANCE_IP}:/tmp/
    
    # Deploy on server
    ssh -i ${KEY_NAME}.pem ec2-user@${INSTANCE_IP} << 'ENDSSH'
cd /home/ec2-user/treasurehunter
docker-compose down
tar -xzf /tmp/treasurehunter.tar.gz
docker-compose build
docker-compose up -d
rm /tmp/treasurehunter.tar.gz
echo "✅ Deployment complete!"
ENDSSH
    
    # Clean up local file
    rm treasurehunter.tar.gz
}

# Main script logic
case "$1" in
    launch)
        create_security_group
        launch_instance
        ;;
    deploy)
        deploy_update $2
        ;;
    *)
        echo "Usage: $0 {launch|deploy <instance-ip>}"
        echo ""
        echo "Commands:"
        echo "  launch - Create new EC2 instance and deploy"
        echo "  deploy - Deploy updates to existing instance"
        exit 1
        ;;
esac