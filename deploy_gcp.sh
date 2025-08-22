#!/bin/bash

# Google Cloud Platform Deployment Script for TreasureHunter
# Deploys to Google Cloud Run (serverless) or Compute Engine

set -e

echo "🚀 TreasureHunter GCP Deployment Script"
echo "========================================"

# Configuration
PROJECT_ID=${GCP_PROJECT_ID:-"your-project-id"}
REGION=${GCP_REGION:-"us-central1"}
SERVICE_NAME="treasurehunter"
IMAGE_NAME="gcr.io/${PROJECT_ID}/${SERVICE_NAME}"

# Check gcloud CLI is installed
if ! command -v gcloud &> /dev/null; then
    echo "❌ gcloud CLI not found. Please install it first."
    echo "Visit: https://cloud.google.com/sdk/docs/install"
    exit 1
fi

# Function to deploy to Cloud Run (Serverless)
deploy_cloud_run() {
    echo "☁️ Deploying to Google Cloud Run..."
    
    # Enable required APIs
    echo "Enabling required APIs..."
    gcloud services enable run.googleapis.com \
        cloudbuild.googleapis.com \
        artifactregistry.googleapis.com \
        --project=${PROJECT_ID}
    
    # Build and push Docker image
    echo "Building Docker image..."
    gcloud builds submit --tag ${IMAGE_NAME} \
        --project=${PROJECT_ID} \
        --timeout=20m
    
    # Deploy to Cloud Run
    echo "Deploying to Cloud Run..."
    gcloud run deploy ${SERVICE_NAME} \
        --image ${IMAGE_NAME} \
        --platform managed \
        --region ${REGION} \
        --allow-unauthenticated \
        --memory 2Gi \
        --cpu 2 \
        --timeout 300 \
        --max-instances 10 \
        --min-instances 1 \
        --port 5000 \
        --set-env-vars "PRODUCTION_MODE=true" \
        --set-env-vars "FLASK_ENV=production" \
        --project=${PROJECT_ID}
    
    # Get service URL
    SERVICE_URL=$(gcloud run services describe ${SERVICE_NAME} \
        --platform managed \
        --region ${REGION} \
        --format 'value(status.url)' \
        --project=${PROJECT_ID})
    
    echo "✅ Deployment complete!"
    echo "Service URL: ${SERVICE_URL}"
}

# Function to deploy to Compute Engine (VM)
deploy_compute_engine() {
    echo "🖥️ Deploying to Google Compute Engine..."
    
    INSTANCE_NAME="${SERVICE_NAME}-vm"
    ZONE="${REGION}-a"
    
    # Create firewall rules
    echo "Creating firewall rules..."
    gcloud compute firewall-rules create ${SERVICE_NAME}-allow-http \
        --allow tcp:80 \
        --source-ranges 0.0.0.0/0 \
        --target-tags ${SERVICE_NAME} \
        --project=${PROJECT_ID} 2>/dev/null || true
    
    gcloud compute firewall-rules create ${SERVICE_NAME}-allow-https \
        --allow tcp:443 \
        --source-ranges 0.0.0.0/0 \
        --target-tags ${SERVICE_NAME} \
        --project=${PROJECT_ID} 2>/dev/null || true
    
    # Create startup script
    cat > startup-script.sh << 'EOF'
#!/bin/bash
# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh
usermod -aG docker $USER

# Install Docker Compose
curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
chmod +x /usr/local/bin/docker-compose

# Clone repository (update with your repo)
cd /opt
git clone https://github.com/yourusername/treasurehunter.git
cd treasurehunter

# Set up environment
cp .env.example .env
# TODO: Add your environment variables

# Start application
docker-compose up -d
EOF
    
    # Create instance
    echo "Creating VM instance..."
    gcloud compute instances create ${INSTANCE_NAME} \
        --zone=${ZONE} \
        --machine-type=e2-standard-2 \
        --network-interface=network-tier=PREMIUM,subnet=default \
        --maintenance-policy=MIGRATE \
        --provisioning-model=STANDARD \
        --tags=${SERVICE_NAME} \
        --create-disk=auto-delete=yes,boot=yes,device-name=${INSTANCE_NAME},image=projects/ubuntu-os-cloud/global/images/ubuntu-2204-lts,mode=rw,size=50,type=pd-balanced \
        --metadata-from-file startup-script=startup-script.sh \
        --project=${PROJECT_ID}
    
    # Get external IP
    EXTERNAL_IP=$(gcloud compute instances describe ${INSTANCE_NAME} \
        --zone=${ZONE} \
        --format='get(networkInterfaces[0].accessConfigs[0].natIP)' \
        --project=${PROJECT_ID})
    
    echo "✅ VM created!"
    echo "External IP: ${EXTERNAL_IP}"
    echo ""
    echo "Next steps:"
    echo "1. SSH into the instance: gcloud compute ssh ${INSTANCE_NAME} --zone=${ZONE}"
    echo "2. Configure environment variables in /opt/treasurehunter/.env"
    echo "3. Restart the application: cd /opt/treasurehunter && docker-compose restart"
    
    # Clean up
    rm -f startup-script.sh
}

# Function to deploy to Google Kubernetes Engine (GKE)
deploy_gke() {
    echo "⚓ Deploying to Google Kubernetes Engine..."
    
    CLUSTER_NAME="${SERVICE_NAME}-cluster"
    
    # Create GKE cluster if it doesn't exist
    if ! gcloud container clusters describe ${CLUSTER_NAME} --zone=${ZONE} --project=${PROJECT_ID} 2>/dev/null; then
        echo "Creating GKE cluster..."
        gcloud container clusters create ${CLUSTER_NAME} \
            --zone=${ZONE} \
            --num-nodes=2 \
            --machine-type=e2-standard-2 \
            --enable-autoscaling \
            --min-nodes=1 \
            --max-nodes=5 \
            --project=${PROJECT_ID}
    fi
    
    # Get cluster credentials
    gcloud container clusters get-credentials ${CLUSTER_NAME} \
        --zone=${ZONE} \
        --project=${PROJECT_ID}
    
    # Build and push image
    echo "Building Docker image..."
    gcloud builds submit --tag ${IMAGE_NAME} \
        --project=${PROJECT_ID}
    
    # Create Kubernetes deployment file
    cat > k8s-deployment.yaml << EOF
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ${SERVICE_NAME}
spec:
  replicas: 2
  selector:
    matchLabels:
      app: ${SERVICE_NAME}
  template:
    metadata:
      labels:
        app: ${SERVICE_NAME}
    spec:
      containers:
      - name: ${SERVICE_NAME}
        image: ${IMAGE_NAME}
        ports:
        - containerPort: 5000
        env:
        - name: PRODUCTION_MODE
          value: "true"
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
---
apiVersion: v1
kind: Service
metadata:
  name: ${SERVICE_NAME}-service
spec:
  type: LoadBalancer
  selector:
    app: ${SERVICE_NAME}
  ports:
  - port: 80
    targetPort: 5000
EOF
    
    # Apply Kubernetes configuration
    kubectl apply -f k8s-deployment.yaml
    
    # Wait for external IP
    echo "Waiting for external IP..."
    sleep 30
    EXTERNAL_IP=$(kubectl get service ${SERVICE_NAME}-service -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
    
    echo "✅ GKE deployment complete!"
    echo "External IP: ${EXTERNAL_IP}"
    
    # Clean up
    rm -f k8s-deployment.yaml
}

# Function to set up Cloud SQL (if needed)
setup_cloud_sql() {
    echo "🗄️ Setting up Cloud SQL..."
    
    INSTANCE_NAME="${SERVICE_NAME}-db"
    
    # Create Cloud SQL instance
    gcloud sql instances create ${INSTANCE_NAME} \
        --database-version=POSTGRES_14 \
        --tier=db-f1-micro \
        --region=${REGION} \
        --project=${PROJECT_ID}
    
    # Create database
    gcloud sql databases create treasurehunter \
        --instance=${INSTANCE_NAME} \
        --project=${PROJECT_ID}
    
    # Set root password
    gcloud sql users set-password root \
        --host=% \
        --instance=${INSTANCE_NAME} \
        --password=your-secure-password \
        --project=${PROJECT_ID}
    
    echo "✅ Cloud SQL setup complete!"
}

# Main script logic
case "$1" in
    cloudrun)
        deploy_cloud_run
        ;;
    compute)
        deploy_compute_engine
        ;;
    gke)
        deploy_gke
        ;;
    sql)
        setup_cloud_sql
        ;;
    *)
        echo "Usage: $0 {cloudrun|compute|gke|sql}"
        echo ""
        echo "Deployment options:"
        echo "  cloudrun - Deploy to Cloud Run (serverless, auto-scaling)"
        echo "  compute  - Deploy to Compute Engine (traditional VM)"
        echo "  gke      - Deploy to Google Kubernetes Engine"
        echo "  sql      - Set up Cloud SQL database"
        echo ""
        echo "Make sure to set GCP_PROJECT_ID environment variable"
        exit 1
        ;;
esac