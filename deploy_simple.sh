#!/bin/bash

echo "🚀 TreasureHunter Easy Deployment Script"
echo "========================================"
echo ""
echo "Choose deployment method:"
echo "1) Local (Python)"
echo "2) Docker" 
echo "3) Streamlit Cloud"
echo "4) Heroku"
echo "5) Railway"
echo ""
read -p "Enter choice (1-5): " choice

case $choice in
  1)
    echo "📦 Setting up local deployment..."
    python -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    echo ""
    echo "✅ Setup complete! Run with:"
    echo "   python treasure_api.py"
    echo "   Access at http://localhost:5000"
    ;;
    
  2)
    echo "🐳 Deploying with Docker..."
    docker-compose up -d
    echo "✅ Running at http://localhost:8080"
    ;;
    
  3)
    echo "☁️ Deploying to Streamlit Cloud..."
    echo ""
    echo "1. Fork this repo to GitHub"
    echo "2. Go to share.streamlit.io"
    echo "3. Connect your GitHub"
    echo "4. Deploy streamlit_app.py"
    echo "5. Add MAPBOX_ACCESS_TOKEN in Secrets"
    echo ""
    echo "Or run locally:"
    pip install streamlit streamlit-folium
    streamlit run streamlit_app.py
    ;;
    
  4)
    echo "🚀 Deploying to Heroku..."
    cat > Procfile << 'EOF'
web: python treasure_api.py
EOF
    cat > runtime.txt << 'EOF'
python-3.11.0
EOF
    heroku create treasurehunter-app
    heroku config:set MAPBOX_ACCESS_TOKEN=your_token_here
    git add .
    git commit -m "Deploy to Heroku"
    git push heroku main
    echo "✅ Deployed to Heroku!"
    ;;
    
  5)
    echo "🚂 Deploying to Railway..."
    cat > railway.json << 'EOF'
{
  "$schema": "https://railway.app/railway.schema.json",
  "build": {
    "builder": "NIXPACKS"
  },
  "deploy": {
    "startCommand": "python treasure_api.py",
    "restartPolicyType": "ON_FAILURE",
    "restartPolicyMaxRetries": 10
  }
}
EOF
    echo ""
    echo "1. Go to railway.app"
    echo "2. New Project → Deploy from GitHub"
    echo "3. Add environment variable:"
    echo "   MAPBOX_ACCESS_TOKEN=your_token"
    echo "4. Deploy!"
    ;;
esac