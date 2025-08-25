# CNN Training Cell - Add this after Cell 16 in satellite_production_production_strict_final.ipynb

"""
# CNN Model Training for Archaeological Site Detection
'''
Train the CNN model using labeled satellite imagery data.
Requires GPU runtime for efficient training.
'''

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Check GPU availability
try:
    TORCH_AVAILABLE
except NameError:
    try:
        import torch  # type: ignore
        TORCH_AVAILABLE = True
    except Exception:
        TORCH_AVAILABLE = False

if TORCH_AVAILABLE:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🎮 Using device: {device}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print("❌ PyTorch not available. Install with: pip install torch torchvision")

def create_training_dataset(num_samples=1000):
    '''
    Create synthetic training dataset from known archaeological sites.
    In production, replace with real labeled satellite imagery.
    '''
    
    print("📊 Creating training dataset...")
    
    # Known archaeological sites (positive samples)
    archaeological_sites = [
        (29.9792, 31.1342),   # Giza Pyramids
        (13.1631, 72.5450),   # Machu Picchu
        (13.4125, 103.8670),  # Angkor Wat
        (20.6843, -88.5678),  # Chichen Itza
        (30.3285, 35.4444),   # Petra
        (51.1789, -1.8262),   # Stonehenge
        (27.1751, 78.0421),   # Taj Mahal
        (41.8902, 12.4922),   # Colosseum
    ]
    
    # Non-archaeological sites (negative samples)
    non_sites = [
        (40.7128, -74.0060),  # New York City
        (51.5074, -0.1278),   # London
        (35.6762, 139.6503),  # Tokyo
        (48.8566, 2.3522),    # Paris
        (-33.8688, 151.2093), # Sydney
        (37.7749, -122.4194), # San Francisco
        (41.8781, -87.6298),  # Chicago
        (34.0522, -118.2437), # Los Angeles
    ]
    
    X_data = []
    y_data = []
    
    # Generate positive samples
    samples_per_site = num_samples // (2 * len(archaeological_sites))
    
    for lat, lon in archaeological_sites:
        for _ in range(samples_per_site):
            try:
                # Fetch real satellite data
                # Use deterministic offset for reproducibility
                offset = (_ - samples_per_site/2) * 0.002
                img_data = fetch_satellite_image(lat + offset, 
                                                lon + offset, 
                                                size=IMAGE_SIZE)
                X_data.append(img_data)
                y_data.append(1)  # Positive label
            except Exception as e:
                # If fetch fails, skip this sample - training requires real data
                print(f"Warning: Failed to fetch satellite data for ({lat}, {lon}): {e}")
                continue
    
    # Generate negative samples
    for lat, lon in non_sites:
        for _ in range(samples_per_site):
            try:
                # Fetch real satellite data
                # Use deterministic offset for reproducibility
                offset = (_ - samples_per_site/2) * 0.002
                img_data = fetch_satellite_image(lat + offset,
                                                lon + offset,
                                                size=IMAGE_SIZE)
                X_data.append(img_data)
                y_data.append(0)  # Negative label
            except Exception as e:
                # If fetch fails, skip this sample - training requires real data
                print(f"Warning: Failed to fetch satellite data for ({lat}, {lon}): {e}")
                continue
    
    X = np.array(X_data, dtype=np.float32)
    y = np.array(y_data, dtype=np.float32)
    
    print(f"✅ Dataset created: {len(X)} samples")
    print(f"   Positive (archaeological): {np.sum(y == 1)}")
    print(f"   Negative (non-archaeological): {np.sum(y == 0)}")
    
    return X, y

def train_cnn_model(X, y, epochs=50, batch_size=16, learning_rate=0.001):
    '''
    Train the CNN model for archaeological site detection.
    '''
    
    if not TORCH_AVAILABLE:
        print("❌ PyTorch not available for training")
        return None
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\n🏋️ Training CNN Model")
    print(f"   Training samples: {len(X_train)}")
    print(f"   Test samples: {len(X_test)}")
    print(f"   Epochs: {epochs}")
    print(f"   Batch size: {batch_size}")
    print(f"   Learning rate: {learning_rate}")
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test)
    
    # Create data loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    model = SatelliteAnomalyCNN().to(device)
    
    # Loss and optimizer
    criterion = nn.BCELoss()  # Binary cross-entropy
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
    
    # Training history
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []
    
    # Training loop
    print("\n📈 Training Progress:")
    print("-" * 50)
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            
            # Forward pass
            outputs = model(batch_X).squeeze()
            loss = criterion(outputs, batch_y)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            predictions = (outputs > 0.5).float()
            train_correct += (predictions == batch_y).sum().item()
            train_total += batch_y.size(0)
        
        # Testing phase
        model.eval()
        test_loss = 0
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)
                
                outputs = model(batch_X).squeeze()
                loss = criterion(outputs, batch_y)
                
                test_loss += loss.item()
                predictions = (outputs > 0.5).float()
                test_correct += (predictions == batch_y).sum().item()
                test_total += batch_y.size(0)
        
        # Calculate metrics
        avg_train_loss = train_loss / len(train_loader)
        avg_test_loss = test_loss / len(test_loader)
        train_acc = train_correct / train_total
        test_acc = test_correct / test_total
        
        train_losses.append(avg_train_loss)
        test_losses.append(avg_test_loss)
        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)
        
        # Update learning rate
        scheduler.step(avg_test_loss)
        
        # Print progress
        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{epochs}]")
            print(f"  Train Loss: {avg_train_loss:.4f}, Acc: {train_acc:.3f}")
            print(f"  Test Loss: {avg_test_loss:.4f}, Acc: {test_acc:.3f}")
    
    print("\n✅ Training Complete!")
    print(f"Final Test Accuracy: {test_accuracies[-1]:.3f}")
    
    # Plot training history
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(train_losses, label='Train Loss')
    ax1.plot(test_losses, label='Test Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Test Loss')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(train_accuracies, label='Train Accuracy')
    ax2.plot(test_accuracies, label='Test Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Test Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return model

def save_trained_model(model, filepath='archaeological_cnn_model.pth'):
    '''Save the trained model to disk.'''
    if model is not None:
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_architecture': 'SatelliteAnomalyCNN',
            'num_channels': NUM_CHANNELS,
            'image_size': IMAGE_SIZE,
        }, filepath)
        print(f"✅ Model saved to {filepath}")
        return filepath
    return None

def load_trained_model(filepath='archaeological_cnn_model.pth'):
    '''Load a trained model from disk.'''
    if not TORCH_AVAILABLE:
        print("❌ PyTorch not available")
        return None
    
    try:
        checkpoint = torch.load(filepath, map_location=device)
        model = SatelliteAnomalyCNN()
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        print(f"✅ Model loaded from {filepath}")
        return model
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return None

# Main training pipeline
def run_training_pipeline(num_samples=500, epochs=30):
    '''
    Complete training pipeline for the CNN model.
    '''
    print("="*60)
    print("🚀 CNN TRAINING PIPELINE")
    print("="*60)
    
    # Step 1: Create dataset
    X, y = create_training_dataset(num_samples)
    
    # Step 2: Train model
    model = train_cnn_model(X, y, epochs=epochs)
    
    # Step 3: Save model
    if model is not None:
        filepath = save_trained_model(model)
        
        # Update global model
        global satellite_cnn
        satellite_cnn = model
        print("\n✅ Global CNN model updated with trained weights")
    
    return model

# Quick training command
print("="*60)
print("🎮 CNN TRAINING MODULE LOADED")
print("="*60)
print("\nTo train the CNN model, run:")
print("  model = run_training_pipeline(num_samples=500, epochs=30)")
print("\nFor quick test (fewer samples):")
print("  model = run_training_pipeline(num_samples=100, epochs=10)")
print("\nTo load a saved model:")
print("  model = load_trained_model('archaeological_cnn_model.pth')")
print("="*60)
"""