import os
import numpy as np
from scipy.io import wavfile
from scipy.signal import convolve
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def load_wav_file(file_path):
    _, data = wavfile.read(file_path)
    return data.astype(float)

def apply_smoothing_filter(data, filter_size=256):
    kernel = np.ones(filter_size) / filter_size
    return convolve(data, kernel, mode='same')

def create_sequences(data, sequence_length):
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i+sequence_length])
        y.append(data[i+sequence_length])
    return np.array(X), np.array(y)

class MLP(nn.Module):
    def __init__(self, input_size):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 100),
            nn.ReLU(),
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, 1)
        )

    def forward(self, x):
        return self.layers(x)

def train_model(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for X_batch, y_batch in tqdm(train_loader, desc="Training"):
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(X_batch).squeeze()
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

def evaluate_model(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for X_batch, y_batch in tqdm(test_loader, desc="Evaluating"):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch).squeeze()
            loss = criterion(outputs, y_batch)
            total_loss += loss.item()
    return total_loss / len(test_loader)

def evaluate_models(folder_path, sequence_length=100, sample_rate=.01, batch_size=64, epochs=1, filter_size=256):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    all_data = []
    for file_name in tqdm(os.listdir(folder_path), desc="Loading and processing files"):
        if file_name.endswith('.wav'):
            file_path = os.path.join(folder_path, file_name)
            data = load_wav_file(file_path)
            
            # Apply smoothing filter
            smoothed_data = apply_smoothing_filter(data, filter_size)
            
            # Sampling down to 1%
            sampled_indices = np.random.choice(len(smoothed_data), size=int(len(smoothed_data) * sample_rate), replace=False)
            sampled_data = smoothed_data[sampled_indices]
            
            all_data.extend(sampled_data)

    print(f"Total sampled data points: {len(all_data)}")

    X, y = create_sequences(all_data, sequence_length)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Convert to PyTorch tensors
    X_train = torch.FloatTensor(X_train)
    y_train = torch.FloatTensor(y_train)
    X_test = torch.FloatTensor(X_test)
    y_test = torch.FloatTensor(y_test)

    # Create DataLoaders
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Train autoregressive MLP
    model = MLP(sequence_length).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())

    for epoch in range(epochs):
        train_loss = train_model(model, train_loader, criterion, optimizer, device)
        test_loss = evaluate_model(model, test_loader, criterion, device)
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")

    # Final evaluation
    mlp_mse = evaluate_model(model, test_loader, criterion, device)

    # Average prediction
    avg_prediction = torch.mean(y_train)
    avg_mse = criterion(torch.full_like(y_test, avg_prediction), y_test).item()

    # Calculate difference in MSE
    mse_difference = avg_mse - mlp_mse

    print(f"Autoregressive MLP MSE: {mlp_mse}")
    print(f"Average Prediction MSE: {avg_mse}")
    print(f"Difference in MSE (Average - MLP): {mse_difference}")

    return mse_difference

# Usage
folder_path = 'data'
mse_difference = evaluate_models(folder_path)
print(f"The autoregressive MLP outperforms the average prediction by {mse_difference} MSE")